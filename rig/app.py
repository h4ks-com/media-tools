"""HTTP service that rigs a humanoid GLB with the UniRig vroid skeleton.

Reachable only from n8n, with no egress of its own: every tool is baked into the image
and the pod mounts the model weights read-only. One request runs on the GPU at a time.
"""

import json
import os
import re
import shutil
import signal
import struct
import subprocess  # nosec B404: we run only gltfpack and UniRig's scripts, with arguments we build
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import PlainTextResponse
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

UNIRIG_DIR = Path(os.environ.get("UNIRIG_DIR", "/opt/unirig/repo"))
GLTFPACK = os.environ.get("GLTFPACK", "/opt/tools/gltfpack")
MAX_MODEL_BYTES = 300 * 1024 * 1024
STAGE_TIMEOUT_SECONDS = float(os.environ.get("STAGE_TIMEOUT_SECONDS", "600"))
SKELETON_TASK = "configs/task/quick_inference_skeleton_vroid_forced.yaml"
MAX_IN_FLIGHT = 2
RIG_ATTEMPTS = 3
# UniRig's VRoid template lists its 22 bones in this order, each with the parent at that index.
VROID_ORDER = (
    "J_Bip_C_Hips",
    "J_Bip_C_Spine",
    "J_Bip_C_Chest",
    "J_Bip_C_UpperChest",
    "J_Bip_C_Neck",
    "J_Bip_C_Head",
    "J_Bip_L_Shoulder",
    "J_Bip_L_UpperArm",
    "J_Bip_L_LowerArm",
    "J_Bip_L_Hand",
    "J_Bip_R_Shoulder",
    "J_Bip_R_UpperArm",
    "J_Bip_R_LowerArm",
    "J_Bip_R_Hand",
    "J_Bip_L_UpperLeg",
    "J_Bip_L_LowerLeg",
    "J_Bip_L_Foot",
    "J_Bip_L_ToeBase",
    "J_Bip_R_UpperLeg",
    "J_Bip_R_LowerLeg",
    "J_Bip_R_Foot",
    "J_Bip_R_ToeBase",
)
VROID_PARENTS = (-1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20)
VROID_BONES = frozenset(VROID_ORDER)
GENERIC_BONE = re.compile(r"bone_\d+")
LIMB_BONES = (
    *(
        f"J_Bip_{side}_{bone}"
        for side in ("L", "R")
        for bone in ("UpperArm", "LowerArm", "UpperLeg", "LowerLeg", "Foot")
    ),
    "J_Bip_C_Head",
)
MIN_BONE_SHARE = 0.005
MIN_WEIGHT = 0.1
HEAD_HOLD = 0.5
FLOAT = 5126
WEIGHT_ROW = 16
BIN_CHUNK = 0x004E4942
COMPONENTS = {5121: numpy.uint8, 5123: numpy.uint16, 5126: numpy.float32}
NORMALIZED_MAX = {5121: 255.0, 5123: 65535.0}
JSON_CHUNK = 0x4E4F534A
JSON_START = 20

gpu_slot = threading.Lock()
in_flight = 0

app = FastAPI(title="unirig", docs_url=None, redoc_url=None, openapi_url=None)


class RigError(Exception):
    pass


@contextmanager
def gpu_turn() -> Iterator[None]:
    """Count a request that waits for or holds the GPU, refusing it when too many do.

    :raises HTTPException: 429 when MAX_IN_FLIGHT requests are already in.
    """
    global in_flight
    if in_flight >= MAX_IN_FLIGHT:
        raise HTTPException(429, "busy, try again")
    in_flight += 1
    try:
        yield
    finally:
        in_flight -= 1


def run_stage(command: list[str], produces: Path, cwd: Path | None = None) -> None:
    # UniRig's scripts run python as a child of bash, so on timeout we kill the whole session.
    with subprocess.Popen(  # nosec B603
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=STAGE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise RigError(f"{command[0]} timed out") from error
    if process.returncode != 0:
        output = (stderr or stdout).decode(errors="replace").strip()
        raise RigError(output[-2000:] or f"{command[0]} failed")
    # UniRig's scripts exit 0 even when python fails, so we check for the file each stage makes.
    if not produces.exists():
        output = (stdout + stderr).decode(errors="replace").strip()
        raise RigError(output[-2000:] or f"{command[0]} made no {produces.name}")


def refuse_constant(name: str) -> float:
    raise RigError(f"the GLB JSON holds {name}")


def read_document(data: bytes) -> dict[str, object]:
    """Read the JSON document of a GLB.

    :raises RigError: when the data holds no readable JSON object.
    """
    if len(data) < JSON_START:
        raise RigError("the GLB is too short")
    chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != JSON_CHUNK or JSON_START + chunk_length > len(data):
        raise RigError("the GLB has no JSON chunk")
    try:
        json_chunk = data[JSON_START : JSON_START + chunk_length]
        document = json.loads(json_chunk, parse_constant=refuse_constant)
    except (ValueError, RecursionError) as error:
        raise RigError("the GLB JSON cannot be read") from error
    if not isinstance(document, dict):
        raise RigError("the GLB JSON is not an object")
    return document


def check_embedded(data: bytes) -> None:
    """Refuse a GLB that points at files outside itself, since gltfpack and Blender read them."""
    document = read_document(data)
    for key in ("buffers", "images"):
        entries = document.get(key, [])
        if not isinstance(entries, list):
            raise RigError(f"the GLB {key} are not a list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise RigError(f"the GLB {key} are not a list of objects")
            uri = entry.get("uri")
            if uri is not None and not (isinstance(uri, str) and uri.startswith("data:")):
                raise RigError(f"the GLB {key} must embed their data")


def joint_names(data: bytes) -> set[str]:
    document = read_document(data)
    nodes = document.get("nodes", [])
    skins = document.get("skins", [])
    if not isinstance(nodes, list) or not isinstance(skins, list):
        return set()
    names = set()
    for skin in skins:
        joints = skin.get("joints", []) if isinstance(skin, dict) else []
        for joint in joints if isinstance(joints, list) else []:
            node = nodes[joint] if isinstance(joint, int) and 0 <= joint < len(nodes) else None
            if isinstance(node, dict) and isinstance(node.get("name"), str):
                names.add(node["name"])
    return names


def write_document(data: bytes, document: dict[str, object]) -> bytes:
    """Return the GLB with its JSON chunk replaced by `document`, keeping the binary chunk."""
    old_length = struct.unpack_from("<I", data, 12)[0]
    rest = data[JSON_START + old_length :]
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    chunk = struct.pack("<II", len(text), JSON_CHUNK) + text
    return data[:8] + struct.pack("<I", 12 + len(chunk) + len(rest)) + chunk + rest


def skin_parents(nodes: list[object], joints: list[int]) -> tuple[int, ...]:
    parent_of = {
        child: index
        for index, node in enumerate(nodes)
        if isinstance(node, dict)
        for child in node.get("children", [])
        if isinstance(child, int)
    }
    return tuple(joints.index(parent_of[j]) if parent_of.get(j) in joints else -1 for j in joints)


def name_vroid_bones(data: bytes) -> bytes:
    """Give VRoid names to a skin UniRig left with names bone_0 to bone_21.

    UniRig sometimes writes its VRoid template with generic names, so we rename the bones by
    their place in the template once their parents match it exactly.
    """
    document = read_document(data)
    nodes, skins = document.get("nodes"), document.get("skins")
    if not isinstance(nodes, list) or not isinstance(skins, list) or len(skins) != 1:
        return data
    joints = skins[0].get("joints") if isinstance(skins[0], dict) else None
    if not isinstance(joints, list) or len(joints) != len(VROID_ORDER):
        return data
    if not all(isinstance(j, int) and 0 <= j < len(nodes) for j in joints):
        return data
    named = [nodes[j] for j in joints]
    if not all(
        isinstance(node, dict) and GENERIC_BONE.fullmatch(str(node.get("name"))) for node in named
    ):
        return data
    if skin_parents(nodes, joints) != VROID_PARENTS:
        return data
    for node, name in zip(named, VROID_ORDER, strict=True):
        node["name"] = name
    return write_document(data, document)


def read_attribute(
    document: dict[str, object], binary: bytes, index: object, width: int
) -> numpy.ndarray:
    accessors, views = document.get("accessors"), document.get("bufferViews")
    if not isinstance(accessors, list) or not isinstance(views, list) or not isinstance(index, int):
        raise RigError("the rigged GLB has no readable skin weights")
    accessor = accessors[index]
    view = views[accessor["bufferView"]]
    component = accessor["componentType"]
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    values = numpy.frombuffer(binary, COMPONENTS[component], int(accessor["count"]) * width, start)
    rows: numpy.ndarray = values.reshape(-1, width).astype(numpy.float64)
    if accessor.get("normalized") and component in NORMALIZED_MAX:
        return rows / NORMALIZED_MAX[component]
    return rows


def binary_chunk(data: bytes) -> tuple[int, bytes]:
    """Return where the GLB binary chunk's data starts, and that data."""
    json_length = struct.unpack_from("<I", data, 12)[0]
    bin_start = JSON_START + json_length
    if len(data) < bin_start + 8:
        raise RigError("the rigged GLB has no binary chunk")
    bin_length, bin_type = struct.unpack_from("<II", data, bin_start)
    if bin_type != BIN_CHUNK:
        raise RigError("the rigged GLB has no binary chunk")
    return bin_start + 8, data[bin_start + 8 : bin_start + 8 + bin_length]


def bone_shares(data: bytes) -> dict[str, float]:
    """Return, per skin joint name, the share of skinned vertices it is the main bone of."""
    document = read_document(data)
    _, binary = binary_chunk(data)
    nodes, skins, meshes = document.get("nodes"), document.get("skins"), document.get("meshes")
    if not isinstance(nodes, list) or not isinstance(skins, list) or not isinstance(meshes, list):
        raise RigError("the rigged GLB has no skin")
    names = [nodes[joint]["name"] for joint in skins[0]["joints"]]
    counts = numpy.zeros(len(names))
    for mesh in meshes:
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            if "JOINTS_0" not in attributes or "WEIGHTS_0" not in attributes:
                continue
            joints = read_attribute(document, binary, attributes["JOINTS_0"], 4).astype(int)
            weights = read_attribute(document, binary, attributes["WEIGHTS_0"], 4)
            main = joints[numpy.arange(len(joints)), weights.argmax(axis=1)]
            counts += numpy.bincount(main, minlength=len(names))[: len(names)]
    total = counts.sum()
    return {
        name: float(count / total) if total else 0.0
        for name, count in zip(names, counts, strict=True)
    }


def cleaned(joints: numpy.ndarray, weights: numpy.ndarray, head: int) -> numpy.ndarray:
    """Return skin weights without stray small influences and with the head held rigid.

    UniRig spreads small weights to far bones, which drags belts, straps and faces along with
    them, so we keep each vertex's main bone, drop influences under MIN_WEIGHT and give head
    vertices wholly to the head.
    """
    rows = numpy.arange(len(weights))
    main = weights.argmax(axis=1)
    kept = numpy.where(weights >= MIN_WEIGHT, weights, 0.0)
    kept[rows, main] = weights[rows, main]
    rigid = (joints[rows, main] == head) & (weights[rows, main] >= HEAD_HOLD)
    kept[rigid] = 0.0
    kept[rigid, main[rigid]] = 1.0
    total = kept.sum(axis=1, keepdims=True)
    result: numpy.ndarray = numpy.divide(kept, total, out=weights.copy(), where=total > 0)
    return result


def clean_weights(data: bytes) -> bytes:
    """Return the GLB with its packed float skin weights cleaned; other formats stay as they are."""
    document = read_document(data)
    start, binary = binary_chunk(data)
    nodes, skins, meshes = document.get("nodes"), document.get("skins"), document.get("meshes")
    accessors, views = document.get("accessors"), document.get("bufferViews")
    if not (
        isinstance(nodes, list)
        and isinstance(skins, list)
        and isinstance(meshes, list)
        and isinstance(accessors, list)
        and isinstance(views, list)
    ):
        return data
    names = [nodes[joint]["name"] for joint in skins[0]["joints"]]
    if "J_Bip_C_Head" not in names:
        return data
    head = names.index("J_Bip_C_Head")
    patched = bytearray(binary)
    for mesh in meshes:
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            if "JOINTS_0" not in attributes or "WEIGHTS_0" not in attributes:
                continue
            accessor = accessors[attributes["WEIGHTS_0"]]
            view = views[accessor["bufferView"]]
            if (
                accessor["componentType"] != FLOAT
                or view.get("byteStride", WEIGHT_ROW) != WEIGHT_ROW
            ):
                continue
            joints = read_attribute(document, binary, attributes["JOINTS_0"], 4).astype(int)
            weights = read_attribute(document, binary, attributes["WEIGHTS_0"], 4)
            offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
            values = cleaned(joints, weights, head).astype(numpy.float32).tobytes()
            patched[offset : offset + len(values)] = values
    return data[:start] + bytes(patched) + data[start + len(binary) :]


def weak_limbs(data: bytes) -> list[str]:
    """Return the limb bones that own almost no vertices, which leaves those limbs unskinned."""
    shares = bone_shares(data)
    return [bone for bone in LIMB_BONES if shares.get(bone, 0.0) < MIN_BONE_SHARE]


def rig_once(decompressed: Path, attempt: Path, repo: Path, seed: int) -> bytes:
    attempt.mkdir()
    skeleton = attempt / "skeleton.fbx"
    skin = attempt / "skin.fbx"
    rigged = attempt / "rigged.glb"
    run_stage(
        [
            "bash",
            "launch/inference/generate_skeleton.sh",
            "--input",
            str(decompressed),
            "--output",
            str(skeleton),
            "--skeleton_task",
            SKELETON_TASK,
            "--seed",
            str(seed),
        ],
        skeleton,
        cwd=repo,
    )
    run_stage(
        [
            "bash",
            "launch/inference/generate_skin.sh",
            "--input",
            str(skeleton),
            "--output",
            str(skin),
        ],
        skin,
        cwd=repo,
    )
    run_stage(
        [
            "bash",
            "launch/inference/merge.sh",
            "--source",
            str(skin),
            "--target",
            str(decompressed),
            "--output",
            str(rigged),
        ],
        rigged,
        cwd=repo,
    )
    return rigged.read_bytes()


def rig(data: bytes) -> bytes:
    """Rig a humanoid GLB with the 22 VRoid body bones.

    :raises RigError: when a stage fails or no attempt finds every VRoid bone.
    """
    with tempfile.TemporaryDirectory() as workdir_name:
        workdir = Path(workdir_name)
        source = workdir / "input.glb"
        decompressed = workdir / "decompressed.glb"
        source.write_bytes(data)
        # UniRig writes logs and scratch files inside its own folder, so we run each job in a copy.
        repo = workdir / "repo"
        shutil.copytree(UNIRIG_DIR, repo, symlinks=True)
        run_stage([GLTFPACK, "-i", str(source), "-o", str(decompressed), "-noq"], decompressed)
        # UniRig samples the skeleton, so an attempt can miss bones that a new seed finds.
        problem = ""
        for seed in range(RIG_ATTEMPTS):
            try:
                rigged = rig_once(decompressed, workdir / f"attempt-{seed}", repo, seed)
            except RigError as error:
                problem = str(error)
                continue
            rigged = name_vroid_bones(rigged)
            missing = VROID_BONES - joint_names(rigged)
            if missing:
                problem = "the skeleton missed " + ", ".join(sorted(missing))
                continue
            weak = weak_limbs(rigged)
            if not weak:
                return clean_weights(rigged)
            problem = "the skin left almost no vertices on " + ", ".join(weak)
        raise RigError(f"no humanoid skeleton after {RIG_ATTEMPTS} tries: {problem[-1500:]}")


@app.exception_handler(HTTPException)
async def plain_error(_: Request, error: HTTPException) -> PlainTextResponse:
    return PlainTextResponse(str(error.detail), status_code=error.status_code)


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"


async def read_glb_body(request: Request) -> bytes:
    declared = request.headers.get("content-length", "")
    if not declared.isdigit() or not 0 < int(declared) <= MAX_MODEL_BYTES:
        raise HTTPException(
            400, f"send a GLB body of 1 to {MAX_MODEL_BYTES} bytes with its Content-Length"
        )
    body = await request.body()
    if len(body) != int(declared) or body[:4] != b"glTF":
        raise HTTPException(400, "send a binary GLB (glTF magic header)")
    try:
        check_embedded(body)
    except RigError as error:
        raise HTTPException(400, str(error)) from error
    return body


def rig_on_gpu(data: bytes) -> bytes:
    with gpu_slot:
        return rig(data)


@app.post("/rig")
async def rig_route(request: Request) -> Response:
    with gpu_turn():
        body = await read_glb_body(request)
        try:
            glb = await run_in_threadpool(rig_on_gpu, body)
        except RigError as error:
            raise HTTPException(422, str(error)) from error
    return Response(
        glb,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": 'inline; filename="rigged.glb"'},
    )
