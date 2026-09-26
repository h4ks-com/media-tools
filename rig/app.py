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
            if not missing:
                return rigged
            problem = "the skeleton missed " + ", ".join(sorted(missing))
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
