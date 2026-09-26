"""HTTP service that rigs a humanoid GLB with the UniRig vroid skeleton.

Reachable only from n8n, with no egress of its own: every tool is baked into the image
and the pod mounts the model weights read-only. One request runs on the GPU at a time.
"""

import json
import os
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


def check_embedded(data: bytes) -> None:
    """Refuse a GLB that points at files outside itself, since gltfpack and Blender read them."""
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


def rig(data: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as workdir_name:
        workdir = Path(workdir_name)
        source = workdir / "input.glb"
        decompressed = workdir / "decompressed.glb"
        skeleton = workdir / "skeleton.fbx"
        skin = workdir / "skin.fbx"
        rigged = workdir / "rigged.glb"
        source.write_bytes(data)
        # UniRig writes logs and scratch files inside its own folder, so we run each job in a copy.
        repo = workdir / "repo"
        shutil.copytree(UNIRIG_DIR, repo, symlinks=True)

        run_stage([GLTFPACK, "-i", str(source), "-o", str(decompressed), "-noq"], decompressed)
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
