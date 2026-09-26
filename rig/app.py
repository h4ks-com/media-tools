"""HTTP service that rigs a humanoid GLB with the UniRig vroid skeleton.

Reachable only from n8n, with no egress of its own: every tool and model weight is
baked into the image at build time. One request runs on the GPU at a time.
"""

import os
import subprocess
import tempfile
import threading
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

gpu_slot = threading.Lock()

app = FastAPI(title="unirig", docs_url=None, redoc_url=None, openapi_url=None)


class RigError(Exception):
    pass


def run_stage(command: list[str], cwd: Path | None = None) -> None:
    try:
        result = subprocess.run(  # nosec B603
            command, cwd=cwd, capture_output=True, timeout=STAGE_TIMEOUT_SECONDS, check=False
        )
    except subprocess.TimeoutExpired as error:
        raise RigError(f"{command[0]} timed out") from error
    if result.returncode != 0:
        raise RigError(result.stderr.decode(errors="replace")[-2000:] or f"{command[0]} failed")


def rig(data: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as workdir_name:
        workdir = Path(workdir_name)
        source = workdir / "input.glb"
        decompressed = workdir / "decompressed.glb"
        skeleton = workdir / "skeleton.fbx"
        skin = workdir / "skin.fbx"
        rigged = workdir / "rigged.glb"
        source.write_bytes(data)

        run_stage([GLTFPACK, "-i", str(source), "-o", str(decompressed), "-noq"])
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
                "--npz_dir",
                str(workdir / "npz_skeleton"),
            ],
            cwd=UNIRIG_DIR,
        )
        run_stage(
            [
                "bash",
                "launch/inference/generate_skin.sh",
                "--input",
                str(skeleton),
                "--output",
                str(skin),
                "--npz_dir",
                str(workdir / "npz_skin"),
            ],
            cwd=UNIRIG_DIR,
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
            cwd=UNIRIG_DIR,
        )
        if not rigged.exists():
            raise RigError("no rigged output was produced")
        return rigged.read_bytes()


@app.exception_handler(HTTPException)
async def plain_error(_: Request, error: HTTPException) -> PlainTextResponse:
    return PlainTextResponse(str(error.detail), status_code=error.status_code)


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"


@app.post("/rig")
async def rig_route(request: Request) -> Response:
    declared = request.headers.get("content-length", "")
    if not declared.isdigit() or not 0 < int(declared) <= MAX_MODEL_BYTES:
        raise HTTPException(
            400, f"send a GLB body of 1 to {MAX_MODEL_BYTES} bytes with its Content-Length"
        )
    body = await request.body()
    if len(body) != int(declared) or body[:4] != b"glTF":
        raise HTTPException(400, "send a binary GLB (glTF magic header)")

    def locked() -> bytes:
        with gpu_slot:
            return rig(body)

    try:
        glb = await run_in_threadpool(locked)
    except RigError as error:
        raise HTTPException(422, str(error)) from error
    return Response(glb, media_type="model/gltf-binary")
