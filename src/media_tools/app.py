"""HTTP routes for the n8n workflows, reachable only inside the cluster.

Each POST takes raw file bytes as its body, since n8n sends one binary body per request; several
files go back to back with their byte lengths in the `lengths` query.
"""

import os
import threading
from collections.abc import Callable
from typing import Annotated

import numpy
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi.responses import PlainTextResponse
from fastapi.responses import Response
from PIL import Image
from PIL import UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from media_tools import mesh
from media_tools import pictures
from media_tools import poses
from media_tools.glb import GlbError
from media_tools.retarget import RetargetError
from media_tools.retarget import retarget

MAX_PICTURE_BYTES = 30 * 1024 * 1024
MAX_MODEL_BYTES = 300 * 1024 * 1024
MAX_MOTION_BYTES = 5 * 1024 * 1024
MAX_FRAMES = 16
MAX_CLIPS = 8
GLTFPACK = os.environ.get("GLTFPACK", "/opt/tools/gltfpack")
TIMEOUT_SECONDS = float(os.environ.get("TIMEOUT_SECONDS", "600"))
PICTURE_ERRORS = (ValueError, UnidentifiedImageError, Image.DecompressionBombError)

cutter = pictures.Cutter(os.environ.get("CUTOUT_MODEL", "/opt/tools/isnet-general-use.onnx"))
# We run one heavy job at a time, since a cutout or a big mesh can take gigabytes.
heavy_slot = threading.Lock()

app = FastAPI(title="media tools", docs_url=None, redoc_url=None, openapi_url=None)


def in_slot[T](work: Callable[[], T]) -> T:
    with heavy_slot:
        return work()


async def read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if not declared.isdigit() or not 0 < int(declared) <= limit:
        raise HTTPException(400, f"send a body of 1 to {limit} bytes with its Content-Length")
    body = await request.body()
    if len(body) != int(declared):
        raise HTTPException(400, "upload ended early")
    return body


def split_body(body: bytes, lengths: str, count: range) -> list[bytes]:
    try:
        sizes = [int(value) for value in lengths.split(",") if value]
    except ValueError as error:
        raise HTTPException(400, "lengths is a comma separated list of byte counts") from error
    if len(sizes) not in count or any(size < 1 for size in sizes) or sum(sizes) != len(body):
        raise HTTPException(400, "send the files back to back with their byte lengths in lengths")
    offsets = [sum(sizes[:index]) for index in range(len(sizes))]
    return [body[offset : offset + size] for offset, size in zip(offsets, sizes, strict=True)]


@app.exception_handler(HTTPException)
async def plain_error(_: Request, error: HTTPException) -> PlainTextResponse:
    return PlainTextResponse(str(error.detail), status_code=error.status_code)


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"


@app.get("/pose")
async def pose(
    move: str,
    frames: Annotated[int, Query(ge=2, le=MAX_FRAMES)] = 4,
    frame: int = 0,
) -> Response:
    try:
        png = poses.skeleton_png(move, frames, frame)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return Response(png, media_type="image/png")


@app.post("/cutout")
async def cutout(request: Request, method: pictures.CutMethod = "isnet") -> Response:
    picture = await read_body(request, MAX_PICTURE_BYTES)
    try:
        png = await run_in_threadpool(in_slot, lambda: cutter.cut(picture, method))
    except PICTURE_ERRORS as error:
        raise HTTPException(400, str(error)) from error
    return Response(png, media_type="image/png")


@app.post("/pixelate")
async def pixelate(
    request: Request,
    size: Annotated[int, Query(ge=8, le=512)] = 64,
    colors: Annotated[int, Query(ge=2, le=256)] = 16,
    scale: Annotated[int, Query(ge=1, le=32)] = 1,
) -> Response:
    picture = await read_body(request, MAX_PICTURE_BYTES)
    try:
        png = await run_in_threadpool(pictures.pixel_art, picture, size, colors, scale)
    except PICTURE_ERRORS as error:
        raise HTTPException(400, str(error)) from error
    return Response(png, media_type="image/png")


@app.post("/sprite-frames")
async def sprite_frames(
    request: Request,
    lengths: str,
    format: Annotated[str, Query(pattern="^(png|gif)$")] = "png",
    size: Annotated[int, Query(ge=8, le=512)] = 64,
    colors: Annotated[int, Query(ge=2, le=256)] = 24,
    scale: Annotated[int, Query(ge=1, le=32)] = 1,
) -> Response:
    body = await read_body(request, MAX_FRAMES * MAX_PICTURE_BYTES)
    frames = split_body(body, lengths, range(2, MAX_FRAMES + 1))
    render = pictures.sprite_gif if format == "gif" else pictures.sprite_sheet
    try:
        data = await run_in_threadpool(render, frames, size, colors, scale)
    except (UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise HTTPException(400, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return Response(data, media_type=f"image/{format}")


@app.post("/mesh")
async def simplify_mesh(
    request: Request, triangles: Annotated[int, Query(ge=1000, le=5_000_000)] = 300_000
) -> Response:
    model = await read_body(request, MAX_MODEL_BYTES)
    try:
        glb = await run_in_threadpool(
            in_slot, lambda: mesh.simplify(model, triangles, GLTFPACK, TIMEOUT_SECONDS)
        )
    except (GlbError, KeyError, IndexError) as error:
        raise HTTPException(400, f"not a GLB: {error}") from error
    except mesh.MeshError as error:
        raise HTTPException(422, str(error)) from error
    return Response(glb, media_type="model/gltf-binary")


@app.post("/retarget")
async def retarget_motions(
    request: Request, lengths: str, names: str, in_place: bool = True
) -> Response:
    body = await read_body(request, MAX_MODEL_BYTES + MAX_CLIPS * MAX_MOTION_BYTES)
    parts = split_body(body, lengths, range(2, MAX_CLIPS + 2))
    clip_names = [name.strip() for name in names.split(",")]
    if len(clip_names) != len(parts) - 1 or not all(clip_names):
        raise HTTPException(400, "give one clip name per motion in names")
    if len(parts[0]) > MAX_MODEL_BYTES or any(len(part) > MAX_MOTION_BYTES for part in parts[1:]):
        raise HTTPException(400, "the character or a motion is too large")
    motions = list(zip(clip_names, parts[1:], strict=True))
    try:
        glb = await run_in_threadpool(in_slot, lambda: retarget(parts[0], motions, in_place))
    except (GlbError, RetargetError, KeyError, IndexError, numpy.linalg.LinAlgError) as error:
        raise HTTPException(422, f"cannot retarget: {error}") from error
    return Response(glb, media_type="model/gltf-binary")
