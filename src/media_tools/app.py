"""HTTP routes for the n8n workflows, reachable only inside the cluster.

Each POST takes raw file bytes as its body, since n8n sends one binary body per request; several
files go back to back with their byte lengths in the `lengths` query.
"""

import itertools
import json
import math
import os
import threading
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi.responses import PlainTextResponse
from fastapi.responses import Response
from pydantic import BaseModel
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from media_tools import audio
from media_tools import karaoke
from media_tools import mesh
from media_tools import pictures
from media_tools import poses
from media_tools.cover import cover
from media_tools.ffmpeg import FfmpegError
from media_tools.glb import GlbError
from media_tools.retarget import RetargetError
from media_tools.retarget import retarget

MAX_PICTURE_BYTES = 30 * 1024 * 1024
MAX_MODEL_BYTES = 300 * 1024 * 1024
MAX_MOTION_BYTES = 5 * 1024 * 1024
MAX_FRAMES = 16
MAX_CLIPS = 8
MAX_LIBRARY_CLIPS = 24
MAX_MEDIA_BYTES = 300 * 1024 * 1024
MAX_WORDS = 20_000
MAX_KARAOKE_PICTURES = 24
MAX_CHAPTERS = 100
MAX_TEXT_CHARS = 120
GLTFPACK = os.environ.get("GLTFPACK", "/opt/tools/gltfpack")
COVER_FONT = Path(karaoke.FONTS_DIR) / "LilitaOne-Regular.ttf"
TIMEOUT_SECONDS = float(os.environ.get("TIMEOUT_SECONDS", "600"))
MAX_IN_FLIGHT = 4
MEDIA_TYPES = {
    "png": "image/png",
    "gif": "image/gif",
    "glb": "model/gltf-binary",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "mp4": "video/mp4",
}
LIBRARY = {path.stem: path for path in sorted((Path(__file__).parent / "library").glob("*.glb"))}
FRIENDLY_NAMES = {
    "Idle_Loop": "Idle",
    "Walk_Loop": "Walk",
    "Jog_Fwd_Loop": "Run",
    "Sprint_Loop": "Sprint",
    "Jump_Start": "Jump Start",
    "Jump_Loop": "Jump Air",
    "Jump_Land": "Jump Land",
    "Punch_Jab": "Punch",
    "Punch_Cross": "Cross Punch",
    "Sword_Attack": "Sword Attack",
    "Hit_Chest": "Hit",
    "Death01": "Death",
    "Roll": "Roll",
    "Dance_Loop": "Dance",
    "Crouch_Idle_Loop": "Crouch",
    "Crouch_Fwd_Loop": "Crouch Walk",
    "Sitting_Idle_Loop": "Sit",
    "Swim_Fwd_Loop": "Swim",
    "Push_Loop": "Push",
    "PickUp_Table": "Pick Up",
    "Interact": "Interact",
    "Spell_Simple_Shoot": "Cast Spell",
    "Pistol_Shoot": "Shoot",
}


def clip_name(key: str) -> str:
    """Return the clip name we give a library animation, readable even for unmapped keys."""
    return FRIENDLY_NAMES.get(key) or key.removesuffix("_Loop").replace("_", " ")


cutter = pictures.Cutter(os.environ.get("CUTOUT_MODEL", "/opt/tools/isnet-general-use.onnx"))
# We run one heavy job at a time, since a cutout or a big mesh can take gigabytes.
heavy_slot = threading.Lock()
in_flight = 0

app = FastAPI(title="media tools", docs_url=None, redoc_url=None, openapi_url=None)


def in_slot[T](work: Callable[[], T]) -> T:
    with heavy_slot:
        return work()


@contextmanager
def heavy_turn() -> Iterator[None]:
    """Count a request that waits for or holds the heavy slot, refusing it when too many do.

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


def named_file(data: bytes, filename: str) -> Response:
    """Answer with a file named `filename`, since n8n otherwise names it after the URL path."""
    extension = filename.rsplit(".", 1)[1]
    return Response(
        data,
        media_type=MEDIA_TYPES[extension],
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


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
    return named_file(png, "pose.png")


@app.get("/pose/points")
async def pose_points(move: str) -> dict[str, tuple[float, float]]:
    """Return a still pose's COCO-18 points, where the rig finds a drawn character's joints."""
    try:
        points = poses.still_points(move)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return {str(index): point for index, point in points.items()}


@app.post("/cutout")
async def cutout(request: Request, method: pictures.CutMethod = "isnet") -> Response:
    with heavy_turn():
        picture = await read_body(request, MAX_PICTURE_BYTES)
        try:
            png = await run_in_threadpool(in_slot, lambda: cutter.cut(picture, method))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    return named_file(png, "cutout.png")


@app.post("/pixelate")
async def pixelate(
    request: Request,
    size: Annotated[int, Query(ge=8, le=512)] = 64,
    colors: Annotated[int, Query(ge=2, le=256)] = 16,
    scale: Annotated[int, Query(ge=1, le=32)] = 1,
) -> Response:
    with heavy_turn():
        picture = await read_body(request, MAX_PICTURE_BYTES)
        try:
            png = await run_in_threadpool(
                in_slot, lambda: pictures.pixel_art(picture, size, colors, scale)
            )
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    return named_file(png, "pixel-art.png")


@app.post("/sprite-frames")
async def sprite_frames(
    request: Request,
    lengths: str,
    format: Annotated[str, Query(pattern="^(png|gif)$")] = "png",
    size: Annotated[int, Query(ge=8, le=512)] = 64,
    colors: Annotated[int, Query(ge=2, le=256)] = 24,
    scale: Annotated[int, Query(ge=1, le=32)] = 1,
) -> Response:
    render = pictures.sprite_gif if format == "gif" else pictures.sprite_sheet
    with heavy_turn():
        body = await read_body(request, MAX_FRAMES * MAX_PICTURE_BYTES)
        frames = split_body(body, lengths, range(2, MAX_FRAMES + 1))
        try:
            data = await run_in_threadpool(in_slot, lambda: render(frames, size, colors, scale))
        except pictures.PictureError as error:
            raise HTTPException(400, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
    return named_file(data, f"sprite-sheet.{format}")


@app.post("/mesh")
async def simplify_mesh(
    request: Request, triangles: Annotated[int, Query(ge=1000, le=5_000_000)] = 300_000
) -> Response:
    with heavy_turn():
        model = await read_body(request, MAX_MODEL_BYTES)
        try:
            glb = await run_in_threadpool(
                in_slot, lambda: mesh.simplify(model, triangles, GLTFPACK, TIMEOUT_SECONDS)
            )
        except GlbError as error:
            raise HTTPException(400, f"not a usable GLB: {error}") from error
        except mesh.MeshError as error:
            raise HTTPException(422, str(error)) from error
    return named_file(glb, "model.glb")


@app.get("/library")
async def library_clips() -> list[dict[str, str]]:
    return [{"key": key, "name": clip_name(key)} for key in LIBRARY]


@app.post("/retarget")
async def retarget_motions(
    request: Request, lengths: str, names: str = "", library: str = "", in_place: bool = True
) -> Response:
    picked = [key.strip() for key in library.split(",") if key.strip()]
    if any(key not in LIBRARY for key in picked):
        raise HTTPException(400, "library takes clip keys that GET /library lists")
    if len(picked) > MAX_LIBRARY_CLIPS or len(set(picked)) != len(picked):
        raise HTTPException(400, f"ask for at most {MAX_LIBRARY_CLIPS} different library clips")
    with heavy_turn():
        body = await read_body(request, MAX_MODEL_BYTES + MAX_CLIPS * MAX_MOTION_BYTES)
        parts = split_body(body, lengths, range(1, MAX_CLIPS + 2))
        clip_names = [name.strip() for name in names.split(",")] if names else []
        if len(clip_names) != len(parts) - 1 or not all(clip_names):
            raise HTTPException(400, "give one clip name per motion in names")
        if not clip_names and not picked:
            raise HTTPException(400, "ask for at least one motion or library clip")
        if len(parts[0]) > MAX_MODEL_BYTES or any(
            len(part) > MAX_MOTION_BYTES for part in parts[1:]
        ):
            raise HTTPException(400, "the character or a motion is too large")
        motions = list(zip(clip_names, parts[1:], strict=True)) + [
            (clip_name(key), LIBRARY[key].read_bytes()) for key in picked
        ]
        try:
            glb = await run_in_threadpool(in_slot, lambda: retarget(parts[0], motions, in_place))
        except GlbError as error:
            raise HTTPException(400, f"not a usable GLB: {error}") from error
        except RetargetError as error:
            raise HTTPException(422, f"cannot retarget: {error}") from error
    return named_file(glb, "model.glb")


async def run_media[T](work: Callable[[], T]) -> T:
    """Run an ffmpeg job in the heavy slot, answering 422 when ffmpeg cannot handle the files."""
    try:
        return await run_in_threadpool(in_slot, work)
    except FfmpegError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/convert")
async def convert_audio(
    request: Request,
    format: audio.AudioFormat = "wav",
    ar: Annotated[int | None, Query(ge=8000, le=192_000)] = None,
    ac: Annotated[int | None, Query(ge=1, le=8)] = None,
) -> Response:
    with heavy_turn():
        track = await read_body(request, MAX_MEDIA_BYTES)
        data = await run_media(lambda: audio.convert(track, format, ar, ac, TIMEOUT_SECONDS))
    return named_file(data, f"audio.{format}")


@app.post("/mix-bed")
async def mix_bed(request: Request, lengths: str) -> Response:
    """Take the spoken track then the music, and lay the voices over the music."""
    with heavy_turn():
        body = await read_body(request, MAX_MEDIA_BYTES)
        voice, bed = split_body(body, lengths, range(2, 3))
        data = await run_media(lambda: audio.mix_bed(voice, bed, TIMEOUT_SECONDS))
    return named_file(data, "mixed.wav")


def picture_starts(starts: str) -> list[float]:
    try:
        values = [float(value) for value in starts.split(",") if value]
    except ValueError as error:
        raise HTTPException(400, "picture_starts is a comma separated list of seconds") from error
    gaps = [later - earlier for earlier, later in itertools.pairwise(values)]
    finite = all(math.isfinite(value) and value >= 0 for value in values)
    if len(values) > MAX_KARAOKE_PICTURES or not finite or any(gap < 1 for gap in gaps):
        raise HTTPException(
            400,
            f"give at most {MAX_KARAOKE_PICTURES} picture_starts in seconds, each at least 1 s "
            "after the one before",
        )
    return values


def parsed_words(data: bytes) -> list[karaoke.Word]:
    try:
        return karaoke.parse_words(json.loads(data), MAX_WORDS)
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(400, f"words: {error}") from error


class KaraokeOptions(BaseModel):
    lengths: str
    look: karaoke.LookName = "neon"
    highlight: karaoke.Highlight = "sweep"
    background: karaoke.Background = "bars"
    vocals_cut: float = Field(0, ge=0, le=1)
    title: str = Field("", max_length=MAX_TEXT_CHARS)
    picture_starts: str = ""


@app.post("/karaoke")
async def render_karaoke(request: Request, options: Annotated[KaraokeOptions, Query()]) -> Response:
    """Take the track, the words JSON, the vocal stem when vocals_cut is set, then the pictures."""
    seconds = picture_starts(options.picture_starts)
    stems = 1 if options.vocals_cut > 0 else 0
    style = karaoke.Style(options.look, options.highlight, options.background)
    with heavy_turn():
        body = await read_body(request, MAX_MEDIA_BYTES)
        count = 2 + stems + len(seconds)
        parts = split_body(body, options.lengths, range(count, count + 1))
        vocals = karaoke.Vocals(parts[2], options.vocals_cut) if stems else None
        song = karaoke.Song(parts[0], parsed_words(parts[1]), options.title, vocals)
        shown = [
            karaoke.Picture(data, start)
            for data, start in zip(parts[2 + stems :], seconds, strict=True)
        ]
        data = await run_media(lambda: karaoke.render(song, style, shown, TIMEOUT_SECONDS))
    return named_file(data, "karaoke.mp4")


@app.post("/cover")
async def cover_title(
    request: Request,
    title: Annotated[str, Query(min_length=1, max_length=MAX_TEXT_CHARS)],
    subtitle: Annotated[str, Query(max_length=MAX_TEXT_CHARS)] = "",
) -> Response:
    with heavy_turn():
        picture = await read_body(request, MAX_PICTURE_BYTES)
        try:
            png = await run_in_threadpool(
                in_slot, lambda: cover(picture, title, subtitle, COVER_FONT)
            )
        except pictures.PictureError as error:
            raise HTTPException(400, str(error)) from error
    return named_file(png, "cover.png")


def parsed_chapters(data: bytes) -> list[audio.Chapter]:
    try:
        raw = json.loads(data)
        if not isinstance(raw, list) or len(raw) > MAX_CHAPTERS:
            raise ValueError(f"send a list of at most {MAX_CHAPTERS} chapters")
        chapters = [
            audio.Chapter(str(item["title"])[:MAX_TEXT_CHARS], float(item["start"])) for item in raw
        ]
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(400, f"chapters: {error}") from error
    if not all(math.isfinite(chapter.start) and chapter.start >= 0 for chapter in chapters):
        raise HTTPException(400, "chapters start at a finite 0 seconds or later")
    return sorted(chapters, key=lambda chapter: chapter.start)


@app.post("/mp3")
async def package_mp3(
    request: Request,
    lengths: str,
    title: Annotated[str, Query(max_length=MAX_TEXT_CHARS)] = "",
    artist: Annotated[str, Query(max_length=MAX_TEXT_CHARS)] = "",
) -> Response:
    """Take the audio, the chapters JSON and optionally the cover picture."""
    with heavy_turn():
        body = await read_body(request, MAX_MEDIA_BYTES)
        track, chapter_list, *art = split_body(body, lengths, range(2, 4))
        chapters = parsed_chapters(chapter_list)
        tags = audio.BookTags(title, artist)
        cover_art = art[0] if art else None
        data = await run_media(
            lambda: audio.package_mp3(track, chapters, cover_art, tags, TIMEOUT_SECONDS)
        )
    return named_file(data, "audiobook.mp3")
