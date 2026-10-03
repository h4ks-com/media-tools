import copy
import io
import math
import shutil
import struct
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from media_tools import ffmpeg
from media_tools.glb import CHUNK
from media_tools.glb import GLB_MAGIC
from media_tools.glb import HEADER
from media_tools.glb import JSON_CHUNK
from media_tools.glb import Document
from media_tools.glb import read_glb
from media_tools.glb import write_glb
from media_tools.retarget import KIMODO

DATA = Path(__file__).parent / "data"
SIDES = (("l", "L"), ("r", "R"))
LIMBS = (
    ("shoulder", "Shoulder"),
    ("upperarm", "UpperArm"),
    ("lowerarm", "LowerArm"),
    ("hand", "Hand"),
    ("upperleg", "UpperLeg"),
    ("lowerleg", "LowerLeg"),
    ("foot", "Foot"),
    ("toe", "ToeBase"),
)
VROID = {
    "hips": "J_Bip_C_Hips",
    "spine": "J_Bip_C_Spine",
    "chest": "J_Bip_C_Chest",
    "upperchest": "J_Bip_C_UpperChest",
    "neck": "J_Bip_C_Neck",
    "head": "J_Bip_C_Head",
    **{f"{bone}_{side}": f"J_Bip_{mark}_{name}" for side, mark in SIDES for bone, name in LIMBS},
}


@pytest.fixture(scope="session")
def walk() -> bytes:
    return (DATA / "walk.glb").read_bytes()


@pytest.fixture(scope="session")
def jump() -> bytes:
    return (DATA / "jump.glb").read_bytes()


def glb_with_json(text: bytes) -> bytes:
    """Return a GLB whose JSON chunk holds exactly `text`, parsable or not."""
    text += b" " * (-len(text) % 4)
    chunks = CHUNK.pack(len(text), JSON_CHUNK) + text
    return HEADER.pack(GLB_MAGIC, 2, HEADER.size + len(chunks)) + chunks


def edit(glb: bytes, change: Callable[[Document, bytearray], object]) -> bytes:
    """Return the GLB after `change` edits its document and binary buffer in place."""
    document, binary = read_glb(glb)
    change(document, binary)
    return write_glb(document, binary)


def add_root(document: Document, root: Document) -> None:
    """Make a new `root` node the parent of the whole skeleton."""
    document["nodes"].append({**root, "children": [0]})
    document["scenes"][0]["nodes"] = [len(document["nodes"]) - 1]


def vroid_rig(motion: bytes, root: Document | None = None) -> bytes:
    """Return Kimodo's skeleton renamed to VRoid bones and skinned, as a stand-in rigged character.

    A `root` node, when given, becomes the parent of the whole skeleton.
    """
    document, binary = read_glb(motion)
    rig = copy.deepcopy(document)
    for node in rig["nodes"]:
        if node["name"] in KIMODO:
            node["name"] = VROID[KIMODO[node["name"]]]
    rig["skins"] = [{"joints": list(range(len(rig["nodes"])))}]
    rig.pop("animations")
    if root is not None:
        add_root(rig, root)
    return write_glb(rig, binary)


def png(size: tuple[int, int], box: tuple[int, int, int, int] | None = None) -> bytes:
    """Return a PNG on a flat white background, with a black box where `box` says."""
    picture = Image.new("RGB", size, "white")
    if box is not None:
        picture.paste((0, 0, 0), box)
    output = io.BytesIO()
    picture.save(output, format="PNG")
    return output.getvalue()


def tone(seconds: float, rate: int = 22050) -> bytes:
    """Return a mono 16-bit WAV of a 440 Hz tone."""
    output = io.BytesIO()
    with wave.open(output, "wb") as track:
        track.setnchannels(1)
        track.setsampwidth(2)
        track.setframerate(rate)
        samples = (
            round(8000 * math.sin(2 * math.pi * 440 * n / rate)) for n in range(int(seconds * rate))
        )
        track.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
    return output.getvalue()


@pytest.fixture(autouse=True)
def local_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the ffmpeg on PATH when the image's pinned one is not there."""
    for name in ("FFMPEG", "FFPROBE"):
        pinned = getattr(ffmpeg, name)
        found = shutil.which(Path(pinned).name)
        if not Path(pinned).exists() and found:
            monkeypatch.setattr(ffmpeg, name, found)


def has_libass() -> bool:
    binary = ffmpeg.FFMPEG if Path(ffmpeg.FFMPEG).exists() else shutil.which("ffmpeg")
    if binary is None:
        return False
    filters = subprocess.run([binary, "-hide_banner", "-filters"], capture_output=True, check=False)
    return b" ass " in filters.stdout


needs_libass = pytest.mark.skipif(not has_libass(), reason="this ffmpeg has no libass")
