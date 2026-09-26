import copy
import io
from pathlib import Path

import pytest
from PIL import Image

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
        rig["nodes"].append({**root, "children": [0]})
        rig["scenes"][0]["nodes"] = [len(rig["nodes"]) - 1]
    return write_glb(rig, binary)


def png(size: tuple[int, int], box: tuple[int, int, int, int] | None = None) -> bytes:
    """Return a PNG on a flat white background, with a black box where `box` says."""
    picture = Image.new("RGB", size, "white")
    if box is not None:
        picture.paste((0, 0, 0), box)
    output = io.BytesIO()
    picture.save(output, format="PNG")
    return output.getvalue()
