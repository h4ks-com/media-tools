"""Split Quaternius's Universal Animation Library into one small skeleton GLB per clip.

Run it on the glTF of https://github.com/J-Ponzo/gltf-universal-animation-library:

    uv run python scripts/split_animation_library.py glTF/AnimationLibrary_Godot_Standard.gltf

Each GLB keeps the humanoid bones with their ancestors and the channels that move them.
"""

import json
import sys
from pathlib import Path

import numpy

from media_tools.glb import Document
from media_tools.glb import entries
from media_tools.glb import write_glb
from media_tools.retarget import IDENTITY
from media_tools.retarget import ORIGIN
from media_tools.retarget import TRACK_TYPES
from media_tools.retarget import UNIT_SCALE
from media_tools.retarget import Floats
from media_tools.retarget import append_floats
from media_tools.retarget import bone_map
from media_tools.retarget import numbers
from media_tools.retarget import read_floats
from media_tools.retarget import skeleton

LIBRARY = Path(__file__).parents[1] / "src" / "media_tools" / "library"
SKIPPED = {"A_TPose"}
REST = {"rotation": IDENTITY, "translation": ORIGIN, "scale": UNIT_SCALE}
POSE_KEYS = ("name", "rotation", "translation", "scale")


def humanoid_nodes(document: Document) -> list[int]:
    tree = skeleton(document)
    bones = bone_map(tree, range(len(tree.nodes)))
    return sorted({index for joint in bones.values() for index in tree.chain(joint)})


def still_at_rest(node: Document, path: str, values: Floats) -> bool:
    rest = numbers(node, path, REST[path])
    same = numpy.isclose(values, rest, atol=1e-5).all(axis=1)
    if path == "rotation":
        same |= numpy.isclose(values, -rest, atol=1e-5).all(axis=1)
    return bool(same.all())


def pose_node(node: Document, new_index: dict[int, int]) -> Document:
    kept: Document = {key: node[key] for key in POSE_KEYS if key in node}
    children = [new_index[child] for child in node.get("children", []) if child in new_index]
    if children:
        kept["children"] = children
    return kept


def split(document: Document, binary: bytearray, clip: Document, kept: list[int]) -> bytes:
    """Return a GLB of the kept nodes and the clip's channels that move them away from rest."""
    new_index = {old: new for new, old in enumerate(kept)}
    nodes = entries(document, "nodes")
    out: Document = {"asset": {"version": "2.0"}, "scene": 0}
    out["nodes"] = [pose_node(nodes[old], new_index) for old in kept]
    children = {child for node in out["nodes"] for child in node.get("children", [])}
    out["scenes"] = [{"nodes": [index for index in range(len(kept)) if index not in children]}]
    data = bytearray()
    samplers: list[Document] = []
    channels: list[Document] = []
    for channel in clip["channels"]:
        target = channel["target"]
        if target.get("node") not in new_index:
            continue
        sampler = clip["samplers"][channel["sampler"]]
        interpolation = sampler.get("interpolation", "LINEAR")
        if interpolation not in ("LINEAR", "STEP"):
            sys.exit(f"{clip['name']} has {interpolation} keys, which the retargeter refuses")
        path = target["path"]
        values = read_floats(document, binary, sampler["output"], TRACK_TYPES[path])
        if still_at_rest(nodes[target["node"]], path, values):
            continue
        times = read_floats(document, binary, sampler["input"], "SCALAR")
        samplers.append(
            {
                "input": append_floats(out, data, times, "SCALAR"),
                "interpolation": interpolation,
                "output": append_floats(out, data, values, TRACK_TYPES[path]),
            }
        )
        channels.append(
            {
                "sampler": len(samplers) - 1,
                "target": {"node": new_index[target["node"]], "path": path},
            }
        )
    out["animations"] = [{"name": clip["name"], "channels": channels, "samplers": samplers}]
    out["buffers"] = [{"byteLength": len(data)}]
    return write_glb(out, data)


def main(source: Path) -> None:
    document = json.loads(source.read_text())
    binary = bytearray(source.with_suffix(".bin").read_bytes())
    kept = humanoid_nodes(document)
    for old in LIBRARY.glob("*.glb"):
        old.unlink()
    for clip in entries(document, "animations"):
        name = clip["name"]
        if name in SKIPPED or name.endswith("_RM"):
            continue
        (LIBRARY / f"{name}.glb").write_bytes(split(document, binary, clip, kept))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
