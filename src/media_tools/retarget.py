"""Move skeleton animations onto a rigged humanoid GLB, one named clip per motion.

For every humanoid bone we take how far the source bone turned from its rest pose, in world space,
and apply that turn to the character's bone from its own rest pose. The motion is turned about the
vertical axis when the two skeletons face different ways, and hip travel is scaled by hip height.
"""

import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy
from numpy.typing import NDArray

from media_tools.glb import Document
from media_tools.glb import read_glb
from media_tools.glb import write_glb

type Floats = NDArray[numpy.float64]

FLOAT = 5126
COMPONENTS = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}
IDENTITY = numpy.array([0.0, 0.0, 0.0, 1.0])
REQUIRED_BONES = {"hips", "upperleg_l", "upperleg_r", "upperarm_l", "upperarm_r"}

# Every rig names humanoid bones its own way, so we map rigged characters (VRoid, Mixamo and
# Blender-style names) to one vocabulary; Kimodo's SOMA skeleton has its own table below.
BONE_ALIASES = {
    "hips": {"hips", "pelvis"},
    "spine": {"spine"},
    "chest": {"chest", "spine1"},
    "upperchest": {"upperchest", "spine2"},
    "neck": {"neck"},
    "head": {"head"},
    "shoulder": {"shoulder", "clavicle"},
    "upperarm": {"upperarm", "arm"},
    "lowerarm": {"lowerarm", "forearm"},
    "hand": {"hand"},
    "upperleg": {"upperleg", "upleg", "thigh"},
    "lowerleg": {"lowerleg", "leg", "shin", "calf"},
    "foot": {"foot"},
    "toe": {"toebase", "toe", "toes"},
}
SIDED = {"shoulder", "upperarm", "lowerarm", "hand", "upperleg", "lowerleg", "foot", "toe"}
SIDE_PATTERNS = [
    (re.compile(r"^(j_bip_l_|left|l_)|(\.l|_l|left)$"), "_l"),
    (re.compile(r"^(j_bip_r_|right|r_)|(\.r|_r|right)$"), "_r"),
]
PREFIX = re.compile(r"^(mixamorig\d*:|def-)")
SIDE_MARK = re.compile(r"^(j_bip_[clr]_|left|right|[lr]_)|(\.[lr]|_[lr]|left|right)$")
KIMODO = {
    "Hips": "hips",
    "Spine1": "spine",
    "Spine2": "chest",
    "Chest": "upperchest",
    "Neck1": "neck",
    "Head": "head",
    "LeftShoulder": "shoulder_l",
    "LeftArm": "upperarm_l",
    "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l",
    "RightShoulder": "shoulder_r",
    "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r",
    "RightHand": "hand_r",
    "LeftLeg": "upperleg_l",
    "LeftShin": "lowerleg_l",
    "LeftFoot": "foot_l",
    "LeftToeBase": "toe_l",
    "RightLeg": "upperleg_r",
    "RightShin": "lowerleg_r",
    "RightFoot": "foot_r",
    "RightToeBase": "toe_r",
}


class RetargetError(ValueError):
    pass


@dataclass(frozen=True)
class Motion:
    """A skeleton animation: frame times and each node's world rotation and position per frame."""

    times: Floats
    rest_rotation: dict[str, Floats]
    rest_position: dict[str, Floats]
    rotation: dict[str, Floats]
    position: dict[str, Floats]
    bones: dict[str, str]


def canonical(name: str) -> str | None:
    """Map a VRoid, Mixamo or Blender-style bone name to a shared name such as upperarm_l."""
    lowered = PREFIX.sub("", name.lower())
    side = next((suffix for pattern, suffix in SIDE_PATTERNS if pattern.search(lowered)), "")
    bare = SIDE_MARK.sub("", lowered).replace("_", "")
    for bone, aliases in BONE_ALIASES.items():
        if bare in aliases:
            if bone not in SIDED:
                return bone
            return bone + side if side else None
    return None


def quat_mul(a: Floats, b: Floats) -> Floats:
    ax, ay, az, aw = numpy.moveaxis(a, -1, 0)
    bx, by, bz, bw = numpy.moveaxis(b, -1, 0)
    return numpy.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_inv(q: Floats) -> Floats:
    inverted: Floats = q * numpy.array([-1.0, -1.0, -1.0, 1.0])
    return inverted


def quat_rotate(q: Floats, vector: Floats) -> Floats:
    pure = numpy.concatenate([vector, numpy.zeros((*vector.shape[:-1], 1))], axis=-1)
    rotated: Floats = quat_mul(quat_mul(q, pure), quat_inv(q))[..., :3]
    return rotated


def quat_from_matrix(matrix: Floats) -> Floats:
    """Return the rotation of a 4x4 matrix as a quaternion, ignoring its scale."""
    m = matrix[:3, :3] / numpy.linalg.norm(matrix[:3, :3], axis=0)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * numpy.sqrt(trace + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * numpy.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * numpy.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * numpy.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    quaternion = numpy.array(q, dtype=numpy.float64)
    normalized: Floats = quaternion / numpy.linalg.norm(quaternion)
    return normalized


def local_matrix(node: dict[str, Any]) -> Floats:
    if "matrix" in node:
        return numpy.array(node["matrix"], dtype=numpy.float64).reshape(4, 4).T
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    rotation = numpy.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix = numpy.eye(4)
    matrix[:3, :3] = rotation * numpy.array(node.get("scale", [1.0, 1.0, 1.0]))
    matrix[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    return matrix


def parents(document: Document) -> dict[int, int]:
    return {
        child: index
        for index, node in enumerate(document.get("nodes", []))
        for child in node.get("children", [])
    }


def topological(count: int, parent_of: dict[int, int]) -> list[int]:
    """Order node indices so every parent comes before its children."""
    order: list[int] = []
    seen: set[int] = set()
    for start in range(count):
        chain = []
        index: int | None = start
        while index is not None and index not in seen:
            chain.append(index)
            seen.add(index)
            index = parent_of.get(index)
        order.extend(reversed(chain))
    return order


def world_matrices(document: Document) -> dict[int, Floats]:
    nodes = document.get("nodes", [])
    parent_of = parents(document)
    worlds: dict[int, Floats] = {}
    for index in topological(len(nodes), parent_of):
        local = local_matrix(nodes[index])
        worlds[index] = worlds[parent_of[index]] @ local if index in parent_of else local
    return worlds


def read_floats(document: Document, binary: bytes | bytearray, index: int) -> Floats:
    spec = document["accessors"][index]
    if spec["componentType"] != FLOAT or spec["type"] not in COMPONENTS:
        raise RetargetError("only float animation data is supported")
    view = document["bufferViews"][spec["bufferView"]]
    width = COMPONENTS[spec["type"]]
    start = view.get("byteOffset", 0) + spec.get("byteOffset", 0)
    stride = view.get("byteStride", 4 * width)
    row = struct.Struct(f"<{width}f")
    rows = [row.unpack_from(binary, start + number * stride) for number in range(spec["count"])]
    return numpy.array(rows, dtype=numpy.float64).reshape(spec["count"], width)


def append_floats(document: Document, binary: bytearray, values: Floats, kind: str) -> int:
    data = numpy.ascontiguousarray(values, dtype=numpy.float32)
    binary.extend(b"\0" * (-len(binary) % 4))
    view = {"buffer": 0, "byteOffset": len(binary), "byteLength": data.nbytes}
    document.setdefault("bufferViews", []).append(view)
    binary.extend(data.tobytes())
    spec: dict[str, Any] = {
        "bufferView": len(document["bufferViews"]) - 1,
        "componentType": FLOAT,
        "count": len(data),
        "type": kind,
    }
    if kind == "SCALAR":
        spec["min"], spec["max"] = [float(data.min())], [float(data.max())]
    document.setdefault("accessors", []).append(spec)
    return len(document["accessors"]) - 1


def _channels(
    document: Document, binary: bytes | bytearray, clip: dict[str, Any]
) -> tuple[Floats, dict[str, dict[str, Floats]]]:
    nodes = document["nodes"]
    times: Floats | None = None
    tracks: dict[str, dict[str, Floats]] = {"rotation": {}, "translation": {}}
    for channel in clip.get("channels", []):
        sampler = clip["samplers"][channel["sampler"]]
        path = channel["target"].get("path")
        if path not in tracks:
            continue
        if times is None:
            times = read_floats(document, binary, sampler["input"])[:, 0]
        name = nodes[channel["target"]["node"]].get("name", "")
        tracks[path][name] = read_floats(document, binary, sampler["output"])
    if times is None:
        raise RetargetError("the motion has no rotation or translation channels")
    return times, tracks


def read_motion(data: bytes) -> Motion:
    """Read the first animation of a skeleton GLB as world rotations and positions per frame.

    :raises GlbError: when the data is not a GLB.
    :raises RetargetError: when it has no usable animation.
    """
    document, binary = read_glb(data)
    if not document.get("animations"):
        raise RetargetError("the motion has no animation")
    times, tracks = _channels(document, binary, document["animations"][0])
    frames = len(times)
    nodes = document["nodes"]
    parent_of = parents(document)
    rest_rotation: dict[str, Floats] = {}
    rest_position: dict[str, Floats] = {}
    rotation: dict[str, Floats] = {}
    position: dict[str, Floats] = {}
    for index in topological(len(nodes), parent_of):
        name = nodes[index].get("name", str(index))
        local_rest = numpy.array(nodes[index].get("rotation", IDENTITY), dtype=numpy.float64)
        local_offset = numpy.array(
            nodes[index].get("translation", [0.0, 0.0, 0.0]), dtype=numpy.float64
        )
        local_rotation = tracks["rotation"].get(name, numpy.tile(local_rest, (frames, 1)))
        local_position = tracks["translation"].get(name, numpy.tile(local_offset, (frames, 1)))
        if index not in parent_of:
            rest_rotation[name], rest_position[name] = local_rest, local_offset
            rotation[name], position[name] = local_rotation, local_position
            continue
        parent = nodes[parent_of[index]].get("name", str(parent_of[index]))
        rest_rotation[name] = quat_mul(rest_rotation[parent], local_rest)
        rest_position[name] = rest_position[parent] + quat_rotate(
            rest_rotation[parent], local_offset
        )
        rotation[name] = quat_mul(rotation[parent], local_rotation)
        position[name] = position[parent] + quat_rotate(rotation[parent], local_position)
    bones = {bone: name for name, bone in KIMODO.items() if name in rest_rotation}
    if "hips" not in bones:
        bones = {bone: name for name in rest_rotation if (bone := canonical(name))}
    return Motion(times, rest_rotation, rest_position, rotation, position, bones)


def rig_bones(document: Document) -> tuple[dict[str, int], list[int]]:
    """Return the character's humanoid bones by shared name, and all its joints.

    :raises RetargetError: when it has no skin or lacks the core humanoid bones.
    """
    skins = document.get("skins") or []
    if not skins:
        raise RetargetError("the character has no skin; rig it first")
    joints = [int(joint) for joint in skins[0]["joints"]]
    found: dict[str, int] = {}
    for joint in joints:
        bone = canonical(document["nodes"][joint].get("name", ""))
        if bone and bone not in found:
            found[bone] = joint
    missing = REQUIRED_BONES - set(found)
    if missing:
        raise RetargetError(f"the character's skeleton lacks humanoid bones: {sorted(missing)}")
    return found, joints


def facing_turn(motion: Motion, worlds: dict[int, Floats], bones: dict[str, int]) -> Floats:
    """Return the turn about the vertical axis that lines the source's hips up with the rig's."""
    missing = {"upperleg_l", "upperleg_r"} - set(motion.bones)
    if missing:
        raise RetargetError(f"the motion's skeleton lacks {sorted(missing)}")
    source = (
        motion.rest_position[motion.bones["upperleg_l"]]
        - motion.rest_position[motion.bones["upperleg_r"]]
    )
    target = worlds[bones["upperleg_l"]][:3, 3] - worlds[bones["upperleg_r"]][:3, 3]
    angle = numpy.arctan2(target[0], target[2]) - numpy.arctan2(source[0], source[2])
    return numpy.array([0.0, numpy.sin(angle / 2), 0.0, numpy.cos(angle / 2)])


def hip_height(heights: Sequence[float], hips: float) -> float:
    return hips - (min(heights) if heights else 0.0)


def hip_track(
    motion: Motion, document: Document, bones: dict[str, int], turn: Floats, in_place: bool
) -> Floats:
    """Return the character's hip translation per frame: the source's hip travel, scaled."""
    worlds = world_matrices(document)
    hips = bones["hips"]
    source_hips = motion.bones["hips"]
    target_feet = [
        float(worlds[bones[side]][1, 3]) for side in ("foot_l", "foot_r") if side in bones
    ]
    source_feet = [
        float(motion.rest_position[motion.bones[side]][1])
        for side in ("foot_l", "foot_r")
        if side in motion.bones
    ]
    target_height = hip_height(target_feet, float(worlds[hips][1, 3]))
    source_height = hip_height(source_feet, float(motion.rest_position[source_hips][1]))
    if source_height <= 0 or target_height <= 0:
        raise RetargetError("could not measure the hip heights")
    # Motion files may keep the rest hips at the origin, so travel counts from the first frame.
    travel = motion.position[source_hips] - motion.position[source_hips][0]
    if in_place:
        travel[:, [0, 2]] = 0.0
    moved = quat_rotate(turn, travel) * (target_height / source_height)
    parent = parents(document).get(hips)
    parent_world = worlds[parent] if parent is not None else numpy.eye(4)
    rest = local_matrix(document["nodes"][hips])[:3, 3]
    track: Floats = rest + moved @ numpy.linalg.inv(parent_world[:3, :3]).T
    return track


def joint_rotations(
    motion: Motion, document: Document, bones: dict[str, int], joints: list[int], turn: Floats
) -> dict[int, Floats]:
    """Return each joint's local rotation per frame that gives its bone the source's world turn."""
    frames = len(motion.times)
    nodes = document["nodes"]
    parent_of = parents(document)
    worlds = world_matrices(document)
    bone_of = {joint: bone for bone, joint in bones.items()}
    joint_set = set(joints)
    world: dict[int, Floats] = {}
    local: dict[int, Floats] = {}
    for joint in topological(len(nodes), parent_of):
        if joint not in joint_set:
            continue
        parent = parent_of.get(joint)
        parent_rotation = world.get(parent) if parent is not None else None
        if parent_rotation is None:
            base = quat_from_matrix(worlds[parent]) if parent is not None else IDENTITY
            parent_rotation = numpy.tile(base, (frames, 1))
        source = motion.bones.get(bone_of.get(joint, ""))
        if source is None:
            rest_local = quat_from_matrix(local_matrix(nodes[joint]))
            world[joint] = quat_mul(parent_rotation, numpy.tile(rest_local, (frames, 1)))
        else:
            delta = quat_mul(motion.rotation[source], quat_inv(motion.rest_rotation[source]))
            delta = quat_mul(quat_mul(turn, delta), quat_inv(turn))
            world[joint] = quat_mul(delta, quat_from_matrix(worlds[joint]))
        rotation = quat_mul(quat_inv(parent_rotation), world[joint])
        local[joint] = rotation / numpy.linalg.norm(rotation, axis=-1, keepdims=True)
    return local


def add_clip(
    document: Document, binary: bytearray, motion: Motion, name: str, in_place: bool
) -> None:
    """Retarget one motion onto the character and add it as an animation named `name`."""
    bones, joints = rig_bones(document)
    turn = facing_turn(motion, world_matrices(document), bones)
    times = append_floats(document, binary, motion.times, "SCALAR")
    samplers: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []

    def track(node: int, path: str, values: Floats, kind: str) -> None:
        output = append_floats(document, binary, values, kind)
        samplers.append({"input": times, "interpolation": "LINEAR", "output": output})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})

    for joint, rotation in joint_rotations(motion, document, bones, joints, turn).items():
        track(joint, "rotation", rotation, "VEC4")
    track(bones["hips"], "translation", hip_track(motion, document, bones, turn, in_place), "VEC3")
    document.setdefault("animations", []).append(
        {"name": name, "channels": channels, "samplers": samplers}
    )


def retarget(
    character: bytes, motions: Sequence[tuple[str, bytes]], in_place: bool = True
) -> bytes:
    """Return the character GLB with one animation per (name, skeleton GLB) motion.

    :raises GlbError: when a file is not a GLB.
    :raises RetargetError: when the character or a motion cannot be matched up.
    """
    document, binary = read_glb(character)
    buffers = document.get("buffers") or []
    if len(buffers) != 1 or "uri" in buffers[0]:
        raise RetargetError("the character must be a GLB with one embedded buffer")
    for name, data in motions:
        add_clip(document, binary, read_motion(data), name, in_place)
    buffers[0]["byteLength"] = len(binary) + (-len(binary) % 4)
    return write_glb(document, binary)
