"""Move skeleton animations onto a rigged humanoid GLB, one named clip per motion.

For every humanoid bone we take how far the source bone turned from its rest pose, in world space,
and apply that turn to the character's bone from its own rest pose. The motion is turned about the
vertical axis when the two skeletons face different ways. The hips keep the source's height above
the ground and its travel, both scaled by the ratio of the two standing hip heights.
"""

import re
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from graphlib import CycleError
from graphlib import TopologicalSorter

import numpy
from numpy.typing import NDArray

from media_tools.glb import Document
from media_tools.glb import GlbError
from media_tools.glb import entries
from media_tools.glb import index_into
from media_tools.glb import natural
from media_tools.glb import read_glb
from media_tools.glb import write_glb

type Floats = NDArray[numpy.float64]

FLOAT = 5126
COMPONENTS = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}
TRACK_TYPES = {"rotation": "VEC4", "translation": "VEC3", "scale": "VEC3"}
IDENTITY = numpy.array([0.0, 0.0, 0.0, 1.0])
ORIGIN = numpy.zeros(3)
UNIT_SCALE = numpy.ones(3)
REQUIRED_BONES = {"hips", "upperleg_l", "upperleg_r", "upperarm_l", "upperarm_r"}
MOTION_BONES = {"hips", "upperleg_l", "upperleg_r"}
FEET = ("foot_l", "foot_r", "toe_l", "toe_r")
LIMB_CHILDREN = {
    "shoulder": "upperarm",
    "upperarm": "lowerarm",
    "lowerarm": "hand",
    "upperleg": "lowerleg",
    "lowerleg": "foot",
    "foot": "toe",
}
LIMB_ENDS = {"hand": "lowerarm", "toe": "foot"}
MAX_NODES = 2000
MAX_JOINTS = 1000
MAX_FRAMES = 10_000
MAX_MOTION_NODES = 128
TINY = 1e-9

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
# Mixamo also has Hips and LeftLeg, so we know Kimodo only by the shin name no other rig uses.
KIMODO_MARK = "LeftShin"
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
    """A skeleton animation: frame times, and each humanoid bone's world pose at rest and per frame.

    Every dictionary is keyed by the shared bone name, such as upperarm_l.
    """

    times: Floats
    rest_rotation: dict[str, Floats]
    rest_position: dict[str, Floats]
    rotation: dict[str, Floats]
    position: dict[str, Floats]


@dataclass(frozen=True)
class Skeleton:
    """A glTF node tree: each node's parent, a parents-first order and each node's rest world."""

    nodes: list[Document]
    parent_of: dict[int, int]
    order: list[int]
    rest: list[Floats]

    def chain(self, index: int) -> list[int]:
        """Return the node and its ancestors, nearest first."""
        chain = [index]
        while chain[-1] in self.parent_of:
            chain.append(self.parent_of[chain[-1]])
        return chain


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


def unit(q: Floats) -> Floats:
    """Return quaternions scaled to length one.

    :raises RetargetError: when one has no length.
    """
    lengths = numpy.linalg.norm(q, axis=-1, keepdims=True)
    if (lengths < TINY).any():
        raise RetargetError("a rotation has zero length")
    scaled: Floats = q / lengths
    return scaled


def quat_from_matrix(matrix: Floats) -> Floats:
    """Return the rotation of 4x4 matrices as quaternions, ignoring their scale.

    :raises RetargetError: when a matrix scales an axis to nothing.
    """
    basis = matrix[..., :3, :3]
    lengths = numpy.linalg.norm(basis, axis=-2, keepdims=True)
    if (lengths < TINY).any():
        raise RetargetError("a node has zero scale")
    m = basis / lengths
    xx, yy, zz = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    xy, yx = m[..., 0, 1] + m[..., 1, 0], m[..., 1, 0] - m[..., 0, 1]
    xz, zx = m[..., 0, 2] + m[..., 2, 0], m[..., 0, 2] - m[..., 2, 0]
    yz, zy = m[..., 1, 2] + m[..., 2, 1], m[..., 2, 1] - m[..., 1, 2]
    candidates = numpy.stack(
        [
            numpy.stack([zy, zx, yx, 1 + xx + yy + zz], axis=-1),
            numpy.stack([1 + xx - yy - zz, xy, xz, zy], axis=-1),
            numpy.stack([xy, 1 - xx + yy - zz, yz, zx], axis=-1),
            numpy.stack([xz, yz, 1 - xx - yy + zz, yx], axis=-1),
        ],
        axis=-2,
    )
    best = numpy.argmax(candidates[..., [0, 1, 2, 3], [3, 0, 1, 2]], axis=-1)
    chosen = numpy.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :]
    return unit(chosen)


def matrix_from_quat(q: Floats) -> Floats:
    x, y, z, w = numpy.moveaxis(q, -1, 0)
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return numpy.stack([numpy.stack(row, axis=-1) for row in rows], axis=-2)


def compose(rotation: Floats, translation: Floats, scale: Floats) -> Floats:
    """Return the 4x4 matrices of translation, rotation and scale, over any leading axes."""
    shape = numpy.broadcast_shapes(rotation.shape[:-1], translation.shape[:-1], scale.shape[:-1])
    matrix = numpy.zeros((*shape, 4, 4))
    matrix[..., :3, :3] = matrix_from_quat(rotation) * scale[..., None, :]
    matrix[..., :3, 3] = translation
    matrix[..., 3, 3] = 1.0
    return matrix


def numbers(node: Document, key: str, default: Floats) -> Floats:
    """Return a node's list of numbers under `key`, or the default when it has none.

    :raises GlbError: when it is not a list of as many finite numbers as the default.
    """
    value = node.get(key)
    if value is None:
        return default
    if (
        not isinstance(value, list)
        or len(value) != len(default)
        or not all(isinstance(item, int | float) and not isinstance(item, bool) for item in value)
    ):
        raise GlbError(f"a node's {key} is not {len(default)} numbers")
    try:
        array = numpy.array(value, dtype=numpy.float64)
    except OverflowError as error:
        raise GlbError(f"a node's {key} is not finite") from error
    if not numpy.isfinite(array).all():
        raise GlbError(f"a node's {key} is not finite")
    return array


def local_matrix(node: Document) -> Floats:
    if "matrix" in node:
        matrix: Floats = numbers(node, "matrix", numpy.eye(4).ravel()).reshape(4, 4).T
        return matrix
    return compose(
        unit(numbers(node, "rotation", IDENTITY)),
        numbers(node, "translation", ORIGIN),
        numbers(node, "scale", UNIT_SCALE),
    )


def parents_first(count: int, parent_of: dict[int, int]) -> list[int]:
    """Order node indices so every parent comes before its children.

    :raises GlbError: when the nodes form a loop.
    """
    graph = {index: [parent_of[index]] if index in parent_of else [] for index in range(count)}
    try:
        return list(TopologicalSorter(graph).static_order())
    except CycleError as error:
        raise GlbError("the nodes form a loop") from error


def skeleton(document: Document) -> Skeleton:
    """Read a document's node tree and every node's rest world matrix.

    :raises GlbError: when the nodes do not form a tree.
    :raises RetargetError: when there are more than MAX_NODES.
    """
    nodes = entries(document, "nodes")
    if len(nodes) > MAX_NODES:
        raise RetargetError(f"a file has more than {MAX_NODES} nodes")
    parent_of: dict[int, int] = {}
    for index, node in enumerate(nodes):
        children = node.get("children", [])
        if not isinstance(children, list):
            raise GlbError("a node's children are not a list")
        for child in children:
            child_index = index_into(child, nodes, "a node's child")
            if child_index in parent_of:
                raise GlbError("a node has two parents")
            parent_of[child_index] = index
    order = parents_first(len(nodes), parent_of)
    rest = [numpy.eye(4)] * len(nodes)
    for index in order:
        local = local_matrix(nodes[index])
        rest[index] = rest[parent_of[index]] @ local if index in parent_of else local
    return Skeleton(nodes, parent_of, order, rest)


def bone_map(tree: Skeleton, candidates: Iterable[int]) -> dict[str, int]:
    """Return the humanoid bones among the candidate nodes; the first node wins a shared name.

    :raises GlbError: when a node's name is not text.
    """
    names: dict[int, str] = {}
    for index in candidates:
        name = tree.nodes[index].get("name", "")
        if not isinstance(name, str):
            raise GlbError("a node's name is not text")
        names[index] = name
    kimodo = KIMODO_MARK in names.values()
    found: dict[str, int] = {}
    for index, name in names.items():
        bone = KIMODO.get(name) if kimodo else canonical(name)
        if bone and bone not in found:
            found[bone] = index
    return found


def accessor_span(document: Document, spec: Document, count: int, size: int) -> tuple[int, int]:
    """Return where an accessor starts in the binary buffer and the stride between its rows.

    :raises GlbError: when its rows overlap or reach past its bufferView or the buffer.
    :raises RetargetError: when the data lives outside the GLB's binary buffer.
    """
    views = entries(document, "bufferViews")
    view = views[index_into(spec.get("bufferView"), views, "an accessor's bufferView")]
    if view.get("buffer") != 0:
        raise RetargetError("the animation data must live in the GLB")
    view_start = natural(view.get("byteOffset", 0), "a bufferView byteOffset")
    view_end = view_start + natural(view.get("byteLength"), "a bufferView byteLength")
    start = view_start + natural(spec.get("byteOffset", 0), "an accessor byteOffset")
    stride = natural(view.get("byteStride", 0), "a byteStride") or size
    if stride < size:
        raise GlbError("a byteStride is shorter than its element")
    if start + (count - 1) * stride + size > view_end:
        raise GlbError("an accessor reaches past its bufferView")
    return start, stride


def read_floats(document: Document, binary: bytearray, index: object, kind: str) -> Floats:
    """Return an accessor of float SCALAR, VEC3 or VEC4 rows as a (count, width) array.

    :raises GlbError: when the accessor does not fit its buffer.
    :raises RetargetError: when it holds other data, too many rows or numbers that are not finite.
    """
    accessors = entries(document, "accessors")
    spec = accessors[index_into(index, accessors, "an animation sampler")]
    if spec.get("componentType") != FLOAT or spec.get("type") != kind or "sparse" in spec:
        raise RetargetError(f"only plain float {kind} animation data is supported")
    count = natural(spec.get("count"), "an accessor count")
    if not 0 < count <= MAX_FRAMES:
        raise RetargetError(f"an animation needs 1 to {MAX_FRAMES} keys")
    width = COMPONENTS[kind]
    start, stride = accessor_span(document, spec, count, 4 * width)
    if start + (count - 1) * stride + 4 * width > len(binary):
        raise GlbError("an accessor reaches past the binary buffer")
    rows = numpy.ndarray(
        (count, width), dtype="<f4", buffer=binary, offset=start, strides=(stride, 4)
    )
    values = rows.astype(numpy.float64)
    if not numpy.isfinite(values).all():
        raise RetargetError("the animation holds numbers that are not finite")
    return values


def read_track(
    document: Document, binary: bytearray, sampler: Document, kind: str
) -> tuple[Floats, Floats, bool]:
    """Return a sampler's key times, its values and whether it steps between keys.

    :raises RetargetError: when it is cubic, or its times and values do not line up.
    """
    interpolation = sampler.get("interpolation", "LINEAR")
    if interpolation not in ("LINEAR", "STEP"):
        raise RetargetError("only LINEAR and STEP animation is supported")
    times = read_floats(document, binary, sampler.get("input"), "SCALAR")[:, 0]
    values = read_floats(document, binary, sampler.get("output"), kind)
    if len(values) != len(times):
        raise RetargetError("an animation sampler has more or fewer values than key times")
    if (numpy.diff(times) <= 0).any():
        raise RetargetError("an animation sampler's key times do not rise")
    return times, values, interpolation == "STEP"


def slerp(a: Floats, b: Floats, weight: Floats) -> Floats:
    dot = (a * b).sum(axis=-1, keepdims=True)
    b = numpy.where(dot < 0, -b, b)
    angle = numpy.arccos(numpy.clip(numpy.abs(dot), 0.0, 1.0))
    sine = numpy.sin(angle)
    near = sine < TINY
    safe = numpy.where(near, 1.0, sine)
    weight_a = numpy.where(near, 1 - weight, numpy.sin((1 - weight) * angle) / safe)
    weight_b = numpy.where(near, weight, numpy.sin(weight * angle) / safe)
    return unit(weight_a * a + weight_b * b)


def resample(times: Floats, values: Floats, step: bool, timeline: Floats, spin: bool) -> Floats:
    """Return a track's values at the timeline's times, holding its first and last keys."""
    after = numpy.searchsorted(times, timeline, side="right")
    if step or len(times) == 1:
        held: Floats = values[numpy.clip(after - 1, 0, len(times) - 1)]
        return held
    first = numpy.clip(after - 1, 0, len(times) - 2)
    span = times[first + 1] - times[first]
    weight = numpy.clip((timeline - times[first]) / span, 0.0, 1.0)[:, None]
    if spin:
        return slerp(values[first], values[first + 1], weight)
    blended: Floats = values[first] + (values[first + 1] - values[first]) * weight
    return blended


def sample_channels(
    document: Document, binary: bytearray, clip: Document, tree: Skeleton, wanted: set[int]
) -> tuple[Floats, dict[int, dict[str, Floats]]]:
    """Return one timeline, the union of all key times, and the wanted nodes' tracks on it.

    :raises RetargetError: when no channel moves a wanted node, or the timeline is too long.
    """
    samplers = entries(clip, "samplers")
    picked: dict[tuple[int, str], Document] = {}
    for channel in entries(clip, "channels"):
        target = channel.get("target")
        if not isinstance(target, dict):
            raise GlbError("an animation channel has no target")
        path = target.get("path")
        if not isinstance(path, str) or path not in TRACK_TYPES or "node" not in target:
            continue
        node = index_into(target["node"], tree.nodes, "an animation channel")
        if node in wanted:
            sampler = index_into(channel.get("sampler"), samplers, "an animation channel")
            picked.setdefault((node, path), samplers[sampler])
    if not picked:
        raise RetargetError("the motion does not animate its humanoid bones")
    keys = {
        key: read_track(document, binary, sampler, TRACK_TYPES[key[1]])
        for key, sampler in picked.items()
    }
    timeline = numpy.unique(numpy.concatenate([times for times, _, _ in keys.values()]))
    if len(timeline) > MAX_FRAMES:
        raise RetargetError(f"the motion has more than {MAX_FRAMES} frames")
    tracks: dict[int, dict[str, Floats]] = {}
    for (node, path), (times, values, step) in keys.items():
        tracks.setdefault(node, {})[path] = resample(
            times, values, step, timeline, path == "rotation"
        )
    return timeline, tracks


def animated_local(node: Document, tracks: dict[str, Floats], frames: int) -> Floats:
    """Return a node's local matrix per frame, from its animated or its rest TRS.

    :raises RetargetError: when a channel animates a node that has a matrix.
    """
    if "matrix" in node:
        if tracks:
            raise RetargetError("a motion animates a node that has a matrix")
        return numpy.broadcast_to(local_matrix(node), (frames, 4, 4))
    rotation = tracks.get("rotation", numbers(node, "rotation", IDENTITY))
    return compose(
        numpy.broadcast_to(unit(rotation), (frames, 4)),
        numpy.broadcast_to(
            tracks.get("translation", numbers(node, "translation", ORIGIN)), (frames, 3)
        ),
        numpy.broadcast_to(tracks.get("scale", numbers(node, "scale", UNIT_SCALE)), (frames, 3)),
    )


def read_motion(data: bytes) -> Motion:
    """Read the first animation of a skeleton GLB as world rotations and positions per frame.

    :raises GlbError: when the data is not a GLB.
    :raises RetargetError: when it has no usable animation.
    """
    document, binary = read_glb(data)
    animations = entries(document, "animations")
    if not animations:
        raise RetargetError("the motion has no animation")
    tree = skeleton(document)
    bones = bone_map(tree, range(len(tree.nodes)))
    missing = MOTION_BONES - set(bones)
    if missing:
        raise RetargetError(f"the motion's skeleton lacks {sorted(missing)}")
    wanted = {index for joint in bones.values() for index in tree.chain(joint)}
    if len(wanted) > MAX_MOTION_NODES:
        raise RetargetError(f"the motion's bones hang under more than {MAX_MOTION_NODES} nodes")
    times, tracks = sample_channels(document, binary, animations[0], tree, wanted)
    world: dict[int, Floats] = {}
    for index in tree.order:
        if index in wanted:
            local = animated_local(tree.nodes[index], tracks.get(index, {}), len(times))
            parent = tree.parent_of.get(index)
            world[index] = local if parent is None else world[parent] @ local
    return Motion(
        times,
        rest_rotation={bone: quat_from_matrix(tree.rest[index]) for bone, index in bones.items()},
        rest_position={bone: tree.rest[index][:3, 3] for bone, index in bones.items()},
        rotation={bone: quat_from_matrix(world[index]) for bone, index in bones.items()},
        position={bone: world[index][:, :3, 3] for bone, index in bones.items()},
    )


def rig_bones(document: Document, tree: Skeleton) -> dict[str, int]:
    """Return the character's humanoid bones by shared name, from the joints of all its skins.

    :raises RetargetError: when it has no skin, too many joints or lacks the core humanoid bones.
    """
    skins = entries(document, "skins")
    if not skins:
        raise RetargetError("the character has no skin; rig it first")
    joints: set[int] = set()
    for skin in skins:
        listed = skin.get("joints")
        if not isinstance(listed, list):
            raise GlbError("a skin's joints are not a list")
        joints.update(index_into(joint, tree.nodes, "a skin joint") for joint in listed)
    if len(joints) > MAX_JOINTS:
        raise RetargetError(f"the character has more than {MAX_JOINTS} joints")
    found = bone_map(tree, sorted(joints))
    missing = REQUIRED_BONES - set(found)
    if missing:
        raise RetargetError(f"the character's skeleton lacks humanoid bones: {sorted(missing)}")
    return found


def facing_turn(motion: Motion, tree: Skeleton, bones: dict[str, int]) -> Floats:
    """Return the turn about the vertical axis that lines the source's hips up with the rig's."""
    source = motion.rest_position["upperleg_l"] - motion.rest_position["upperleg_r"]
    target = tree.rest[bones["upperleg_l"]][:3, 3] - tree.rest[bones["upperleg_r"]][:3, 3]
    angle = numpy.arctan2(target[0], target[2]) - numpy.arctan2(source[0], source[2])
    return numpy.array([0.0, numpy.sin(angle / 2), 0.0, numpy.cos(angle / 2)])


def standing(hips: Floats, feet: Sequence[Floats]) -> tuple[float, float]:
    """Return the ground under a rest skeleton, at its lowest foot, and the hips' height over it."""
    ground = min(float(foot[1]) for foot in feet) if feet else 0.0
    return ground, float(hips[1]) - ground


def hip_track(
    motion: Motion, tree: Skeleton, bones: dict[str, int], turn: Floats, in_place: bool
) -> Floats:
    """Return the character's local hip translation per frame.

    The motion's ground is at height zero, so we keep its hip height above that ground; its
    travel counts from the first frame. Both scale by the ratio of the standing hip heights.
    """
    hips = tree.rest[bones["hips"]][:3, 3]
    ground, target_height = standing(hips, [tree.rest[bones[f]][:3, 3] for f in FEET if f in bones])
    _, source_height = standing(
        motion.rest_position["hips"],
        [motion.rest_position[f] for f in FEET if f in motion.rest_position],
    )
    if source_height <= 0 or target_height <= 0:
        raise RetargetError("could not measure the hip heights")
    scale = target_height / source_height
    path = motion.position["hips"]
    travel = path - path[0]
    travel[:, 1] = 0.0
    if in_place:
        travel[:, [0, 2]] = 0.0
    world = hips + quat_rotate(turn, travel) * scale
    world[:, 1] = ground + path[:, 1] * scale
    parent = tree.parent_of.get(bones["hips"])
    parent_world = tree.rest[parent] if parent is not None else numpy.eye(4)
    if abs(numpy.linalg.det(parent_world)) < TINY:
        raise RetargetError("the hips hang under a node with zero scale")
    points = numpy.column_stack([world, numpy.ones(len(world))])
    local: Floats = (points @ numpy.linalg.inv(parent_world).T)[:, :3]
    return local


def parent_rotation(tree: Skeleton, joint: int, world: dict[int, Floats]) -> Floats:
    """Return a joint's parent world rotation per frame.

    That is its nearest animated ancestor's, carried through the still nodes in between.
    """
    ancestors = tree.chain(joint)[1:]
    if not ancestors:
        return IDENTITY
    rest = quat_from_matrix(tree.rest[ancestors[0]])
    animated = next((index for index in ancestors if index in world), None)
    if animated is None:
        return rest
    between = quat_mul(quat_inv(quat_from_matrix(tree.rest[animated])), rest)
    return quat_mul(world[animated], between)


def shortest_arc(start: Floats, end: Floats) -> Floats:
    """Return the smallest rotation that turns the direction `start` onto the direction `end`."""
    start, end = start / numpy.linalg.norm(start), end / numpy.linalg.norm(end)
    dot = float(numpy.dot(start, end))
    if dot < -1 + TINY:
        axis = numpy.cross(start, [1.0, 0.0, 0.0])
        if numpy.linalg.norm(axis) < TINY:
            axis = numpy.cross(start, [0.0, 1.0, 0.0])
        return numpy.append(axis / numpy.linalg.norm(axis), 0.0)
    return unit(numpy.append(numpy.cross(start, end), 1.0 + dot))


def rest_alignment(
    motion: Motion, tree: Skeleton, bones: dict[str, int], turn: Floats
) -> dict[str, Floats]:
    """Return, per limb bone, the turn that points the character's rest limb like the motion's.

    A character modelled with its arms down (A-pose) and a motion recorded from a T-pose rest
    differ by that turn, which we apply before the motion so the limbs follow the source.
    """
    align: dict[str, Floats] = {}
    for side in ("_l", "_r"):
        for bone, child in LIMB_CHILDREN.items():
            names = (bone + side, child + side)
            if all(name in bones and name in motion.rest_position for name in names):
                source = motion.rest_position[names[1]] - motion.rest_position[names[0]]
                target = tree.rest[bones[names[1]]][:3, 3] - tree.rest[bones[names[0]]][:3, 3]
                if numpy.linalg.norm(source) > TINY and numpy.linalg.norm(target) > TINY:
                    align[names[0]] = shortest_arc(target, quat_rotate(turn, source))
        for leaf, parent in LIMB_ENDS.items():
            if parent + side in align:
                align[leaf + side] = align[parent + side]
    return align


def joint_rotations(
    motion: Motion, tree: Skeleton, bones: dict[str, int], turn: Floats
) -> dict[int, Floats]:
    """Return each mapped joint's local rotation per frame that gives it the source's world turn."""
    mapped = {joint: bone for bone, joint in bones.items() if bone in motion.rotation}
    align = rest_alignment(motion, tree, bones, turn)
    world: dict[int, Floats] = {}
    local: dict[int, Floats] = {}
    for joint in tree.order:
        bone = mapped.get(joint)
        if bone is None:
            continue
        delta = quat_mul(motion.rotation[bone], quat_inv(motion.rest_rotation[bone]))
        delta = quat_mul(quat_mul(turn, delta), quat_inv(turn))
        rest = quat_mul(align.get(bone, IDENTITY), quat_from_matrix(tree.rest[joint]))
        world[joint] = quat_mul(delta, rest)
        local[joint] = unit(quat_mul(quat_inv(parent_rotation(tree, joint, world)), world[joint]))
    return local


def append_floats(document: Document, binary: bytearray, values: Floats, kind: str) -> int:
    data = numpy.ascontiguousarray(values, dtype=numpy.float32)
    binary.extend(b"\0" * (-len(binary) % 4))
    views = entries(document, "bufferViews")
    views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": data.nbytes})
    document["bufferViews"] = views
    binary.extend(data.tobytes())
    spec: Document = {
        "bufferView": len(views) - 1,
        "componentType": FLOAT,
        "count": len(data),
        "type": kind,
    }
    if kind == "SCALAR":
        spec["min"], spec["max"] = [float(data.min())], [float(data.max())]
    accessors = entries(document, "accessors")
    accessors.append(spec)
    document["accessors"] = accessors
    return len(accessors) - 1


def add_clip(
    document: Document,
    binary: bytearray,
    rig: tuple[Skeleton, dict[str, int]],
    motion: Motion,
    name: str,
    in_place: bool,
) -> None:
    """Retarget one motion onto the character and add it as an animation named `name`."""
    tree, bones = rig
    turn = facing_turn(motion, tree, bones)
    times = append_floats(document, binary, motion.times, "SCALAR")
    samplers: list[Document] = []
    channels: list[Document] = []

    def track(node: int, path: str, values: Floats, kind: str) -> None:
        output = append_floats(document, binary, values, kind)
        samplers.append({"input": times, "interpolation": "LINEAR", "output": output})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})

    for joint, rotation in joint_rotations(motion, tree, bones, turn).items():
        track(joint, "rotation", rotation, "VEC4")
    track(bones["hips"], "translation", hip_track(motion, tree, bones, turn, in_place), "VEC3")
    animations = entries(document, "animations")
    animations.append({"name": name, "channels": channels, "samplers": samplers})
    document["animations"] = animations


def retarget(
    character: bytes, motions: Sequence[tuple[str, bytes]], in_place: bool = True
) -> bytes:
    """Return the character GLB with one animation per (name, skeleton GLB) motion.

    :raises GlbError: when a file is not a GLB.
    :raises RetargetError: when the character or a motion cannot be matched up.
    """
    document, binary = read_glb(character)
    buffers = entries(document, "buffers")
    if not buffers or "uri" in buffers[0]:
        raise RetargetError("the character must keep its data inside the GLB")
    tree = skeleton(document)
    rig = (tree, rig_bones(document, tree))
    for name, data in motions:
        add_clip(document, binary, rig, read_motion(data), name, in_place)
    buffers[0]["byteLength"] = len(binary) + (-len(binary) % 4)
    return write_glb(document, binary)
