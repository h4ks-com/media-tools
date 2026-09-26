from collections.abc import Callable

import numpy
import pytest

from conftest import DATA
from conftest import VROID
from conftest import add_root
from conftest import edit
from conftest import vroid_rig
from media_tools.glb import Document
from media_tools.glb import GlbError
from media_tools.glb import read_glb
from media_tools.glb import write_glb
from media_tools.mesh import make_opaque
from media_tools.retarget import FEET
from media_tools.retarget import KIMODO
from media_tools.retarget import Floats
from media_tools.retarget import Motion
from media_tools.retarget import RetargetError
from media_tools.retarget import append_floats
from media_tools.retarget import canonical
from media_tools.retarget import parents_first
from media_tools.retarget import quat_from_matrix
from media_tools.retarget import quat_mul
from media_tools.retarget import quat_rotate
from media_tools.retarget import read_floats
from media_tools.retarget import read_motion
from media_tools.retarget import retarget

TURNED = {"name": "Armature", "rotation": [0, 1, 0, 0], "scale": [2, 2, 2]}
STILL = numpy.array([0, 0, 0, 1.0])
HALF_TURN = numpy.array([0, 1.0, 0, 0])
MIXAMO = {
    "hips": "Hips",
    "spine": "Spine",
    "chest": "Spine1",
    "upperchest": "Spine2",
    "neck": "Neck",
    "head": "Head",
    **{
        f"{bone}_{side}": f"{mark}{name}"
        for side, mark in (("l", "Left"), ("r", "Right"))
        for bone, name in (
            ("shoulder", "Shoulder"),
            ("upperarm", "Arm"),
            ("lowerarm", "ForeArm"),
            ("hand", "Hand"),
            ("upperleg", "UpLeg"),
            ("lowerleg", "Leg"),
            ("foot", "Foot"),
            ("toe", "ToeBase"),
        )
    },
}
HIPS, SPINE, LEFT_ARM = 0, 1, 11


def worst_errors(
    result: Motion,
    source: Motion,
    turn: Floats,
    scale: float = 1.0,
    bones: set[str] | None = None,
) -> tuple[float, float]:
    """Compare the bones' world rotations and positions with the turned, scaled source.

    The rig stands its rest hips over the motion's first spot, with its feet on the motion's
    ground, so we expect the source shifted by that first spot sideways and by its rest ground.
    """
    ground = min(source.rest_position[foot][1] for foot in FEET)
    start = source.position["hips"][0]
    offset = numpy.array([start[0], -ground, start[2]])
    worst_angle, worst_distance = 0.0, 0.0
    for bone in bones or set(source.rotation):
        expected = quat_mul(turn, source.rotation[bone])
        dot = numpy.abs((result.rotation[bone] * expected).sum(axis=1))
        angle = numpy.degrees(2 * numpy.arccos(numpy.clip(dot, -1, 1))).max()
        distance = numpy.linalg.norm(
            result.position[bone] - scale * quat_rotate(turn, source.position[bone] - offset),
            axis=1,
        ).max()
        worst_angle, worst_distance = max(worst_angle, angle), max(worst_distance, distance)
    return worst_angle, worst_distance


def clip(glb: bytes, index: int) -> bytes:
    """Keep one animation of a GLB, so read_motion reads it."""
    document, binary = read_glb(glb)
    document["animations"] = [document["animations"][index]]
    return write_glb(document, binary)


def sampler_of(document: Document, node: int, path: str) -> Document:
    animation = document["animations"][0]
    channel = next(
        channel
        for channel in animation["channels"]
        if channel["target"] == {"node": node, "path": path}
    )
    sampler: Document = animation["samplers"][channel["sampler"]]
    return sampler


def output_accessor(document: Document, node: int, path: str) -> Document:
    accessor: Document = document["accessors"][sampler_of(document, node, path)["output"]]
    return accessor


def every_other_key(
    node: int, path: str, interpolation: str
) -> Callable[[Document, bytearray], None]:
    """Return an edit that gives one channel every other key of its own, under `interpolation`."""

    def change(document: Document, binary: bytearray) -> None:
        sampler = sampler_of(document, node, path)
        kind = "VEC4" if path == "rotation" else "VEC3"
        times = read_floats(document, binary, sampler["input"], "SCALAR")[::2]
        values = read_floats(document, binary, sampler["output"], kind)[::2]
        sampler["input"] = append_floats(document, binary, times, "SCALAR")
        sampler["output"] = append_floats(document, binary, values, kind)
        sampler["interpolation"] = interpolation
        document["buffers"][0]["byteLength"] = len(binary)

    return change


def lowest_foot(motion: Motion) -> float:
    return min(float(motion.position[foot][:, 1].min()) for foot in FEET)


@pytest.mark.parametrize(
    ("name", "bone"),
    [
        ("J_Bip_C_Chest", "chest"),
        ("J_Bip_C_UpperChest", "upperchest"),
        ("J_Bip_L_LowerArm", "lowerarm_l"),
        ("J_Bip_R_UpperLeg", "upperleg_r"),
        ("mixamorig:LeftUpLeg", "upperleg_l"),
        ("mixamorig:LeftLeg", "lowerleg_l"),
        ("mixamorig:RightForeArm", "lowerarm_r"),
        ("mixamorig:Spine2", "upperchest"),
        ("thigh.L", "upperleg_l"),
        ("J_Bip_L_Thumb1", None),
        ("Hand", None),
        ("Jaw", None),
    ],
)
def test_canonical_names(name: str, bone: str | None) -> None:
    assert canonical(name) == bone


def test_retargeting_onto_the_same_skeleton_keeps_the_motion(walk: bytes) -> None:
    result = retarget(vroid_rig(walk), [("Walk", walk)], in_place=False)

    angle, distance = worst_errors(read_motion(clip(result, 0)), read_motion(walk), STILL)

    assert angle < 0.5
    assert distance < 0.01


def test_a_turned_and_scaled_rig_gets_the_motion_turned_and_scaled(walk: bytes) -> None:
    result = retarget(vroid_rig(walk, TURNED), [("Walk", walk)], in_place=False)

    angle, distance = worst_errors(
        read_motion(clip(result, 0)), read_motion(walk), HALF_TURN, scale=2.0
    )

    assert angle < 0.5
    assert distance < 0.02


@pytest.mark.parametrize("name", ["walk", "jump"])
def test_in_place_clips_keep_the_feet_on_the_ground_and_the_hips_on_the_spot(
    name: str, request: pytest.FixtureRequest
) -> None:
    motion: bytes = request.getfixturevalue(name)
    character = vroid_rig(motion)
    rest_feet = min(read_motion(motion).rest_position[foot][1] for foot in FEET)

    result = read_motion(clip(retarget(character, [("Clip", motion)]), 0))

    hips = result.position["hips"]
    assert abs(lowest_foot(result) - rest_feet) < 0.03
    assert numpy.abs(hips[:, [0, 2]] - hips[0, [0, 2]]).max() < 1e-6
    assert numpy.ptp(hips[:, 1]) > 0.01


def test_hips_keep_the_motion_height_when_the_first_frame_crouches(walk: bytes) -> None:
    def crouch_first_frame(document: Document, binary: bytearray) -> None:
        accessor = output_accessor(document, HIPS, "translation")
        start = document["bufferViews"][accessor["bufferView"]]["byteOffset"] + 4
        binary[start : start + 4] = numpy.array([0.5], dtype="<f4").tobytes()

    crouching = edit(walk, crouch_first_frame)
    rest_feet = min(read_motion(walk).rest_position[foot][1] for foot in FEET)

    result = read_motion(clip(retarget(vroid_rig(walk), [("Walk", crouching)]), 0))

    feet = min(float(result.position[foot][1:, 1].min()) for foot in FEET)
    assert abs(feet - rest_feet) < 0.03


def test_a_motion_under_a_turned_and_scaled_armature_reads_in_world_space(walk: bytes) -> None:
    plain = read_motion(walk)

    moved = read_motion(edit(walk, lambda document, _: add_root(document, TURNED)))

    for bone in plain.rotation:
        assert numpy.allclose(
            numpy.abs((moved.rotation[bone] * quat_mul(HALF_TURN, plain.rotation[bone])).sum(1)),
            1.0,
            atol=1e-5,
        )
        assert numpy.allclose(
            moved.position[bone], 2 * quat_rotate(HALF_TURN, plain.position[bone]), atol=1e-4
        )


def test_a_still_node_between_two_joints_follows_the_joint_above(walk: bytes) -> None:
    def drop_spine_joint(document: Document, _: bytearray) -> None:
        document["skins"][0]["joints"].remove(SPINE)

    character = edit(vroid_rig(walk), drop_spine_joint)

    result = read_motion(clip(retarget(character, [("Walk", walk)], in_place=False), 0))

    angle, _ = worst_errors(result, read_motion(walk), STILL, bones=set(VROID) - {"spine"})
    _, distance = worst_errors(
        result, read_motion(walk), STILL, bones={bone for bone in VROID if "leg" in bone}
    )
    assert angle < 0.5
    assert distance < 0.01


def test_joints_come_from_every_skin(walk: bytes) -> None:
    def split_skin(document: Document, _: bytearray) -> None:
        joints = document["skins"][0]["joints"]
        document["skins"] = [{"joints": joints[:12]}, {"joints": joints[12:]}]

    result = retarget(edit(vroid_rig(walk), split_skin), [("Walk", walk)], in_place=False)

    angle, distance = worst_errors(read_motion(clip(result, 0)), read_motion(walk), STILL)
    assert angle < 0.5
    assert distance < 0.01


def test_channels_with_their_own_key_times_meet_on_one_timeline(walk: bytes) -> None:
    plain = read_motion(walk)

    sparse = read_motion(edit(walk, every_other_key(LEFT_ARM, "rotation", "LINEAR")))

    assert len(sparse.times) == len(plain.times)
    assert numpy.allclose(sparse.rotation["upperarm_l"][::2], plain.rotation["upperarm_l"][::2])
    assert numpy.allclose(sparse.rotation["hips"], plain.rotation["hips"])


def test_step_channels_hold_each_key(walk: bytes) -> None:
    plain = read_motion(walk)

    stepped = read_motion(edit(walk, every_other_key(HIPS, "translation", "STEP")))

    assert numpy.allclose(stepped.position["hips"][1::2], plain.position["hips"][0:-1:2])


def test_cubic_channels_are_refused(walk: bytes) -> None:
    def make_cubic(document: Document, _: bytearray) -> None:
        sampler_of(document, LEFT_ARM, "rotation")["interpolation"] = "CUBICSPLINE"

    with pytest.raises(RetargetError, match="LINEAR and STEP"):
        read_motion(edit(walk, make_cubic))


def test_unprefixed_mixamo_names_use_the_alias_table(walk: bytes) -> None:
    def rename(document: Document, _: bytearray) -> None:
        for node in document["nodes"]:
            if node["name"] in KIMODO:
                node["name"] = MIXAMO[KIMODO[node["name"]]]

    mixamo = read_motion(edit(walk, rename))

    plain = read_motion(walk)
    for bone in plain.rest_position:
        assert numpy.allclose(mixamo.rest_position[bone], plain.rest_position[bone])


def test_a_later_node_with_a_taken_name_does_not_replace_the_bone(walk: bytes) -> None:
    def name_jaw_hips(document: Document, _: bytearray) -> None:
        document["nodes"][7]["name"] = "Hips"

    twin = read_motion(edit(walk, name_jaw_hips))

    assert numpy.allclose(twin.position["hips"], read_motion(walk).position["hips"])


def test_a_character_with_a_meshopt_fallback_buffer_is_accepted(walk: bytes) -> None:
    def add_buffers(document: Document, _: bytearray) -> None:
        document["buffers"] += [
            {"byteLength": 8, "extensions": {"EXT_meshopt_compression": {"fallback": True}}},
            {"byteLength": 4, "uri": "data:application/octet-stream;base64,AAAAAA=="},
        ]

    result = retarget(edit(vroid_rig(walk), add_buffers), [("Walk", walk)])

    document, binary = read_glb(result)
    assert len(document["buffers"]) == 3
    assert document["buffers"][0]["byteLength"] == len(binary)


def test_a_character_from_mesh_takes_the_motion(walk: bytes) -> None:
    character = make_opaque((DATA / "packed-character.glb").read_bytes())

    result = retarget(character, [("Walk", walk)], in_place=False)

    angle, distance = worst_errors(read_motion(clip(result, 0)), read_motion(walk), STILL)
    assert angle < 0.5
    assert distance < 0.01


def test_a_character_whose_data_lives_in_a_data_uri_is_refused(walk: bytes) -> None:
    def move_data(document: Document, _: bytearray) -> None:
        document["buffers"][0]["uri"] = "data:application/octet-stream;base64,AAAA"

    with pytest.raises(RetargetError, match="inside the GLB"):
        retarget(edit(vroid_rig(walk), move_data), [("Walk", walk)])


def test_a_matrix_node_reads_like_its_rotation() -> None:
    quarter = numpy.array([0, numpy.sin(numpy.pi / 4), 0, numpy.cos(numpy.pi / 4)])
    matrix = numpy.eye(4)
    matrix[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]

    assert numpy.allclose(numpy.abs(quat_from_matrix(matrix)), numpy.abs(quarter))


@pytest.mark.parametrize(
    "matrix",
    [
        numpy.diag([1.0, -1.0, -1.0, 1.0]),
        numpy.diag([-1.0, 1.0, -1.0, 1.0]),
        numpy.diag([-1.0, -1.0, 1.0, 1.0]),
    ],
)
def test_half_turn_matrices_read_as_unit_quaternions(matrix: numpy.ndarray) -> None:
    assert numpy.isclose(numpy.linalg.norm(quat_from_matrix(matrix)), 1.0)


def test_a_zero_scale_is_refused(walk: bytes) -> None:
    with pytest.raises(RetargetError, match="zero scale"):
        retarget(vroid_rig(walk, {"scale": [0, 0, 0]}), [("Walk", walk)])


def test_a_character_without_a_skin_is_refused(walk: bytes) -> None:
    with pytest.raises(RetargetError, match="no skin"):
        retarget(walk, [("Walk", walk)])


def test_a_character_without_humanoid_bones_is_refused(walk: bytes) -> None:
    document, binary = read_glb(vroid_rig(walk))
    for node in document["nodes"]:
        node["name"] = f"bone{id(node)}"

    with pytest.raises(RetargetError, match="lacks humanoid bones"):
        retarget(write_glb(document, binary), [("Walk", walk)])


def test_a_motion_without_animation_is_refused(walk: bytes) -> None:
    with pytest.raises(RetargetError, match="no animation"):
        retarget(vroid_rig(walk), [("Walk", vroid_rig(walk))])


def test_non_finite_animation_data_is_refused(walk: bytes) -> None:
    def spoil(document: Document, binary: bytearray) -> None:
        accessor = output_accessor(document, HIPS, "translation")
        start = document["bufferViews"][accessor["bufferView"]]["byteOffset"]
        binary[start : start + 4] = numpy.array([numpy.nan], dtype="<f4").tobytes()

    with pytest.raises(RetargetError, match="not finite"):
        retarget(vroid_rig(walk), [("Walk", edit(walk, spoil))])


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"count": 20_000}, RetargetError, "keys"),
        ({"byteOffset": 10**9}, GlbError, "past its bufferView"),
    ],
)
def test_accessors_are_bounded(
    walk: bytes, change: Document, error: type[Exception], match: str
) -> None:
    def spoil(document: Document, _: bytearray) -> None:
        output_accessor(document, HIPS, "rotation").update(change)

    with pytest.raises(error, match=match):
        read_motion(edit(walk, spoil))


def test_a_view_that_claims_more_than_the_buffer_is_refused(walk: bytes) -> None:
    def spoil(document: Document, _: bytearray) -> None:
        accessor = output_accessor(document, HIPS, "rotation")
        view = document["bufferViews"][accessor["bufferView"]]
        view["byteOffset"], view["byteLength"] = 10**9, 2400

    with pytest.raises(GlbError, match="past the binary buffer"):
        read_motion(edit(walk, spoil))


def test_a_stride_shorter_than_its_element_is_refused(walk: bytes) -> None:
    def spoil(document: Document, _: bytearray) -> None:
        accessor = output_accessor(document, HIPS, "rotation")
        document["bufferViews"][accessor["bufferView"]]["byteStride"] = 4

    with pytest.raises(GlbError, match="byteStride"):
        read_motion(edit(walk, spoil))


def test_mismatched_sampler_counts_are_refused(walk: bytes) -> None:
    def spoil(document: Document, _: bytearray) -> None:
        output_accessor(document, HIPS, "rotation")["count"] = 100

    with pytest.raises(RetargetError, match="more or fewer values"):
        read_motion(edit(walk, spoil))


def test_a_node_name_that_is_not_text_is_refused(walk: bytes) -> None:
    def spoil(document: Document, _: bytearray) -> None:
        document["nodes"][3]["name"] = 5

    with pytest.raises(GlbError, match="not text"):
        read_motion(edit(walk, spoil))


@pytest.mark.parametrize(
    ("limit", "value", "match"),
    [
        ("MAX_NODES", 10, "nodes"),
        ("MAX_JOINTS", 10, "joints"),
        ("MAX_MOTION_NODES", 10, "hang under"),
    ],
)
def test_skeletons_are_bounded(
    walk: bytes, monkeypatch: pytest.MonkeyPatch, limit: str, value: int, match: str
) -> None:
    monkeypatch.setattr(f"media_tools.retarget.{limit}", value)

    with pytest.raises(RetargetError, match=match):
        retarget(vroid_rig(walk), [("Walk", walk)])


def test_nodes_that_form_a_loop_are_refused(walk: bytes) -> None:
    def spoil(document: Document, _: bytearray) -> None:
        document["nodes"][7]["children"] = [0]

    with pytest.raises(GlbError, match="loop"):
        read_motion(edit(walk, spoil))


def test_parents_come_before_their_children() -> None:
    parent_of = {0: 3, 1: 0, 2: 1, 4: 3}

    order = parents_first(5, parent_of)

    assert sorted(order) == [0, 1, 2, 3, 4]
    assert all(order.index(parent) < order.index(child) for child, parent in parent_of.items())
