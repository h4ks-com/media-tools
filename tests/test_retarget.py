import numpy
import pytest

from conftest import VROID
from conftest import vroid_rig
from media_tools.glb import read_glb
from media_tools.glb import write_glb
from media_tools.retarget import KIMODO
from media_tools.retarget import Motion
from media_tools.retarget import RetargetError
from media_tools.retarget import canonical
from media_tools.retarget import quat_from_matrix
from media_tools.retarget import quat_mul
from media_tools.retarget import quat_rotate
from media_tools.retarget import read_motion
from media_tools.retarget import retarget

TURNED = {"name": "Armature", "rotation": [0, 1, 0, 0], "scale": [2, 2, 2]}
BY_BONE = {bone: name for name, bone in KIMODO.items()}


def worst_errors(result: Motion, source: Motion, turn: numpy.ndarray) -> tuple[float, float]:
    """Compare every mapped bone's world rotation and position with the turned source."""
    frames = len(source.times)
    turns = numpy.tile(turn, (frames, 1))
    worst_angle, worst_distance = 0.0, 0.0
    # The rig keeps its rest hips where they stand, and the motion's travel counts from frame one.
    start = source.position["Hips"][0] - source.rest_position["Hips"]
    for bone, name in VROID.items():
        expected = quat_mul(turns, source.rotation[BY_BONE[bone]])
        dot = numpy.abs((result.rotation[name] * expected).sum(axis=1))
        angle = numpy.degrees(2 * numpy.arccos(numpy.clip(dot, -1, 1))).max()
        distance = numpy.linalg.norm(
            result.position[name] - quat_rotate(turns, source.position[BY_BONE[bone]] - start),
            axis=1,
        ).max()
        worst_angle, worst_distance = max(worst_angle, angle), max(worst_distance, distance)
    return worst_angle, worst_distance


def clip(glb: bytes, index: int) -> bytes:
    """Keep one animation of a GLB, so read_motion reads it."""
    document, binary = read_glb(glb)
    document["animations"] = [document["animations"][index]]
    return write_glb(document, binary)


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

    angle, distance = worst_errors(
        read_motion(clip(result, 0)), read_motion(walk), numpy.array([0, 0, 0, 1.0])
    )

    assert angle < 0.5
    assert distance < 0.01


def test_a_turned_rig_gets_the_motion_turned(walk: bytes) -> None:
    result = retarget(vroid_rig(walk, TURNED), [("Walk", walk)], in_place=False)

    # read_motion ignores node scale, so positions compare before the rig's 2x scale.
    angle, distance = worst_errors(
        read_motion(clip(result, 0)), read_motion(walk), numpy.array([0, 1.0, 0, 0])
    )

    assert angle < 0.5
    assert distance < 0.01


def test_every_motion_becomes_a_named_clip(walk: bytes, jump: bytes) -> None:
    result = retarget(vroid_rig(walk), [("Walk", walk), ("Jump", jump)])

    document, _ = read_glb(result)

    assert [animation["name"] for animation in document["animations"]] == ["Walk", "Jump"]
    assert document["buffers"][0]["byteLength"] % 4 == 0


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


def test_in_place_clips_keep_the_hips_over_their_rest_spot(walk: bytes) -> None:
    result = read_motion(clip(retarget(vroid_rig(walk), [("Walk", walk)]), 0))

    hips = result.position["J_Bip_C_Hips"]

    assert numpy.abs(hips[:, [0, 2]] - hips[0, [0, 2]]).max() < 1e-6
    assert numpy.ptp(hips[:, 1]) > 0.01
