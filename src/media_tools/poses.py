"""Preset animation poses for a side-view character facing right, as OpenPose COCO-18 skeletons.

The skeletons steer a pose-transfer model, one image per frame. Angles are in degrees: 0 hangs
straight down and positive swings forward (to the right). A limb is (thigh, knee bend, upper arm,
elbow bend); the near side is the character's right, which faces the viewer.
"""

import io
import math
from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image
from PIL import ImageDraw

type Point = tuple[float, float]
type Limb = tuple[float, float, float, float]

SIZE = 512
HIP_Y, TORSO, THIGH, SHIN, UPPER_ARM, FOREARM = 300, 110, 80, 80, 62, 58
LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]  # fmt: skip
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
]  # fmt: skip
LINE_SHADE = 0.6
JOINT_RADIUS = 5
TPOSE = "tpose"


@dataclass(frozen=True)
class Pose:
    bob: float = 0.0
    lean: float = 0.0
    near: Limb = (0, 0, 0, 0)
    far: Limb = (0, 0, 0, 0)


@dataclass(frozen=True)
class Gait:
    stride: float
    knee: float
    arm: float
    elbow: float
    bob: float
    lean: float


def cycle(frames: int, gait: Gait) -> list[Pose]:
    """Return a looping gait: limbs swing in opposite phase and the body dips at each contact."""
    poses = []
    for index in range(frames):
        phase = 2 * math.pi * index / frames
        swing = math.sin(phase)
        poses.append(
            Pose(
                bob=-gait.bob * abs(math.cos(phase)),
                lean=gait.lean,
                near=(
                    gait.stride * swing,
                    max(0.0, gait.knee * math.sin(phase + math.pi / 2)),
                    -gait.arm * swing,
                    gait.elbow,
                ),
                far=(
                    -gait.stride * swing,
                    max(0.0, gait.knee * math.sin(phase - math.pi / 2)),
                    gait.arm * swing,
                    gait.elbow,
                ),
            )
        )
    return poses


def _mix(first: Limb, second: Limb, share: float) -> Limb:
    a, b, c, d = (x + (y - x) * share for x, y in zip(first, second, strict=True))
    return a, b, c, d


def blend(keys: list[Pose], frames: int, loop: bool) -> list[Pose]:
    """Spread the key poses over the frames, easing linearly between neighbours."""
    spans = len(keys) if loop else len(keys) - 1
    poses = []
    for index in range(frames):
        position = index * spans / (frames if loop else frames - 1)
        first = min(int(position), spans if loop else spans - 1)
        share = position - first
        a, b = keys[first % len(keys)], keys[(first + 1) % len(keys)]
        poses.append(
            Pose(
                bob=a.bob + (b.bob - a.bob) * share,
                lean=a.lean + (b.lean - a.lean) * share,
                near=_mix(a.near, b.near, share),
                far=_mix(a.far, b.far, share),
            )
        )
    return poses


MOVES: dict[str, Callable[[int], list[Pose]]] = {
    "idle": lambda frames: blend(
        [Pose(0, 2, (3, 4, -4, 12), (-3, 4, 4, 12)), Pose(-4, 0, (3, 6, -8, 18), (-3, 6, 8, 18))],
        frames,
        loop=True,
    ),
    "walk": lambda frames: cycle(frames, Gait(stride=38, knee=45, arm=42, elbow=22, bob=8, lean=4)),
    "run": lambda frames: cycle(
        frames, Gait(stride=55, knee=95, arm=65, elbow=85, bob=14, lean=14)
    ),
    "jump": lambda frames: blend(
        [
            Pose(0, 2, (3, 4, -5, 10), (-3, 4, 5, 10)),
            Pose(40, 18, (60, 110, -40, 30), (50, 100, -30, 30)),
            Pose(-60, 5, (30, 80, 150, 20), (20, 70, 140, 20)),
            Pose(-50, 0, (70, 120, 120, 40), (60, 110, 110, 40)),
            Pose(35, 15, (55, 100, 40, 30), (45, 90, 30, 30)),
        ],
        frames,
        loop=False,
    ),
    "attack": lambda frames: blend(
        [
            Pose(0, -8, (10, 10, 170, -30), (-15, 5, 160, -20)),
            Pose(-4, -12, (15, 20, 200, -40), (-20, 10, 190, -30)),
            Pose(4, 12, (30, 35, 110, 0), (-25, 15, 100, 10)),
            Pose(10, 20, (35, 45, 40, 10), (-30, 20, 30, 15)),
            Pose(2, 5, (15, 20, 90, 60), (-15, 10, 80, 60)),
        ],
        frames,
        loop=False,
    ),
    "hurt": lambda frames: blend(
        [
            Pose(0, 2, (3, 4, -5, 10), (-3, 4, 5, 10)),
            Pose(6, -22, (-10, 25, 60, 70), (-25, 20, 70, 80)),
            Pose(12, -28, (-5, 40, 90, 60), (-30, 35, 100, 70)),
            Pose(4, -10, (0, 15, 20, 30), (-15, 10, 25, 30)),
        ],
        frames,
        loop=False,
    ),
}


def end(origin: Point, length: float, degrees: float) -> Point:
    radians = math.radians(degrees)
    return origin[0] + length * math.sin(radians), origin[1] + length * math.cos(radians)


def keypoints(body: Pose) -> dict[int, Point]:
    hip = (SIZE / 2, HIP_Y + body.bob)
    neck = end(hip, TORSO, 180 - body.lean)
    points = {
        1: neck,
        0: (neck[0] + 22, neck[1] - 30),
        15: (neck[0] + 16, neck[1] - 38),
        17: (neck[0] - 4, neck[1] - 34),
    }
    for limb, shoulder_index, hip_index, offset in ((body.near, 2, 8, 3), (body.far, 5, 11, -3)):
        thigh, knee, arm, elbow = limb
        shoulder = (neck[0] + offset, neck[1] + 6)
        hip_point = (hip[0] + offset, hip[1])
        elbow_point = end(shoulder, UPPER_ARM, arm)
        knee_point = end(hip_point, THIGH, thigh)
        points[shoulder_index] = shoulder
        points[shoulder_index + 1] = elbow_point
        points[shoulder_index + 2] = end(elbow_point, FOREARM, arm + elbow)
        points[hip_index] = hip_point
        points[hip_index + 1] = knee_point
        points[hip_index + 2] = end(knee_point, SHIN, thigh - knee)
    return points


def front_tpose() -> dict[int, Point]:
    """Return a front-view T-pose: arms straight out at shoulder height, legs slightly apart.

    The character faces the viewer, so its right side is on the left of the picture.
    """
    middle = SIZE / 2
    points: dict[int, Point] = {
        0: (middle, 92),
        1: (middle, 130),
        14: (middle - 10, 84),
        15: (middle + 10, 84),
        16: (middle - 20, 90),
        17: (middle + 20, 90),
    }
    for side, shoulder_index, hip_index in ((-1, 2, 8), (1, 5, 11)):
        points[shoulder_index] = (middle + side * 38, 136)
        points[shoulder_index + 1] = (middle + side * 118, 136)
        points[shoulder_index + 2] = (middle + side * 196, 136)
        points[hip_index] = (middle + side * 28, 262)
        points[hip_index + 1] = (middle + side * 36, 358)
        points[hip_index + 2] = (middle + side * 44, 458)
    return points


def draw(points: dict[int, Point]) -> bytes:
    """Draw OpenPose COCO-18 points and limbs as a PNG on black."""
    picture = Image.new("RGB", (SIZE, SIZE), "black")
    pen = ImageDraw.Draw(picture)
    for index, (a, b) in enumerate(LIMBS):
        if a in points and b in points:
            red, green, blue = (int(channel * LINE_SHADE) for channel in COLORS[index])
            pen.line([points[a], points[b]], fill=(red, green, blue), width=8)
    for index, point in points.items():
        x, y = point
        box = [x - JOINT_RADIUS, y - JOINT_RADIUS, x + JOINT_RADIUS, y + JOINT_RADIUS]
        pen.ellipse(box, fill=COLORS[index])
    output = io.BytesIO()
    picture.save(output, format="PNG")
    return output.getvalue()


def skeleton_png(move: str, frames: int, frame: int) -> bytes:
    """Return the OpenPose skeleton for one frame of a preset move, as a PNG on black.

    The move tpose is one front-view frame, for drawing a character ready to rig.

    :raises ValueError: when the move is unknown or the frame is out of range.
    """
    if move == TPOSE:
        return draw(front_tpose())
    if move not in MOVES:
        raise ValueError(f"move is one of {', '.join([*MOVES, TPOSE])}")
    if not 0 <= frame < frames:
        raise ValueError("frame must be below frames")
    return draw(keypoints(MOVES[move](frames)[frame]))
