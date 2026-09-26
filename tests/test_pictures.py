import io

import numpy
import pytest
from PIL import Image
from PIL import ImageSequence

from conftest import png
from media_tools import pictures
from media_tools import poses


def test_key_cutout_clears_the_flat_background() -> None:
    cut = Image.open(
        io.BytesIO(pictures.Cutter("unused").cut(png((64, 64), (20, 20, 44, 44)), "key"))
    )

    assert cut.mode == "RGBA"
    alpha = numpy.asarray(cut)[..., 3]
    assert (alpha[2, 2], alpha[32, 32]) == (0, 255)


def test_pixel_art_fits_the_size_and_scales_up() -> None:
    art = Image.open(io.BytesIO(pictures.pixel_art(png((200, 100), (50, 25, 150, 75)), 32, 4, 2)))

    assert art.size == (64, 32)


def test_sprite_frames_share_one_crop_and_height() -> None:
    frames = [png((100, 100), (30, 20, 50 + shift, 80)) for shift in (0, 10, 20)]

    sheet = Image.open(io.BytesIO(pictures.sprite_sheet(frames, 32, 8, 1)))
    gif = Image.open(io.BytesIO(pictures.sprite_gif(frames, 32, 8, 2)))

    assert sheet.height == 32
    assert sheet.width % 3 == 0
    assert (len(list(ImageSequence.Iterator(gif))), gif.height) == (3, 64)


def test_sprite_frames_of_different_sizes_are_refused() -> None:
    with pytest.raises(ValueError, match="differ in size"):
        pictures.sprite_frames(
            [png((100, 100), (30, 20, 50, 80)), png((90, 90), (30, 20, 50, 80))], 32, 8
        )


def test_a_frame_without_a_character_is_refused() -> None:
    with pytest.raises(ValueError, match="no character"):
        pictures.sprite_frames([png((100, 100)), png((100, 100))], 32, 8)


@pytest.mark.parametrize("move", sorted(poses.MOVES))
def test_every_move_draws_every_frame(move: str) -> None:
    for frame in range(6):
        picture = Image.open(io.BytesIO(poses.skeleton_png(move, 6, frame)))
        assert picture.size == (poses.SIZE, poses.SIZE)
        assert picture.getbbox() is not None


@pytest.mark.parametrize(
    ("move", "frame", "error"), [("dance", 0, "move is one of"), ("walk", 9, "below frames")]
)
def test_bad_pose_requests_are_refused(move: str, frame: int, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        poses.skeleton_png(move, 4, frame)
