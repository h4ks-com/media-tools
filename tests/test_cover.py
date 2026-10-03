import io
from pathlib import Path

import numpy
import pytest
from PIL import Image
from PIL import ImageFont

from conftest import png
from media_tools import cover
from media_tools.pictures import PictureError

FONT = Path(__file__).parent / "data" / "LilitaOne-Regular.ttf"


def test_the_cover_is_a_square_with_the_title_on_a_dark_band() -> None:
    data = cover.cover(png((800, 600)), "The Keeper's Light", "An h4ks audiobook", FONT)

    picture = numpy.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=numpy.float32)
    assert picture.shape == (cover.SIZE, cover.SIZE)
    assert picture[: cover.TOP].mean() < picture[cover.SIZE // 2].mean()
    assert picture[cover.SIZE - cover.MARGIN - 20].min() < 50


def test_a_long_title_shrinks_into_three_lines() -> None:
    font, lines = cover.fitted_title("A VERY LONG TITLE " * 4, FONT)

    assert len(lines) <= cover.MAX_TITLE_LINES
    assert font.size < cover.TITLE_SIZES[0]


def test_a_short_title_keeps_the_largest_size() -> None:
    font, lines = cover.fitted_title("FOG", FONT)

    assert (font.size, lines) == (cover.TITLE_SIZES[0], ["FOG"])


def test_wrapping_keeps_words_whole() -> None:
    font = ImageFont.truetype(str(FONT), 100)

    width = round(font.getlength("ONE TWO"))

    assert cover.wrap("ONE TWO THREE", font, width) == ["ONE TWO", "THREE"]


def test_a_cover_needs_a_picture() -> None:
    with pytest.raises(PictureError):
        cover.cover(b"nope", "Title", "", FONT)
