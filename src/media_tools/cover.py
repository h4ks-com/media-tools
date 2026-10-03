"""Set a title on cover art in a real font, since picture models misspell words."""

from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

from media_tools.pictures import open_picture
from media_tools.pictures import png_bytes

SIZE = 1024
MARGIN = 64
TOP = 64
MAX_TITLE_LINES = 3
TITLE_SIZES = (120, 104, 92, 80, 70, 60)
SUBTITLE_SIZE = 44
LINE_SPACING = 1.08
BAND_ALPHA = 110
STROKE = 5


def wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        joined = f"{lines[-1]} {word}" if lines else word
        if lines and font.getlength(joined) <= width:
            lines[-1] = joined
        else:
            lines.append(word)
    return lines


def fitted_title(title: str, font_path: Path) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Pick the largest title size whose wrapped lines fit the width in at most three lines."""
    width = SIZE - 2 * MARGIN
    for size in TITLE_SIZES[:-1]:
        font = ImageFont.truetype(str(font_path), size)
        lines = wrap(title, font, width)
        if len(lines) <= MAX_TITLE_LINES and all(font.getlength(line) <= width for line in lines):
            return font, lines
    font = ImageFont.truetype(str(font_path), TITLE_SIZES[-1])
    return font, wrap(title, font, width)[:MAX_TITLE_LINES]


def centred(draw: ImageDraw.ImageDraw, text: str, top: int, font: ImageFont.FreeTypeFont) -> None:
    left = (SIZE - font.getlength(text)) / 2
    draw.text(
        (left, top), text, font=font, fill="white", stroke_width=STROKE, stroke_fill=(0, 0, 0, 200)
    )


def cover(data: bytes, title: str, subtitle: str, font_path: Path) -> bytes:
    """Crop the picture to a 1024 px square PNG with the title on top and the subtitle below.

    :raises PictureError: when the data is no PNG, JPEG or WebP picture.
    """
    picture = ImageOps.fit(open_picture(data).convert("RGB"), (SIZE, SIZE)).convert("RGBA")
    font, lines = fitted_title(title.upper(), font_path)
    line_height = round(font.size * LINE_SPACING)
    band = Image.new("RGBA", picture.size, (0, 0, 0, 0))
    ImageDraw.Draw(band).rectangle(
        (0, 0, SIZE, TOP + line_height * len(lines) + TOP // 2), fill=(0, 0, 0, BAND_ALPHA)
    )
    picture = Image.alpha_composite(picture, band)
    draw = ImageDraw.Draw(picture)
    for index, line in enumerate(lines):
        centred(draw, line, TOP + index * line_height, font)
    if subtitle:
        small = ImageFont.truetype(str(font_path), SUBTITLE_SIZE)
        centred(draw, subtitle, SIZE - MARGIN - SUBTITLE_SIZE, small)
    return png_bytes(picture.convert("RGB"))
