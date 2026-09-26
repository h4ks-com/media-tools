"""Cut objects out of pictures, turn them into pixel art and build sprite sheets."""

import io
from collections.abc import Sequence
from typing import Literal

import numpy
import onnxruntime
from numpy.typing import NDArray
from PIL import Image

type CutMethod = Literal["isnet", "key"]

MODEL_SIZE = (1024, 1024)
MAX_PIXELS = 4096 * 4096
# Colours within KEY_NEAR of the background become fully transparent, and fade in until KEY_FAR.
KEY_NEAR = 40
KEY_FAR = 80
MIN_LINE_PIXELS = 3
FRAME_PADDING = 8
GIF_FRAME_MS = 150
SOLID = 128

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def open_picture(data: bytes) -> Image.Image:
    """Decode a picture fully.

    :raises PIL.UnidentifiedImageError: when the data is no picture PIL reads.
    :raises PIL.Image.DecompressionBombError: when the picture is larger than MAX_PIXELS.
    """
    picture = Image.open(io.BytesIO(data))
    picture.load()
    return picture


def png_bytes(picture: Image.Image) -> bytes:
    output = io.BytesIO()
    picture.save(output, format="PNG")
    return output.getvalue()


def background_key_mask(picture: Image.Image) -> Image.Image:
    """Mask out the flat background colour we asked for, read from the picture's border."""
    rgb: NDArray[numpy.float32] = numpy.asarray(picture.convert("RGB"), dtype=numpy.float32)
    border = numpy.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = numpy.median(border, axis=0)
    distance = numpy.sqrt(((rgb - background) ** 2).sum(axis=-1))
    alpha = numpy.clip((distance - KEY_NEAR) * 255 / (KEY_FAR - KEY_NEAR), 0, 255)
    return Image.fromarray(alpha.astype(numpy.uint8), mode="L")


class Cutter:
    """Cuts the main object out of a picture with the ISNet general-use model, as rembg does.

    We load the model per picture, since keeping it loaded holds a gigabyte the mesh work needs.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def mask(self, picture: Image.Image) -> Image.Image:
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 4
        session = onnxruntime.InferenceSession(
            self.model_path, options, providers=["CPUExecutionProvider"]
        )
        resized = picture.convert("RGB").resize(MODEL_SIZE, Image.Resampling.LANCZOS)
        pixels = numpy.asarray(resized, dtype=numpy.float32)
        pixels = pixels / max(float(pixels.max()), 1e-6) - 0.5
        batch = pixels.transpose((2, 0, 1))[numpy.newaxis]
        prediction = session.run(None, {session.get_inputs()[0].name: batch})[0][0, 0]
        low, high = float(prediction.min()), float(prediction.max())
        scaled = (prediction - low) / max(high - low, 1e-6)
        mask = Image.fromarray((scaled * 255).astype(numpy.uint8), mode="L")
        return mask.resize(picture.size, Image.Resampling.LANCZOS)

    def cut(self, data: bytes, method: CutMethod) -> bytes:
        """Return the picture as a PNG whose alpha keeps only the main object.

        `isnet` finds the object in any photo; `key` removes a flat background colour we asked for.
        """
        picture = open_picture(data)
        cut = picture.convert("RGBA")
        cut.putalpha(background_key_mask(picture) if method == "key" else self.mask(picture))
        return png_bytes(cut)


def pixelate(picture: Image.Image, width: int, height: int, colors: int) -> Image.Image:
    """Shrink an RGBA picture to width x height in `colors` colours with hard transparent edges."""
    small = picture.resize((width, height), Image.Resampling.BOX)
    alpha = small.getchannel("A").point(lambda value: 255 if value >= SOLID else 0)
    visible = Image.new("RGB", small.size)
    visible.paste(small.convert("RGB"), mask=alpha)
    art = visible.quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    ).convert("RGBA")
    art.putalpha(alpha)
    return art


def upscale(picture: Image.Image, scale: int) -> Image.Image:
    if scale <= 1:
        return picture
    return picture.resize((picture.width * scale, picture.height * scale), Image.Resampling.NEAREST)


def pixel_art(data: bytes, size: int, colors: int, scale: int) -> bytes:
    """Turn a picture into pixel art `size` across, in `colors` colours, scaled up `scale` times."""
    picture = open_picture(data).convert("RGBA")
    ratio = size / max(picture.size)
    width = max(1, round(picture.width * ratio))
    height = max(1, round(picture.height * ratio))
    return png_bytes(upscale(pixelate(picture, width, height, colors), scale))


def keyed(data: bytes) -> Image.Image:
    picture = open_picture(data)
    cut = picture.convert("RGBA")
    cut.putalpha(background_key_mask(picture))
    return cut


def solid_box(alpha: Image.Image) -> tuple[int, int, int, int]:
    """Return the box around lines with MIN_LINE_PIXELS solid pixels, so specks don't count.

    :raises ValueError: when the frame has no character.
    """
    solid = numpy.asarray(alpha) >= SOLID
    rows = numpy.flatnonzero(solid.sum(axis=1) >= MIN_LINE_PIXELS)
    columns = numpy.flatnonzero(solid.sum(axis=0) >= MIN_LINE_PIXELS)
    if not len(rows) or not len(columns):
        raise ValueError("found no character in a frame")
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def sprite_frames(frame_files: Sequence[bytes], size: int, colors: int) -> list[Image.Image]:
    """Key out each frame's background and return the frames as pixel art in one palette.

    Frames are `size` tall. We crop every frame to one box that holds the character in all of them,
    so it keeps its place.

    :raises ValueError: when the frames differ in size or one has no character.
    """
    frames = [keyed(data) for data in frame_files]
    if len({frame.size for frame in frames}) != 1:
        raise ValueError("the frames differ in size")
    boxes = [solid_box(frame.getchannel("A")) for frame in frames]
    width, height = frames[0].size
    box = (
        max(0, min(b[0] for b in boxes) - FRAME_PADDING),
        max(0, min(b[1] for b in boxes) - FRAME_PADDING),
        min(width, max(b[2] for b in boxes) + FRAME_PADDING),
        min(height, max(b[3] for b in boxes) + FRAME_PADDING),
    )
    crops = [frame.crop(box) for frame in frames]
    cell_width, cell_height = crops[0].size
    sheet = Image.new("RGBA", (cell_width * len(crops), cell_height))
    for index, crop in enumerate(crops):
        sheet.paste(crop, (index * cell_width, 0))
    frame_width = max(1, round(cell_width * size / cell_height))
    art = pixelate(sheet, frame_width * len(crops), size, colors)
    return [
        art.crop((index * frame_width, 0, (index + 1) * frame_width, size))
        for index in range(len(crops))
    ]


def sprite_sheet(frame_files: Sequence[bytes], size: int, colors: int, scale: int) -> bytes:
    """Return the frames side by side as one PNG sprite sheet."""
    frames = sprite_frames(frame_files, size, colors)
    sheet = Image.new("RGBA", (frames[0].width * len(frames), size))
    for index, frame in enumerate(frames):
        sheet.paste(frame, (index * frame.width, 0))
    return png_bytes(upscale(sheet, scale))


def sprite_gif(frame_files: Sequence[bytes], size: int, colors: int, scale: int) -> bytes:
    """Return the frames played as a looping GIF with a transparent background."""
    frames = [upscale(frame, scale) for frame in sprite_frames(frame_files, size, colors)]
    output = io.BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_MS,
        loop=0,
        disposal=2,
    )
    return output.getvalue()
