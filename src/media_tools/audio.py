"""Convert audio, lay voices over a music bed and package audiobooks as MP3s with chapters."""

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from media_tools import ffmpeg

type AudioFormat = Literal["wav", "flac", "mp3"]

CODECS: dict[AudioFormat, list[str]] = {
    "wav": ["-c:a", "pcm_s16le"],
    "flac": ["-c:a", "flac"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
}
BED_PADDING_SECONDS = 8
COVER_ART_SIZE = 600


@dataclass(frozen=True)
class Chapter:
    title: str
    start: float


@dataclass(frozen=True)
class BookTags:
    title: str
    artist: str


def convert(
    audio: bytes, audio_format: AudioFormat, rate: int | None, channels: int | None, timeout: float
) -> bytes:
    """Convert audio to wav, flac or mp3, at that sample rate and channel count when given.

    :raises FfmpegError: when ffmpeg cannot read or write it.
    """
    options = ["-ar", str(rate)] if rate else []
    options += ["-ac", str(channels)] if channels else []
    with tempfile.TemporaryDirectory() as workdir:
        source, target = Path(workdir) / "input", Path(workdir) / f"output.{audio_format}"
        source.write_bytes(audio)
        arguments = [*ffmpeg.AUDIO_GUARD, "-i", str(source), "-vn", *options]
        ffmpeg.run([*arguments, *CODECS[audio_format], "-y", str(target)], timeout)
        return target.read_bytes()


def bed_mix_graph(seconds: float, rate: int, bed_volume: float) -> str:
    """Lay voices over a looping bed that ducks under speech, with 8 s of music around them."""
    total = seconds + 2 * BED_PADDING_SECONDS
    return (
        f"[0:a]aresample={rate},aformat=channel_layouts=stereo,"
        f"adelay={BED_PADDING_SECONDS}s:all=1,apad=pad_dur={BED_PADDING_SECONDS},asplit[voice][key];"
        f"[1:a]aresample={rate},aformat=channel_layouts=stereo,"
        f"aloop=loop=-1:size=2147483647,volume={bed_volume}[bed];"
        "[bed][key]sidechaincompress=threshold=0.015:ratio=8:attack=150:release=1500[ducked];"
        f"[ducked]afade=t=in:d=2,afade=t=out:st={total - 4:.3f}:d=4[music];"
        "[voice][music]amix=inputs=2:normalize=0:duration=first[out]"
    )


def mix_bed(voice: bytes, bed: bytes, bed_volume: float, timeout: float) -> bytes:
    """Lay the voices over the bed at that volume; answer with a stereo WAV at the voices' rate.

    :raises FfmpegError: when ffmpeg cannot read the tracks.
    """
    with tempfile.TemporaryDirectory() as workdir:
        voice_path, bed_path = Path(workdir) / "voice", Path(workdir) / "bed"
        target = Path(workdir) / "mixed.wav"
        voice_path.write_bytes(voice)
        bed_path.write_bytes(bed)
        facts = ffmpeg.probe(voice_path, timeout)
        arguments = [*ffmpeg.AUDIO_GUARD, "-i", str(voice_path), *ffmpeg.AUDIO_GUARD]
        arguments += ["-i", str(bed_path), "-filter_complex"]
        arguments += [bed_mix_graph(facts.seconds, facts.sample_rate, bed_volume), "-map", "[out]"]
        ffmpeg.run([*arguments, "-c:a", "pcm_s16le", "-y", str(target)], timeout)
        return target.read_bytes()


def metadata_value(text: str) -> str:
    """Escape a tag for ffmpeg's metadata file, kept on one line."""
    return re.sub(r"([=;#\\])", r"\\\1", re.sub(r"\s+", " ", text))


def ffmetadata(tags: BookTags, chapters: list[Chapter], total_seconds: float) -> str:
    """Write ffmpeg's metadata: the tags, then each chapter ending where the next one starts."""
    lines = [";FFMETADATA1", f"title={metadata_value(tags.title)}"]
    lines.append(f"artist={metadata_value(tags.artist)}")
    for index, chapter in enumerate(chapters):
        end = chapters[index + 1].start if index + 1 < len(chapters) else total_seconds
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={round(chapter.start * 1000)}",
            f"END={round(max(end, chapter.start) * 1000)}",
            f"title={metadata_value(chapter.title)}",
        ]
    return "\n".join(lines) + "\n"


def package_mp3(
    audio: bytes, chapters: list[Chapter], cover: bytes | None, tags: BookTags, timeout: float
) -> bytes:
    """Encode an audiobook as MP3 with its tags, chapter marks and, when given, the cover picture.

    :raises FfmpegError: when ffmpeg cannot read the audio or the cover.
    """
    with tempfile.TemporaryDirectory() as workdir:
        audio_path, metadata = Path(workdir) / "audio", Path(workdir) / "metadata.txt"
        target = Path(workdir) / "audiobook.mp3"
        audio_path.write_bytes(audio)
        total = ffmpeg.probe(audio_path, timeout).seconds
        metadata.write_text(ffmetadata(tags, chapters, total), encoding="utf-8")
        inputs = [
            *ffmpeg.AUDIO_GUARD,
            "-i",
            str(audio_path),
            "-f",
            "ffmetadata",
            "-i",
            str(metadata),
        ]
        maps = ["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
        if cover:
            cover_path = Path(workdir) / "cover"
            cover_path.write_bytes(cover)
            inputs += [*ffmpeg.PICTURE_GUARD, "-i", str(cover_path)]
            maps += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v", "attached_pic"]
            maps += ["-vf", f"scale={COVER_ART_SIZE}:{COVER_ART_SIZE}"]
        encode = ["-c:a", "libmp3lame", "-b:a", "128k", "-id3v2_version", "3", "-y", str(target)]
        ffmpeg.run([*inputs, *maps, *encode], timeout)
        return target.read_bytes()
