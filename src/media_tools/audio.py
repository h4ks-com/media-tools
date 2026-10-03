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
BED_CROSSFADE_SECONDS = 3
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


def bed_sections(seconds: float, bed_starts: list[float]) -> list[float]:
    """Say how long each bed plays, the first from the start and each later one from its start.

    Each later bed starts BED_CROSSFADE_SECONDS early and fades in over the bed before it, so the
    new music is in full when the voices reach its start.

    :raises ValueError: when a bed would start too close to the one before or after the voices end.
    """
    ends = [*bed_starts[1:], seconds]
    gaps = [end - start for start, end in zip(bed_starts, ends, strict=True)]
    if len(bed_starts) > 1 and min(gaps) < 2 * BED_CROSSFADE_SECONDS:
        raise ValueError(
            f"start each bed at least {2 * BED_CROSSFADE_SECONDS} s after the one before "
            "and before the voices end"
        )
    sections = [gap + BED_CROSSFADE_SECONDS for gap in gaps]
    sections[0] += BED_PADDING_SECONDS - BED_CROSSFADE_SECONDS
    sections[-1] += BED_PADDING_SECONDS
    return sections


def bed_mix_graph(seconds: float, rate: int, bed_volume: float, bed_starts: list[float]) -> str:
    """Lay voices over looping beds that cross-fade at their starts and duck under speech.

    Music plays alone for 8 s before and after the voices.
    """
    total = seconds + 2 * BED_PADDING_SECONDS
    stereo = f"aresample={rate},aformat=channel_layouts=stereo"
    graph = [
        f"[0:a]{stereo},adelay={BED_PADDING_SECONDS}s:all=1,"
        f"apad=pad_dur={BED_PADDING_SECONDS},asplit[voice][key]"
    ]
    for index, length in enumerate(bed_sections(seconds, bed_starts), start=1):
        graph.append(
            f"[{index}:a]{stereo},aloop=loop=-1:size=2147483647,"
            f"atrim=duration={length:.3f},asetpts=PTS-STARTPTS[section{index}]"
        )
    joined = "section1"
    for index in range(2, len(bed_starts) + 1):
        graph.append(
            f"[{joined}][section{index}]acrossfade=d={BED_CROSSFADE_SECONDS}[joined{index}]"
        )
        joined = f"joined{index}"
    graph += [
        f"[{joined}]volume={bed_volume}[bed]",
        "[bed][key]sidechaincompress=threshold=0.015:ratio=8:attack=150:release=1500[ducked]",
        f"[ducked]afade=t=in:d=2,afade=t=out:st={total - 4:.3f}:d=4[music]",
        "[voice][music]amix=inputs=2:normalize=0:duration=first[out]",
    ]
    return ";".join(graph)


def mix_bed(
    voice: bytes, beds: list[bytes], bed_starts: list[float], bed_volume: float, timeout: float
) -> bytes:
    """Lay the voices over the beds at that volume, each bed from its start in the voices' seconds.

    Answers with a stereo WAV at the voices' rate.

    :raises FfmpegError: when ffmpeg cannot read the tracks.
    :raises ValueError: when the beds start too close together or after the voices end.
    """
    with tempfile.TemporaryDirectory() as workdir:
        voice_path, target = Path(workdir) / "voice", Path(workdir) / "mixed.wav"
        voice_path.write_bytes(voice)
        facts = ffmpeg.probe(voice_path, timeout)
        graph = bed_mix_graph(facts.seconds, facts.sample_rate, bed_volume, bed_starts)
        arguments = [*ffmpeg.AUDIO_GUARD, "-i", str(voice_path)]
        for index, bed in enumerate(beds):
            bed_path = Path(workdir) / f"bed{index}"
            bed_path.write_bytes(bed)
            arguments += [*ffmpeg.AUDIO_GUARD, "-i", str(bed_path)]
        arguments += ["-filter_complex", graph, "-map", "[out]"]
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
