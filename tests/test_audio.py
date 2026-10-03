import json
import subprocess
from pathlib import Path

import pytest

from conftest import png
from conftest import tone
from media_tools import audio
from media_tools import ffmpeg
from media_tools.ffmpeg import FfmpegError

type Entries = list[dict[str, str | int | dict[str, str]]]


def probed(data: bytes, tmp_path: Path, section: str, entries: str) -> Entries:
    """Return one section of ffprobe's JSON for the file, such as its streams or chapters."""
    path = tmp_path / "probe"
    path.write_bytes(data)
    result = subprocess.run(
        [ffmpeg.FFPROBE, "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
        capture_output=True,
        check=True,
    )
    section_entries: Entries = json.loads(result.stdout)[section]
    return section_entries


def test_convert_resamples_to_stereo_mp3(tmp_path: Path) -> None:
    mp3 = audio.convert(tone(1.0), "mp3", 44100, 2, 60)

    streams = probed(mp3, tmp_path, "streams", "stream=codec_name,sample_rate,channels")
    assert streams == [{"codec_name": "mp3", "sample_rate": "44100", "channels": 2}]


def test_convert_refuses_what_is_no_audio() -> None:
    with pytest.raises(FfmpegError):
        audio.convert(b"not audio", "wav", None, None, 60)


def test_the_bed_plays_before_and_after_the_voices(tmp_path: Path) -> None:
    path = tmp_path / "mixed.wav"
    path.write_bytes(audio.mix_bed(tone(2.0), tone(3.0, rate=44100), 0.3, 60))

    facts = ffmpeg.probe(path, 60)

    assert facts.sample_rate == 22050
    assert facts.seconds == pytest.approx(2.0 + 2 * audio.BED_PADDING_SECONDS, abs=0.1)


def test_metadata_escapes_tags_and_closes_the_last_chapter_at_the_end() -> None:
    chapters = [audio.Chapter("One; = two", 0), audio.Chapter("Three", 12.5)]

    text = audio.ffmetadata(audio.BookTags("A #1 book", "h4ks"), chapters, 30)

    assert text.splitlines() == [
        ";FFMETADATA1",
        "title=A \\#1 book",
        "artist=h4ks",
        "[CHAPTER]",
        "TIMEBASE=1/1000",
        "START=0",
        "END=12500",
        "title=One\\; \\= two",
        "[CHAPTER]",
        "TIMEBASE=1/1000",
        "START=12500",
        "END=30000",
        "title=Three",
    ]


def test_metadata_values_stay_on_one_line() -> None:
    assert audio.metadata_value("one\r[CHAPTER]\ntwo") == "one [CHAPTER] two"


def test_the_mp3_carries_chapters_and_cover(tmp_path: Path) -> None:
    chapters = [audio.Chapter("Start", 0), audio.Chapter("Middle", 1.0)]

    mp3 = audio.package_mp3(tone(2.0), chapters, png((64, 64)), audio.BookTags("Book", "h4ks"), 60)

    marks = probed(mp3, tmp_path, "chapters", "chapter_tags=title")
    streams = probed(mp3, tmp_path, "streams", "stream=codec_name")
    assert marks == [{"tags": {"title": "Start"}}, {"tags": {"title": "Middle"}}]
    assert streams == [{"codec_name": "mp3"}, {"codec_name": "mjpeg"}]


def test_the_mp3_needs_no_cover(tmp_path: Path) -> None:
    mp3 = audio.package_mp3(tone(1.0), [], None, audio.BookTags("", ""), 60)

    assert probed(mp3, tmp_path, "streams", "stream=codec_name") == [{"codec_name": "mp3"}]


def test_a_broken_track_cannot_be_probed(tmp_path: Path) -> None:
    path = tmp_path / "broken"
    path.write_bytes(b"nothing")

    with pytest.raises(FfmpegError, match="could not read the audio"):
        ffmpeg.probe(path, 60)
