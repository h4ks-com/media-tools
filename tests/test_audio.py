import json
import subprocess
from pathlib import Path

import numpy
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
    path.write_bytes(audio.mix_bed(tone(2.0), [tone(3.0, rate=44100)], [0], 0.3, "wav", 60))

    facts = ffmpeg.probe(path, 60)

    assert facts.sample_rate == 22050
    assert facts.seconds == pytest.approx(2.0 + 2 * audio.BED_PADDING_SECONDS, abs=0.1)


def test_later_beds_cross_fade_in_and_keep_the_length(tmp_path: Path) -> None:
    path = tmp_path / "mixed.wav"
    path.write_bytes(audio.mix_bed(tone(20.0), [tone(3.0), tone(2.0)], [0, 10], 0.3, "wav", 60))

    assert ffmpeg.probe(path, 60).seconds == pytest.approx(
        20.0 + 2 * audio.BED_PADDING_SECONDS, abs=0.1
    )


def test_the_mix_comes_as_mp3_when_asked(tmp_path: Path) -> None:
    mp3 = audio.mix_bed(tone(2.0), [tone(3.0)], [0], 0.3, "mp3", 60)

    assert probed(mp3, tmp_path, "streams", "stream=codec_name,channels") == [
        {"codec_name": "mp3", "channels": 2}
    ]


def joined_samples(flac: bytes, tmp_path: Path) -> numpy.ndarray:
    path, raw = tmp_path / "joined.flac", tmp_path / "joined.f32"
    path.write_bytes(flac)
    subprocess.run(
        [ffmpeg.FFMPEG, "-v", "error", "-i", str(path), "-f", "f32le", "-y", str(raw)], check=True
    )
    return numpy.fromfile(raw, dtype=numpy.float32)


def test_join_places_each_clip_at_its_start_at_one_rate(tmp_path: Path) -> None:
    clips = [tone(1.0), tone(0.5, rate=44100)]
    placements = [audio.Placement(0, 1), audio.Placement(2.0, 1)]

    flac = audio.join(clips, placements, "flac", 60, 60)

    streams = probed(flac, tmp_path, "streams", "stream=codec_name,sample_rate,channels")
    assert streams == [{"codec_name": "flac", "sample_rate": "44100", "channels": 1}]
    samples = joined_samples(flac, tmp_path)
    assert len(samples) == pytest.approx(2.5 * audio.JOIN_RATE, abs=50)
    assert numpy.abs(samples[round(1.2 * audio.JOIN_RATE) : 2 * audio.JOIN_RATE]).max() == 0


def test_overlapping_clips_add_up_with_their_gains(tmp_path: Path) -> None:
    placements = [audio.Placement(0, 1), audio.Placement(0, 0.5)]

    alone = joined_samples(audio.join([tone(1.0)], placements[:1], "flac", 60, 60), tmp_path)
    both = joined_samples(audio.join([tone(1.0), tone(1.0)], placements, "flac", 60, 60), tmp_path)

    assert numpy.abs(both).max() == pytest.approx(1.5 * numpy.abs(alone).max(), rel=0.01)


def test_the_joined_track_comes_as_mp3_when_asked(tmp_path: Path) -> None:
    mp3 = audio.join([tone(1.0)], [audio.Placement(0, 1)], "mp3", 60, 60)

    assert probed(mp3, tmp_path, "streams", "stream=codec_name,channels") == [
        {"codec_name": "mp3", "channels": 1}
    ]


def test_join_refuses_a_track_longer_than_the_limit() -> None:
    with pytest.raises(ValueError, match="longer than"):
        audio.join([tone(1.0)], [audio.Placement(5, 1)], "flac", 3, 60)


def test_join_refuses_what_is_no_audio() -> None:
    with pytest.raises(FfmpegError):
        audio.join([b"not audio"], [audio.Placement(0, 1)], "flac", 60, 60)


def test_each_bed_plays_until_the_next_starts_with_the_fades_counted() -> None:
    fade, padding = audio.BED_CROSSFADE_SECONDS, audio.BED_PADDING_SECONDS

    sections = audio.bed_sections(60, [0, 20, 45])

    assert sections == [20 + padding, 25 + fade, 15 + fade + padding]
    assert sum(sections) - 2 * fade == 60 + 2 * padding


def test_beds_cannot_start_too_close_together_or_after_the_voices() -> None:
    with pytest.raises(ValueError, match="at least"):
        audio.bed_sections(60, [0, 2])
    with pytest.raises(ValueError, match="at least"):
        audio.bed_sections(60, [0, 58])


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
