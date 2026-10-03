import json
import math
import subprocess
from pathlib import Path

import pytest

from conftest import needs_libass
from conftest import png
from conftest import tone
from media_tools import ffmpeg
from media_tools import karaoke
from media_tools.karaoke import Word

STYLE = karaoke.Style("classic", "sweep", "bars")


def words(*timed: tuple[str, float, float, int | None]) -> list[Word]:
    return [Word(text, start, end, line) for text, start, end, line in timed]


def test_words_are_read_sorted_and_trimmed() -> None:
    raw = [
        {"text": "b" * 100, "start": 2, "end": 3},
        {"text": "a", "start": 0.5, "end": 1, "line": 0},
    ]

    parsed = karaoke.parse_words(raw, 10)

    assert parsed == [Word("a", 0.5, 1.0, 0), Word("b" * karaoke.MAX_WORD_CHARS, 2.0, 3.0, None)]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "list of 1 to 2"),
        ([{}, {}, {}], "list of 1 to 2"),
        (["word"], "object"),
        ([{"text": "a", "start": 2, "end": 1}], "start <= end"),
        ([{"text": "a", "start": 0, "end": math.inf}], "finite"),
        ([{"text": "a", "start": 0, "end": 1, "line": 1.5}], "whole number"),
    ],
)
def test_bad_words_are_refused(raw: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        karaoke.parse_words(raw, 2)


def test_words_without_line_numbers_break_at_pauses() -> None:
    timed = words(("one", 0, 0.5, None), ("two", 0.6, 1, None), ("three", 3, 3.5, None))

    assert [[word.text for word in line] for line in karaoke.group_by_pauses(timed)] == [
        ["one", "two"],
        ["three"],
    ]


def test_numbered_words_keep_their_lines() -> None:
    timed = words(("one", 0, 0.5, 0), ("two", 0.6, 1, 1), ("three", 1.1, 1.5, 1))

    assert [[word.text for word in line] for line in karaoke.group_by_line(timed)] == [
        ["one"],
        ["two", "three"],
    ]


def test_the_subtitles_show_the_title_the_line_and_the_next_one() -> None:
    timed = words(("Hello", 2, 2.5, 0), ("{world}", 2.5, 3, 0), ("again", 6, 6.5, 1))

    ass = karaoke.build_ass(timed, "My Song", karaoke.Style("neon", "glow", "bars"))

    events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert events[0].endswith("Title,,0,0,0,,My Song")
    assert "(world)" in events[1]
    assert "\\bord9" in events[1]
    assert events[2].endswith("again")
    assert len(events) == 4


def test_the_next_line_can_stay_hidden() -> None:
    timed = words(("Hello", 2, 2.5, 0), ("again", 6, 6.5, 1))

    ass = karaoke.build_ass(timed, "", karaoke.Style("neon", "sweep", "bars", upcoming=False))

    events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert [",Current," in event for event in events] == [True, True]


def test_speakers_are_read_trimmed_and_blank_means_none() -> None:
    raw = [
        {"text": "a", "start": 0, "end": 1, "speaker": " " + "x" * 50},
        {"text": "b", "start": 1, "end": 2, "speaker": ""},
    ]

    parsed = karaoke.parse_words(raw, 10)

    assert [word.speaker for word in parsed] == ["x" * karaoke.MAX_SPEAKER_CHARS, None]


def test_each_speaker_line_starts_with_their_name_in_their_own_colour() -> None:
    timed = [
        Word("Run", 1, 1.5, 0, "Pip"),
        Word("Night", 4, 4.5, 1),
        Word("Stay", 7, 7.5, 2, "Thomas"),
        Word("Now", 10, 10.5, 3, "Pip"),
    ]

    ass = karaoke.build_ass(timed, "", STYLE)

    lines = [line for line in ass.splitlines() if ",Current," in line]
    pip = karaoke.inline_colour(karaoke.SPEAKER_COLOURS[0])
    thomas = karaoke.inline_colour(karaoke.SPEAKER_COLOURS[1])
    assert f"{pip}}}PIP: " in lines[0]
    assert "PIP" not in lines[1]
    assert f"{thomas}}}THOMAS: " in lines[2]
    assert f"{pip}}}PIP: " in lines[3]


def test_a_new_speaker_starts_a_new_line() -> None:
    timed = [Word("one", 0, 0.5, None, "A"), Word("two", 0.6, 1, None, "B")]

    assert len(karaoke.group_by_pauses(timed)) == 2


def test_text_cannot_start_a_new_subtitle_line() -> None:
    assert karaoke.ass_text("a\rDialogue: {x}\nb") == "a Dialogue: (x) b"


def test_the_glow_is_only_for_the_glow_highlight() -> None:
    assert karaoke.word_effect("sweep", karaoke.LOOKS["neon"], 100) == ""


def test_a_long_lead_in_is_trimmed() -> None:
    line = karaoke.trim_lead_in(words(("late", 0, 5, 0)))

    assert line[0].start == 5 - karaoke.FIRST_WORD_MAX_SECONDS


def test_timestamps_have_centiseconds() -> None:
    assert karaoke.timestamp(3725.5) == "1:02:05.50"


def test_vocals_are_cut_from_the_track_only_when_asked() -> None:
    assert karaoke.track_filter(False, 0.5) == "[0:a]asplit[track][heard];"
    assert "c0-0.5*c2" in karaoke.track_filter(True, 0.5)


def test_each_picture_lasts_until_the_next_starts() -> None:
    inputs = karaoke.slideshow_inputs([Path("a"), Path("b")], [3.0, 10.0], 25.0)

    durations = [inputs[index + 1] for index, value in enumerate(inputs) if value == "-t"]
    assert durations == ["10.000", "15.000"]


def video_facts(data: bytes, tmp_path: Path) -> list[dict[str, str | int]]:
    path = tmp_path / "video.mp4"
    path.write_bytes(data)
    entries = ["-show_entries", "stream=codec_type,width", "-of", "json"]
    result = subprocess.run(
        [ffmpeg.FFPROBE, "-v", "error", *entries, str(path)],
        capture_output=True,
        check=True,
    )
    streams: list[dict[str, str | int]] = json.loads(result.stdout)["streams"]
    return streams


@needs_libass
def test_karaoke_renders_over_a_visualizer(tmp_path: Path) -> None:
    song = karaoke.Song(tone(2.0), words(("la", 0.2, 1.0, 0)), "Song", None)

    video = karaoke.render(song, STYLE, [], 120)

    assert video_facts(video, tmp_path) == [
        {"codec_type": "video", "width": karaoke.WIDTH},
        {"codec_type": "audio"},
    ]


@needs_libass
def test_karaoke_renders_over_pictures_with_the_vocals_cut(tmp_path: Path) -> None:
    vocals = karaoke.Vocals(tone(2.0), 0.5)
    song = karaoke.Song(tone(2.0), words(("la", 0.2, 1.0, 0)), "", vocals)
    shown = [karaoke.Picture(png((64, 48)), 0), karaoke.Picture(png((48, 64)), 1.0)]

    video = karaoke.render(song, STYLE, shown, 120)

    assert video_facts(video, tmp_path)[0] == {"codec_type": "video", "width": karaoke.WIDTH}
