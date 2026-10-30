from pathlib import Path

import pytest

from conftest import clip
from conftest import tone
from media_tools import video
from media_tools.ffmpeg import FfmpegError
from test_audio import probed

STREAMS = "stream=codec_type,width,height,duration"


def stream_seconds(mp4: bytes, tmp_path: Path) -> dict[str, float]:
    streams = probed(mp4, tmp_path, "streams", STREAMS)
    return {str(stream["codec_type"]): float(str(stream["duration"])) for stream in streams}


def test_clips_join_end_to_end_at_the_asked_size(tmp_path: Path) -> None:
    mp4 = video.cut([clip(1.0), clip(1.5)], None, 72, 128, 60)

    picture = probed(mp4, tmp_path, "streams", "stream=codec_name,width,height")[0]
    seconds = stream_seconds(mp4, tmp_path)
    assert picture == {"codec_name": "h264", "width": 72, "height": 128}
    assert seconds["video"] == pytest.approx(2.5, abs=0.05)
    assert seconds["audio"] == pytest.approx(2.5, abs=0.05)


def test_a_clip_without_sound_plays_silence(tmp_path: Path) -> None:
    mp4 = video.cut([clip(1.0, sound=False), clip(1.0)], None, 72, 128, 60)

    assert stream_seconds(mp4, tmp_path)["audio"] == pytest.approx(2.0, abs=0.05)


def test_music_gives_silent_clips_their_sound(tmp_path: Path) -> None:
    mp4 = video.cut([clip(1.0, sound=False), clip(1.0, sound=False)], tone(5.0), 72, 128, 60)

    seconds = stream_seconds(mp4, tmp_path)
    assert seconds["video"] == pytest.approx(2.0, abs=0.05)
    assert seconds["audio"] == pytest.approx(2.0, abs=0.05)


def test_music_shorter_than_the_clips_is_padded_to_their_end(tmp_path: Path) -> None:
    mp4 = video.cut([clip(2.0, sound=False)], tone(0.5), 72, 128, 60)

    assert stream_seconds(mp4, tmp_path)["audio"] == pytest.approx(2.0, abs=0.05)


def test_a_clip_that_is_no_video_is_refused() -> None:
    with pytest.raises(FfmpegError):
        video.cut([b"not a video"], None, 72, 128, 60)
