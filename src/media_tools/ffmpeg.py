"""Run the pinned ffmpeg and ffprobe on files we wrote ourselves."""

import json
import os
import subprocess  # nosec B404: we run only the ffmpeg binaries, with arguments we build
from dataclasses import dataclass
from pathlib import Path

FFMPEG = os.environ.get("FFMPEG", "/opt/tools/ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "/opt/tools/ffprobe")
# We only accept plain media read from the uploaded file itself. Playlist demuxers such as hls or
# concat would let a crafted upload make ffmpeg open other local files or URLs.
AUDIO_GUARD = [
    "-protocol_whitelist",
    "file",
    "-format_whitelist",
    "mp3,wav,flac,ogg,mov,matroska,aac,w64,aiff",
]
PICTURE_GUARD = ["-protocol_whitelist", "file", "-format_whitelist", "image2,png_pipe,jpeg_pipe"]
ERROR_TAIL = 2000


class FfmpegError(ValueError):
    pass


@dataclass(frozen=True)
class AudioFacts:
    seconds: float
    sample_rate: int


def run(arguments: list[str], timeout: float) -> None:
    """Run ffmpeg quietly.

    :raises FfmpegError: when ffmpeg fails or runs out of time.
    """
    command = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", *arguments]
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)  # nosec B603
    except subprocess.TimeoutExpired as error:
        raise FfmpegError("ffmpeg timed out") from error
    if result.returncode != 0:
        raise FfmpegError(result.stderr[-ERROR_TAIL:].decode(errors="replace") or "ffmpeg failed")


def probe(path: Path, timeout: float) -> AudioFacts:
    """Read an audio file's length and its first stream's sample rate.

    :raises FfmpegError: when ffprobe cannot read it.
    """
    command = [FFPROBE, "-v", "error", *AUDIO_GUARD, "-select_streams", "a:0", "-show_entries"]
    command += ["format=duration:stream=sample_rate", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)  # nosec B603
        facts = json.loads(result.stdout or b"{}")
        return AudioFacts(
            float(facts["format"]["duration"]), int(facts["streams"][0]["sample_rate"])
        )
    except (subprocess.TimeoutExpired, KeyError, IndexError, ValueError) as error:
        raise FfmpegError("could not read the audio") from error
