"""Cut generated clips into one video at one size, over a music track or the clips' own sound."""

import tempfile
from pathlib import Path

from media_tools import ffmpeg

# We list only mov because the mov demuxer also reads MP4.
VIDEO_GUARD = ["-protocol_whitelist", "file", "-format_whitelist", "mov"]
FPS = 24
SOUND_RATE = 44100
MUSIC_FADE_SECONDS = 2.0
ENCODING = [
    *("-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"),
    *("-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"),
]


def framed(index: int, width: int, height: int) -> str:
    """Scale one clip to cover width x height, crop the overflow and set the shared frame rate."""
    return (
        f"[{index}:v]setpts=PTS-STARTPTS,scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1,fps={FPS}[v{index}]"
    )


def has_sound(path: Path, timeout: float) -> bool:
    try:
        ffmpeg.probe(path, timeout)
    except ffmpeg.FfmpegError:
        return False
    return True


def clip_sound(index: int, length: float, sounded: bool) -> str:
    """Give one clip a stereo track exactly as long as its video, silence when it has no sound."""
    source = (
        f"[{index}:a]asetpts=PTS-STARTPTS," if sounded else f"anullsrc=r={SOUND_RATE}:cl=stereo,"
    )
    return (
        f"{source}aresample={SOUND_RATE},aformat=channel_layouts=stereo,"
        f"apad,atrim=0:{length:.3f}[a{index}]"
    )


def cut(clips: list[bytes], music: bytes | None, width: int, height: int, timeout: float) -> bytes:
    """Join the clips end to end as one MP4 of width x height.

    With music, the music plays under the whole video and fades out at its end. Without it, each
    clip keeps its own sound.

    :raises FfmpegError: when ffmpeg cannot read a clip or the music.
    """
    with tempfile.TemporaryDirectory() as workdir:
        folder = Path(workdir)
        inputs: list[str] = []
        lengths: list[float] = []
        sounded: list[bool] = []
        for index, clip in enumerate(clips):
            path = folder / f"clip{index}"
            path.write_bytes(clip)
            lengths.append(ffmpeg.video_seconds(path, VIDEO_GUARD, timeout))
            sounded.append(has_sound(path, timeout))
            inputs += [*VIDEO_GUARD, "-i", str(path)]
        seconds = sum(lengths)
        graph = [framed(index, width, height) for index in range(len(clips))]
        if music is None:
            graph += [
                clip_sound(index, length, sound)
                for index, (length, sound) in enumerate(zip(lengths, sounded, strict=True))
            ]
            pairs = "".join(f"[v{index}][a{index}]" for index in range(len(clips)))
            graph.append(f"{pairs}concat=n={len(clips)}:v=1:a=1[video][sound]")
        else:
            track = folder / "music"
            track.write_bytes(music)
            inputs += [*ffmpeg.AUDIO_GUARD, "-i", str(track)]
            fade_start = max(seconds - MUSIC_FADE_SECONDS, 0)
            videos = "".join(f"[v{index}]" for index in range(len(clips)))
            graph.append(f"{videos}concat=n={len(clips)}:v=1:a=0[video]")
            graph.append(
                f"[{len(clips)}:a]aresample={SOUND_RATE},apad,atrim=0:{seconds:.3f},"
                f"afade=t=out:st={fade_start:.3f}:d={MUSIC_FADE_SECONDS}[sound]"
            )
        target = folder / "video.mp4"
        mapping = ["-filter_complex", ";".join(graph), "-map", "[video]", "-map", "[sound]"]
        ffmpeg.run([*inputs, *mapping, *ENCODING, "-y", str(target)], timeout)
        return target.read_bytes()
