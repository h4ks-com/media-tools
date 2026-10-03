"""Karaoke videos: words lighting up as they are sung, over an audio visualizer or pictures."""

import math
import os
import re
import tempfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Literal

from media_tools import ffmpeg

type LookName = Literal["neon", "classic", "sunset", "ocean", "mono"]
type Highlight = Literal["sweep", "word", "glow"]
type Background = Literal["bars", "waves", "spectrum"]

FONTS_DIR = os.environ.get("FONTS_DIR", "/opt/tools/fonts")
WIDTH, HEIGHT, FPS = 1280, 720, 30
# Over still pictures only the words move, and the highlight sweep reads smoothly at half the
# visualizer's frame rate for half the encoding work.
SLIDESHOW_FPS = 15
FONT = "Lilita One"
LINE_GAP_SECONDS = 1.2
MAX_WORDS_PER_LINE = 8
LEAD_IN_SECONDS = 1.5
# A line leaves the screen this long after it is sung, so instrumental breaks show no stale lyrics.
LINGER_SECONDS = 2.0
FIRST_WORD_MAX_SECONDS = 1.0
MAX_WORD_CHARS = 60
PICTURE_FADE_SECONDS = 0.8


@dataclass(frozen=True)
class Look:
    """Colours of a look; ASS colours are &HAABBGGRR, sung words primary, upcoming secondary."""

    sung: str
    unsung: str
    outline: str
    outline_width: int
    blur: int
    left: str
    right: str
    palette: str


LOOKS: dict[LookName, Look] = {
    "neon": Look("&H00FFFF00", "&H00FF40FF", "&H00300030", 4, 3, "0x00ffff", "0xff40ff", "plasma"),
    "classic": Look(
        "&H0000D7FF", "&H00FFFFFF", "&H00000000", 3, 0, "0xffd700", "0xffffff", "fiery"
    ),
    "sunset": Look("&H003C8CFF", "&H00E0F0FF", "&H00201040", 3, 1, "0xff8c3c", "0xff4f9a", "magma"),
    "ocean": Look("&H00E0D040", "&H00FFF0E0", "&H00401808", 3, 2, "0x40d0e0", "0x2060ff", "cool"),
    "mono": Look(
        "&H00FFFFFF", "&H00707070", "&H00000000", 3, 0, "0xffffff", "0x808080", "intensity"
    ),
}

BACKGROUNDS: dict[Background, str] = {
    "bars": "showcqt=s={w}x{h}:r={fps}:axis=0:sono_h=0:bar_g=2:bar_v=3:basefreq=40:endfreq=10000"
    ":cscheme={cscheme}",
    # The waves sit in a band at the bottom so they never run through the lyrics.
    "waves": "showwaves=s={w}x220:r={fps}:mode=cline:scale=sqrt:colors={left}|{right},format=rgba,"
    "pad={w}:{h}:0:{h}-250:color=black@0",
    # We draw a narrow spectrum and stretch it so it fills the screen within seconds.
    "spectrum": "showspectrum=s=96x{h}:fps={fps}:mode=combined:slide=scroll:scale=cbrt"
    ":color={palette},scale={w}:{h}",
}


# Tag colours for the speakers in order of first appearance, as ASS &HAABBGGRR.
SPEAKER_COLOURS = (
    "&H005F5AFF",
    "&H00D8B400",
    "&H0000A1F4",
    "&H00E55D9B",
    "&H00B5C42E",
    "&H00B55BF1",
)
MAX_SPEAKER_CHARS = 40


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    line: int | None
    speaker: str | None = None


@dataclass(frozen=True)
class Style:
    """How the words look; `upcoming` also shows the next line under the one being sung."""

    look: LookName
    highlight: Highlight
    background: Background
    upcoming: bool = True


@dataclass(frozen=True)
class Vocals:
    """The song's vocal stem and the share of it to take out of the track, 0 to 1."""

    stem: bytes
    cut: float


@dataclass(frozen=True)
class Song:
    audio: bytes
    words: list[Word]
    title: str
    vocals: Vocals | None


@dataclass(frozen=True)
class Picture:
    """A picture shown from `start` seconds until the next one fades in."""

    data: bytes
    start: float


def parse_words(raw: object, limit: int) -> list[Word]:
    """Read timed words from JSON: a list of {text, start, end, line, speaker}.

    line and speaker may be missing; a speaker's name shows above the lines they say.

    :raises ValueError: when it is no list of 1 to `limit` words with 0 <= start <= end.
    """
    if not isinstance(raw, list) or not 0 < len(raw) <= limit:
        raise ValueError(f"words must be a list of 1 to {limit} timed words")
    words = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("every word is an object with text, start and end")
        start, end, line = float(item["start"]), float(item["end"]), item.get("line")
        if not (math.isfinite(end) and 0 <= start <= end):
            raise ValueError("every word needs finite 0 <= start <= end")
        if line is not None and not (isinstance(line, int) and line >= 0):
            raise ValueError("a word's line is a whole number from 0")
        text = str(item["text"])[:MAX_WORD_CHARS]
        speaker = str(item.get("speaker") or "").strip()[:MAX_SPEAKER_CHARS] or None
        words.append(Word(text, start, end, line, speaker))
    return sorted(words, key=lambda word: word.start)


def channels(hex_colour: str) -> tuple[int, int, int]:
    value = int(hex_colour, 16)
    return value >> 16, (value >> 8) & 255, value & 255


def background_filter(background: Background, look: Look) -> str:
    cscheme = "|".join(
        f"{part / 255:.2f}" for part in (*channels(look.left), *channels(look.right))
    )
    return BACKGROUNDS[background].format(
        w=WIDTH,
        h=HEIGHT,
        fps=FPS,
        cscheme=cscheme,
        left=look.left,
        right=look.right,
        palette=look.palette,
    )


def timestamp(seconds: float) -> str:
    hours, rest = divmod(max(0.0, seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"


def ass_text(text: str) -> str:
    """Keep text on one ASS line, since libass breaks lines at a carriage return too."""
    return re.sub(r"\s+", " ", text.replace("\\", "").replace("{", "(").replace("}", ")"))


def group_by_pauses(words: list[Word]) -> list[list[Word]]:
    lines: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        pause = current and word.start - current[-1].end > LINE_GAP_SECONDS
        new_speaker = current and word.speaker != current[-1].speaker
        if current and (pause or new_speaker or len(current) >= MAX_WORDS_PER_LINE):
            lines.append(current)
            current = []
        current.append(word)
    if current:
        lines.append(current)
    return lines


def group_by_line(words: list[Word]) -> list[list[Word]]:
    """Split the words at the lyric line each one belongs to, keeping long lines readable."""
    lines: list[list[Word]] = []
    for word in words:
        current = lines[-1] if lines else None
        if current and current[-1].line == word.line and len(current) < MAX_WORDS_PER_LINE * 2:
            current.append(word)
        else:
            lines.append([word])
    return lines


def inline_colour(style_colour: str) -> str:
    return f"&H{style_colour[-6:]}&"


def word_effect(highlight: Highlight, look: Look, starts_at_ms: int) -> str:
    """Return ASS tags that flash a thick glow as the word starts, keeping the line layout."""
    if highlight != "glow":
        return ""
    rest = f"\\bord{look.outline_width}\\blur{look.blur}\\3c{inline_colour(look.outline)}"
    flash = f"\\bord9\\blur6\\3c{inline_colour(look.sung)}"
    return (
        f"{rest}\\t({starts_at_ms},{starts_at_ms + 80},{flash})"
        f"\\t({starts_at_ms + 80},{starts_at_ms + 450},{rest})"
    )


def karaoke_line(words: list[Word], shown_from: float, highlight: Highlight, look: Look) -> str:
    fill = "kf" if highlight == "sweep" else "k"
    parts = [f"{{\\k{round((words[0].start - shown_from) * 100)}}}"]
    for index, word in enumerate(words):
        until = words[index + 1].start if index + 1 < len(words) else word.end
        effect = word_effect(highlight, look, round((word.start - shown_from) * 1000))
        length = max(1, round((until - word.start) * 100))
        parts.append(f"{{\\{fill}{length}{effect}}}{ass_text(word.text)} ")
    return "".join(parts).rstrip()


def trim_lead_in(line: list[Word]) -> list[Word]:
    """Start a line's first word at most a second before it ends.

    The aligner stretches a line's first word back over the music before it.
    """
    first = line[0]
    return [replace(first, start=max(first.start, first.end - FIRST_WORD_MAX_SECONDS)), *line[1:]]


STYLE_FIELDS = (
    "Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
    "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding"
)


def style_line(name: str, size: int, colours: str, outline: str, layout: str) -> str:
    """Return one ASS style; `layout` is spacing through encoding after the border style."""
    return f"Style: {name},{FONT},{size},{colours},{outline},&H80000000,0,0,0,0,100,100,{layout}"


def ass_header(look: Look) -> str:
    styles = [
        style_line(
            "Current",
            64,
            f"{look.sung},{look.unsung}",
            look.outline,
            f"1,0,1,{look.outline_width},2,5,60,60,0,1",
        ),
        style_line("Next", 44, "&H90FFFFFF,&H90FFFFFF", look.outline, "1,0,1,2,0,5,60,60,0,1"),
        style_line("Title", 34, "&H00FFFFFF,&H00FFFFFF", look.outline, "2,0,1,2,0,8,60,60,40,1"),
        # Border style 3 draws the outline colour as a box behind the name.
        style_line("Speaker", 30, "&H00FFFFFF,&H00FFFFFF", "&H00000000", "1,0,3,8,0,5,60,60,0,1"),
    ]
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {WIDTH}",
        f"PlayResY: {HEIGHT}",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        f"Format: {STYLE_FIELDS}",
        *styles,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    return "\n".join(lines) + "\n"


def build_ass(words: list[Word], title: str, style: Style) -> str:
    look, highlight = LOOKS[style.look], style.highlight
    words = [word for word in words if word.text.strip()]
    grouped = group_by_line(words) if all(word.line is not None for word in words) else None
    lines = [trim_lead_in(line) for line in grouped or group_by_pauses(words)]
    events = [f"Dialogue: 1,0:00:00.00,9:00:00.00,Title,,0,0,0,,{ass_text(title)}"] if title else []
    speakers = list(dict.fromkeys(word.speaker for word in words if word.speaker))
    blur = f"\\blur{look.blur}" if look.blur else ""
    previous_end = 0.0
    for index, line in enumerate(lines):
        upcoming = lines[index + 1] if index + 1 < len(lines) else None
        shown_from = max(previous_end, line[0].start - LEAD_IN_SECONDS)
        shown_until = line[-1].end + LINGER_SECONDS
        if upcoming:
            shown_until = min(shown_until, upcoming[0].start - 0.05)
        shown_until = max(shown_until, line[-1].end + 0.2)
        span = f"{timestamp(shown_from)},{timestamp(shown_until)}"
        events.append(
            f"Dialogue: 2,{span},Current,,0,0,0,,{{\\pos({WIDTH // 2},{HEIGHT // 2 + 40}){blur}}}"
            f"{karaoke_line(line, shown_from, highlight, look)}"
        )
        if speaker := line[0].speaker:
            colour = SPEAKER_COLOURS[speakers.index(speaker) % len(SPEAKER_COLOURS)]
            events.append(
                f"Dialogue: 3,{span},Speaker,,0,0,0,,{{\\pos({WIDTH // 2},{HEIGHT // 2 - 30})"
                f"\\3c{inline_colour(colour)}}}{ass_text(speaker.upper())}"
            )
        if upcoming and style.upcoming:
            text = " ".join(ass_text(word.text) for word in upcoming)
            events.append(
                f"Dialogue: 2,{span},Next,,0,0,0,,{{\\pos({WIDTH // 2},{HEIGHT // 2 + 130})}}{text}"
            )
        previous_end = shown_until
    return ass_header(look) + "\n".join(events) + "\n"


def track_filter(with_vocals: bool, cut: float) -> str:
    """Return the filter that makes the track: the song minus that share of its vocal stem."""
    if not with_vocals or cut <= 0:
        return "[0:a]asplit[track][heard];"
    # We subtract with pan, since amix silently drops negative weights.
    return (
        f"[0:a][1:a]amerge=inputs=2,pan=stereo|c0=c0-{cut}*c2|c1=c1-{cut}*c3,asplit[track][heard];"
    )


def visualizer_graph(style: Style, ass_path: Path, track: str) -> str:
    visual = background_filter(style.background, LOOKS[style.look])
    return (
        track + f"color=c=0x0b0e16:s={WIDTH}x{HEIGHT}:r={FPS}[base];"
        f"[heard]{visual},format=rgba,colorchannelmixer=aa=0.55[visual];"
        f"[base][visual]overlay=shortest=1,ass={ass_path}:fontsdir={FONTS_DIR},format=yuv420p[video]"
    )


def slideshow_inputs(paths: list[Path], starts: list[float], duration: float) -> list[str]:
    inputs: list[str] = []
    for index, path in enumerate(paths):
        until = starts[index + 1] if index + 1 < len(starts) else duration
        seconds = max(1.0, until - (0 if index == 0 else starts[index]))
        inputs += [*ffmpeg.PICTURE_GUARD, "-loop", "1", "-framerate", str(SLIDESHOW_FPS)]
        inputs += ["-t", f"{seconds:.3f}", "-i", str(path)]
    return inputs


def slideshow_graph(first_input: int, count: int, ass_path: Path) -> str:
    """Show the pictures in turn, each fading in, with the words over them."""
    chains = [
        f"[{first_input + index}:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},setsar=1,fps={SLIDESHOW_FPS},format=yuv420p,"
        f"fade=t=in:st=0:d={PICTURE_FADE_SECONDS}[picture{index}]"
        for index in range(count)
    ]
    labels = "".join(f"[picture{index}]" for index in range(count))
    return (
        ";".join(chains)
        + f";{labels}concat=n={count}:v=1:a=0,"
        + f"ass={ass_path}:fontsdir={FONTS_DIR},format=yuv420p[video]"
    )


def render(song: Song, style: Style, pictures: list[Picture], timeout: float) -> bytes:
    """Render the karaoke MP4: over the pictures when there are any, else over a visualizer.

    :raises FfmpegError: when ffmpeg cannot read the inputs.
    """
    vocals, words = song.vocals, song.words
    with tempfile.TemporaryDirectory() as workdir:
        folder = Path(workdir)
        (folder / "audio").write_bytes(song.audio)
        ass_path, target = folder / "lyrics.ass", folder / "karaoke.mp4"
        lyrics = build_ass(words, song.title, style)
        ass_path.write_text(lyrics, encoding="utf-8")
        inputs = [*ffmpeg.AUDIO_GUARD, "-i", str(folder / "audio")]
        if vocals:
            (folder / "vocals").write_bytes(vocals.stem)
            inputs += [*ffmpeg.AUDIO_GUARD, "-i", str(folder / "vocals")]
        track = track_filter(vocals is not None, vocals.cut if vocals else 0)
        if pictures:
            paths = [folder / f"picture-{index}" for index in range(len(pictures))]
            for path, picture in zip(paths, pictures, strict=True):
                path.write_bytes(picture.data)
            duration = ffmpeg.probe(folder / "audio", timeout).seconds
            starts = [picture.start for picture in pictures]
            inputs += slideshow_inputs(paths, starts, duration)
            graph = track + "[heard]anullsink;"
            graph += slideshow_graph(2 if vocals else 1, len(pictures), ass_path)
            video = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-tune", "stillimage"]
        else:
            graph = visualizer_graph(style, ass_path, track)
            video = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
        maps = ["-filter_complex", graph, "-map", "[video]", "-map", "[track]"]
        sound = ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest"]
        ffmpeg.run([*inputs, *maps, *video, *sound, "-y", str(target)], timeout)
        return target.read_bytes()
