import math
import re
from dataclasses import dataclass


FPS = 24
#MAX_RAW_FRAMES = 345
MAX_RAW_FRAMES = 158   #test with 158 frames to see if it works better for long videos, since 345 is too much for the model to handle


@dataclass(frozen=True)
class Segment:
    index: int
    raw_frames: int
    head_frames: int
    output_start: int
    output_frames: int

    @property
    def prompt_start_seconds(self):
        return (self.output_start - self.head_frames) / FPS

    @property
    def prompt_end_seconds(self):
        return self.prompt_start_seconds + self.raw_frames / FPS


def plan_segments(output_frames, context_frames, has_initial_latent):
    if output_frames < 1:
        raise ValueError("output_frames must be positive")
    if context_frames not in (22, 39):
        raise ValueError("context_frames must be 22 or 39")

    segments = []
    remaining = int(output_frames)
    output_start = 0
    index = 0
    while remaining:
        head = context_frames if has_initial_latent or index else 0
        if head:
            capacity = MAX_RAW_FRAMES - head
            wanted = min(remaining, capacity)
            generated = min(capacity, math.ceil(wanted / 17) * 17)
            raw_frames = head + generated
        else:
            wanted = min(remaining, MAX_RAW_FRAMES)
            raw_frames = wanted
            while raw_frames % 17 != 5:
                raw_frames += 1
            generated = raw_frames

        delivered = min(remaining, generated)
        segments.append(Segment(index, raw_frames, head, output_start, delivered))
        remaining -= delivered
        output_start += delivered
        index += 1
    return segments


_TIMESTAMP = re.compile(r"(?<!\d)(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{3})(?!\d)")
_SHOT = re.compile(
    r"\[Shot\s+(\d+)\](?:\s+At\s+((?:(?:\d{1,2}):)?\d{2}:\d{2}\.\d{3}))?\s*,?",
    re.IGNORECASE,
)
_INTEGRATED = re.compile(
    r"integrated_multimodal_description\s*:\s*(.*?)(?=\n\s*overall_soundscape\s*:|\n\s*non_diegetic_music\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_SOUNDSCAPE = re.compile(
    r"overall_soundscape\s*:\s*(.*?)(?=\n\s*non_diegetic_music\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_MUSIC = re.compile(r"non_diegetic_music\s*:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)
_GLOBAL_TAIL = re.compile(
    r"(?m)^(?:Camera-cut requirement|Rear-drone limitation|Cut-in duration requirement|Action-editing rhythm)\s*:",
    re.IGNORECASE,
)


def parse_timestamp(value):
    match = _TIMESTAMP.fullmatch(value.strip())
    if match is None:
        raise ValueError("invalid H3 timestamp: {}".format(value))
    hours, minutes, seconds, millis = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def format_timestamp(seconds):
    millis = max(0, int(round(seconds * 1000.0)))
    minutes, remainder = divmod(millis, 60_000)
    secs, ms = divmod(remainder, 1000)
    return "{:02d}:{:02d}.{:03d}".format(minutes, secs, ms)


def _fallback_prompt(prompt, start_seconds, head_seconds):
    prefix = ""
    if head_seconds:
        prefix = (
            "For the first {:.3f} seconds, faithfully preserve the supplied motion and sound context. "
            "At {}, continue forward from its final moment with new motion. Do not restart, replay, "
            "or return to an earlier pose or composition.\n\n"
        ).format(head_seconds, format_timestamp(head_seconds))

    def rebase(match):
        absolute = parse_timestamp(match.group(0))
        return format_timestamp(absolute - start_seconds)

    return prefix + _TIMESTAMP.sub(rebase, prompt)


def slice_prompt(prompt, start_seconds, end_seconds, head_seconds=0.0):
    integrated = _INTEGRATED.search(prompt)
    global_tail = ""
    if integrated is not None:
        body = integrated.group(1).strip()
        prefix = prompt[:integrated.start()].rstrip()
        wrapped = True
    else:
        first_shot = _SHOT.search(prompt)
        if first_shot is None:
            return _fallback_prompt(prompt, start_seconds, head_seconds)
        tail = _GLOBAL_TAIL.search(prompt, first_shot.end())
        body_end = tail.start() if tail is not None else len(prompt)
        body = prompt[first_shot.start():body_end].strip()
        prefix = prompt[:first_shot.start()].rstrip()
        global_tail = prompt[body_end:].strip()
        wrapped = False

    matches = list(_SHOT.finditer(body))
    if not matches:
        return _fallback_prompt(prompt, start_seconds, head_seconds)

    shots = []
    for index, match in enumerate(matches):
        timestamp = match.group(2)
        if timestamp is None:
            if shots:
                raise ValueError("every H3 shot after Shot 1 needs an At MM:SS.mmm timestamp")
            start = 0.0
        else:
            start = parse_timestamp(timestamp)
        if shots and start <= shots[-1][0]:
            raise ValueError("H3 shot timestamps must be strictly increasing")
        text_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        shots.append((start, body[match.end():text_end].strip()))

    selected = []
    for index, (shot_start, text) in enumerate(shots):
        shot_end = shots[index + 1][0] if index + 1 < len(shots) else float("inf")
        if shot_start < end_seconds and shot_end > start_seconds:
            selected.append((shot_start, text))
    if not selected:
        selected.append(shots[-1])

    rendered = []
    for index, (shot_start, text) in enumerate(selected):
        if index == 0:
            marker = "[Shot 1]"
            if head_seconds:
                text = (
                    "For the first {:.3f} seconds, faithfully preserve the supplied motion and sound context. "
                    "At {}, continue this ongoing shot forward from its final moment. Advance the action with "
                    "new motion; do not restart, replay, or return to an earlier pose or composition. {}"
                ).format(head_seconds, format_timestamp(head_seconds), text)
            elif shot_start < start_seconds:
                text = "Continuing seamlessly from the preceding segment, " + text
        else:
            marker = "[Shot {}] At {},".format(index + 1, format_timestamp(shot_start - start_seconds))
        rendered.append("{} {}".format(marker, text).strip())

    parts = []
    if prefix:
        parts.append(prefix)
    timeline = " ".join(rendered)
    parts.append("integrated_multimodal_description: " + timeline if wrapped else timeline)
    if wrapped:
        soundscape = _SOUNDSCAPE.search(prompt)
        if soundscape is not None:
            parts.append("overall_soundscape: " + soundscape.group(1).strip())
        music = _MUSIC.search(prompt)
        if music is not None:
            parts.append("non_diegetic_music: " + music.group(1).strip())
    elif global_tail:
        parts.append(global_tail)
    return "\n\n".join(parts)
