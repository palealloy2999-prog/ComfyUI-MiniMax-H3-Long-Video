import math
import re
from dataclasses import dataclass


FPS = 24


@dataclass(frozen=True)
class Segment:
    index: int
    raw_frames: int
    context_frames: int
    output_start: int
    output_frames: int

    @property
    def prompt_start_seconds(self):
        return self.output_start / FPS

    @property
    def prompt_end_seconds(self):
        return self.prompt_start_seconds + (self.raw_frames - self.context_frames) / FPS


def plan_segments(output_frames, context_frames, has_initial_latent, max_raw_frames):
    if output_frames < 1:
        raise ValueError("output_frames must be positive")
    if context_frames not in (22, 39):
        raise ValueError("context_frames must be 22 or 39")
    if max_raw_frames < 5 or max_raw_frames % 17 != 5:
        raise ValueError("max_raw_frames must use the MiniMax H3 17k+5 frame grid")
    if max_raw_frames <= context_frames:
        raise ValueError("max_raw_frames must be greater than context_frames")

    segments = []
    remaining = int(output_frames)
    output_start = 0
    index = 0
    while remaining:
        context = context_frames if has_initial_latent or index else 0
        if context:
            capacity = max_raw_frames - context
            wanted = min(remaining, capacity)
            generated = math.ceil(wanted / 17) * 17
            raw_frames = context + generated
        else:
            raw_frames = min(remaining, max_raw_frames)
            while raw_frames % 17 != 5:
                raw_frames += 1
            generated = raw_frames

        delivered = min(remaining, generated)
        segments.append(Segment(index, raw_frames, context, output_start, delivered))
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


def _guide_instruction(context_seconds):
    return (
        "[Shot 1] For the first {:.3f} seconds, follow the supplied AV guide exactly. "
        "The guide is preceding context only. At {}, continue forward from its final moment "
        "with new motion and do not repeat the guided action."
    ).format(context_seconds, format_timestamp(context_seconds))


def _fallback_prompt(prompt, start_seconds, context_seconds):
    def rebase(match):
        absolute = parse_timestamp(match.group(0))
        return format_timestamp(context_seconds + absolute - start_seconds)

    rebased = _TIMESTAMP.sub(rebase, prompt)
    if context_seconds:
        return _guide_instruction(context_seconds) + "\n\n" + rebased
    return rebased


def slice_prompt(prompt, start_seconds, end_seconds, context_seconds=0.0,
                 segment_index=None, segment_count=None,
                 timeline_duration_seconds=None):
    integrated = _INTEGRATED.search(prompt)
    global_tail = ""
    if integrated is not None:
        body = integrated.group(1).strip()
        prefix = prompt[:integrated.start()].rstrip()
        wrapped = True
    else:
        first_shot = _SHOT.search(prompt)
        if first_shot is None:
            return _fallback_prompt(prompt, start_seconds, context_seconds)
        tail = _GLOBAL_TAIL.search(prompt, first_shot.end())
        body_end = tail.start() if tail is not None else len(prompt)
        body = prompt[first_shot.start():body_end].strip()
        prefix = prompt[:first_shot.start()].rstrip()
        global_tail = prompt[body_end:].strip()
        wrapped = False

    matches = list(_SHOT.finditer(body))
    if not matches:
        return _fallback_prompt(prompt, start_seconds, context_seconds)

    parsed = []
    for index, match in enumerate(matches):
        timestamp = match.group(2)
        start = parse_timestamp(timestamp) if timestamp is not None else None
        text_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        parsed.append((start, body[match.end():text_end].strip()))

    explicit = [(index, start) for index, (start, _) in enumerate(parsed) if start is not None]
    if any(current[1] <= previous[1] for previous, current in zip(explicit, explicit[1:])):
        raise ValueError("H3 shot timestamps must be strictly increasing")

    if not explicit:
        if segment_index is None or segment_count is None:
            return _fallback_prompt(prompt, start_seconds, context_seconds)
        if len(parsed) >= segment_count:
            per_segment, extra = divmod(len(parsed), segment_count)
            first = segment_index * per_segment + min(segment_index, extra)
            count = per_segment + (segment_index < extra)
            assigned = parsed[first:first + count]
        elif len(parsed) == 1:
            assigned = parsed if segment_index == 0 else []
        else:
            assigned = [
                shot for index, shot in enumerate(parsed)
                if round(index * (segment_count - 1) / (len(parsed) - 1)) == segment_index
            ]
        duration = end_seconds - start_seconds
        shots = [
            (start_seconds + offset * duration / len(assigned), text)
            for offset, (_, text) in enumerate(assigned)
        ] if assigned else []
    else:
        starts = [start for start, _ in parsed]
        if starts[0] is None:
            starts[0] = 0.0
        anchors = [index for index, start in enumerate(starts) if start is not None]
        for left, right in zip(anchors, anchors[1:]):
            gap = right - left - 1
            if gap:
                step = (starts[right] - starts[left]) / (gap + 1)
                for offset in range(1, gap + 1):
                    starts[left + offset] = starts[left] + step * offset
        last = anchors[-1]
        if last < len(starts) - 1:
            timeline_end = timeline_duration_seconds
            if timeline_end is None:
                timeline_end = end_seconds
            if timeline_end <= starts[last]:
                raise ValueError("H3 timeline must end after its final timestamped Shot")
            gap = len(starts) - last - 1
            step = (timeline_end - starts[last]) / (gap + 1)
            for offset in range(1, gap + 1):
                starts[last + offset] = starts[last] + step * offset
        shots = [(start, text) for start, (_, text) in zip(starts, parsed)]

    selected = [
        (shot_start, text)
        for shot_start, text in shots
        if start_seconds <= shot_start < end_seconds
    ]

    rendered = [_guide_instruction(context_seconds)] if context_seconds else []
    for index, (shot_start, text) in enumerate(selected):
        shot_number = index + 1 + bool(context_seconds)
        local_start = context_seconds + shot_start - start_seconds
        if shot_number == 1 and local_start == 0:
            marker = "[Shot 1]"
        else:
            marker = "[Shot {}] At {},".format(
                shot_number, format_timestamp(local_start))
        rendered.append("{} {}".format(marker, text).strip())
    if not rendered:
        rendered.append(
            "[Shot 1] Continue forward from the supplied preceding AV context. "
            "Do not restart or repeat any action already shown."
        )

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
