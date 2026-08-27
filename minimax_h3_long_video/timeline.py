import re
from dataclasses import dataclass


FPS = 24
PROMPT_PLAN_SCHEMA_VERSION = 1


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
        return (self.output_start + self.output_frames) / FPS


def _h3_grid_frames(minimum_frames):
    frames = max(5, int(minimum_frames))
    return frames + ((5 - frames) % 17)


def _segment_output_frames(max_raw_frames):
    """Reverse the common whole-second input before its 17k+5 padding."""
    nearest_seconds = round(max_raw_frames / FPS)
    for seconds in (nearest_seconds, nearest_seconds - 1, nearest_seconds + 1):
        whole_seconds_frames = seconds * FPS
        if seconds > 0 and _h3_grid_frames(whole_seconds_frames) == max_raw_frames:
            return whole_seconds_frames
    return max_raw_frames


def plan_segments(output_frames, context_frames, has_initial_latent, max_raw_frames):
    if output_frames < 1:
        raise ValueError("output_frames must be positive")
    if context_frames not in (22, 39):
        raise ValueError("context_frames must be 22 or 39")
    if max_raw_frames < 5 or max_raw_frames % 17 != 5:
        raise ValueError("max_raw_frames must use the MiniMax H3 17k+5 frame grid")

    # max_raw_frames selects a human-scale timeline window. For example, 362
    # frames means a 15-second master window, not a 15.083-second prompt cut.
    # The removable AV guide and H3 grid padding are internal generation detail.
    segment_frames = _segment_output_frames(max_raw_frames)

    segments = []
    remaining = _segment_output_frames(int(output_frames))
    output_start = 0
    index = 0
    while remaining:
        context = context_frames if has_initial_latent or index else 0
        delivered = min(remaining, segment_frames)
        raw_frames = _h3_grid_frames(context + delivered)
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
_DETAILED = re.compile(
    r"detailed_description\s*:\s*(.*?)(?=\n\s*overall_soundscape\s*:|\n\s*non_diegetic_music\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_SOUNDSCAPE = re.compile(
    r"overall_soundscape\s*:\s*(.*?)(?=\n\s*non_diegetic_music\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_MUSIC = re.compile(r"non_diegetic_music\s*:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)
_GLOBAL_INSTRUCTIONS = re.compile(
    r"(?mi)^[ \t]*\[Global Instructions\][ \t]*$",
)
_SUMMARY = re.compile(
    r"(?ims)^([ \t]*summary\s*:\s*)(.*?)(?=^[ \t]*retention_analysis\s*:)",
)
_KEYFRAME_ALIGNMENT = re.compile(
    r"(?mi)^(?:For the target video, at 0\.00 seconds into the target video,.*fully referenced\.|"
    r"How the reference pictures align with the target video.*)$\s*",
)
_OUTER_CODE_FENCE = re.compile(
    r"\A\s*```(?:text|txt)?[ \t]*\r?\n(.*?)\r?\n```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
_BARE_S_DEFINITION = re.compile(r"(?mi)^[ \t]*(S\d+)[ \t]*(?:=|:)")


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


def _timestamp_millis(seconds):
    return int(round(seconds * 1000.0))


def _guide_instruction(context_seconds, active_text=""):
    instruction = (
        "The supplied AV guide is preceding context and is outside this segment's "
        "timestamp clock. Segment time 00:00.000 begins immediately after the guide's "
        "final moment. Continue forward without repeating the guided action."
    )
    if active_text:
        instruction += (
            " Continue the master-timeline Shot already in progress from its current state; "
            "do not replay its beginning: {}"
        ).format(active_text)
    return instruction


def _unguided_continuation(active_text):
    return (
        "[Shot 1] Continue the master-timeline Shot already in progress from its current "
        "state; do not restart or replay its beginning: {}"
    ).format(active_text)


def _segment_scope(start_seconds, master_end_seconds, context_seconds,
                   generated_end_seconds=None):
    if generated_end_seconds is None:
        generated_end_seconds = master_end_seconds
    local_duration = generated_end_seconds - start_seconds
    return (
        "Long-video segment scope: the master timeline range is {}-{}. "
        "This H3 generation pass is {:.3f} seconds long"
    ).format(
        format_timestamp(start_seconds),
        format_timestamp(master_end_seconds),
        local_duration,
    ) + (
        ". The preceding AV guide is outside this local timestamp clock."
        if context_seconds else "."
    )


def _scope_reference_prefix(prefix, start_seconds, master_end_seconds,
                            context_seconds, generated_end_seconds):
    match = _SUMMARY.search(prefix)
    if match is None:
        return prefix
    scope = _segment_scope(
        start_seconds, master_end_seconds, context_seconds,
        generated_end_seconds)
    if start_seconds:
        scoped_summary = (
            "This is a continuation pass from the complete long-video master. {} "
            "Continue from the supplied preceding AV guide and the local Shot timeline. "
            "Do not restart the opening, recap completed Shots, or execute master-timeline "
            "beats outside this segment."
        ).format(scope)
    else:
        scoped_summary = "{}\n{}".format(match.group(2).rstrip(), scope)
    suffix = prefix[match.end(2):].lstrip()
    return prefix[:match.start(2)] + scoped_summary + "\n\n" + suffix


def _strip_outer_code_fence(prompt):
    match = _OUTER_CODE_FENCE.fullmatch(prompt)
    return match.group(1) if match is not None else prompt


def _validate_reference_labels(prompt, field_name):
    if field_name != "detailed_description":
        return
    invalid = sorted(set(_BARE_S_DEFINITION.findall(prompt)))
    if invalid:
        raise ValueError(
            "Invalid Ref2VA subject label(s): {}. Define visual references as "
            "<Subject N> in subject_definitions and use (Sx) only for speakers."
            .format(", ".join(invalid))
        )


def _localize_audio(value, kind, start_seconds):
    value = value.strip()
    if not start_seconds:
        return value
    if kind == "soundscape":
        continuity = (
            "Continue the soundscape established by the supplied AV guide without "
            "restarting it. Follow only sound events in the local Shot timeline."
        )
    else:
        continuity = (
            "Continue the already-playing master score seamlessly from the supplied AV "
            "guide. Do not restart its intro, drop, vocals, or earlier musical phases. "
            "Follow only music cues in the local Shot timeline."
        )
    # Absolute master times become local times after segmentation and would replay the
    # opening/drop/finale in every continuation pass. Keep timeless style constraints,
    # but replace a time-bearing master audio timeline with an explicit continuation.
    if _TIMESTAMP.search(value):
        return continuity
    return "{} {}".format(continuity, value) if value else continuity


def _timeline_field(prompt):
    matches = []
    for field_name, pattern in (
            ("integrated_multimodal_description", _INTEGRATED),
            ("detailed_description", _DETAILED)):
        match = pattern.search(prompt)
        if match is not None:
            matches.append((match.start(), field_name, match))
    if not matches:
        return None, None
    _, field_name, match = min(matches, key=lambda item: item[0])
    return field_name, match


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
    prompt = _strip_outer_code_fence(prompt)
    master_end_seconds = end_seconds
    if timeline_duration_seconds is not None:
        master_end_seconds = min(master_end_seconds, timeline_duration_seconds)
    field_name, field = _timeline_field(prompt)
    _validate_reference_labels(prompt, field_name)
    if field is not None:
        content = field.group(1).strip()
        prefix = prompt[:field.start()].rstrip()
        wrapped = True
    else:
        content = prompt
        prefix = ""
        wrapped = False

    all_matches = list(_SHOT.finditer(content))
    if not all_matches:
        return _fallback_prompt(prompt, start_seconds, context_seconds)
    first_shot = all_matches[0]
    global_markers = list(_GLOBAL_INSTRUCTIONS.finditer(content))
    if len(global_markers) > 1:
        raise ValueError("H3 prompt must contain at most one [Global Instructions] marker")
    tail = global_markers[0] if global_markers else None
    if tail is not None and tail.start() < all_matches[-1].end():
        raise ValueError("[Global Instructions] must appear after the final Shot")
    body_end = tail.start() if tail is not None else len(content)
    common_intro = content[:first_shot.start()].strip()
    global_tail = content[tail.end():].strip() if tail is not None else ""
    body = content[first_shot.start():body_end].strip()

    matches = list(_SHOT.finditer(body))
    if not matches:
        return _fallback_prompt(prompt, start_seconds, context_seconds)

    parsed = []
    for index, match in enumerate(matches):
        timestamp = match.group(2)
        start = parse_timestamp(timestamp) if timestamp is not None else None
        text_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        parsed.append((int(match.group(1)), start, body[match.end():text_end].strip()))

    shot_numbers = [number for number, _, _ in parsed]
    if shot_numbers != list(range(1, len(parsed) + 1)):
        raise ValueError("H3 Shot numbers must be sequential starting at 1")

    explicit = [
        (index, start)
        for index, (_, start, _) in enumerate(parsed)
        if start is not None
    ]
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
        duration = master_end_seconds - start_seconds
        shots = [
            (number, start_seconds + offset * duration / len(assigned), text)
            for offset, (number, _, text) in enumerate(assigned)
        ] if assigned else []
    else:
        starts = [start for _, start, _ in parsed]
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
                timeline_end = master_end_seconds
            if timeline_end <= starts[last]:
                raise ValueError("H3 timeline must end after its final timestamped Shot")
            gap = len(starts) - last - 1
            step = (timeline_end - starts[last]) / (gap + 1)
            for offset in range(1, gap + 1):
                starts[last + offset] = starts[last] + step * offset
        shots = [
            (number, start, text)
            for start, (number, _, text) in zip(starts, parsed)
        ]

    window_start_millis = _timestamp_millis(start_seconds)
    window_end_millis = _timestamp_millis(master_end_seconds)
    selected = [
        (number, shot_start, text)
        for number, shot_start, text in shots
        if window_start_millis <= _timestamp_millis(shot_start) < window_end_millis
    ]
    starts_with_new_shot = bool(
        selected and _timestamp_millis(selected[0][1]) == window_start_millis)
    active = None if starts_with_new_shot else next(
        (
            (number, shot_start, text)
            for number, shot_start, text in reversed(shots)
            if _timestamp_millis(shot_start) < window_start_millis
        ),
        None,
    )

    active_text = active[2] if active is not None else ""
    guide_note = _guide_instruction(context_seconds) if context_seconds else ""
    if start_seconds and active_text:
        rendered = [_unguided_continuation(active_text)]
    else:
        rendered = []
    shot_number_offset = len(rendered)
    for index, (_, shot_start, text) in enumerate(selected):
        shot_number = index + 1 + shot_number_offset
        local_start = shot_start - start_seconds
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

    if start_seconds and prefix:
        prefix = _KEYFRAME_ALIGNMENT.sub("", prefix).rstrip()
    if field_name == "detailed_description" and prefix:
        prefix = _scope_reference_prefix(
            prefix, start_seconds, master_end_seconds, context_seconds,
            end_seconds)

    parts = []
    if prefix:
        parts.append(prefix)
    timeline = " ".join(rendered)
    if wrapped:
        if field_name == "integrated_multimodal_description":
            common_intro = "{}\n{}".format(
                _segment_scope(
                    start_seconds, master_end_seconds, context_seconds,
                    end_seconds),
                common_intro,
            ).strip()
        timeline_parts = [
            value for value in (common_intro, global_tail, guide_note, timeline) if value
        ]
        parts.append("{}: {}".format(field_name, "\n\n".join(timeline_parts)))
        soundscape = _SOUNDSCAPE.search(prompt)
        if soundscape is not None:
            parts.append(
                "overall_soundscape: "
                + _localize_audio(soundscape.group(1), "soundscape", start_seconds)
            )
        music = _MUSIC.search(prompt)
        if music is not None:
            parts.append(
                "non_diegetic_music: "
                + _localize_audio(music.group(1), "music", start_seconds)
            )
    else:
        if common_intro:
            parts.append(common_intro)
        if guide_note:
            parts.append(guide_note)
        parts.append(timeline)
        if global_tail:
            parts.append(global_tail)
    return "\n\n".join(parts)


def _segment_record(segment):
    return {
        "index": segment.index,
        "raw_frames": segment.raw_frames,
        "context_frames": segment.context_frames,
        "output_start": segment.output_start,
        "output_frames": segment.output_frames,
    }


def build_prompt_plan(master_prompt, length, max_raw_frames, context_frames,
                      has_initial_latent=False, overrides=None):
    """Build the exact local prompts consumed by a long-video sampler run."""
    context_frames = int(context_frames)
    segments = plan_segments(
        length, context_frames, bool(has_initial_latent), max_raw_frames)
    delivered_length = sum(item.output_frames for item in segments)
    prompts = [
        slice_prompt(
            master_prompt,
            item.prompt_start_seconds,
            item.prompt_end_seconds,
            item.context_frames / FPS,
            item.index,
            len(segments),
            delivered_length / FPS,
        )
        for item in segments
    ]
    for name, value in (overrides or {}).items():
        if value is None or not str(value).strip():
            continue
        try:
            index = int(name.rsplit("_", 1)[-1])
        except (AttributeError, TypeError, ValueError):
            raise ValueError(
                "segment prompt override has an invalid index: {}".format(name))
        if index < 0 or index >= len(prompts):
            raise ValueError(
                "segment prompt override {} is outside the {}-segment plan"
                .format(index, len(prompts)))
        prompts[index] = str(value).strip()
    return {
        "schema": PROMPT_PLAN_SCHEMA_VERSION,
        "length_input": int(length),
        "delivered_length": delivered_length,
        "max_raw_frames": int(max_raw_frames),
        "context_frames": context_frames,
        "has_initial_latent": bool(has_initial_latent),
        "segments": [
            {**_segment_record(item), "prompt": local_prompt}
            for item, local_prompt in zip(segments, prompts)
        ],
    }


def prompt_plan_prompts(prompt_plan, segments, length, max_raw_frames,
                        context_frames, has_initial_latent):
    """Validate a prompt plan against sampler settings and return its prompts."""
    if not isinstance(prompt_plan, dict):
        raise ValueError("prompt_plan is not a MiniMax H3 Long prompt plan")
    expected = {
        "schema": PROMPT_PLAN_SCHEMA_VERSION,
        "length_input": int(length),
        "max_raw_frames": int(max_raw_frames),
        "context_frames": int(context_frames),
        "has_initial_latent": bool(has_initial_latent),
    }
    if any(prompt_plan.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "prompt_plan settings do not match length, max_raw_frames, "
            "context_frames, or initial_latent")
    entries = prompt_plan.get("segments")
    if not isinstance(entries, list) or len(entries) != len(segments):
        raise ValueError("prompt_plan segment count does not match the sampler")
    prompts = []
    for segment, entry in zip(segments, entries):
        if not isinstance(entry, dict):
            raise ValueError("prompt_plan contains an invalid segment")
        expected_segment = _segment_record(segment)
        if any(entry.get(key) != value for key, value in expected_segment.items()):
            raise ValueError("prompt_plan segment layout does not match the sampler")
        local_prompt = entry.get("prompt")
        if not isinstance(local_prompt, str) or not local_prompt.strip():
            raise ValueError("prompt_plan contains an empty segment prompt")
        prompts.append(local_prompt.strip())
    return prompts
