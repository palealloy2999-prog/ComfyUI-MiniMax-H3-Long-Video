# Prompt format for ComfyUI-MiniMax-H3-Long-Video

## Recommended structure

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description:
[Shot 1] Establish the opening composition and begin the first action.

[Shot 2] At 00:04.500, cut to the next viewpoint and continue with a new action.

[Shot 3] At 00:09.250, continue forward into the next distinct beat.

[Global Instructions]
Preserve the same subject identity, clothing, environment logic, and visual style throughout.
Maintain continuous motion and audio across generated segment boundaries.

overall_soundscape: Describe ambience, physical sounds, and non-verbal human sounds across the complete timeline.

non_diegetic_music: Describe one continuous score and its changes across the complete timeline, or use N/A.
```

Omit the first reference-alignment line when the selected H3 mode does not use it. Keep `overall_soundscape` and `non_diegetic_music` outside `integrated_multimodal_description`.

For full-reference Ref2VA, retain H3's `subject_definitions`, `summary`, and `retention_analysis` sections, then use the same timeline inside `detailed_description`:

```text
detailed_description:
[Shot 1] Establish the referenced subjects and begin the first action.
[Shot 2] At 00:05.000, continue with the next distinct action.

[Global Instructions]
Keep every reference label and subject identity consistent across the complete timeline.

overall_soundscape: ...
non_diegetic_music: ...
```

## How the node divides the prompt

- A Shot is assigned only to the segment containing that Shot's global start time.
- The node rebases selected Shot timestamps to each segment's local time.
- Text before the first Shot is common to all segments.
- The standalone `[Global Instructions]` marker and everything after it inside the multimodal description are common to all segments.
- Text after the final Shot but before `[Global Instructions]` belongs only to the final Shot.
- The soundscape and music fields are copied to every segment.

Consequently, a Shot should contain only the action beginning at its own timestamp. Do not repeat the preceding Shot's full action as a continuity reminder. If a long action crosses an internal segment boundary, describe it once at its real start; the node's AV latent guide carries its motion forward.

## Shot time and segment windows

The node works at 24 fps. When its settings are available, derive the delivered timeline as follows:

- Total delivered duration is `length / 24` seconds.
- Without `initial_latent`, segment 0 can deliver up to `max_raw_frames` frames.
- Each continuation segment generates `context_frames` of removable guide content, so its normal delivered capacity is `max_raw_frames - context_frames` frames.
- With `initial_latent`, segment 0 is also a continuation segment and uses that reduced capacity.
- The final segment may deliver fewer frames than its generated H3 frame-grid length.

Build consecutive half-open windows `[start, end)`. A Shot is sent only to the window satisfying `start <= shot_time < end`. A Shot exactly on a boundary belongs to the later segment. The node then adds `context_frames / 24` to its local timestamp because the removable guide occupies the beginning of that sampling pass.

Example with `length=720`, `max_raw_frames=124`, `context_frames=22`, and no `initial_latent`:

```text
segment 0: frames [0, 124)   = 00:00.000–00:05.167
segment 1: frames [124, 226) = 00:05.167–00:09.417
segment 2: frames [226, 328) = 00:09.417–00:13.667
...
```

The first continuation Shot in segment 1 is rebased behind its 22-frame guide. For example, a global Shot at `00:06.000` becomes local time `22/24 + (6.000 - 124/24) = 00:01.750`.

When producing the requested summary, list these windows if `length`, `max_raw_frames`, and `context_frames` were provided. Use them to verify Shot ownership, not to duplicate actions at the boundaries.

## Timeline rules

- Use sequential Shot numbers.
- Use strictly increasing global timestamps with millisecond precision.
- Keep every timestamp earlier than the requested delivered duration.
- Prefer a new Shot for a cut, new viewpoint, new state, new location, or distinct action beat.
- Use camera motion within the same Shot when only framing distance or angle changes continuously.
- Put local music cues, impacts, dialogue, and lyrics in the Shot where they occur. Put only whole-video audio continuity rules under `[Global Instructions]` or in the two audio fields.

## Global versus local example

```text
[Shot 7] At 00:24.000, she lands on the roof, bends both knees under the impact, and immediately accelerates toward the ledge.

[Global Instructions]
Keep her face, mechanical limbs, outfit, and proportions identical in every Shot.
Keep the same music track continuous across the full video.
```

The landing instruction stays in Shot 7. It must not be written as an unmarked `Landing requirement:` after the final Shot, because arbitrary `requirement:` headings are not global markers.
