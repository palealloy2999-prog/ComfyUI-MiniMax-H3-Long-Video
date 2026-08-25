---
name: minimax-h3-long-video-prompt-writing
description: Write or revise timestamped long-form MiniMax H3 prompts specifically for ComfyUI-MiniMax-H3-Long-Video, including segment-safe Shot progression and the node's explicit [Global Instructions] section. Use for prompts intended for this custom node; do not use for ordinary H3 prompts unless this long-video format is requested.
---

# MiniMax H3 Long Video Prompt Writing

Create one paste-ready prompt whose global timeline can be divided by ComfyUI-MiniMax-H3-Long-Video without replaying earlier actions.

## Workflow

1. Determine the intended total duration, reference assets, dialogue or lyrics, continuous audio, and required visual beats. When the user supplies node settings, treat `length / 24` as the delivered duration and use `max_raw_frames` plus `context_frames` to audit the segment windows.
2. Use the standard H3 audiovisual fields and reference labels appropriate to the requested generation mode.
3. Build a fully timestamped `[Shot N]` timeline. Give each action or camera beat to one Shot only; do not recap it in a later Shot to create continuity.
4. Put reusable identity, style, continuity, camera, and audio rules after the final Shot beneath a standalone `[Global Instructions]` line.
5. Check that Shot timestamps are strictly increasing, remain inside the requested duration, and describe forward progress through the final frame. If node settings are known, map every Shot start time to its exact segment and report the segment windows in the summary.
6. Return the finished prompt in one code block, followed by a compact duration and reference-role summary.

Read [references/prompt-format.md](references/prompt-format.md) before writing or revising a prompt. Its marker placement and timeline rules are part of this node's parsing contract.

## Constraints

- Write `[Shot 1]` without a timestamp and use `[Shot N] At MM:SS.mmm,` for later Shots. Use `HH:MM:SS.mmm` when the timeline exceeds 59 minutes.
- Use `[Global Instructions]` exactly once, on its own line, after the final Shot. Do not assume that headings ending in `requirement:` are global.
- Keep global rules out of individual Shot descriptions unless they are also a visible or audible event at that time.
- For a full-reference Ref2VA prompt, place the timed Shots and `[Global Instructions]` inside `detailed_description`; keep `subject_definitions`, `summary`, and `retention_analysis` before it.
- Do not describe the node's latent overlap or repeat the end of one segment at the start of the next. The node supplies continuation context and removes its guided overlap automatically.
- Preserve the user's dialogue, lyrics, and visible text verbatim. Keep reference labels stable throughout the prompt.
- Prefer explicit observable actions, camera changes, and sound events over abstract quality adjectives.
