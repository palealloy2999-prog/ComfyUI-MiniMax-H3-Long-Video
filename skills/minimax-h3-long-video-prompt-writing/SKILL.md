---
name: minimax-h3-long-video-prompt-writing
description: Extend the official MiniMax H3 prompt format into a timestamped long-video master that ComfyUI-MiniMax-H3-Long-Video can render as self-consistent short H3 segment prompts. Use for writing, revising, or auditing prompts for this custom node; do not use it instead of the official H3 format for ordinary single-clip generation.
---

# MiniMax H3 Long Video Prompt Writing

Build a segment-safe long-video master on top of MiniMax's official
[`h3-prompt-writing`](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing)
format. The custom node turns that master into one short, locally timed H3 prompt per sampling pass.

## Workflow

1. Apply the official H3 prompt-writing rules for the selected T2VA, I2VA, FL2VA, L2VA, or full-reference Ref2VA mode.
2. Read [references/prompt-format.md](references/prompt-format.md) and add only the long-video master conventions defined there.
3. Use a fully timestamped master Shot timeline. `[Shot 1]` starts at zero; every later Shot uses a strictly increasing global `At MM:SS.mmm` cut time.
4. Before writing reusable fields, make an event-ownership ledger: assign every concrete prop, wipe, transformation, wardrobe or environment result, camera change, sound cue, and music phase to exactly one Shot.
5. Put reusable cross-segment rules after the final Shot beneath one standalone `[Global Instructions]` marker. This is an internal delimiter removed before H3 conditioning, not an official H3 output field.
6. Keep every concrete visual, dialogue, lyric, sound, and music event in its owning Shot. The absence of a timestamp does not make an event reusable.
7. When node settings are known, calculate the exact segment windows and audit every Shot near a boundary. Report the windows and any Shot that will be continued across one.
8. Before returning the prompt, perform the event-leakage audit below. Do not return a prompt with a once-only event in any reusable field.

## Official-format invariants

- Base modes retain `integrated_multimodal_description`, `overall_soundscape`, and `non_diegetic_music` in official order.
- Full-reference Ref2VA retains `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, and `non_diegetic_music` in official order.
- Use `<Subject N>`, `<Picture N>`, `<Video N>`, and `<Audio N>` for full-reference entities. Reserve `(S1)`, `(S2)`, and later IDs for actual vocal sources.
- Preserve dialogue, lyrics, and visible text exactly as required by the official Skill.
- Do not put a keyframe alignment instruction at the start of an independently written continuation segment. The node retains it for the first pass and supplies AV guide context to later passes.

## Required semantic separation

Write the master as three layers. Never mix their responsibilities.

1. **Reference identity:** `subject_definitions` maps `<Subject N>` to reference inputs. `retention_analysis` states only which reference properties remain fixed. Neither field may describe an opening pose, future action, choreography phase, camera event, or ending.
2. **Repeat-safe premise:** Ref2VA `summary`, prose before the first Shot, `[Global Instructions]`, `overall_soundscape`, and `non_diegetic_music` contain only facts that remain true if read in any segment. Do not put the total duration, “begins with,” “then,” “after,” “builds toward,” “finally,” or any once-only beat in these fields. Do not summarize or enumerate the Shot sequence there.
3. **Single-execution timeline:** every ordered or timed event appears once in `detailed_description` or `integrated_multimodal_description` as a Shot. This includes the opening reference pose, reveals, drops, lyrics, typography, transitions, climax, final pose, music stop, and fade.

Reusable prose must be closed-world: it may state invariant identity, continuity,
style, physics, camera-axis, and no-restart rules, but it must not name the
master's concrete event inventory. In particular, never place any of these in a
reusable field:

- a mapping or recap such as `blue umbrella -> blue suit; red cloth -> crimson gown`;
- a list of future props, outfits, colors, locations, transformations, or camera beats;
- a list of per-Shot wipe sounds or score phases, even when no timestamps appear;
- a future-state instruction such as `until the finale` or `from the close-up onward`.

Use abstract invariant wording instead: for example, `Each foreground occluder
must fully hide the subject before the already-specified Shot transformation is
revealed.` The owning Shot alone names the occluder and transformation result.

## Mandatory event-leakage audit

After drafting, inspect each concrete event from the ownership ledger against
all repeat-safe fields. For every event, require both conditions:

1. its actionable description occurs in exactly one owning Shot;
2. none of its distinctive prop, result, sound, or music-phase descriptions
   occurs in `subject_definitions`, `summary`, `retention_analysis`, prose before
   the first Shot, `[Global Instructions]`, `overall_soundscape`, or
   `non_diegetic_music`.

Apply the segment-zero counterfactual: if a reusable sentence were copied into
segment 0, could it prompt any object, transformation, outfit, environment,
sound, or camera behavior owned by a later segment? If yes, move that sentence
into its owning Shot or rewrite it as a genuinely event-agnostic invariant.

When segment settings are known, also compare the expanded local prompts by
event ownership. Shared invariant language may repeat; concrete events may
appear only in their owning segment, except for an explicitly marked
in-progress boundary continuation. Report `shared-field event leakage: none`
only after this check passes.

For voices, write `(S1)`, `(S2)`, and so on only inside the Shot containing the speech or singing. Do not define a vocalist as bare `S2` in `subject_definitions`.

## Long-video invariants

- Treat Shot timestamps as positions on the delivered master timeline, never as manually calculated segment-local times.
- Derive the human-scale segment and total duration from supplied H3-grid values. Values produced by `n=max(5, round(a*24)); n+(5-n%17)%17` map back to `a`; in particular, 362 means 15 seconds and total length 736 means 30 seconds.
- Keep segment 0 timestamps unchanged. For every later segment, subtract only that segment's master start time. Do not add `context_frames / 24`; the AV guide is preceding context outside the local timestamp clock.
- Prefer a coarse creative schedule using whole seconds or simple half-second beats. Do not derive Shot timestamps from H3 grid padding or write millisecond-level boundaries such as `00:15.083`; use the recovered human-scale boundary such as `00:15.000`.
- Do not put another absolute master timestamp inside a Shot body. Start a new timestamped Shot for that beat, or describe its timing relative to the current Shot.
- Describe an action once at its real start. If it crosses a boundary, make the intended continuation clear inside that Shot; the renderer carries it into the next segment with an explicit no-restart instruction.
- Keep Ref2VA `summary` short, timeless, and free of the master duration or narrative sequence. The renderer appends the first pass's local scope and replaces the summary in continuation passes with a local no-restart scope.
- Avoid Shot-number citations in prose outside the timeline when they would become ambiguous after local renumbering. Prefer stable reference labels and master timestamps.
- Never use `[Global Instructions]` for a timed event. Never place it before or inside a Shot.

Return one paste-ready master prompt without Markdown code fences, followed by a compact audit containing total duration, reference roles, segment windows, boundary-spanning Shots, the event-ownership ledger, and `shared-field event leakage: none`.
