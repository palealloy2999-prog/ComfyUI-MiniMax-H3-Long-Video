# Long-video extension to the official MiniMax H3 prompt format

This document defines only the conventions added by
ComfyUI-MiniMax-H3-Long-Video. MiniMax's official `h3-prompt-writing` Skill
remains authoritative for H3 field order, reference labels, camera language,
dialogue, lyrics, visible text, and audio categories.

## Master formats

### Base T2VA/I2VA/FL2VA/L2VA modes

```text
<official keyframe-alignment instruction when required>

integrated_multimodal_description:
<timeless master-wide visual premise; do not list any Shot's props, transformations, or results>

[Shot 1] Establish the opening and begin the first action.
[Shot 2] At 00:04.500, the camera cuts and the next action begins.
[Shot 3] At 00:09.250, continue into the next distinct beat.

[Global Instructions]
Preserve identity, physical continuity, style, and forward progression. Refer only to the active Shot for concrete events.

overall_soundscape: A timeless description of ambience and mix character; concrete sounds occur only in their owning Shots.

non_diegetic_music: A timeless description of the score's genre and instrumentation; phases occur only in their owning Shots.
```

### Full-reference Ref2VA

```text
subject_definitions:
<Subject 1> is the person visually defined by <Picture 1>.

summary:
A continuous high-energy dance performance starring <Subject 1>, with evolving choreography, camera work, and integrated motion graphics.

retention_analysis:
<Subject 1>: fully_preserved - preserve identity, facial appearance, body proportions, clothing, materials, and design details from <Picture 1>.

detailed_description:
<timeless visual language and environment rules only>

[Shot 1] Begin in the reference pose, then immediately depart from it as the music intro starts.
[Shot 2] At 00:03.000, the beat drops once; <Subject 1> starts the main choreography and (S1) sings: <d>[English] EXACT LYRIC</d>.
[Shot 3] At 00:28.800, <Subject 1> completes the final pose; motion and music stop, then the image fades to black.

[Global Instructions]
Preserve <Subject 1> and all reference relationships across the master timeline.

overall_soundscape: ...

non_diegetic_music: ...
```

The template's separation is intentional:

- `<Subject 1>` is a visual reference entity. Bare `S1 = ...` is invalid here.
- `(S1)` is a voice identifier and appears only where that voice speaks or sings.
- The reference pose is not a subject property to repeat; its one-time use belongs in Shot 1.
- `summary` identifies the repeat-safe premise, not the 30-second story arc.
- The intro, drop, final pose, stop, and fade are each executed only by their owning Shot.

`[Global Instructions]` is a parser delimiter. The renderer copies its contents
into every local timeline but removes the marker itself, so H3 receives only
ordinary prompt prose and official fields.

## Put timed changes in Shots

Any event tied to master time belongs in the Shot where it starts:

- a cut, action, pose, state, location, or lighting change;
- dialogue, lyrics, typography, an impact, or another synchronized sound;
- a music intro, drop, break, climax, or outro.

Do not write `at 28.8 seconds`, `from 15-30 seconds`, or similar master timing
inside `summary`, `[Global Instructions]`, `overall_soundscape`, or
`non_diegetic_music`. Those sections are reused by multiple locally timed H3
passes. Express the change in a timestamped Shot and keep the two audio fields
as timeline-independent continuity summaries.

The same rule applies even without a numeric timestamp. Phrases such as
`the video begins`, `after three seconds`, `the remaining timeline`,
`build toward the finale`, `the final movement`, and `then stop` encode a
once-only sequence and must not appear in reusable fields. Put each clause in
its owning Shot.

Concrete nouns and outcomes can leak events even when the prose contains no
sequence word. A reusable field must not recap the master with mappings or
lists such as:

```text
blue umbrella -> cobalt suit; flowers -> floral dress; red fabric -> crimson gown
```

It must not enumerate the corresponding umbrella sound, taxi pass, metal-panel
ring, crimson-cloth flutter, or the score used for each transformation. Because
the renderer copies reusable fields into every local prompt, such a list tells
segment 0 about transformations owned by later segments. Keep only an abstract
rule such as `Each physical foreground wipe reveals the transformation specified
in the active Shot.` Put every named object, result, and synchronized sound in
that Shot.

## Field ownership table

| Field | Allowed | Move into a Shot |
| --- | --- | --- |
| `subject_definitions` | `<Subject N>` to reference mapping and stable identity | opening pose, motion freedom after opening, performer actions, voice definitions |
| `summary` | short timeless genre and premise | duration, opening, ordered phases, climax, final pose, fade |
| `retention_analysis` | stable reference properties and allowed variation | changes that occur at a particular phase |
| prose before Shots | timeless environment and visual language | event inventory, props, transformations, results, scheduled camera moves |
| `[Global Instructions]` | abstract identity, physics, and continuity rules safe in every pass | event mappings, concrete examples from the timeline, anything that should happen only once |
| `overall_soundscape` | timeless ambience and mix character | lists of per-Shot sounds, impacts, silence, or scheduled sound changes |
| `non_diegetic_music` | timeless score genre, instrumentation, and mix | lists of per-Shot musical treatments, intro, drop, break, buildup, lyric entrance, stop, outro |
| Shot timeline | all single-execution events | — |

Also avoid a second absolute master timestamp inside a Shot body. Create a new
timestamped Shot for that beat, or use timing relative to the current Shot such
as `two beats later` or `near the end of the shot`. Only Shot-marker timestamps
are automatically rebased.

## Segment windows

The node operates at 24 fps.

- Total delivered duration reverses the same grid formula when `length` is
  grid-encoded; for example, both 720 and 736 represent 30 seconds.
- `max_raw_frames` is expected to be produced from segment seconds `a` with
  `n=max(5, round(a*24)); n+(5-n%17)%17`.
- Common whole-second values are reversed for the master window: 73 -> 3
  seconds, 107 -> 4 seconds, 124 -> 5 seconds, and 362 -> 15 seconds.
- The preceding guide and H3 grid padding are added only to the internal raw
  LATENT. They do not reduce the delivered master window.
- The final pass delivers only the requested remaining master frames.

Build consecutive half-open master windows `[start, end)`. Segment 0 and normal
continuation windows have the same human-scale duration.

Use coarse creative timestamps. Whole seconds and simple half-second beats are
preferred because H3 will not reliably express grid-padding precision. Never
turn a padded value such as 362/24 into a `00:15.083` Shot boundary when 362 was
produced from `a=15`; the master boundary is `00:15.000`.

## How a master becomes local H3 prompts

For each pass, the renderer:

1. preserves the official field order and reusable reference sections;
2. adds the first pass's range and H3-local duration to a Ref2VA `summary`, then replaces the full-master summary in continuation passes with a local no-restart scope, or
   to the base multimodal description;
3. removes a first/last-frame alignment instruction after the first pass;
4. describes the preceding AV guide as context outside the local timestamp clock;
5. carries the Shot active at the boundary with an explicit instruction to
   continue from its current state without replaying its beginning;
6. selects new Shots whose global starts fall inside `[start, end)`;
7. rebases timestamps by subtracting only the segment's master start time and
   renumbers them sequentially for that local prompt;
8. copies the contents below `[Global Instructions]` as common prose before the
   local Shot timeline but removes the marker;
9. preserves timeless audio fields with a continuation instruction; if a shared audio field contains an absolute timestamp, replaces its continuation body with a no-restart instruction instead of replaying it.

A Shot beginning just before a boundary therefore appears once at its start and
again only as the explicitly marked in-progress action in the next pass. This
intentional carry avoids losing a complex action after only a few generated
frames while telling H3 not to restart it.

## Boundary audit example

With `max_raw_frames=362`, `context_frames=39`, no `initial_latent`, and a
30-second (`length=720`) delivered master:

```text
segment 0: 00:00.000-00:15.000 (360 delivered; 362 raw grid frames)
segment 1: 00:15.000-00:30.000 (360 delivered; 39 guide + raw grid padding internally)
```

A master Shot at exactly `00:15.000` belongs only to segment 1 and becomes local
`[Shot 1]`. A master Shot at `00:18.000` becomes local `At 00:03.000`. Segment 0
keeps its original timestamps unchanged.

## Audit checklist

- The master duration matches `length / 24`.
- Shot numbers are sequential and timestamps strictly increase.
- Every timestamp is earlier than the delivered duration.
- Full-reference entities use `<Subject N>` etc.; `(Sx)` is used only for voices.
- `subject_definitions` and `retention_analysis` contain identity/retention facts only.
- `summary` is short and repeat-safe, with no duration, opening, sequence, or ending.
- Timed audio and visual changes occur in Shots, not shared fields.
- Non-numeric sequence words do not hide once-only events in reusable fields.
- Every concrete prop, transformation result, camera beat, sound cue, and music phase has exactly one owning Shot.
- Reusable fields contain no event mappings or lists of the master's props, outfits, colors, locations, transformations, wipe sounds, or score phases.
- Copying the reusable fields into segment 0 cannot trigger an event owned by a later segment.
- Expanded local prompts share invariant rules only; concrete event overlap is limited to an explicitly marked in-progress boundary continuation.
- `[Global Instructions]` occurs exactly once after the last Shot.
- Important actions near boundaries explicitly describe how they progress.
- The reported master windows are not shortened or shifted by continuation context.
