# ComfyUI-MiniMax-H3-Long-Video

[English](README.md) | [日本語](README_ja.md)

Experimental long-form MiniMax H3 reference-video generation for current ComfyUI.

## Node

`MiniMax H3 Long Reference Sampler` combines the reference inputs from ComfyUI's built-in `MiniMax H3 Reference to Video` node with custom sampler inputs. It splits a long 24 fps timeline into model-sized AV latent segments, uses 22 or 39 frames of sampled video and audio latent as a frame-zero guide for the next segment, saves every segment to SSD, and decodes the selected checkpoints to one MP4 without retaining the full decoded movie in RAM. Guided frames are generated at the head of each continuation segment and removed from the master video.

Prompts may use the official base-mode `integrated_multimodal_description:`, the full-reference Ref2VA `detailed_description:`, or a plain `[Shot 1] ... [Shot N] At MM:SS.mmm, ...` timeline. Each Shot starts in the segment containing its global timestamp. If it is still active at a segment boundary, the next segment receives it with an explicit instruction to continue from its current state without replaying its beginning. Segment 0 keeps the master timestamps unchanged. Later prompts simply subtract the segment's master start time; the removable AV guide is preceding context outside that local timestamp clock.

Place instructions that must apply to every segment after the final Shot under a standalone `[Global Instructions]` line. This is an internal long-master delimiter: its contents remain common, but the marker itself is removed before H3 conditioning. Labels such as `Character-consistency requirement:` have no special meaning by themselves; without the marker, text after the final Shot remains part of that Shot. Use the marker exactly once and do not place it inside a Shot action.

```text
[Shot 1] Begin running.
[Shot 2] At 00:05.000, jump over the barrier.

[Global Instructions]
Preserve the same character identity, outfit, and continuous music across every segment.
```

A segment with no new Shot receives the preceding AV context plus the Shot active at its boundary, explicitly marked as a continuation rather than a restart.

> **Shot markers are required for intentional long-form progression.** A fully timestamped Shot timeline gives the most precise control. If every Shot omits its timestamp, the node distributes the Shots evenly across the segment count calculated from `length` and `max_raw_frames`, then assigns local timestamps automatically. Timestamped and untimed Shots may be mixed: explicit timestamps remain fixed, while untimed Shots are spaced evenly between the surrounding timestamped Shots or between the final timestamp and the end of the video. Without Shot markers, the node cannot divide actions by meaning and repeats the full prompt for every segment. This can cause each segment to restart or repeat the same action.

Every segment uses the same `noise_seed`. Its local timeline prompt and preceding AV latent context provide the changes between segments.

`max_raw_frames` is the H3-grid value produced from the intended segment duration `a` by `n=max(5, round(a*24)); n+(5-n%17)%17`. The node reverses common whole-second values for the master windows: 73 = 3 seconds, 107 = 4 seconds, 124 = 5 seconds, and 362 = 15 seconds. The same reversal is applied to `length`, so both 720 and its grid-encoded form 736 mean a 30-second master. With 362, either form produces exactly two windows, 0-15 and 15-30 seconds. The continuation guide and H3 padding are added only to the internal raw LATENT; they do not shorten or shift the prompt window.

The exact prompt sent to each sampling pass is saved beside the latent checkpoints as `prompts/segment_NNNN.txt`. The manifest records the delivered timeline and prompt window for each segment.

To inspect or revise those prompts before sampling, add `MiniMax H3 Long Prompt Planner` and connect its `prompt_plan` output to `MiniMax H3 Long Reference Sampler.prompt_plan`. Give both nodes the same `length`, `max_raw_frames`, and `context_frames`; also enable the planner's `has_initial_latent` only when the sampler receives one. The planner's `preview` is an ordered STRING list with one exact localized prompt per segment. Add `segment_prompt_N` fields only for segments that need a complete manual replacement. When a plan is connected, it takes precedence over the sampler's internal split; the sampler's existing `prompt` input remains present for workflow compatibility.

An installable Agent Skill extending MiniMax's official [`h3-prompt-writing`](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing) format for long masters is included at [`skills/minimax-h3-long-video-prompt-writing`](skills/minimax-h3-long-video-prompt-writing). It preserves the official field order, Ref2VA `<Subject N>` reference labels, and `(Sx)` speaker IDs while defining segment boundaries, `[Global Instructions]`, and timed-event placement. Put every timed visual, dialogue, lyric, sound, and music change in its owning Shot; do not put master timestamps in the shared `overall_soundscape` or `non_diegetic_music` fields.

For continuation segments, the renderer replaces the full-master Ref2VA `summary` with a local no-restart scope. Timeless soundscape and music constraints are retained with a continuation instruction; a shared audio field containing absolute timestamps is replaced by that instruction so its intro, drop, or finale cannot replay in every segment. An outer Markdown `text` code fence is stripped automatically before encoding. Ambiguous bare `S1 = ...` definitions are rejected before generation: visual references must use `<Subject N>`, while `(Sx)` is reserved for speakers.

```bash
npx skills add . --skill minimax-h3-long-video-prompt-writing
```

An optional `initial_latent` is context only. Its tail guides the removable head of this node's first generated segment, but its frames are not included in the output and this node's prompt timeline begins again at 0 seconds.

## Checkpoints and rerolls

`cache_name` is always treated as an output-relative bundle directory and supports the same substitutions as ComfyUI's Save Video node. A trailing `/` is optional. For example, `h3_long_video/%seed.seed%/` writes everything together:

- `output/h3_long_video/<seed>/master.mp4`
- `output/h3_long_video/<seed>/latents/segment_XXXX.safetensors`
- `output/h3_long_video/<seed>/prompts/segment_XXXX.txt`
- `output/h3_long_video/<seed>/manifest.json`

Patterns such as `%date:yyyy-MM-dd%` and `%Node name.widget_name%` are expanded by the node. `Node name` must uniquely match the referenced node's title or type, so giving a seed node the title `seed` makes `h3_long_video/%seed.seed%/` resolve from its `seed` input. With `resume` disabled, an existing non-empty bundle is not overwritten: `_2`, `_3`, and so on are appended to its folder name. With `resume` enabled, the exact expanded folder is opened so its checkpoints can be reused.

Enable `resume` to reuse compatible checkpoints. Keep `reroll_from_segment` at `-1` to continue from the first missing or incompatible segment, or set it to `N` to keep segments before `N` and regenerate segment `N` and everything after it. Compatibility includes the local prompt, seed and frame plan, upstream model/CLIP/sampler/sigma settings, reference inputs, `initial_latent`, and predecessor lineage. Checkpoints from older manifest schemas are regenerated.

Changing an earlier prompt window requires rerolling from that segment or earlier because every later segment inherits its predecessor's latent tail.

## Upscaling a saved long video

`MiniMax H3 Long Latent Upscale & Assemble` reads a completed Long H3 bundle and processes its checkpoints one at a time with [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler). Install that custom node and place a compatible model under `ComfyUI/models/latent_upscale_models/` before using this node.

`source_path` accepts an output-relative bundle folder, its `manifest.json`, or its `master.mp4`. An absolute path is also accepted when it stays inside ComfyUI's output folder. The node loads one source checkpoint from SSD, upscales only its 24-channel video stream, preserves its audio stream, and saves the result to a separate output bundle before loading the next segment.

The complete raw segment, including its continuation guide, is upscaled before any frames are removed. During MP4 assembly the same `context_frames` and final padding rules recorded in the source manifest are applied, so the upscaled master has the same delivered timeline as the source master. Upscaled segment checkpoints can be reused with `resume`, or regenerated from `reroll_from_segment` onward.

Use `target_width` and `target_height` for the requested pixel size. The default latent-grid `align` of 2 preserves dimensions that are multiples of 32 pixels; a larger alignment may round the actual output resolution upward. `last_latent` is the final upscaled raw AV segment, while `video` and `master_path` refer to the assembled result.

Example output:

```text
output/h3_long_upscaled/
├── master.mp4
├── manifest.json
└── latents/
    ├── segment_0000.safetensors
    └── segment_0001.safetensors
```

### Diffusion re-sampling with MMH3 Ultimate Upscale

For diffusion-based latent enlargement, use the four loop support nodes with EasyUse's `For Loop Start` / `For Loop End` and `MMH3 Ultimate Upscale`. A ready-to-edit graph is included at [`sample/minimax_h3_r2v-longtime_upscale.json`](sample/minimax_h3_r2v-longtime_upscale.json).

Connect `MiniMax H3 Long Reference Sampler.master_path` directly to `MiniMax H3 Long Upscale Prepare.master_path`. A bundle folder or its `manifest.json` is also accepted when entered manually.

The final bundle path starts at `upscale/` under the source bundle. For example, `h3_long_video/123/master.mp4` produces `h3_long_video/123/upscale/master.mp4`. If that folder already exists, a new `upscale_2/`, `upscale_3/`, and so on is created instead of overwriting it. Prepare allocates this persistent bundle before the loop, and Segment Save writes processed checkpoints and prompts directly into it. If processing stops, completed segments and a `processing` manifest remain under the output folder. Assemble validates the saved files and creates the MP4 without moving them through a temporary job folder.

To continue an interrupted upscale, enable Prepare's advanced `resume` input and keep the same Ultimate Upscale workflow settings. Prepare reopens the newest incomplete bundle for that source, validates the source checkpoints and saved outputs, and sends only missing segments through the loop. If every segment was already saved but MP4 decoding failed, the final segment is processed once more so Assemble receives a fresh progress value and can retry the decode.

The loop wiring is `Prepare -> For Loop Start -> Segment Load -> MiniMax H3 Reference to Video -> MMH3 Ultimate Upscale -> Segment Save -> For Loop End -> Assemble`. `segment_count` drives the loop and EasyUse's `index` drives `segment_index`. The loader emits exactly one source latent plus its segment-local prompt, seed, source width and height, and raw frame count. The save node writes the processed latent to SSD immediately; only a small progress value is carried through Loop End. Assemble starts after the final iteration, decodes the saved checkpoints one at a time, removes the recorded continuation overlap, and writes one MP4.

Connect the same reference images, videos, and audio used for the original generation to `MiniMax H3 Reference to Video`. Its `prompt` and `length` inputs come from Segment Load, and its width and height must match the target size configured for MMH3 Ultimate Upscale. Its empty-latent output is intentionally unused; Ultimate Upscale receives the loaded source latent.

This workflow requires separately installed [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) and [Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale), plus their required model weights. Replace the sample model names and reference image before running it.

## Current limits

- MiniMax H3 AV latents only, batch size 1
- 24 fps output
- H.264/AAC MP4 output
- `width` and `height` must match a connected `initial_latent`
- Long latent upscaling requires the separately installed H3 latent upscaler custom node and model weights. **Experimental and untested.**

## License

[GNU General Public License v3.0](LICENSE)

This project contains portions adapted and modified in 2026 from ComfyUI's built-in MiniMax H3 implementation, which is licensed under GPL-3.0.
