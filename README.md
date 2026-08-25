# ComfyUI-MiniMax-H3-Long-Video

[English](README.md) | [日本語](README_ja.md)

Experimental long-form MiniMax H3 reference-video generation for current ComfyUI.

## Node

`MiniMax H3 Long Reference Sampler` combines the reference inputs from ComfyUI's built-in `MiniMax H3 Reference to Video` node with custom sampler inputs. It splits a long 24 fps timeline into model-sized AV latent segments, uses 22 or 39 frames of sampled video and audio latent as a frame-zero guide for the next segment, saves every segment to SSD, and decodes the selected checkpoints to one MP4 without retaining the full decoded movie in RAM. Guided frames are generated at the head of each continuation segment and removed from the master video.

Prompts may use either an `integrated_multimodal_description:` block or a plain `[Shot 1] ... [Shot N] At MM:SS.mmm, ...` timeline. Each shot is assigned once, to the segment containing its global start timestamp; a shot already started in the previous segment is not copied into the next one. Timestamps are rebased to the segment-local timeline while text before the first shot and recognized camera-editing requirements after the final shot remain global instructions. A segment with no new shot receives only a neutral instruction to continue from its preceding AV context without replaying earlier action.

> **Shot markers are required for intentional long-form progression.** A fully timestamped Shot timeline gives the most precise control. If every Shot omits its timestamp, the node distributes the Shots evenly across the segment count calculated from `length` and `max_raw_frames`, then assigns local timestamps automatically. Timestamped and untimed Shots may be mixed: explicit timestamps remain fixed, while untimed Shots are spaced evenly between the surrounding timestamped Shots or between the final timestamp and the end of the video. Without Shot markers, the node cannot divide actions by meaning and repeats the full prompt for every segment. This can cause each segment to restart or repeat the same action.

Every segment uses the same `noise_seed`. Its local timeline prompt and preceding AV latent context provide the changes between segments.

`max_raw_frames` controls the VRAM-sensitive total generated length of each segment, including its removable guide, on MiniMax H3's `17k+5` frame grid. At 24 fps, useful values are 73 (~3.0 seconds), 90 (3.75 seconds), 107 (~4.5 seconds), and 124 (~5.2 seconds). The default is 124.

The exact prompt sent to each sampling pass is saved beside the latent checkpoints as `prompts/segment_NNNN.txt`. The manifest records the delivered timeline and prompt window for each segment.

An optional `initial_latent` is context only. Its tail guides the removable head of this node's first generated segment, but its frames are not included in the output and this node's prompt timeline begins again at 0 seconds.

## Checkpoints and rerolls

`cache_name` is always treated as an output-relative bundle directory and supports the same substitutions as ComfyUI's Save Video node. A trailing `/` is optional. For example, `h3_long_video/%seed.seed%/` writes everything together:

- `output/h3_long_video/<seed>/master.mp4`
- `output/h3_long_video/<seed>/latents/segment_XXXX.safetensors`
- `output/h3_long_video/<seed>/prompts/segment_XXXX.txt`
- `output/h3_long_video/<seed>/manifest.json`

Patterns such as `%date:yyyy-MM-dd%` and `%Node name.widget_name%` are expanded by the node. `Node name` must uniquely match the referenced node's title or type, so giving a seed node the title `seed` makes `h3_long_video/%seed.seed%/` resolve from its `seed` input. With `resume` disabled, an existing non-empty bundle is not overwritten: `_2`, `_3`, and so on are appended to its folder name. With `resume` enabled, the exact expanded folder is opened so its checkpoints can be reused.

Enable `resume` to reuse compatible checkpoints. Keep `reroll_from_segment` at `-1` to continue from the first missing or incompatible segment, or set it to `N` to keep segments before `N` and regenerate segment `N` and everything after it.

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

The final bundle path starts at `upscale/` under the source bundle. For example, `h3_long_video/123/master.mp4` produces `h3_long_video/123/upscale/master.mp4`. If that folder already exists, a new `upscale_2/`, `upscale_3/`, and so on is created instead of overwriting it. Segment Save uses an internal ComfyUI temporary folder while the loop is running; after Assemble moves the processed checkpoints and prompts into the selected output bundle, the temporary job folder is removed.

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
