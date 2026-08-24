# ComfyUI-MiniMax-H3-Long-Video

[English](README.md) | [日本語](README_ja.md)

Experimental long-form MiniMax H3 reference-video generation for current ComfyUI.

## Node

`MiniMax H3 Long Reference Sampler` combines the reference inputs from ComfyUI's built-in `MiniMax H3 Reference to Video` node with custom sampler inputs. It splits a long 24 fps timeline into model-sized AV latent segments, uses 22 or 39 frames of sampled video and audio latent as a frame-zero guide for the next segment, saves every segment to SSD, and decodes the selected checkpoints to one MP4 without retaining the full decoded movie in RAM. Guided frames are generated at the head of each continuation segment and removed from the master video.

Prompts may use either an `integrated_multimodal_description:` block or a plain `[Shot 1] ... [Shot N] At MM:SS.mmm, ...` timeline. Each shot is assigned once, to the segment containing its global start timestamp; a shot already started in the previous segment is not copied into the next one. Timestamps are rebased to the segment-local timeline while text before the first shot and recognized camera-editing requirements after the final shot remain global instructions. A segment with no new shot receives only a neutral instruction to continue from its preceding AV context without replaying earlier action.

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

## Current limits

- MiniMax H3 AV latents only, batch size 1
- 24 fps output
- H.264/AAC MP4 output
- BasicGuider sampling, matching the existing H3 multishot path
- `width` and `height` must match a connected `initial_latent`
