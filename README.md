# ComfyUI-MiniMax-H3-Long-Video

Experimental long-form MiniMax H3 reference-video generation for current ComfyUI.

## Node

`MiniMax H3 Long Reference Sampler` combines the reference inputs from ComfyUI's built-in `MiniMax H3 Reference to Video` node with custom sampler inputs. It splits a long 24 fps timeline into model-sized AV latent segments, carries 22 or 39 frames of sampled video and audio latent context across each boundary, saves every segment to SSD, and decodes the selected checkpoints to one MP4 without retaining the full decoded movie in RAM.

Prompts may use either an `integrated_multimodal_description:` block or a plain `[Shot 1] ... [Shot N] At MM:SS.mmm, ...` timeline. Each segment receives only the shots overlapping its global time window; timestamps are rebased to the segment-local timeline while text before the first shot and recognized camera-editing requirements after the final shot remain global instructions.

The exact prompt sent to each sampling pass is saved as `output/h3_long/<cache_name>/prompts/segment_NNNN.txt`. The manifest records both the delivered timeline range and the wider prompt window that includes continuation context.

An optional `initial_latent` is context only. Its tail becomes preroll for this node's first segment, but its frames are not included in the output and this node's prompt timeline begins again at 0 seconds.

## Checkpoints and rerolls

Files are written below `ComfyUI/output/h3_long/<cache_name>/`:

- `latents/segment_XXXX.safetensors`: sampled H3 video+audio latent
- `manifest.json`: current segment plan and completion status
- `master.mp4`: current joined result

Enable `resume` to reuse compatible checkpoints. Keep `reroll_from_segment` at `-1` to continue from the first missing or incompatible segment, or set it to `N` to keep segments before `N` and regenerate segment `N` and everything after it.

Changing an earlier prompt window requires rerolling from that segment or earlier because every later segment inherits its predecessor's latent tail.

## Current limits

- MiniMax H3 AV latents only, batch size 1
- 24 fps output
- H.264/AAC MP4 output
- BasicGuider sampling, matching the existing H3 multishot path
- `width` and `height` must match a connected `initial_latent`
