import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path

import av
import torch

import comfy.nested_tensor
import comfy.utils
import folder_paths
import node_helpers
from comfy_api.latest import ComfyExtension, InputImpl, io, ui
from comfy_extras import nodes_custom_sampler as custom_sampler
from comfy_extras import nodes_minimax_h3 as h3
from comfy_extras.nodes_audio import vae_decode_audio
from typing_extensions import override

from .timeline import FPS, plan_segments, slice_prompt


SCHEMA_VERSION = 2
CROSSFADE_FRAMES = 8


def _streams(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None or not getattr(samples, "is_nested", False):
        raise ValueError("initial_latent must be a sampled MiniMax H3 AV latent")
    parts = samples.unbind()
    if len(parts) != 2:
        raise ValueError("initial_latent must contain H3 video and audio streams")
    video, audio = parts
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError("initial_latent has an invalid H3 video stream")
    if audio.ndim != 4 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError("initial_latent has an invalid H3 audio stream")
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("MiniMax H3 Long Video currently supports batch size 1")
    return video, audio


def _project_directory(cache_name):
    if not cache_name or cache_name in (".", ".."):
        raise ValueError("cache_name must not be empty")
    if any(ord(char) < 32 or char in '<>:"/\\|?*' for char in cache_name):
        raise ValueError("cache_name contains characters that are not valid in a folder name")
    output_root = Path(folder_paths.get_output_directory()).resolve()
    project = (output_root / "h3_long" / cache_name).resolve()
    if os.path.commonpath((str(output_root), str(project))) != str(output_root):
        raise ValueError("cache_name must stay inside the ComfyUI output folder")
    project.mkdir(parents=True, exist_ok=True)
    (project / "latents").mkdir(exist_ok=True)
    return project


def _atomic_json(path, data):
    temporary = path.with_name("{}.tmp-{}".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _atomic_text(path, value):
    temporary = path.with_name("{}.tmp-{}".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
    os.replace(temporary, path)


def _cpu_latent(latent):
    video, audio = _streams(latent)
    return {
        "samples": comfy.nested_tensor.NestedTensor((
            video.detach().to(device="cpu", copy=True).contiguous(),
            audio.detach().to(device="cpu", copy=True).contiguous(),
        ))
    }


def _save_segment(path, latent, metadata):
    video, audio = _streams(latent)
    state = {
        "video": video.detach().to(device="cpu", copy=True).contiguous(),
        "audio": audio.detach().to(device="cpu", copy=True).contiguous(),
    }
    temporary = path.with_name("{}.tmp-{}.safetensors".format(path.stem, os.getpid()))
    comfy.utils.save_torch_file(state, str(temporary), metadata={key: str(value) for key, value in metadata.items()})
    os.replace(temporary, path)


def _load_segment(path):
    state, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    if "video" not in state or "audio" not in state:
        raise ValueError("{} is not an H3 AV checkpoint".format(path.name))
    latent = {"samples": comfy.nested_tensor.NestedTensor((state["video"], state["audio"]))}
    _streams(latent)
    return latent, metadata or {}


def _add_context(latent, previous, context_frames):
    target_video, target_audio = _streams(latent)
    previous_video, previous_audio = _streams(previous)
    if previous_video.shape[3:] != target_video.shape[3:]:
        raise ValueError("initial_latent resolution does not match width and height")

    video_steps = h3.video_latent_t(context_frames)
    if previous_video.shape[2] < video_steps or target_video.shape[2] < video_steps:
        raise ValueError("initial_latent is shorter than the selected context")
    previous_start = previous_video.shape[2] - video_steps
    if previous_start % 5:
        raise ValueError("initial_latent does not end on an H3 temporal cycle boundary")

    audio_steps = round(context_frames / FPS * h3.AUDIO_LATENT_FPS)
    if previous_audio.shape[-1] < audio_steps or target_audio.shape[-1] < audio_steps:
        raise ValueError("initial_latent audio is shorter than the selected context")

    target_video[:, :, :video_steps].copy_(previous_video[:, :, -video_steps:].to(target_video))
    target_audio[..., :audio_steps].copy_(previous_audio[..., -audio_steps:].to(target_audio))
    video_mask = torch.ones_like(target_video)
    audio_mask = torch.ones_like(target_audio)
    video_mask[:, :, :video_steps] = 0.0
    audio_mask[..., :audio_steps] = 0.0
    latent["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return latent


def _prepare_references(vae, audio_vae, width, height, frame_count, ref_image_size,
                        ref_images, ref_videos, ref_video_audios, ref_audios):
    ref_items = []
    ref_blocks = []

    for image in (ref_images or {}).values():
        if image is None:
            continue
        image_height, image_width = image.shape[1], image.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (image_width * image_height)))
        else:
            scale = min(1.0, h3.REF_IMAGE_SHORT_EDGE / min(image_width, image_height))
        target_width = max(h3.CANVAS_MULTIPLE, round(image_width * scale / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        target_height = max(h3.CANVAS_MULTIPLE, round(image_height * scale / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        resized = h3._resize(image[:1], target_width, target_height, "disabled")
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({
            "kind": "image",
            "latent_h": target_height // 16,
            "latent_w": target_width // 16,
            "latent": vae.encode(resized),
        })

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        video_height, video_width = video_frames.shape[1], video_frames.shape[2]
        canvas_width, canvas_height = h3.adapt_canvas(video_width, video_height)
        if video_width * video_height < canvas_width * canvas_height:
            canvas_width = max(h3.CANVAS_MULTIPLE, round(video_width / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
            canvas_height = max(h3.CANVAS_MULTIPLE, round(video_height / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        frames = h3._resize(video_frames, canvas_width, canvas_height, "disabled")[:frame_count]
        count = frames.shape[0]
        if count < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames")
        while count % 17 != 5:
            count -= 1
        frames = frames[:count]
        video_latent = vae.encode(frames)
        audio_latent = None
        audio_length = 0
        if soundtrack is not None:
            audio_latent, audio_length = h3._encode_ref_audio(audio_vae, soundtrack)
            ref_items.append({"type": "audio"})
        sample_indices = list(range(0, frames.shape[0], FPS // 2))
        ref_items.append({
            "type": "video",
            "data": frames[sample_indices],
            "timestamps": [index / 2.0 for index in range(len(sample_indices))],
        })
        ref_blocks.append({
            "kind": "video_audio" if audio_length else "video",
            "latent_t": video_latent.shape[2],
            "latent_h": canvas_height // 16,
            "latent_w": canvas_width // 16,
            "ref_audio_t": audio_length,
            "latent": video_latent,
            "audio_latent": audio_latent,
        })

    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, audio_length = h3._encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": audio_length, "audio_latent": audio_latent})
    return ref_items, ref_blocks


def _conditioning(clip, prompt, ref_items, ref_blocks):
    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": ref_blocks})
    return conditioning


def _add_continuation_guide(conditioning, previous, context_frames):
    video, audio = _streams(previous)
    video_steps = h3.video_latent_t(context_frames)
    audio_steps = round(context_frames / FPS * h3.AUDIO_LATENT_FPS)
    keyframes = list(conditioning[0][1].get("minimax_keyframes", []))
    keyframes.append({
        "resolved_frame_index": 0,
        "latent": video[:, :, -video_steps:].detach().clone(),
        "audio_latent": audio[..., -audio_steps:].detach().clone(),
    })
    return node_helpers.conditioning_set_values(conditioning, {"minimax_keyframes": keyframes})


def _prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _metadata_matches(metadata, segment, prompt_hash, width, height):
    expected = {
        "schema": str(SCHEMA_VERSION),
        "raw_frames": str(segment.raw_frames),
        "head_frames": str(segment.head_frames),
        "output_frames": str(segment.output_frames),
        "width": str(width),
        "height": str(height),
        "prompt_sha256": prompt_hash,
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def _manifest_segment(segment, status, checkpoint, prompt_hash):
    return {
        "index": segment.index,
        "status": status,
        "file": checkpoint.name,
        "timeline_start": segment.output_start / FPS,
        "timeline_end": (segment.output_start + segment.output_frames) / FPS,
        "prompt_window_start": segment.prompt_start_seconds,
        "prompt_window_end": segment.prompt_end_seconds,
        "prompt_file": "prompts/segment_{:04d}.txt".format(segment.index),
        "raw_frames": segment.raw_frames,
        "head_frames": segment.head_frames,
        "prompt_sha256": prompt_hash,
    }


def _decode_segment(vae, audio_vae, latent, raw_frames):
    video, _ = _streams(latent)
    images = vae.decode(video)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    if images.shape[0] < raw_frames:
        raise ValueError("video VAE decoded fewer frames than the H3 segment requires")
    audio = vae_decode_audio(audio_vae, latent)
    return images[:raw_frames], audio


def _write_master(path, segment_paths, segments, vae, audio_vae, width, height, crf):
    temporary = path.with_name("{}.tmp-{}.mp4".format(path.stem, os.getpid()))
    sample_rate = int(getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 32000)))
    try:
        with av.open(str(temporary), mode="w", options={"movflags": "faststart"}) as container:
            video_stream = container.add_stream("h264", rate=Fraction(FPS, 1))
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {"crf": str(crf)}
            audio_stream = container.add_stream("aac", rate=sample_rate, layout="stereo")
            video_pts = 0
            audio_pts = 0
            pending_images = None
            pending_audio = None

            def write_images(images):
                nonlocal video_pts
                for image in images:
                    array = (image[..., :3] * 255).clamp(0, 255).to(device="cpu", dtype=torch.uint8).numpy()
                    frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                    frame.pts = video_pts
                    frame.time_base = Fraction(1, FPS)
                    video_pts += 1
                    for packet in video_stream.encode(frame):
                        container.mux(packet)

            def write_audio(waveform):
                nonlocal audio_pts
                if waveform.shape[-1] == 0:
                    return
                frame = av.AudioFrame.from_ndarray(
                    waveform.float().contiguous().numpy(), format="fltp", layout="stereo")
                frame.sample_rate = sample_rate
                frame.pts = audio_pts
                frame.time_base = Fraction(1, sample_rate)
                audio_pts += waveform.shape[-1]
                for packet in audio_stream.encode(frame):
                    container.mux(packet)

            for segment_index, (checkpoint, segment) in enumerate(zip(segment_paths, segments)):
                latent, _ = _load_segment(checkpoint)
                images, audio = _decode_segment(vae, audio_vae, latent, segment.raw_frames)
                output_images = images[segment.head_frames:segment.head_frames + segment.output_frames]

                waveform = audio["waveform"]
                source_rate = int(audio["sample_rate"])
                if source_rate != sample_rate:
                    raise ValueError("audio VAE changed sample rate between H3 segments")
                raw_samples = round(segment.raw_frames / FPS * sample_rate)
                head_samples = round(segment.head_frames / FPS * sample_rate)
                output_start = round(segment.output_start / FPS * sample_rate)
                output_end = round((segment.output_start + segment.output_frames) / FPS * sample_rate)
                output_samples = output_end - output_start
                waveform = waveform[0]
                if waveform.shape[-1] < raw_samples:
                    waveform = torch.nn.functional.pad(waveform, (0, raw_samples - waveform.shape[-1]))
                if waveform.shape[0] == 1:
                    waveform = waveform.repeat(2, 1)
                elif waveform.shape[0] > 2:
                    waveform = waveform[:2]
                waveform = waveform.to(device="cpu")
                output_audio = waveform[:, head_samples:head_samples + output_samples]
                if output_audio.shape[-1] < output_samples:
                    output_audio = torch.nn.functional.pad(output_audio, (0, output_samples - output_audio.shape[-1]))

                if pending_images is not None:
                    blend_frames = pending_images.shape[0]
                    context_images = images[
                        segment.head_frames - blend_frames:segment.head_frames
                    ].to(device="cpu")
                    weights = torch.linspace(0.0, 1.0, blend_frames, dtype=context_images.dtype).reshape(-1, 1, 1, 1)
                    write_images(pending_images * (1.0 - weights) + context_images * weights)

                    blend_samples = pending_audio.shape[-1]
                    context_audio = waveform[:, head_samples - blend_samples:head_samples]
                    audio_weights = torch.linspace(
                        0.0, 1.0, blend_samples, dtype=context_audio.dtype).reshape(1, -1)
                    write_audio(pending_audio * (1.0 - audio_weights) + context_audio * audio_weights)

                if segment_index + 1 < len(segments):
                    next_blend_frames = min(
                        CROSSFADE_FRAMES, segments[segment_index + 1].head_frames,
                        output_images.shape[0])
                    next_blend_samples = min(
                        round(next_blend_frames / FPS * sample_rate), output_audio.shape[-1])
                    write_images(output_images[:-next_blend_frames])
                    write_audio(output_audio[:, :-next_blend_samples])
                    pending_images = output_images[-next_blend_frames:].detach().to(device="cpu", copy=True)
                    pending_audio = output_audio[:, -next_blend_samples:].detach().clone()
                else:
                    write_images(output_images)
                    write_audio(output_audio)

            for packet in video_stream.encode(None):
                container.mux(packet)
            for packet in audio_stream.encode(None):
                container.mux(packet)
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


class MiniMaxH3LongReferenceSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongReferenceSampler",
            display_name="MiniMax H3 Long Reference Sampler",
            category="sampling/minimax",
            description="Generate a long H3 reference video as sequential AV latent segments. Segment checkpoints are saved under output/h3_long.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=4096, step=32),
                io.Int.Input("height", default=768, min=32, max=4096, step=32),
                io.Int.Input("length", default=720, min=24, max=86400, step=1,
                             tooltip="Delivered frames at 24 fps. 720 frames = 30 seconds."),
                io.Combo.Input("context_frames", options=["22", "39"], default="22",
                               tooltip="Previous sampled AV latent carried into each new segment. The copied head is removed from the delivered video."),
                io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.String.Input("cache_name", default="h3_long_video",
                                tooltip="Folder name below output/h3_long. Segment latents and master.mp4 are stored here."),
                io.Boolean.Input("resume", default=False,
                                 tooltip="Reuse compatible segment checkpoints. Missing or incompatible later segments are regenerated."),
                io.Int.Input("reroll_from_segment", default=-1, min=-1, max=999, step=1,
                             tooltip="With resume enabled: -1 resumes the first missing segment; N keeps segments before N and regenerates N onward."),
                io.Int.Input("crf", default=18, min=0, max=51, step=1, advanced=True),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Latent.Input("initial_latent", optional=True,
                                tooltip="Optional sampled H3 AV latent. Only its tail is used as preroll; this node's timeline and delivered video still start at 0."),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps"),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"), prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                io.Latent.Output(display_name="last_latent"),
                io.String.Output(display_name="master_path"),
                io.Int.Output(display_name="segment_count"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, prompt, width, height, length,
                context_frames, noise_seed, sampler, sigmas, cache_name, resume,
                reroll_from_segment, crf, ref_image_size="match", initial_latent=None,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
        if width % 32 or height % 32:
            raise ValueError("width and height must be multiples of 32")
        context_frames = int(context_frames)
        if initial_latent is not None:
            initial_video, _ = _streams(initial_latent)
            if initial_video.shape[3] * 16 != height or initial_video.shape[4] * 16 != width:
                raise ValueError("initial_latent resolution does not match width and height")

        segments = plan_segments(length, context_frames, initial_latent is not None)
        project = _project_directory(cache_name)
        segment_paths = [project / "latents" / "segment_{:04d}.safetensors".format(segment.index) for segment in segments]
        local_prompts = [
            slice_prompt(
                prompt,
                segment.prompt_start_seconds,
                segment.prompt_end_seconds,
                segment.head_frames / FPS,
            )
            for segment in segments
        ]
        prompt_directory = project / "prompts"
        prompt_directory.mkdir(exist_ok=True)
        for segment, local_prompt in zip(segments, local_prompts):
            _atomic_text(
                prompt_directory / "segment_{:04d}.txt".format(segment.index),
                local_prompt,
            )
        manifest = {
            "schema": SCHEMA_VERSION,
            "status": "sampling",
            "width": width,
            "height": height,
            "length": length,
            "context_frames": context_frames,
            "segments": [],
        }
        _atomic_json(project / "manifest.json", manifest)

        previous = initial_latent
        generated = False
        completed = 0
        ref_items = None
        ref_blocks = None
        for segment, checkpoint, local_prompt in zip(segments, segment_paths, local_prompts):
            prompt_hash = _prompt_hash(local_prompt)
            may_reuse = resume and not generated and (
                reroll_from_segment < 0 or segment.index < reroll_from_segment)
            if may_reuse and checkpoint.exists():
                cached, metadata = _load_segment(checkpoint)
                if _metadata_matches(metadata, segment, prompt_hash, width, height):
                    previous = cached
                    completed += 1
                    manifest["segments"].append(_manifest_segment(
                        segment, "reused", checkpoint, prompt_hash))
                    _atomic_json(project / "manifest.json", manifest)
                    continue
                if reroll_from_segment >= 0:
                    raise ValueError(
                        "segment {} no longer matches this timeline; reroll from this segment or earlier".format(segment.index))

            generated = True
            if ref_items is None:
                ref_items, ref_blocks = _prepare_references(
                    vae, audio_vae, width, height,
                    max(item.raw_frames for item in segments), ref_image_size,
                    ref_images, ref_videos, ref_video_audios, ref_audios)
            latent, _ = h3._empty_av_latent(width, height, segment.raw_frames)
            if segment.head_frames:
                if previous is None:
                    raise ValueError("a previous H3 AV latent is required for this continuation segment")
                latent = _add_context(latent, previous, segment.head_frames)
            conditioning = _conditioning(clip, local_prompt, ref_items, ref_blocks)
            if segment.head_frames:
                conditioning = _add_continuation_guide(
                    conditioning, previous, segment.head_frames)
            guider = custom_sampler.BasicGuider.execute(model, conditioning)[0]
            noise = custom_sampler.RandomNoise.execute((noise_seed + segment.index) & 0xffffffffffffffff)[0]
            sampled = custom_sampler.SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]
            previous = {"samples": sampled["samples"]}
            metadata = {
                "schema": SCHEMA_VERSION,
                "index": segment.index,
                "raw_frames": segment.raw_frames,
                "head_frames": segment.head_frames,
                "output_start": segment.output_start,
                "output_frames": segment.output_frames,
                "width": width,
                "height": height,
                "seed": (noise_seed + segment.index) & 0xffffffffffffffff,
                "prompt_sha256": prompt_hash,
            }
            _save_segment(checkpoint, previous, metadata)
            completed += 1
            manifest["segments"].append(_manifest_segment(
                segment, "generated", checkpoint, prompt_hash))
            _atomic_json(project / "manifest.json", manifest)

        if previous is None:
            raise RuntimeError("MiniMax H3 Long Video did not produce a latent")
        last_latent = _cpu_latent(previous)
        master_path = project / "master.mp4"
        manifest["status"] = "decoding"
        _atomic_json(project / "manifest.json", manifest)
        _write_master(master_path, segment_paths, segments, vae, audio_vae, width, height, crf)
        manifest["status"] = "complete"
        manifest["master"] = master_path.name
        _atomic_json(project / "manifest.json", manifest)

        relative_folder = "h3_long/{}".format(cache_name)
        video = InputImpl.VideoFromFile(str(master_path))
        preview = ui.PreviewVideo([ui.SavedResult(master_path.name, relative_folder, io.FolderType.output)])
        return io.NodeOutput(video, last_latent, str(master_path), completed, ui=preview)


class MiniMaxH3LongVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [MiniMaxH3LongReferenceSampler]


async def comfy_entrypoint():
    return MiniMaxH3LongVideoExtension()
