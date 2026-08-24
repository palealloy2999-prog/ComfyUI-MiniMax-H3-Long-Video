# SPDX-License-Identifier: GPL-3.0-only
# Portions of the MiniMax H3 reference conditioning were adapted and modified
# from ComfyUI's built-in MiniMax H3 implementation in 2026.

import hashlib
import json
import math
import os
import re
import time
from fractions import Fraction
from pathlib import Path

import av
import torch

import comfy.nested_tensor
import comfy.utils
import folder_paths
import node_helpers
import nodes as comfy_nodes
from comfy_api.latest import ComfyExtension, InputImpl, io, ui
from comfy_extras import nodes_custom_sampler as custom_sampler
from comfy_extras import nodes_minimax_h3 as h3
from comfy_extras.nodes_audio import vae_decode_audio
from typing_extensions import override

from .timeline import FPS, Segment, plan_segments, slice_prompt


SCHEMA_VERSION = 9
UPSCALE_SCHEMA_VERSION = 1


_PATTERN = re.compile(r"%([^%]+)%")
_DATE_FIELD = re.compile(r"dd?|MM?|hh?|HH?|mm?|ss?|yyy?y?")


def _expand_date_pattern(value):
    now = time.localtime()
    fields = {
        "d": now.tm_mday,
        "M": now.tm_mon,
        "h": now.tm_hour,
        "H": now.tm_hour,
        "m": now.tm_min,
        "s": now.tm_sec,
    }

    def replace_field(match):
        token = match.group(0)
        if token == "yy":
            return str(now.tm_year)[-2:]
        if token == "yyyy":
            return str(now.tm_year).zfill(4)
        if token[0] in fields:
            return str(fields[token[0]]).zfill(len(token))
        return token

    return _DATE_FIELD.sub(replace_field, value)


def _workflow_nodes(extra_pnginfo):
    if not isinstance(extra_pnginfo, dict):
        return []
    workflow = extra_pnginfo.get("workflow")
    if not isinstance(workflow, dict):
        return []
    nodes = workflow.get("nodes")
    return nodes if isinstance(nodes, list) else []


def _node_names(node):
    names = [str(node.get("id", "")), node.get("title"), node.get("type")]
    properties = node.get("properties")
    if isinstance(properties, dict):
        names.append(properties.get("Node name for S&R"))
    return [name for name in names if isinstance(name, str) and name]


def _node_input_value(node_name, input_name, prompt, extra_pnginfo):
    nodes = _workflow_nodes(extra_pnginfo)
    matches = [node for node in nodes if node_name in _node_names(node)]
    if not matches:
        lowered = node_name.casefold()
        matches = [node for node in nodes if lowered in [name.casefold() for name in _node_names(node)]]
    if not matches:
        raise ValueError("cache_name pattern refers to unknown node {!r}".format(node_name))
    if len(matches) > 1:
        raise ValueError("cache_name pattern node {!r} is ambiguous; give the node a unique title".format(node_name))

    node_id = str(matches[0].get("id"))
    prompt_node = prompt.get(node_id) if isinstance(prompt, dict) else None
    inputs = prompt_node.get("inputs") if isinstance(prompt_node, dict) else None
    if isinstance(inputs, dict) and input_name in inputs:
        value = inputs[input_name]
    elif input_name == "value":
        widget_values = matches[0].get("widgets_values")
        value = widget_values[0] if isinstance(widget_values, list) and widget_values else None
    else:
        value = None
    if value is None:
        raise ValueError(
            "cache_name pattern %{0}.{1}% could not read that node input".format(
                node_name, input_name))
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(
            "cache_name pattern %{0}.{1}% does not resolve to a text or number value".format(
                node_name, input_name))
    return str(value).lower() if isinstance(value, bool) else str(value)


def _expand_cache_name(cache_name, prompt, extra_pnginfo):
    def replace(match):
        pattern = match.group(1)
        if pattern.startswith("date:"):
            return _expand_date_pattern(pattern[5:])
        if "." in pattern:
            node_name, input_name = pattern.rsplit(".", 1)
            return _node_input_value(node_name, input_name, prompt, extra_pnginfo)
        return match.group(0)

    return _PATTERN.sub(replace, cache_name)


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


def _output_paths(cache_name, resume, width, height):
    output_root = Path(folder_paths.get_output_directory()).resolve()
    resolved_name = cache_name.rstrip("/\\")
    if not resolved_name or resolved_name in (".", ".."):
        raise ValueError("cache_name must be an output-relative folder")
    unresolved_project = (output_root / resolved_name).resolve()
    unresolved_inside_output = os.path.commonpath(
        (str(output_root), str(unresolved_project))) == str(output_root)
    existed_before_resolution = (
        "%" not in resolved_name and unresolved_inside_output and unresolved_project.exists())
    full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(
        resolved_name + "/master", str(output_root), width, height)
    if filename != "master":
        raise ValueError("cache_name must resolve to an output-relative folder")

    project = Path(full_output_folder).resolve()
    if "%" in os.path.relpath(project, output_root):
        raise ValueError("cache_name contains an unexpanded %...% pattern")
    existed = existed_before_resolution if project == unresolved_project else any(project.iterdir())
    if not resume and existed:
        base = project
        suffix = 2
        while project.exists():
            project = base.with_name("{}_{}".format(base.name, suffix))
            suffix += 1

    master_path = project / "master.mp4"
    relative_folder = os.path.relpath(project, output_root)

    if os.path.commonpath((str(output_root), str(project))) != str(output_root):
        raise ValueError("output path must stay inside the ComfyUI output folder")
    if os.path.commonpath((str(output_root), str(master_path))) != str(output_root):
        raise ValueError("output path must stay inside the ComfyUI output folder")
    project.mkdir(parents=True, exist_ok=True)
    (project / "latents").mkdir(exist_ok=True)
    return project, master_path, relative_folder


def _source_bundle(source_path):
    output_root = Path(folder_paths.get_output_directory()).resolve()
    candidate = Path(source_path.rstrip("/\\"))
    if not candidate.is_absolute():
        candidate = output_root / candidate
    candidate = candidate.resolve()
    if not folder_paths.is_within_directory(str(output_root), str(candidate)):
        raise ValueError("source_path must stay inside the ComfyUI output folder")
    if candidate.is_file():
        if candidate.name not in ("manifest.json", "master.mp4"):
            raise ValueError("source_path must point to a Long H3 bundle, manifest.json, or master.mp4")
        candidate = candidate.parent
    manifest_path = candidate / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("source_path does not contain a Long H3 manifest.json")
    return candidate, manifest_path


def _manifest_segments(source, manifest):
    if manifest.get("status") != "complete":
        raise ValueError("source Long H3 bundle is not complete")
    entries = manifest.get("segments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source Long H3 manifest has no segments")

    segments = []
    checkpoints = []
    expected_index = 0
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("index") != expected_index:
            raise ValueError("source Long H3 manifest has invalid segment ordering")
        filename = entry.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("source Long H3 manifest has an invalid segment filename")
        checkpoint = (source / "latents" / filename).resolve()
        if checkpoint.parent != (source / "latents").resolve() or not checkpoint.is_file():
            raise ValueError("source Long H3 checkpoint is missing: {}".format(filename))
        raw_frames = entry.get("raw_frames")
        context_frames = entry.get("context_frames")
        output_start = entry.get("output_start")
        output_frames = entry.get("output_frames")
        if output_start is None or output_frames is None:
            timeline_start = entry.get("timeline_start")
            timeline_end = entry.get("timeline_end")
            if not all(isinstance(value, (int, float)) for value in (
                    timeline_start, timeline_end)):
                raise ValueError("source Long H3 manifest has invalid segment timing")
            output_start = round(timeline_start * FPS)
            output_frames = round((timeline_end - timeline_start) * FPS)
        if not all(isinstance(value, (int, float)) for value in (
                output_start, output_frames, raw_frames, context_frames)):
            raise ValueError("source Long H3 manifest has invalid segment timing")
        if (raw_frames < 1 or raw_frames % 17 != 5 or
                context_frames not in (0, 22, 39) or output_frames < 1 or
                output_frames > raw_frames - context_frames or
                output_start != (segments[-1].output_start + segments[-1].output_frames if segments else 0)):
            raise ValueError("source Long H3 manifest has invalid segment lengths")
        segments.append(Segment(
            expected_index, int(raw_frames), int(context_frames),
            int(output_start), int(output_frames)))
        checkpoints.append(checkpoint)
        expected_index += 1
    return segments, checkpoints


def _upscaler_models():
    model_folder = "latent_upscale_models"
    if model_folder not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(
            model_folder, os.path.join(folder_paths.models_dir, model_folder))
    models = [
        name for name in folder_paths.get_filename_list(model_folder)
        if Path(name).suffix.lower() in (".pth", ".safetensors")
    ]
    return models or ["(no H3 latent upscaler models found)"]


def _upscaler_model_path(model_name):
    if Path(model_name).suffix.lower() not in (".pth", ".safetensors"):
        raise ValueError("select a .pth or .safetensors H3 latent upscaler model")
    path = folder_paths.get_full_path("latent_upscale_models", model_name)
    if path is None:
        raise ValueError("H3 latent upscaler model was not found: {}".format(model_name))
    return Path(path)


def _file_fingerprint(path):
    stat = path.stat()
    return str(stat.st_size), str(stat.st_mtime_ns)


def _upscale_metadata(source_checkpoint, model_path, model_name, target_width,
                      target_height, align, device, precision, segment):
    source_size, source_mtime = _file_fingerprint(source_checkpoint)
    model_size, model_mtime = _file_fingerprint(model_path)
    return {
        "upscale_schema": UPSCALE_SCHEMA_VERSION,
        "source_file": source_checkpoint.name,
        "source_size": source_size,
        "source_mtime_ns": source_mtime,
        "model_name": model_name,
        "model_size": model_size,
        "model_mtime_ns": model_mtime,
        "target_width": target_width,
        "target_height": target_height,
        "align": align,
        "device": device,
        "precision": precision,
        "index": segment.index,
        "raw_frames": segment.raw_frames,
        "context_frames": segment.context_frames,
        "output_start": segment.output_start,
        "output_frames": segment.output_frames,
    }


def _upscale_metadata_matches(metadata, expected):
    return all(metadata.get(key) == str(value) for key, value in expected.items())


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
    if video.shape[2] < video_steps or audio.shape[-1] < audio_steps:
        raise ValueError(
            "the previous H3 AV latent is shorter than context_frames")
    keyframes = list(conditioning[0][1].get("minimax_keyframes", []))
    keyframes.append({
        "resolved_frame_index": 0,
        "latent": video[:, :, -video_steps:].detach().clone(),
        "audio_latent": audio[..., -audio_steps:].detach().clone(),
    })
    return node_helpers.conditioning_set_values(
        conditioning, {"minimax_keyframes": keyframes})


def _prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _metadata_matches(metadata, segment, prompt_hash, width, height, noise_seed):
    expected = {
        "schema": str(SCHEMA_VERSION),
        "raw_frames": str(segment.raw_frames),
        "context_frames": str(segment.context_frames),
        "output_frames": str(segment.output_frames),
        "width": str(width),
        "height": str(height),
        "seed": str(noise_seed),
        "prompt_sha256": prompt_hash,
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def _manifest_segment(segment, status, checkpoint, prompt_hash):
    return {
        "index": segment.index,
        "status": status,
        "file": checkpoint.name,
        "output_start": segment.output_start,
        "output_frames": segment.output_frames,
        "timeline_start": segment.output_start / FPS,
        "timeline_end": (segment.output_start + segment.output_frames) / FPS,
        "prompt_window_start": segment.prompt_start_seconds,
        "prompt_window_end": segment.prompt_end_seconds,
        "prompt_file": "prompts/segment_{:04d}.txt".format(segment.index),
        "raw_frames": segment.raw_frames,
        "context_frames": segment.context_frames,
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

            for checkpoint, segment in zip(segment_paths, segments):
                latent, _ = _load_segment(checkpoint)
                images, audio = _decode_segment(vae, audio_vae, latent, segment.raw_frames)
                output_images = images[
                    segment.context_frames:segment.context_frames + segment.output_frames]

                waveform = audio["waveform"]
                source_rate = int(audio["sample_rate"])
                if source_rate != sample_rate:
                    raise ValueError("audio VAE changed sample rate between H3 segments")
                raw_samples = round(segment.raw_frames / FPS * sample_rate)
                context_samples = round(segment.context_frames / FPS * sample_rate)
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
                output_audio = waveform[:, context_samples:context_samples + output_samples]
                if output_audio.shape[-1] < output_samples:
                    output_audio = torch.nn.functional.pad(output_audio, (0, output_samples - output_audio.shape[-1]))

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
                io.Int.Input("max_raw_frames", default=124, min=73, max=362, step=17,
                             tooltip="Maximum generated frames per segment on H3's 17k+5 grid. 73 is ~3.0s, 90 is 3.75s, 107 is ~4.5s, and 124 is ~5.2s at 24 fps."),
                io.Combo.Input("context_frames", options=["22", "39"], default="22",
                               tooltip="Previous sampled AV latent generated as a guide at the start of each continuation segment. Guided frames are removed from the delivered video."),
                io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True,
                             tooltip="Noise seed shared by every segment. Timeline prompts and continuation context change between segments."),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.String.Input("cache_name", default="h3_long_video",
                                tooltip="Output-relative bundle folder. Supports Save Video patterns, for example h3_long_video/%seed.seed%/. Existing folders become _2, _3, and so on unless resume is enabled."),
                io.Boolean.Input("resume", default=False,
                                 tooltip="Reuse compatible segment checkpoints. Missing or incompatible later segments are regenerated."),
                io.Int.Input("reroll_from_segment", default=-1, min=-1, max=999, step=1,
                             tooltip="With resume enabled: -1 resumes the first missing segment; N keeps segments before N and regenerates N onward."),
                io.Int.Input("crf", default=18, min=0, max=51, step=1, advanced=True),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Latent.Input("initial_latent", optional=True,
                                tooltip="Optional sampled H3 AV latent. Its tail guides a removable head before this node's timeline starts at 0."),
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
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            outputs=[
                io.Video.Output(display_name="video"),
                io.Latent.Output(display_name="last_latent"),
                io.String.Output(display_name="master_path"),
                io.Int.Output(display_name="segment_count"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, prompt, width, height, length, max_raw_frames,
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

        hidden = cls.hidden
        cache_name = _expand_cache_name(
            cache_name,
            hidden.prompt if hidden is not None else None,
            hidden.extra_pnginfo if hidden is not None else None,
        )
        segments = plan_segments(
            length, context_frames, initial_latent is not None, max_raw_frames)
        project, master_path, relative_folder = _output_paths(cache_name, resume, width, height)
        segment_paths = [project / "latents" / "segment_{:04d}.safetensors".format(segment.index) for segment in segments]
        local_prompts = [
            slice_prompt(
                prompt,
                segment.prompt_start_seconds,
                segment.prompt_end_seconds,
                segment.context_frames / FPS,
                segment.index,
                len(segments),
                length / FPS,
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
            "fps": FPS,
            "latent_format": "minimax_h3_av",
            "width": width,
            "height": height,
            "length": length,
            "max_raw_frames": max_raw_frames,
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
                if _metadata_matches(metadata, segment, prompt_hash, width, height, noise_seed):
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
            if segment.context_frames:
                if previous is None:
                    raise ValueError("a previous H3 AV latent is required for this continuation segment")
            conditioning = _conditioning(clip, local_prompt, ref_items, ref_blocks)
            if segment.context_frames:
                conditioning = _add_continuation_guide(
                    conditioning, previous, segment.context_frames)
            guider = custom_sampler.BasicGuider.execute(model, conditioning)[0]
            noise = custom_sampler.RandomNoise.execute(noise_seed)[0]
            sampled = custom_sampler.SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]
            previous = {"samples": sampled["samples"]}
            metadata = {
                "schema": SCHEMA_VERSION,
                "index": segment.index,
                "raw_frames": segment.raw_frames,
                "context_frames": segment.context_frames,
                "output_start": segment.output_start,
                "output_frames": segment.output_frames,
                "width": width,
                "height": height,
                "seed": noise_seed,
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
        manifest["status"] = "decoding"
        _atomic_json(project / "manifest.json", manifest)
        _write_master(master_path, segment_paths, segments, vae, audio_vae, width, height, crf)
        manifest["status"] = "complete"
        manifest["master"] = master_path.name
        _atomic_json(project / "manifest.json", manifest)

        video = InputImpl.VideoFromFile(str(master_path))
        preview = ui.PreviewVideo([ui.SavedResult(master_path.name, relative_folder, io.FolderType.output)])
        return io.NodeOutput(video, last_latent, str(master_path), completed, ui=preview)


class MiniMaxH3LongLatentUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongLatentUpscale",
            display_name="MiniMax H3 Long Latent Upscale & Assemble",
            category="sampling/minimax",
            description="Upscale Long H3 AV latent checkpoints one at a time and assemble a new MP4 without loading the full movie into memory.",
            is_output_node=True,
            inputs=[
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("source_path", default="h3_long_video",
                                tooltip="Output-relative Long H3 bundle folder, manifest.json, or master.mp4. An absolute path is accepted only inside ComfyUI's output folder."),
                io.Combo.Input("model_name", options=_upscaler_models()),
                io.Int.Input("target_width", default=1344, min=32, max=8192, step=32),
                io.Int.Input("target_height", default=768, min=32, max=8192, step=32),
                io.Int.Input("align", default=2, min=2, max=64, step=2,
                             tooltip="Target latent-grid alignment. 2 preserves typical Long H3 dimensions that are multiples of 32 pixels."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
                io.Combo.Input("precision", options=["fp32", "fp16", "bf16"], default="fp16"),
                io.String.Input("output_cache_name", default="h3_long_upscaled",
                                tooltip="Output-relative bundle folder for upscaled checkpoints and master.mp4. Save Video patterns are supported."),
                io.Boolean.Input("resume", default=False,
                                 tooltip="Reuse upscaled checkpoints whose source and settings still match."),
                io.Int.Input("reroll_from_segment", default=-1, min=-1, max=999, step=1,
                             tooltip="With resume enabled: -1 processes only missing or incompatible segments; N regenerates N onward."),
                io.Int.Input("crf", default=18, min=0, max=51, step=1, advanced=True),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            outputs=[
                io.Video.Output(display_name="video"),
                io.Latent.Output(display_name="last_latent"),
                io.String.Output(display_name="master_path"),
                io.Int.Output(display_name="segment_count"),
            ],
        )

    @classmethod
    def execute(cls, vae, audio_vae, source_path, model_name, target_width,
                target_height, align, device, precision, output_cache_name,
                resume, reroll_from_segment, crf):
        hidden = cls.hidden
        prompt = hidden.prompt if hidden is not None else None
        extra_pnginfo = hidden.extra_pnginfo if hidden is not None else None
        source_path = _expand_cache_name(source_path, prompt, extra_pnginfo)
        output_cache_name = _expand_cache_name(
            output_cache_name, prompt, extra_pnginfo)

        source, source_manifest_path = _source_bundle(source_path)
        with source_manifest_path.open("r", encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        if source_manifest.get("fps", FPS) != FPS:
            raise ValueError("source Long H3 bundle must use 24 fps")
        if source_manifest.get("latent_format", "minimax_h3_av") != "minimax_h3_av":
            raise ValueError("source manifest is not a MiniMax H3 AV latent bundle")
        segments, source_checkpoints = _manifest_segments(source, source_manifest)

        model_path = _upscaler_model_path(model_name)
        upscaler_class = comfy_nodes.NODE_CLASS_MAPPINGS.get(
            "H3LatentUpscalerNodeResolution")
        if upscaler_class is None:
            raise ValueError(
                "install Comfyui_Minimax_h3_latent_Upscaler and restart ComfyUI")
        upscaler = upscaler_class()

        project, master_path, relative_folder = _output_paths(
            output_cache_name, resume, target_width, target_height)
        if project == source:
            raise ValueError("output_cache_name must be different from the source bundle")
        checkpoints = [
            project / "latents" / "segment_{:04d}.safetensors".format(segment.index)
            for segment in segments
        ]
        manifest = {
            "schema": UPSCALE_SCHEMA_VERSION,
            "status": "upscaling",
            "fps": FPS,
            "latent_format": "minimax_h3_av",
            "source": os.path.relpath(source, Path(folder_paths.get_output_directory()).resolve()),
            "source_schema": source_manifest.get("schema"),
            "source_width": source_manifest.get("width"),
            "source_height": source_manifest.get("height"),
            "length": source_manifest.get("length"),
            "requested_width": target_width,
            "requested_height": target_height,
            "align": align,
            "model_name": model_name,
            "device": device,
            "precision": precision,
            "segments": [],
        }
        _atomic_json(project / "manifest.json", manifest)

        actual_width = None
        actual_height = None
        last_latent = None
        completed = 0
        for segment, source_checkpoint, checkpoint in zip(
                segments, source_checkpoints, checkpoints):
            expected = _upscale_metadata(
                source_checkpoint, model_path, model_name, target_width,
                target_height, align, device, precision, segment)
            may_reuse = resume and (
                reroll_from_segment < 0 or segment.index < reroll_from_segment)
            status = "upscaled"
            latent = None
            upscaled = None
            if may_reuse and checkpoint.exists():
                cached, metadata = _load_segment(checkpoint)
                if _upscale_metadata_matches(metadata, expected):
                    upscaled = cached
                    status = "reused"
                elif reroll_from_segment >= 0:
                    raise ValueError(
                        "upscaled segment {} no longer matches; reroll from this segment or earlier".format(
                            segment.index))

            if upscaled is None:
                latent, _ = _load_segment(source_checkpoint)
                upscaled = upscaler.run(
                    latent, model_name, target_width, target_height,
                    align, device, precision)[0]
                _streams(upscaled)
                _save_segment(checkpoint, upscaled, expected)

            video_latent, audio_latent = _streams(upscaled)
            segment_width = video_latent.shape[-1] * 16
            segment_height = video_latent.shape[-2] * 16
            if actual_width is None:
                actual_width = segment_width
                actual_height = segment_height
            elif segment_width != actual_width or segment_height != actual_height:
                raise ValueError("upscaled H3 segments do not have a consistent resolution")

            completed += 1
            manifest["segments"].append({
                "index": segment.index,
                "status": status,
                "file": checkpoint.name,
                "source_file": source_checkpoint.name,
                "output_start": segment.output_start,
                "output_frames": segment.output_frames,
                "timeline_start": segment.output_start / FPS,
                "timeline_end": (segment.output_start + segment.output_frames) / FPS,
                "raw_frames": segment.raw_frames,
                "context_frames": segment.context_frames,
            })
            _atomic_json(project / "manifest.json", manifest)
            if segment.index == segments[-1].index:
                last_latent = _cpu_latent(upscaled)
            latent = None
            upscaled = None
            video_latent = None
            audio_latent = None

        if last_latent is None or actual_width is None or actual_height is None:
            raise RuntimeError("MiniMax H3 Long Latent Upscale produced no segments")
        manifest["width"] = actual_width
        manifest["height"] = actual_height
        manifest["status"] = "decoding"
        _atomic_json(project / "manifest.json", manifest)
        _write_master(
            master_path, checkpoints, segments, vae, audio_vae,
            actual_width, actual_height, crf)
        manifest["status"] = "complete"
        manifest["master"] = master_path.name
        _atomic_json(project / "manifest.json", manifest)

        video = InputImpl.VideoFromFile(str(master_path))
        preview = ui.PreviewVideo([
            ui.SavedResult(master_path.name, relative_folder, io.FolderType.output)
        ])
        return io.NodeOutput(
            video, last_latent, str(master_path), completed, ui=preview)


class MiniMaxH3LongVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [MiniMaxH3LongReferenceSampler, MiniMaxH3LongLatentUpscale]


async def comfy_entrypoint():
    return MiniMaxH3LongVideoExtension()
