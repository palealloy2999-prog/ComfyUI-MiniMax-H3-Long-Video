import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(COMFY_ROOT), str(PACKAGE_ROOT)]

from comfy_extras import nodes_minimax_h3 as h3
from minimax_h3_long_video import nodes as long_nodes
from minimax_h3_long_video import timeline
from minimax_h3_long_video.nodes import (
    _add_continuation_guide, _expand_cache_name, _load_segment, _output_paths,
    _save_segment, _write_master, MiniMaxH3LongLatentUpscale,
    MiniMaxH3LongSegmentLoad, MiniMaxH3LongSegmentSave,
    MiniMaxH3LongUpscaleAssemble, MiniMaxH3LongUpscalePrepare,
)
from minimax_h3_long_video.timeline import plan_segments


class VideoVAE:
    def decode(self, latent):
        spans = (1, 4, 4, 4, 4)
        frames = sum(spans[index % 5] for index in range(latent.shape[2]))
        values = torch.arange(frames, dtype=torch.float32) / max(1, frames - 1)
        return values.reshape(-1, 1, 1, 1).expand(-1, 32, 32, 3)


class AudioVAE:
    audio_sample_rate = 32000
    audio_sample_rate_output = 32000

    def decode(self, latent):
        return torch.zeros((1, latent.shape[-1] * 800, 2))


class BackendTests(unittest.TestCase):
    def test_schema_uses_cache_name_without_filename_prefix(self):
        schema = long_nodes.MiniMaxH3LongReferenceSampler.define_schema()
        input_ids = [input.id for input in schema.inputs]
        self.assertIn("cache_name", input_ids)
        self.assertIn("prompt_plan", input_ids)
        self.assertNotIn("filename_prefix", input_ids)
        self.assertIn(long_nodes.io.Hidden.prompt, schema.hidden)
        self.assertIn(long_nodes.io.Hidden.extra_pnginfo, schema.hidden)
        self.assertIn(long_nodes.io.Hidden.unique_id, schema.hidden)

        planner_schema = long_nodes.MiniMaxH3LongPromptPlanner.define_schema()
        self.assertEqual(planner_schema.outputs[1].id, "preview")
        self.assertTrue(planner_schema.outputs[1].is_output_list)

        upscale_schema = MiniMaxH3LongLatentUpscale.define_schema()
        upscale_inputs = [input.id for input in upscale_schema.inputs]
        self.assertEqual(upscale_schema.node_id, "MiniMaxH3LongLatentUpscale")
        self.assertIn("source_path", upscale_inputs)
        self.assertIn("output_cache_name", upscale_inputs)
        self.assertIn("target_width", upscale_inputs)
        self.assertIn("target_height", upscale_inputs)

        prepare_schema = MiniMaxH3LongUpscalePrepare.define_schema()
        prepare_inputs = [input.id for input in prepare_schema.inputs]
        self.assertIn("master_path", prepare_inputs)
        self.assertIn("resume", prepare_inputs)
        self.assertNotIn("source_path", prepare_inputs)
        self.assertNotIn("output_cache_name", prepare_inputs)

        assemble_schema = MiniMaxH3LongUpscaleAssemble.define_schema()
        assemble_inputs = [input.id for input in assemble_schema.inputs]
        self.assertNotIn("output_cache_name", assemble_inputs)

    def test_prompt_planner_builds_and_overrides_exact_local_prompts(self):
        prompt = (
            "[Shot 1] Start. "
            "[Shot 2] At 00:03.000, first action. "
            "[Shot 3] At 00:15.000, second segment. "
            "[Shot 4] At 00:19.000, later action."
        )
        result = long_nodes.MiniMaxH3LongPromptPlanner.execute(
            prompt, 736, 362, "39", False,
            {"segment_prompt_1": "[Shot 1] Manually revised second segment."},
        )
        plan, preview, count = result
        self.assertEqual(count, 2)
        self.assertIn("[Shot 2] At 00:03.000, first action", plan["segments"][0]["prompt"])
        self.assertEqual(
            plan["segments"][1]["prompt"],
            "[Shot 1] Manually revised second segment.",
        )
        self.assertIsInstance(preview, list)
        self.assertEqual(len(preview), 2)
        self.assertEqual(preview, [entry["prompt"] for entry in plan["segments"]])
        self.assertIn("[Shot 2] At 00:03.000, first action", preview[0])
        self.assertEqual(preview[1], "[Shot 1] Manually revised second segment.")
        segments = plan_segments(736, 39, False, 362)
        self.assertEqual(
            timeline.prompt_plan_prompts(plan, segments, 736, 362, 39, False),
            [entry["prompt"] for entry in plan["segments"]],
        )
        with self.assertRaisesRegex(ValueError, "settings do not match"):
            timeline.prompt_plan_prompts(plan, segments, 736, 362, 22, False)

    def test_cache_name_expands_node_input_pattern(self):
        prompt = {
            "460": {
                "class_type": "PrimitiveInt",
                "inputs": {"seed": 123456789},
            },
        }
        extra_pnginfo = {
            "workflow": {
                "nodes": [{
                    "id": 460,
                    "type": "PrimitiveInt",
                    "title": "seed",
                    "properties": {},
                }],
            },
        }
        self.assertEqual(
            _expand_cache_name(
                "h3_long_video/%seed.seed%/", prompt, extra_pnginfo),
            "h3_long_video/123456789/",
        )

    def test_generation_fingerprint_tracks_generation_graph_but_not_full_prompt(self):
        graph = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3-a.safetensors"}},
            "10": {
                "class_type": "MiniMaxH3LongReferenceSampler",
                "inputs": {
                    "model": ["1", 0],
                    "prompt": "first timeline",
                    "cache_name": "first/output",
                },
            },
        }

        def fingerprint(value):
            return long_nodes._generation_fingerprint(
                value, "10", None, None, "euler", torch.tensor([1.0, 0.0]),
                "match", None, None, None, None, None)

        first = fingerprint(graph)
        graph["10"]["inputs"]["prompt"] = "changed timeline"
        graph["10"]["inputs"]["cache_name"] = "second/output"
        self.assertEqual(first, fingerprint(graph))
        graph["1"]["inputs"]["unet_name"] = "h3-b.safetensors"
        self.assertNotEqual(first, fingerprint(graph))

    def test_cache_name_expands_frontend_primitive_value(self):
        extra_pnginfo = {
            "workflow": {
                "nodes": [{
                    "id": 455,
                    "type": "PrimitiveNode",
                    "title": "noise_seed",
                    "widgets_values": [987654321, "fixed"],
                }],
            },
        }
        self.assertEqual(
            _expand_cache_name(
                "h3_long_video/%noise_seed.value%/", {}, extra_pnginfo),
            "h3_long_video/987654321/",
        )

    def test_cache_name_pattern_requires_unique_node_title(self):
        prompt = {"1": {"inputs": {"seed": 1}}, "2": {"inputs": {"seed": 2}}}
        extra_pnginfo = {
            "workflow": {
                "nodes": [
                    {"id": 1, "type": "PrimitiveInt", "title": "seed"},
                    {"id": 2, "type": "PrimitiveInt", "title": "seed"},
                ],
            },
        }
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _expand_cache_name("%seed.seed%", prompt, extra_pnginfo)

    def test_cache_name_controls_and_numbers_bundle_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(long_nodes.folder_paths, "get_output_directory", return_value=directory):
                bundle, bundle_master, bundle_relative_folder = _output_paths(
                    "h3_long_video/seed_123/", False, 640, 360)
                self.assertEqual(bundle, Path(directory) / "h3_long_video" / "seed_123")
                self.assertEqual(bundle_master, bundle / "master.mp4")
                self.assertEqual(bundle_relative_folder, str(Path("h3_long_video") / "seed_123"))
                self.assertTrue((bundle / "latents").is_dir())

                (bundle / "manifest.json").write_text("{}", encoding="utf-8")
                numbered, numbered_master, numbered_relative = _output_paths(
                    "h3_long_video/seed_123", False, 640, 360)
                self.assertEqual(numbered, Path(directory) / "h3_long_video" / "seed_123_2")
                self.assertEqual(numbered_master, numbered / "master.mp4")
                self.assertEqual(numbered_relative, str(Path("h3_long_video") / "seed_123_2"))

                third, _, _ = _output_paths(
                    "h3_long_video/seed_123", False, 640, 360)
                self.assertEqual(third, Path(directory) / "h3_long_video" / "seed_123_3")

                empty = Path(directory) / "h3_long_video" / "empty"
                empty.mkdir()
                numbered_empty, _, _ = _output_paths(
                    "h3_long_video/empty", False, 640, 360)
                self.assertEqual(numbered_empty, Path(directory) / "h3_long_video" / "empty_2")

                resumed, resumed_master, resumed_relative = _output_paths(
                    "h3_long_video/seed_123", True, 640, 360)
                self.assertEqual(resumed, bundle)
                self.assertEqual(resumed_master, bundle / "master.mp4")
                self.assertEqual(resumed_relative, bundle_relative_folder)
                with self.assertRaisesRegex(Exception, "outside the output folder"):
                    _output_paths("../outside/video", False, 640, 360)

    def test_continuation_guide_anchors_the_previous_tail_at_frame_zero(self):
        previous, _ = h3._empty_av_latent(32, 32, 56)
        original_guide = {"resolved_frame_index": 12, "latent": torch.zeros((1, 24, 1, 2, 2))}
        conditioning = [[torch.zeros((1, 1, 1)), {
            "marker": True,
            "minimax_keyframes": [original_guide],
        }]]
        conditioned = _add_continuation_guide(conditioning, previous, 22)
        guides = conditioned[0][1]["minimax_keyframes"]
        self.assertIs(guides[0], original_guide)
        guide = guides[-1]
        self.assertEqual(guide["resolved_frame_index"], 0)
        self.assertEqual(guide["latent"].shape[2], 7)
        self.assertEqual(guide["audio_latent"].shape[-1], 37)
        self.assertTrue(conditioned[0][1]["marker"])

    def test_continuation_guide_rejects_a_short_previous_latent(self):
        previous, _ = h3._empty_av_latent(32, 32, 5)
        conditioning = [[torch.zeros((1, 1, 1)), {}]]
        with self.assertRaisesRegex(ValueError, "shorter than context_frames"):
            _add_continuation_guide(conditioning, previous, 22)

    def test_nested_context_checkpoint_and_mp4(self):
        target, _ = h3._empty_av_latent(32, 32, 56)
        target_video, target_audio = target["samples"].unbind()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "segment.safetensors"
            _save_segment(checkpoint, target, {"schema": 1})
            loaded, metadata = _load_segment(checkpoint)
            self.assertEqual(metadata["schema"], "1")
            loaded_video, loaded_audio = loaded["samples"].unbind()
            self.assertTrue(torch.equal(loaded_video, target_video))
            self.assertTrue(torch.equal(loaded_audio, target_audio))

            master = Path(directory) / "master.mp4"
            segment = plan_segments(24, 22, True, 124)[0]
            _write_master(master, [checkpoint], [segment], VideoVAE(), AudioVAE(), 32, 32, 28)
            self.assertTrue(master.is_file())
            import av
            with av.open(str(master)) as container:
                self.assertEqual(len(container.streams.video), 1)
                self.assertEqual(len(container.streams.audio), 1)
                first_frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
                self.assertGreater(first_frame.mean(), 80)

    def test_manifest_upscale_processes_and_reuses_segments_one_at_a_time(self):
        class FakeUpscaler:
            calls = 0

            def run(self, latent, model_name, width, height, align, device, precision):
                self.__class__.calls += 1
                video, audio = latent["samples"].unbind()
                upscaled = video.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
                return ({
                    "samples": long_nodes.comfy.nested_tensor.NestedTensor((upscaled, audio)),
                },)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "upscaled"
            (source / "latents").mkdir(parents=True)
            (destination / "latents").mkdir(parents=True)
            segments = plan_segments(80, 22, False, 73)
            entries = []
            for segment in segments:
                latent, _ = h3._empty_av_latent(32, 32, segment.raw_frames)
                checkpoint = source / "latents" / "segment_{:04d}.safetensors".format(segment.index)
                _save_segment(checkpoint, latent, {"schema": 9})
                entries.append({
                    "index": segment.index,
                    "file": checkpoint.name,
                    "output_start": segment.output_start,
                    "output_frames": segment.output_frames,
                    "timeline_start": segment.output_start / 24,
                    "timeline_end": (segment.output_start + segment.output_frames) / 24,
                    "raw_frames": segment.raw_frames,
                    "context_frames": segment.context_frames,
                })
            (source / "manifest.json").write_text(json.dumps({
                "schema": 9,
                "status": "complete",
                "fps": 24,
                "latent_format": "minimax_h3_av",
                "width": 32,
                "height": 32,
                "length": 80,
                "segments": entries,
            }), encoding="utf-8")
            model_path = root / "upscaler.safetensors"
            model_path.write_bytes(b"model")

            captured = []

            def fake_master(path, checkpoints, planned, vae, audio_vae, width, height, crf):
                captured.append((len(checkpoints), len(planned), width, height))
                path.write_bytes(b"mp4")

            patches = (
                mock.patch.object(long_nodes.folder_paths, "get_output_directory", return_value=directory),
                mock.patch.object(long_nodes, "_upscaler_model_path", return_value=model_path),
                mock.patch.object(
                    long_nodes, "_output_paths",
                    return_value=(destination, destination / "master.mp4", "upscaled"),
                ),
                mock.patch.object(long_nodes, "_write_master", side_effect=fake_master),
                mock.patch.dict(
                    long_nodes.comfy_nodes.NODE_CLASS_MAPPINGS,
                    {"H3LatentUpscalerNodeResolution": FakeUpscaler},
                ),
            )
            for patch in patches:
                patch.start()
            try:
                first = MiniMaxH3LongLatentUpscale.execute(
                    VideoVAE(), AudioVAE(), str(source), "upscaler.safetensors",
                    64, 64, 2, "cuda", "fp16", "upscaled", False, -1, 28)
                self.assertEqual(first[3], 2)
                self.assertEqual(FakeUpscaler.calls, 2)
                self.assertEqual(captured[-1], (2, 2, 64, 64))
                upscaled, _ = _load_segment(destination / "latents" / "segment_0000.safetensors")
                video, audio = upscaled["samples"].unbind()
                self.assertEqual(video.shape[-2:], (4, 4))
                self.assertEqual(audio.shape[1:3], (32, 2))

                second = MiniMaxH3LongLatentUpscale.execute(
                    VideoVAE(), AudioVAE(), str(source), "upscaler.safetensors",
                    64, 64, 2, "cuda", "fp16", "upscaled", True, -1, 28)
                self.assertEqual(second[3], 2)
                self.assertEqual(FakeUpscaler.calls, 2)
                self.assertEqual(captured[-1], (2, 2, 64, 64))
            finally:
                for patch in reversed(patches):
                    patch.stop()

    def test_loop_upscale_nodes_load_save_and_assemble_by_segment_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = source / "upscale"
            (source / "latents").mkdir(parents=True)
            (source / "prompts").mkdir()
            segments = plan_segments(80, 22, False, 73)
            entries = []
            for segment in segments:
                latent, _ = h3._empty_av_latent(32, 32, segment.raw_frames)
                checkpoint = source / "latents" / "segment_{:04d}.safetensors".format(segment.index)
                prompt = source / "prompts" / "segment_{:04d}.txt".format(segment.index)
                _save_segment(checkpoint, latent, {"schema": 9})
                prompt.write_text("segment prompt {}".format(segment.index), encoding="utf-8")
                entries.append({
                    "index": segment.index,
                    "file": checkpoint.name,
                    "prompt_file": "prompts/{}".format(prompt.name),
                    "output_start": segment.output_start,
                    "output_frames": segment.output_frames,
                    "raw_frames": segment.raw_frames,
                    "context_frames": segment.context_frames,
                    "seed": 1234,
                })
            (source / "manifest.json").write_text(json.dumps({
                "schema": 9,
                "status": "complete",
                "fps": 24,
                "latent_format": "minimax_h3_av",
                "width": 32,
                "height": 32,
                "length": 80,
                "segments": entries,
            }), encoding="utf-8")

            captured = []

            def fake_master(path, checkpoints, planned, vae, audio_vae, width, height, crf):
                captured.append((len(checkpoints), len(planned), width, height))
                path.write_bytes(b"mp4")

            def fake_output_paths(cache_name, resume, width, height):
                self.assertEqual(cache_name, str(Path("source") / "upscale"))
                self.assertFalse(resume)
                return destination, destination / "master.mp4", "source/upscale"

            patches = (
                mock.patch.object(long_nodes.folder_paths, "get_output_directory", return_value=directory),
                mock.patch.object(
                    long_nodes, "_output_paths",
                    side_effect=fake_output_paths,
                ),
                mock.patch.object(long_nodes, "_write_master", side_effect=fake_master),
            )
            for patch in patches:
                patch.start()
            try:
                destination.mkdir()
                (destination / "latents").mkdir()
                prepared = MiniMaxH3LongUpscalePrepare.execute(str(source))
                job, count = prepared[0], prepared[1]
                self.assertEqual(count, 2)
                bundle = Path(job["project"])
                self.assertTrue(bundle.is_dir())
                loaded = MiniMaxH3LongSegmentLoad.execute(job, 0)
                latent, prompt, seed, width, height, raw_frames, token = loaded
                self.assertEqual(prompt, "segment prompt 0")
                self.assertEqual(seed, 1234)
                self.assertEqual((width, height), (32, 32))
                self.assertEqual(raw_frames, segments[0].raw_frames)
                video, audio = latent["samples"].unbind()
                upscaled_video = video.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
                invalid = {
                    "samples": long_nodes.comfy.nested_tensor.NestedTensor((
                        upscaled_video, audio[..., :-1],
                    )),
                }
                with self.assertRaisesRegex(ValueError, "changed its audio length"):
                    MiniMaxH3LongSegmentSave.execute(invalid, token)
                upscaled = {
                    "samples": long_nodes.comfy.nested_tensor.NestedTensor((upscaled_video, audio)),
                }
                MiniMaxH3LongSegmentSave.execute(upscaled, token)
                self.assertTrue((destination / "latents" / "segment_0000.safetensors").is_file())

                resumed = MiniMaxH3LongUpscalePrepare.execute(str(source), True)
                resumed_job, remaining = resumed[0], resumed[1]
                self.assertEqual(remaining, 1)
                self.assertEqual(Path(resumed_job["project"]), bundle)
                loaded = MiniMaxH3LongSegmentLoad.execute(resumed_job, 0)
                latent, prompt, seed, width, height, raw_frames, token = loaded
                self.assertEqual(token["index"], 1)
                self.assertEqual(prompt, "segment prompt 1")
                self.assertEqual(raw_frames, segments[1].raw_frames)
                video, audio = latent["samples"].unbind()
                upscaled = {
                    "samples": long_nodes.comfy.nested_tensor.NestedTensor((
                        video.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1), audio,
                    )),
                }
                progress = MiniMaxH3LongSegmentSave.execute(upscaled, token)[0]
                self.assertTrue((destination / "latents" / "segment_0001.safetensors").is_file())

                assembled = MiniMaxH3LongUpscaleAssemble.execute(
                    progress, VideoVAE(), AudioVAE(), 28)
                self.assertEqual(assembled[3], 2)
                self.assertEqual(captured, [(2, 2, 64, 64)])
                self.assertTrue(bundle.exists())
                manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["schema"], long_nodes.LOOP_UPSCALE_SCHEMA_VERSION)
                self.assertTrue(all(item["status"] == "saved" for item in manifest["segments"]))
            finally:
                for patch in reversed(patches):
                    patch.stop()

    def test_execute_reuses_completed_segments(self):
        noise_seeds = []

        def fake_noise(seed):
            noise_seeds.append(seed)
            return ("noise",)

        class FakeSampler:
            calls = 0

            @classmethod
            def execute(cls, noise, guider, sampler, sigmas, latent):
                cls.calls += 1
                if "noise_mask" in latent:
                    raise AssertionError("continuation segments must not use a hard latent mask")
                return ({"samples": latent["samples"]},)

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "latents").mkdir()

            def fake_master(path, *args):
                path.write_bytes(b"mp4")

            patches = (
                mock.patch.object(
                    long_nodes, "_output_paths",
                    return_value=(project, project / "master.mp4", "h3_long_video/test"),
                ),
                mock.patch.object(long_nodes, "_prepare_references", return_value=([], [])),
                mock.patch.object(
                    long_nodes, "_conditioning",
                    return_value=[[torch.zeros((1, 1, 1)), {}]],
                ),
                mock.patch.object(long_nodes, "_write_master", side_effect=fake_master),
                mock.patch.object(long_nodes.custom_sampler.BasicGuider, "execute", return_value=("guider",)),
                mock.patch.object(long_nodes.custom_sampler.RandomNoise, "execute", side_effect=fake_noise),
                mock.patch.object(long_nodes.custom_sampler.SamplerCustomAdvanced, "execute", side_effect=FakeSampler.execute),
            )
            for patch in patches:
                patch.start()
            try:
                first = long_nodes.MiniMaxH3LongReferenceSampler.execute(
                    None, None, VideoVAE(), AudioVAE(), "A continuous shot", 32, 32, 360,
                    345, "22", 1, "sampler", torch.tensor([1.0, 0.0]), "test", False, -1, 28)
                self.assertEqual(first[3], 2)
                self.assertEqual(FakeSampler.calls, 2)
                self.assertEqual(noise_seeds, [1, 1])
                manifest = json.loads((project / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["fps"], 24)
                self.assertEqual(manifest["latent_format"], "minimax_h3_av")
                self.assertEqual(manifest["schema"], long_nodes.SCHEMA_VERSION)
                self.assertIn("generation_fingerprint", manifest)
                self.assertEqual(manifest["segments"][1]["output_start"], 336)
                self.assertEqual(manifest["segments"][1]["output_frames"], 24)

                second = long_nodes.MiniMaxH3LongReferenceSampler.execute(
                    None, None, VideoVAE(), AudioVAE(), "A continuous shot", 32, 32, 360,
                    345, "22", 1, "sampler", torch.tensor([1.0, 0.0]), "test", True, -1, 28)
                self.assertEqual(second[3], 2)
                self.assertEqual(FakeSampler.calls, 2)
                self.assertEqual(noise_seeds, [1, 1])

                third = long_nodes.MiniMaxH3LongReferenceSampler.execute(
                    None, None, VideoVAE(), AudioVAE(), "A continuous shot", 32, 32, 360,
                    345, "22", 1, "sampler", torch.tensor([1.0, 0.5, 0.0]), "test", True, -1, 28)
                self.assertEqual(third[3], 2)
                self.assertEqual(FakeSampler.calls, 4)
                self.assertEqual(noise_seeds, [1, 1, 1, 1])

                manifest_before = (project / "manifest.json").read_text(encoding="utf-8")
                prompt_before = (project / "prompts" / "segment_0000.txt").read_text(encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "current generation inputs"):
                    long_nodes.MiniMaxH3LongReferenceSampler.execute(
                        None, None, VideoVAE(), AudioVAE(), "A continuous shot", 32, 32, 360,
                        345, "22", 1, "sampler", torch.tensor([1.0, 0.25, 0.0]),
                        "test", True, 1, 28)
                self.assertEqual(FakeSampler.calls, 4)
                self.assertEqual(
                    (project / "manifest.json").read_text(encoding="utf-8"), manifest_before)
                self.assertEqual(
                    (project / "prompts" / "segment_0000.txt").read_text(encoding="utf-8"), prompt_before)
            finally:
                for patch in reversed(patches):
                    patch.stop()


if __name__ == "__main__":
    unittest.main()
