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
    _add_context, _add_continuation_guide, _load_segment, _save_segment, _write_master,
)
from minimax_h3_long_video.timeline import plan_segments


class VideoVAE:
    def decode(self, latent):
        spans = (1, 4, 4, 4, 4)
        frames = sum(spans[index % 5] for index in range(latent.shape[2]))
        return torch.zeros((frames, 32, 32, 3))


class AudioVAE:
    audio_sample_rate = 32000
    audio_sample_rate_output = 32000

    def decode(self, latent):
        return torch.zeros((1, latent.shape[-1] * 800, 2))


class BackendTests(unittest.TestCase):
    def test_continuation_guide_anchors_video_and_audio_at_frame_zero(self):
        previous, _ = h3._empty_av_latent(32, 32, 56)
        conditioning = [[torch.zeros((1, 1, 1)), {"marker": True}]]
        guided = _add_continuation_guide(conditioning, previous, 22)
        keyframe = guided[0][1]["minimax_keyframes"][0]
        self.assertEqual(keyframe["resolved_frame_index"], 0)
        self.assertEqual(keyframe["latent"].shape[2], 7)
        self.assertEqual(keyframe["audio_latent"].shape[-1], 37)
        self.assertTrue(guided[0][1]["marker"])

    def test_nested_context_checkpoint_and_mp4(self):
        previous, _ = h3._empty_av_latent(32, 32, 56)
        previous_video, previous_audio = previous["samples"].unbind()
        previous_video.copy_(torch.arange(previous_video.numel()).reshape(previous_video.shape))
        previous_audio.copy_(torch.arange(previous_audio.numel()).reshape(previous_audio.shape))

        target, _ = h3._empty_av_latent(32, 32, 56)
        target = _add_context(target, previous, 22)
        target_video, target_audio = target["samples"].unbind()
        video_mask, audio_mask = target["noise_mask"].unbind()
        self.assertTrue(torch.equal(target_video[:, :, :7], previous_video[:, :, -7:]))
        self.assertTrue(torch.equal(target_audio[..., :37], previous_audio[..., -37:]))
        self.assertEqual(video_mask[:, :, :7].count_nonzero().item(), 0)
        self.assertEqual(audio_mask[..., :37].count_nonzero().item(), 0)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "segment.safetensors"
            _save_segment(checkpoint, target, {"schema": 1})
            loaded, metadata = _load_segment(checkpoint)
            self.assertEqual(metadata["schema"], "1")
            loaded_video, loaded_audio = loaded["samples"].unbind()
            self.assertTrue(torch.equal(loaded_video, target_video))
            self.assertTrue(torch.equal(loaded_audio, target_audio))

            master = Path(directory) / "master.mp4"
            segment = plan_segments(24, 22, True)[0]
            _write_master(master, [checkpoint], [segment], VideoVAE(), AudioVAE(), 32, 32, 28)
            self.assertTrue(master.is_file())
            import av
            with av.open(str(master)) as container:
                self.assertEqual(len(container.streams.video), 1)
                self.assertEqual(len(container.streams.audio), 1)

    def test_execute_reuses_completed_segments(self):
        class FakeSampler:
            calls = 0

            @classmethod
            def execute(cls, noise, guider, sampler, sigmas, latent):
                cls.calls += 1
                return ({"samples": latent["samples"]},)

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "latents").mkdir()

            def fake_master(path, *args):
                path.write_bytes(b"mp4")

            patches = (
                mock.patch.object(long_nodes, "_project_directory", return_value=project),
                mock.patch.object(long_nodes, "_prepare_references", return_value=([], [])),
                mock.patch.object(long_nodes, "_conditioning", return_value="conditioning"),
                mock.patch.object(long_nodes, "_write_master", side_effect=fake_master),
                mock.patch.object(long_nodes.custom_sampler.BasicGuider, "execute", return_value=("guider",)),
                mock.patch.object(long_nodes.custom_sampler.RandomNoise, "execute", return_value=("noise",)),
                mock.patch.object(long_nodes.custom_sampler.SamplerCustomAdvanced, "execute", side_effect=FakeSampler.execute),
                mock.patch.object(timeline, "MAX_RAW_FRAMES", 345),
            )
            for patch in patches:
                patch.start()
            try:
                first = long_nodes.MiniMaxH3LongReferenceSampler.execute(
                    None, None, VideoVAE(), AudioVAE(), "A continuous shot", 32, 32, 360,
                    "22", 1, "sampler", torch.tensor([1.0, 0.0]), "test", False, -1, 28)
                self.assertEqual(first[3], 2)
                self.assertEqual(FakeSampler.calls, 2)

                second = long_nodes.MiniMaxH3LongReferenceSampler.execute(
                    None, None, VideoVAE(), AudioVAE(), "A continuous shot", 32, 32, 360,
                    "22", 1, "sampler", torch.tensor([1.0, 0.0]), "test", True, -1, 28)
                self.assertEqual(second[3], 2)
                self.assertEqual(FakeSampler.calls, 2)
            finally:
                for patch in reversed(patches):
                    patch.stop()


if __name__ == "__main__":
    unittest.main()
