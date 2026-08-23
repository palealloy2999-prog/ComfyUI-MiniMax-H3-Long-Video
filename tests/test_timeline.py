import sys
import unittest
from pathlib import Path
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from minimax_h3_long_video import timeline
from minimax_h3_long_video.timeline import plan_segments, slice_prompt


PROMPT = """integrated_multimodal_description: [Shot 1] A woman walks through a station.
[Shot 2] At 00:12.000, the camera cuts to her hand holding a ticket.
[Shot 3] At 00:24.000, she boards the train.
[Shot 4] At 00:38.000, the doors close.

overall_soundscape: Footsteps and station ambience continue throughout.

non_diegetic_music: Sparse piano at a slow tempo.
"""


class TimelineTests(unittest.TestCase):
    def test_segment_plan_without_initial_latent(self):
        with mock.patch.object(timeline, "MAX_RAW_FRAMES", 345):
            segments = plan_segments(1569, 39, False)
        self.assertEqual(len(segments), 5)
        self.assertEqual(segments[0].raw_frames, 345)
        self.assertEqual(segments[0].head_frames, 0)
        self.assertEqual([segment.raw_frames for segment in segments[1:]], [345, 345, 345, 345])
        self.assertEqual([segment.output_frames for segment in segments], [345, 306, 306, 306, 306])
        self.assertEqual(segments[-1].output_start, 1263)

    def test_segment_plan_with_initial_latent_is_a_fresh_timeline(self):
        with mock.patch.object(timeline, "MAX_RAW_FRAMES", 345):
            segments = plan_segments(720, 22, True)
        self.assertEqual([segment.head_frames for segment in segments], [22, 22, 22])
        self.assertEqual([segment.output_frames for segment in segments], [323, 323, 74])
        self.assertEqual(segments[0].prompt_start_seconds, -22 / 24)
        self.assertEqual(sum(segment.output_frames for segment in segments), 720)

    def test_last_segment_is_grid_aligned_but_delivers_exact_length(self):
        with mock.patch.object(timeline, "MAX_RAW_FRAMES", 345):
            segments = plan_segments(360, 22, False)
        self.assertEqual(
            [(segment.raw_frames, segment.head_frames, segment.output_frames) for segment in segments],
            [(345, 0, 345), (39, 22, 15)],
        )
        self.assertTrue(all(segment.raw_frames % 17 == 5 for segment in segments))

    def test_prompt_window_rebases_shot_times(self):
        sliced = slice_prompt(PROMPT, 12.75, 27.125, 1.625)
        self.assertIn("[Shot 1]", sliced)
        self.assertIn("For the first 1.625 seconds", sliced)
        self.assertIn("[Shot 2] At 00:11.250", sliced)
        self.assertNotIn("00:24.000", sliced)
        self.assertNotIn("doors close", sliced)
        self.assertIn("overall_soundscape:", sliced)
        self.assertIn("non_diegetic_music:", sliced)

    def test_initial_latent_preroll_moves_timed_events_forward(self):
        sliced = slice_prompt(PROMPT, -22 / 24, 13.458, 22 / 24)
        self.assertIn("At 00:00.917, continue this ongoing shot forward", sliced)
        self.assertIn("do not restart, replay", sliced)
        self.assertIn("[Shot 2] At 00:12.917", sliced)

    def test_freeform_prompt_still_rebases_timestamps(self):
        sliced = slice_prompt("At 00:03.000, she turns.", -22 / 24, 5.0, 22 / 24)
        self.assertIn("At 00:03.917", sliced)
        self.assertIn("For the first 0.917 seconds", sliced)

    def test_plain_shot_timeline_is_windowed_and_rebased(self):
        prompt = """Camera-editing priority: hard cuts are mandatory.

[Shot 1] She starts running.
[Shot 2] At 00:05.300, reveal the broken bridge.
[Shot 3] At 00:06.900, HARD CUT to a side tracking shot.
[Shot 4] At 00:08.150, enter bullet time.

Camera-cut requirement: Each insert must be a real cut.
Action-editing rhythm: chase -> insert -> chase.
"""
        sliced = slice_prompt(prompt, 5.667, 8.0, 22 / 24)
        self.assertIn("Camera-editing priority:", sliced)
        self.assertIn("reveal the broken bridge", sliced)
        self.assertIn("[Shot 2] At 00:01.233", sliced)
        self.assertNotIn("She starts running", sliced)
        self.assertNotIn("enter bullet time", sliced)
        self.assertIn("Camera-cut requirement:", sliced)
        self.assertIn("Action-editing rhythm:", sliced)

    def test_later_shots_require_increasing_timestamps(self):
        missing = "integrated_multimodal_description: [Shot 1] A. [Shot 2] B."
        with self.assertRaisesRegex(ValueError, "needs an At"):
            slice_prompt(missing, 0.0, 10.0)
        reversed_times = (
            "integrated_multimodal_description: [Shot 1] A. "
            "[Shot 2] At 00:05.000, B. [Shot 3] At 00:04.000, C."
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            slice_prompt(reversed_times, 0.0, 10.0)


if __name__ == "__main__":
    unittest.main()
