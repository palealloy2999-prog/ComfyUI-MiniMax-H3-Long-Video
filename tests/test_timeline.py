import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

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
        segments = plan_segments(1569, 39, False, 345)
        self.assertEqual(len(segments), 5)
        self.assertEqual(segments[0].raw_frames, 345)
        self.assertEqual(segments[0].context_frames, 0)
        self.assertEqual([segment.raw_frames for segment in segments[1:]], [345, 345, 345, 345])
        self.assertEqual([segment.output_frames for segment in segments], [345, 306, 306, 306, 306])
        self.assertEqual(segments[-1].output_start, 1263)

    def test_segment_plan_with_initial_latent_is_a_fresh_timeline(self):
        segments = plan_segments(720, 22, True, 345)
        self.assertEqual([segment.context_frames for segment in segments], [22, 22, 22])
        self.assertEqual([segment.raw_frames for segment in segments], [345, 345, 107])
        self.assertEqual([segment.output_frames for segment in segments], [323, 323, 74])
        self.assertEqual(segments[0].prompt_start_seconds, 0)
        self.assertEqual(segments[0].prompt_end_seconds, 323 / 24)
        self.assertEqual(sum(segment.output_frames for segment in segments), 720)

    def test_last_segment_is_grid_aligned_but_delivers_exact_length(self):
        segments = plan_segments(360, 22, False, 345)
        self.assertEqual(
            [(segment.raw_frames, segment.context_frames, segment.output_frames) for segment in segments],
            [(345, 0, 345), (39, 22, 15)],
        )
        self.assertTrue(all(segment.raw_frames % 17 == 5 for segment in segments))

    def test_max_raw_frames_controls_segment_duration(self):
        three_seconds = plan_segments(240, 22, False, 73)
        five_seconds = plan_segments(240, 22, False, 124)
        self.assertEqual([segment.raw_frames for segment in three_seconds], [73, 73, 73, 73, 39])
        self.assertEqual([segment.output_frames for segment in three_seconds], [73, 51, 51, 51, 14])
        self.assertEqual([segment.raw_frames for segment in five_seconds], [124, 124, 39])
        self.assertEqual([segment.output_frames for segment in five_seconds], [124, 102, 14])

    def test_max_raw_frames_requires_h3_grid(self):
        with self.assertRaisesRegex(ValueError, "17k\\+5"):
            plan_segments(240, 22, False, 120)

    def test_prompt_window_rebases_shot_times(self):
        sliced = slice_prompt(PROMPT, 12.75, 27.125)
        self.assertIn("[Shot 1] At 00:11.250", sliced)
        self.assertNotIn("ticket", sliced)
        self.assertNotIn("00:24.000", sliced)
        self.assertNotIn("doors close", sliced)
        self.assertIn("overall_soundscape:", sliced)
        self.assertIn("non_diegetic_music:", sliced)

    def test_freeform_prompt_still_rebases_timestamps(self):
        sliced = slice_prompt("At 00:03.000, she turns.", 2.0, 5.0)
        self.assertIn("At 00:01.000", sliced)

    def test_plain_shot_timeline_is_windowed_and_rebased(self):
        prompt = """Camera-editing priority: hard cuts are mandatory.

[Shot 1] She starts running.
[Shot 2] At 00:05.300, reveal the broken bridge.
[Shot 3] At 00:06.900, HARD CUT to a side tracking shot.
[Shot 4] At 00:08.150, enter bullet time.

Camera-cut requirement: Each insert must be a real cut.
Action-editing rhythm: chase -> insert -> chase.
"""
        sliced = slice_prompt(prompt, 5.667, 8.0)
        self.assertIn("Camera-editing priority:", sliced)
        self.assertNotIn("reveal the broken bridge", sliced)
        self.assertIn("[Shot 1] At 00:01.233", sliced)
        self.assertNotIn("She starts running", sliced)
        self.assertNotIn("enter bullet time", sliced)
        self.assertIn("Camera-cut requirement:", sliced)
        self.assertIn("Action-editing rhythm:", sliced)

    def test_shot_is_assigned_only_to_the_segment_containing_its_start(self):
        prompt = (
            "[Shot 1] Start. "
            "[Shot 2] At 00:05.300, spin, kick, land, grab, and swing. "
            "[Shot 3] At 00:06.900, begin the wall-run."
        )
        first = slice_prompt(prompt, 0.0, 158 / 24)
        second = slice_prompt(prompt, 158 / 24, 316 / 24)
        self.assertIn("spin, kick, land, grab, and swing", first)
        self.assertNotIn("spin, kick, land, grab, and swing", second)
        self.assertIn("[Shot 1] At 00:00.317", second)
        self.assertIn("begin the wall-run", second)

    def test_context_guide_offsets_new_shots_without_copying_previous_action(self):
        prompt = (
            "[Shot 1] Start. "
            "[Shot 2] At 00:05.300, spin, kick, land, grab, and swing. "
            "[Shot 3] At 00:06.900, begin the wall-run."
        )
        sliced = slice_prompt(prompt, 158 / 24, 260 / 24, 22 / 24)
        self.assertIn("For the first 0.917 seconds", sliced)
        self.assertIn("context only", sliced)
        self.assertNotIn("spin, kick, land, grab, and swing", sliced)
        self.assertIn("[Shot 2] At 00:01.233", sliced)
        self.assertIn("begin the wall-run", sliced)

    def test_segment_without_a_new_shot_uses_only_neutral_continuation(self):
        prompt = "[Shot 1] Run forward. [Shot 2] At 00:20.000, stop."
        sliced = slice_prompt(prompt, 6.0, 12.0)
        self.assertIn("Continue forward from the supplied preceding AV context", sliced)
        self.assertIn("Do not restart or repeat", sliced)
        self.assertNotIn("Run forward", sliced)
        self.assertNotIn("stop", sliced)

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
