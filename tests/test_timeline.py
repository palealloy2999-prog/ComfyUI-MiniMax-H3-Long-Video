import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from minimax_h3_long_video.timeline import (
    build_prompt_plan,
    plan_segments,
    prompt_plan_prompts,
    slice_prompt,
)


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
        self.assertEqual([segment.raw_frames for segment in segments[1:]], [379, 379, 379, 260])
        self.assertEqual([segment.output_frames for segment in segments], [336, 336, 336, 336, 216])
        self.assertEqual(segments[-1].output_start, 1344)

    def test_segment_plan_with_initial_latent_is_a_fresh_timeline(self):
        segments = plan_segments(720, 22, True, 345)
        self.assertEqual([segment.context_frames for segment in segments], [22, 22, 22])
        self.assertEqual([segment.raw_frames for segment in segments], [362, 362, 73])
        self.assertEqual([segment.output_frames for segment in segments], [336, 336, 48])
        self.assertEqual(segments[0].prompt_start_seconds, 0)
        self.assertEqual(segments[0].prompt_end_seconds, 14)
        self.assertEqual(sum(segment.output_frames for segment in segments), 720)

    def test_last_segment_is_grid_aligned_but_delivers_exact_length(self):
        segments = plan_segments(360, 22, False, 345)
        self.assertEqual(
            [(segment.raw_frames, segment.context_frames, segment.output_frames) for segment in segments],
            [(345, 0, 336), (56, 22, 24)],
        )
        self.assertTrue(all(segment.raw_frames % 17 == 5 for segment in segments))

    def test_max_raw_frames_controls_segment_duration(self):
        three_seconds = plan_segments(240, 22, False, 73)
        five_seconds = plan_segments(240, 22, False, 124)
        self.assertEqual([segment.raw_frames for segment in three_seconds], [73, 107, 107, 56])
        self.assertEqual([segment.output_frames for segment in three_seconds], [72, 72, 72, 24])
        self.assertEqual([segment.raw_frames for segment in five_seconds], [124, 158])
        self.assertEqual([segment.output_frames for segment in five_seconds], [120, 120])

    def test_362_frames_reverse_to_two_fifteen_second_windows(self):
        segments = plan_segments(720, 39, False, 362)
        self.assertEqual(len(segments), 2)
        self.assertEqual([item.output_frames for item in segments], [360, 360])
        self.assertEqual([item.raw_frames for item in segments], [362, 413])
        self.assertEqual(
            [(item.prompt_start_seconds, item.prompt_end_seconds) for item in segments],
            [(0, 15), (15, 30)],
        )

    def test_grid_encoded_thirty_seconds_does_not_create_a_third_segment(self):
        segments = plan_segments(736, 39, False, 362)
        self.assertEqual(len(segments), 2)
        self.assertEqual([item.output_frames for item in segments], [360, 360])
        self.assertEqual(segments[-1].prompt_end_seconds, 30)

    def test_prompt_plan_can_override_one_exact_local_prompt(self):
        prompt = (
            "[Shot 1] Start. "
            "[Shot 2] At 00:03.000, first action. "
            "[Shot 3] At 00:15.000, second segment. "
            "[Shot 4] At 00:19.000, later action."
        )
        override = "[Shot 1] Manually revised second segment."
        plan = build_prompt_plan(
            prompt, 736, 362, "39", False,
            {"segment_prompt_1": override},
        )
        segments = plan_segments(736, 39, False, 362)
        self.assertEqual(len(plan["segments"]), 2)
        self.assertIn(
            "[Shot 2] At 00:03.000, first action",
            plan["segments"][0]["prompt"],
        )
        self.assertEqual(plan["segments"][1]["prompt"], override)
        self.assertEqual(
            prompt_plan_prompts(plan, segments, 736, 362, 39, False),
            [entry["prompt"] for entry in plan["segments"]],
        )
        with self.assertRaisesRegex(ValueError, "settings do not match"):
            prompt_plan_prompts(plan, segments, 736, 362, 22, False)

    def test_sample1_once_only_crimson_wipe_stays_out_of_segment_zero(self):
        prompt = (PACKAGE_ROOT / "sample" / "sample1.txt").read_text(encoding="utf-8")
        segments = plan_segments(736, 39, False, 362)
        total_seconds = sum(item.output_frames for item in segments) / 24
        local = [
            slice_prompt(
                prompt,
                item.prompt_start_seconds,
                item.prompt_end_seconds,
                item.context_frames / 24,
                item.index,
                len(segments),
                total_seconds,
            )
            for item in segments
        ]
        self.assertNotIn("deep crimson fabric", local[0])
        self.assertNotIn("crimson evening gown", local[0])
        self.assertIn("deep crimson fabric", local[1])
        self.assertEqual(local[1].count("deep crimson fabric"), 1)
        self.assertIn("[Shot 5] At 00:14.000", local[1])
        self.assertIn("[Shot 6] At 00:14.700", local[1])

    def test_grid_values_reverse_to_human_scale_segment_frames(self):
        expected = {73: 72, 90: 90, 107: 96, 124: 120, 362: 360}
        for grid_frames, delivered_frames in expected.items():
            with self.subTest(grid_frames=grid_frames):
                first = plan_segments(720, 39, False, grid_frames)[0]
                self.assertEqual(first.output_frames, delivered_frames)

    def test_max_raw_frames_requires_h3_grid(self):
        with self.assertRaisesRegex(ValueError, "17k\\+5"):
            plan_segments(240, 22, False, 120)

    def test_prompt_window_rebases_shot_times(self):
        sliced = slice_prompt(PROMPT, 12.75, 27.125)
        self.assertIn("[Shot 2] At 00:11.250", sliced)
        self.assertIn("already in progress", sliced)
        self.assertIn("ticket", sliced)
        self.assertNotIn("00:24.000", sliced)
        self.assertNotIn("doors close", sliced)
        self.assertIn("overall_soundscape:", sliced)
        self.assertIn("non_diegetic_music:", sliced)

    def test_fifteen_second_windows_keep_segment_zero_and_subtract_fifteen_later(self):
        prompt = (
            "[Shot 1] Opening. "
            "[Shot 2] At 00:03.000, first-window action. "
            "[Shot 3] At 00:15.000, second window starts. "
            "[Shot 4] At 00:18.000, second-window action."
        )
        first = slice_prompt(prompt, 0, 15)
        second = slice_prompt(prompt, 15, 30, 39 / 24)
        self.assertIn("[Shot 2] At 00:03.000, first-window action", first)
        self.assertIn("[Shot 1] second window starts", second)
        self.assertIn("[Shot 2] At 00:03.000, second-window action", second)
        self.assertNotIn("00:18.000", second)

    def test_integrated_prompt_preserves_common_intro_and_global_instructions(self):
        prompt = """integrated_multimodal_description: Preserve the exact character identity.
Use a continuous bright electronic-pop song.

[Shot 1] Start running.
[Shot 2] At 00:05.000, jump over a barrier.
Landing requirement: Bend only during this landing.

[Global Instructions]
Character-consistency requirement: Keep the same face and outfit.
Camera-motion requirement: Keep the camera moving.

overall_soundscape: Footsteps continue.
"""
        first = slice_prompt(prompt, 0.0, 5.0)
        second = slice_prompt(prompt, 5.0, 10.0, 22 / 24)
        for sliced in (first, second):
            self.assertIn("Preserve the exact character identity", sliced)
            self.assertIn("continuous bright electronic-pop song", sliced)
            self.assertIn("Character-consistency requirement", sliced)
            self.assertIn("Camera-motion requirement", sliced)
            self.assertIn("overall_soundscape:", sliced)
            self.assertIn("Footsteps continue", sliced)
            self.assertNotIn("[Global Instructions]", sliced)
        self.assertIn("Start running", first)
        self.assertNotIn("Start running", second)
        self.assertNotIn("Landing requirement", first)
        self.assertIn("jump over a barrier", second)
        self.assertIn("Landing requirement", second)

    def test_global_instructions_marker_must_be_unique_and_after_final_shot(self):
        before = """integrated_multimodal_description:
[Global Instructions]
Keep the same character.
[Shot 1] Start.
"""
        duplicate = """integrated_multimodal_description:
[Shot 1] Start.
[Global Instructions]
Keep the same character.
[Global Instructions]
Keep the same music.
"""
        with self.assertRaisesRegex(ValueError, "after the final Shot"):
            slice_prompt(before, 0.0, 5.0)
        with self.assertRaisesRegex(ValueError, "at most one"):
            slice_prompt(duplicate, 0.0, 5.0)

    def test_freeform_prompt_still_rebases_timestamps(self):
        sliced = slice_prompt("At 00:03.000, she turns.", 2.0, 5.0)
        self.assertIn("At 00:01.000", sliced)

    def test_plain_shot_timeline_is_windowed_and_rebased(self):
        prompt = """Camera-editing priority: hard cuts are mandatory.

[Shot 1] She starts running.
[Shot 2] At 00:05.300, reveal the broken bridge.
[Shot 3] At 00:06.900, HARD CUT to a side tracking shot.
[Shot 4] At 00:08.150, enter bullet time.

[Global Instructions]
Camera-cut requirement: Each insert must be a real cut.
Action-editing rhythm: chase -> insert -> chase.
"""
        sliced = slice_prompt(prompt, 5.667, 8.0)
        self.assertIn("Camera-editing priority:", sliced)
        self.assertIn("reveal the broken bridge", sliced)
        self.assertIn("[Shot 2] At 00:01.233", sliced)
        self.assertNotIn("She starts running", sliced)
        self.assertNotIn("enter bullet time", sliced)
        self.assertIn("Camera-cut requirement:", sliced)
        self.assertIn("Action-editing rhythm:", sliced)

    def test_in_progress_shot_is_carried_without_restarting_it(self):
        prompt = (
            "[Shot 1] Start. "
            "[Shot 2] At 00:05.300, spin, kick, land, grab, and swing. "
            "[Shot 3] At 00:06.900, begin the wall-run."
        )
        first = slice_prompt(prompt, 0.0, 158 / 24)
        second = slice_prompt(prompt, 158 / 24, 316 / 24)
        self.assertIn("spin, kick, land, grab, and swing", first)
        self.assertIn("spin, kick, land, grab, and swing", second)
        self.assertIn("do not restart or replay its beginning", second)
        self.assertIn("[Shot 2] At 00:00.317", second)
        self.assertIn("begin the wall-run", second)

    def test_context_guide_offsets_new_shots_and_carries_active_action(self):
        prompt = (
            "[Shot 1] Start. "
            "[Shot 2] At 00:05.300, spin, kick, land, grab, and swing. "
            "[Shot 3] At 00:06.900, begin the wall-run."
        )
        sliced = slice_prompt(prompt, 158 / 24, 260 / 24, 22 / 24)
        self.assertIn("outside this segment's timestamp clock", sliced)
        self.assertIn("spin, kick, land, grab, and swing", sliced)
        self.assertIn("do not restart or replay its beginning", sliced)
        self.assertIn("[Shot 2] At 00:00.317", sliced)
        self.assertIn("begin the wall-run", sliced)

    def test_segment_without_a_new_shot_carries_the_active_shot(self):
        prompt = "[Shot 1] Run forward. [Shot 2] At 00:20.000, stop."
        sliced = slice_prompt(prompt, 6.0, 12.0)
        self.assertIn("already in progress", sliced)
        self.assertIn("do not restart or replay", sliced)
        self.assertIn("Run forward", sliced)
        self.assertNotIn("stop", sliced)

    def test_untimed_shots_are_distributed_across_segments(self):
        prompt = (
            "integrated_multimodal_description: [Shot 1] A. [Shot 2] B. "
            "[Shot 3] C. [Shot 4] D. [Shot 5] E. [Shot 6] F."
        )
        first = slice_prompt(prompt, 0.0, 5.0, segment_index=0, segment_count=2)
        second = slice_prompt(prompt, 5.0, 10.0, 22 / 24, 1, 2)
        self.assertIn("A.", first)
        self.assertIn("C.", first)
        self.assertNotIn("D.", first)
        self.assertNotIn("C.", second)
        self.assertIn("D.", second)
        self.assertIn("F.", second)
        self.assertIn("[Shot 2] At 00:01.667", second)

    def test_fewer_untimed_shots_are_spread_over_the_full_timeline(self):
        prompt = "[Shot 1] Start. [Shot 2] Middle. [Shot 3] Finish."
        sliced = [
            slice_prompt(prompt, index * 5.0, (index + 1) * 5.0,
                         segment_index=index, segment_count=5)
            for index in range(5)
        ]
        self.assertIn("Start.", sliced[0])
        self.assertIn("Middle.", sliced[2])
        self.assertIn("Finish.", sliced[4])

    def test_untimed_shots_are_interpolated_between_timestamped_shots(self):
        prompt = (
            "[Shot 1] Start. [Shot 2] First gap. "
            "[Shot 3] At 00:06.000, first anchor. [Shot 4] Second gap. "
            "[Shot 5] At 00:10.000, second anchor. [Shot 6] Tail."
        )
        first = slice_prompt(prompt, 0.0, 5.0, timeline_duration_seconds=15.0)
        second = slice_prompt(prompt, 5.0, 10.0, timeline_duration_seconds=15.0)
        third = slice_prompt(prompt, 10.0, 15.0, 22 / 24,
                             timeline_duration_seconds=15.0)
        self.assertIn("[Shot 2] At 00:03.000, First gap.", first)
        self.assertIn("Continue the master-timeline Shot already in progress", second)
        self.assertIn("[Shot 2] At 00:01.000, first anchor.", second)
        self.assertIn("[Shot 3] At 00:03.000, Second gap.", second)
        self.assertIn("[Shot 1] second anchor.", third)
        self.assertIn("[Shot 2] At 00:02.500, Tail.", third)

    def test_official_ref2va_sections_are_preserved_and_scoped_per_segment(self):
        prompt = """subject_definitions:
<Subject 1> is the dancer in <Picture 1>.

summary:
The complete master timeline is a 30-second reference-generation dance video.

retention_analysis:
<Subject 1>: fully_preserved - preserve the dancer's identity.

detailed_description:
The master video uses a luminous digital-pop style.
[Shot 1] The dancer begins a slow turn.
[Shot 2] At 00:15.000, she drops into a floor spin and rises mechanically.
[Shot 3] At 00:17.250, the camera cuts to a rear Tracking Shot.

[Global Instructions]
Preserve <Subject 1> and continue the choreography without replaying it.

overall_soundscape: Footfalls and movement sweeps remain continuous.

non_diegetic_music: One continuous electronic-pop score plays without restarting.
"""
        sliced = slice_prompt(
            prompt,
            15.083333333333334,
            28.541666666666668,
            39 / 24,
        )
        self.assertLess(sliced.index("subject_definitions:"), sliced.index("summary:"))
        self.assertLess(sliced.index("summary:"), sliced.index("retention_analysis:"))
        self.assertLess(sliced.index("retention_analysis:"), sliced.index("detailed_description:"))
        self.assertIn("master timeline range is 00:15.083-00:28.542", sliced)
        self.assertIn("This H3 generation pass is 13.458 seconds long", sliced)
        self.assertIn("preceding AV guide is outside this local timestamp clock", sliced)
        self.assertNotIn("complete master timeline is a 30-second", sliced)
        self.assertIn("This is a continuation pass", sliced)
        self.assertIn("segment.\n\nretention_analysis:", sliced)
        self.assertIn("floor spin and rises mechanically", sliced)
        self.assertIn("do not restart or replay its beginning", sliced)
        self.assertIn("[Shot 2] At 00:02.167", sliced)
        self.assertIn("Preserve <Subject 1>", sliced)
        self.assertLess(
            sliced.index("Preserve <Subject 1>"),
            sliced.index("The supplied AV guide is preceding context"),
        )
        self.assertNotIn("[Global Instructions]", sliced)
        self.assertIn("overall_soundscape:", sliced)
        self.assertIn("non_diegetic_music:", sliced)

    def test_fenced_ref2va_continuation_does_not_replay_global_audio_cues(self):
        prompt = """```text
subject_definitions:
<Subject 1> is the dancer in <Picture 1>.

summary:
The video begins in the reference pose, builds for three seconds, then hits a drop.

retention_analysis:
<Subject 1>: fully_preserved - preserve identity and clothing.

detailed_description:
[Shot 1] The dancer holds the opening pose as WAKE LIGHT appears.
[Shot 2] At 00:15.000, continue the choreography with a new traveling step.

overall_soundscape: At 00:03.000, add an impact and crowd shout.

non_diegetic_music: Begin quietly. At 00:03.000, switch to the beat drop; build toward the finale at 00:28.800.
```"""
        first = slice_prompt(prompt, 0.0, 15.083333333333334)
        continuation = slice_prompt(
            prompt, 15.083333333333334, 28.541666666666668, 39 / 24)

        self.assertNotIn("```", first)
        self.assertNotIn("```", continuation)
        self.assertIn("video begins in the reference pose", first)
        self.assertNotIn("video begins in the reference pose", continuation)
        self.assertNotIn("00:03.000", continuation)
        self.assertNotIn("00:28.800", continuation)
        self.assertIn("Continue the soundscape established", continuation)
        self.assertIn("Do not restart its intro, drop, vocals", continuation)
        self.assertIn("continue the choreography with a new traveling step", continuation)

    def test_ref2va_rejects_bare_s_labels_as_subjects(self):
        prompt = """subject_definitions:
S1 = the dancer in <Picture 1>.
S2 = an off-screen vocalist.

summary:
A continuous dance video.

retention_analysis:
Preserve the dancer.

detailed_description:
[Shot 1] S1 dances while S2 sings.
"""
        with self.assertRaisesRegex(
                ValueError, r"Invalid Ref2VA subject label.*S1, S2.*<Subject N>.*\(Sx\)"):
            slice_prompt(prompt, 0.0, 5.0)

    def test_continuation_removes_master_keyframe_alignment(self):
        prompt = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Begin from <Picture 1> and walk forward.
[Shot 2] At 00:06.000, the camera cuts to a side view.

[Global Instructions]
Preserve the identity from <Picture 1>.

overall_soundscape: Quiet footsteps continue.

non_diegetic_music: N/A
"""
        first = slice_prompt(prompt, 0.0, 5.0)
        second = slice_prompt(prompt, 5.0, 10.0, 22 / 24)
        self.assertIn("at 0.00 seconds into the target video", first)
        self.assertNotIn("at 0.00 seconds into the target video", second)
        self.assertIn("walk forward", second)
        self.assertIn("[Shot 2] At 00:01.000", second)
        self.assertIn("Preserve the identity from <Picture 1>", second)
        self.assertNotIn("[Global Instructions]", second)

    def test_final_grid_padding_is_not_part_of_the_master_prompt_window(self):
        segments = plan_segments(360, 22, False, 345)
        final = segments[-1]
        prompt = """integrated_multimodal_description:
[Shot 1] Continue moving.
[Shot 2] At 00:15.050, this event is outside the delivered master.

[Global Instructions]
Preserve continuity.
"""
        sliced = slice_prompt(
            prompt,
            final.prompt_start_seconds,
            final.prompt_end_seconds,
            final.context_frames / 24,
            final.index,
            len(segments),
            360 / 24,
        )
        self.assertIn("master timeline range is 00:14.000-00:15.000", sliced)
        self.assertIn("This H3 generation pass is 1.000 seconds long", sliced)
        self.assertNotIn("outside the delivered master", sliced)

    def test_displayed_millisecond_boundary_belongs_to_later_segment(self):
        boundary = 158 / 24
        prompt = (
            "[Shot 1] Start. "
            "[Shot 2] At 00:06.583, begin exactly on the displayed boundary."
        )
        first = slice_prompt(prompt, 0.0, boundary)
        second = slice_prompt(prompt, boundary, 10.0, 22 / 24)
        self.assertNotIn("displayed boundary", first)
        self.assertIn("displayed boundary", second)
        self.assertNotIn("Continue the master-timeline Shot already in progress", second)

    def test_reversed_timestamps_are_rejected_even_with_untimed_shots_between(self):
        reversed_times = (
            "integrated_multimodal_description: [Shot 1] A. "
            "[Shot 2] At 00:05.000, B. [Shot 3] C. "
            "[Shot 4] At 00:04.000, D."
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            slice_prompt(reversed_times, 0.0, 10.0)

    def test_shot_numbers_must_be_sequential(self):
        prompt = "[Shot 1] Start. [Shot 3] At 00:04.000, skip a number."
        with self.assertRaisesRegex(ValueError, "sequential starting at 1"):
            slice_prompt(prompt, 0.0, 10.0)


if __name__ == "__main__":
    unittest.main()
