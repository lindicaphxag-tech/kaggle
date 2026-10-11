"""Second GENUINE merged LeRobot source bug: frame0 stack + cached reset."""
import unittest
from research.semrepair.real_merged_lerobot_stacked_state_synthesis import (
    run_real_merged_stacked_source_synthesis,CANDIDATES
)

class VerifiedMergedStackedAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=run_real_merged_stacked_source_synthesis()

    def test_real_original_upstream_issue_merging_identity(self):
        r=self.report
        self.assertIn("4656",r["actual_MERGED_external_PR"])
        self.assertEqual(r["actual_merged_commit"],
                         "30074f7f1358b3c015ae1750017200e86e9c4eb6")
        self.assertTrue(r["original_merged_policy_functions_truly_executed"])
        self.assertEqual(r["candidate_count"],6)
        self.assertEqual(len(r["original_publisher_git_blobs"]),4)
        self.assertFalse(r["actual_VLA_JEPA_neural_network_checkpoint_inference"])
        self.assertEqual(r["actual_robot_task_rollout_episodes"],0)

    def test_unpatched_original_stacked_state_crashes_every_case(self):
        x=self.report["results"]
        self.assertEqual(x["unchanged"]["stacked_exceptions"],96)
        self.assertEqual(x["frame0_both"]["stacked_exceptions"],0)
        self.assertEqual(x["frame0_both"]["conditioning_anchor_pass"],96)
        self.assertEqual(x["frame0_both"]["max_error_against_frame0"],0)

    def test_two_wrong_but_roundtrip_passing_patches_are_real_overfitting(self):
        x=self.report["results"]
        for patch in ("last_both","temporal_mean_both"):
            with self.subTest(patch=patch):
                self.assertEqual(x[patch]["stacked_roundtrip_pass"],96)
                self.assertEqual(x[patch]["conditioning_anchor_pass"],0)
                self.assertGreater(
                    x[patch]["mean_max_relative_action_error_against_frame0"],.1)

    def test_one_sided_repairs_are_not_sufficient(self):
        x=self.report["results"]
        for patch in ("frame0_relative_only","frame0_absolute_only"):
            self.assertEqual(x[patch]["stacked_exceptions"],96)

    def test_preserve_previously_working_2D_policies(self):
        for patch,x in self.report["results"].items():
            with self.subTest(patch=patch):
                self.assertEqual(x["classic_2d_pass"],96)

    def test_real_upstream_reset_prevents_cross_episode_anchor_leak(self):
        x=self.report["original_merged_reset_method_evidence"]
        self.assertTrue(x["publisher_original_reset_AST_executed"])
        self.assertTrue(x["cached_prior_episode_anchor_cleared"])
        flow=self.report["real_eval_rollout_reset_source"]
        self.assertEqual(flow["old_explicit_rollout_reset_targets"],["policy"])
        self.assertTrue({"policy","preprocessor","postprocessor"}.issubset(
            flow["merged_explicit_rollout_reset_targets"]))

if __name__=="__main__":
    unittest.main()
