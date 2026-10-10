"""Third independently merged true LeRobot action middleware bug, PR #4452."""
import unittest

from research.semrepair.real_merged_lerobot_action_order_regression import (
    run_real_merged_compensating_order_issue
)

class GenuineCompensatingSourceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=run_real_merged_compensating_order_issue()

    def test_exact_frozen_publisher_versions_and_no_false_external_credit(self):
        x=self.r
        self.assertIn("/4452",x["real_independently_MERGED_upstream_PR"])
        self.assertEqual(x["external_merge_commit"],
                         "0120909ed1e1df860217fefa78755015b0315744")
        self.assertEqual(len(x["Git_blobs"]),5)
        self.assertEqual(len(x["Git_content_SHA256"]),5)
        self.assertTrue(x["no_robot_or_pretrained_policy_rollout_executed"])

    def test_true_before_program_mislabels_bimanual_command_vectors(self):
        x=self.r
        self.assertEqual(x["distinct_source_regression_samples"],96)
        self.assertGreater(x["baseline_original_mean_action_error"],20)
        self.assertEqual(x["source_corrected_max_action_error"],0)
        for r in x["raw_cases"]:
            self.assertGreater(r["actual_original_sync_max_abs_error"],20)
            self.assertLess(r["actual_merged_sync_max_abs_error"],1e-12)

    def test_real_cross_component_context_aligns_both_checkpoint_arrays(self):
        x=self.r
        self.assertTrue(x["real_merged_context_alignment_and_fail_closed_guard_executed"])
        self.assertTrue(x["correct_automatic_state_action_name_reordering"])
        self.assertTrue(x["true_state_action_permutation_refused"])
        self.assertTrue(x["different_state_sensor_feature_set_not_refused"])

    def test_negative_original_could_be_correct_for_already_aligned_names(self):
        x=self.r
        self.assertTrue(x["already_aligned_original_remains_correct"])
        self.assertTrue(x["real_vendor_make_robot_action_implementation_executed"])

if __name__=="__main__":
    unittest.main()
