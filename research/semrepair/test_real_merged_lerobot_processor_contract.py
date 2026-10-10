"""Pin and execute REAL independently merged upstream before/after LeRobot code."""
import unittest

from research.semrepair.real_merged_lerobot_processor_contract import (
    audit_actual_merged_lerobot_source,
    BEFORE_COMMIT,AFTER_COMMIT,BEFORE_SOURCE_BLOB,AFTER_SOURCE_BLOB
)

class ActualMergedLeRobotRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=audit_actual_merged_lerobot_source()

    def test_two_externally_published_byte_exact_versions_and_merge(self):
        r=self.r
        self.assertEqual(r["exact_before_git_commit"],BEFORE_COMMIT)
        self.assertEqual(r["exact_after_merge_commit"],AFTER_COMMIT)
        self.assertEqual(r["verified_before_actual_source_git_blob"],BEFORE_SOURCE_BLOB)
        self.assertEqual(r["verified_after_actual_source_git_blob"],AFTER_SOURCE_BLOB)
        self.assertIn("3425",r["public_issue"])
        self.assertIn("3711",r["actual_MERGED_publisher_PR"])
        self.assertEqual(r["merge_confirmed_date"],"2026-06-03")
        self.assertTrue(r["frozen_original_publisher_train_setup_source_ran"])

    def test_reported_pretrained_relative_rename_failure_is_present_before_and_fixed_after(self):
        r=self.r["scenarios"][0]
        self.assertEqual(r["scenario"],"pretrained_relative_with_renames")
        self.assertFalse(r["old_software_contract"]["valid"])
        self.assertTrue(r["new_software_contract"]["valid"])
        self.assertIsNone(r["before"]["processor_checkpoint_path_actually_used"])
        self.assertEqual(r["after"]["processor_checkpoint_path_actually_used"],
                         "external/PI05-checkpoint")
        overrides=r["after"]["actual_original_factory_kwargs"]["preprocessor_overrides"]
        self.assertTrue(overrides["relative_actions_processor"]["enabled"])
        self.assertEqual(overrides["rename_observations_processor"]["rename_map"],
                         {"raw.camera":"observation.images.left_wrist_0_rgb"})
        self.assertTrue(r["after"]["actual_original_factory_kwargs"][
                        "postprocessor_overrides"]["absolute_actions_processor"]["enabled"])

    def test_pretrained_nonrelative_renames_still_preserve_checkpoint_and_contract(self):
        r=next(x for x in self.r["scenarios"] if
               x["scenario"]=="pretrained_absolute_with_renames")
        self.assertTrue(r["old_software_contract"]["valid"])
        self.assertTrue(r["new_software_contract"]["valid"])

    def test_fresh_initialized_training_does_not_claim_pretrained_provenance(self):
        r=next(x for x in self.r["scenarios"] if
               x["scenario"]=="fresh_relative_no_renames")
        self.assertEqual(r["old_software_contract"]["decision"],
                         "SCRATCH_NO_CHECKPOINT_CONTRACT")
        self.assertEqual(r["new_software_contract"]["decision"],
                         "SCRATCH_NO_CHECKPOINT_CONTRACT")
        self.assertFalse(r["after"]["checkpoint_provenance_preserved"])

    def test_real_three_file_vendor_merge_not_claimed_as_our_patch(self):
        r=self.r
        self.assertEqual(r["actual_robot_task_or_training_steps_run"],0)
        self.assertFalse(r["actual_pretrained_model_loaded"])
        self.assertTrue(r["programmatically_reimplemented_factory"])
        self.assertEqual(r["scenario_count"],6)

if __name__=="__main__":
    unittest.main()
