"""External merged #4656 two-source-file temporal software bug vs weak roundtrip."""
import unittest
from research.semrepair.real_merged_lerobot_4656_multifile_patch_search import (
    run_real_4656_joint_temporal_source_repairs
)


class TrueExternalMergedLeRobotTemporalTwoFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=run_real_4656_joint_temporal_source_repairs()

    def test_two_real_external_git_blob_files_and_true_MERGED_upstream_reference(self):
        x=self.x
        self.assertIn("/4656",x["genuinely_publisher_MERGED_PR"])
        self.assertFalse(x["author_claims_upstream_patch_was_ours"])
        self.assertTrue(x["actual_source_evaluator_executes_original_AST"])
        self.assertEqual(x["native_robot_task_trials"],0)
        self.assertEqual(x["typed_two_file_candidate_program_count"],12)
        self.assertEqual(len(x["original_before_source_git_blobs"]),2)

    def test_weak_roundtrip_accepts_five_unsafe_programs_but_joint_contract_only_one(self):
        x=self.x
        self.assertEqual(x["single_roundtrip_test_plausible_program_count"],6)
        self.assertEqual(x["weak_plausible_but_semantically_invalid_program_count"],5)
        self.assertEqual(x["full_crosscomponent_semantics_valid_count"],1)

    def test_actual_two_source_file_fix_required(self):
        x=self.x["minimum_cost_full_semantics_candidate"]
        self.assertEqual(x["converter_AST_variant"],"frame0_both")
        self.assertTrue(x["eval_reset_AST_modified"])
        self.assertEqual(x["source_files_modified"],2)
        self.assertEqual(x["declared_source_edit_cost"],4)
        self.assertTrue(x["actual_cross_component_frame0_contract"])
        self.assertTrue(x["actual_new_epoch_cache_reset_contract"])
        self.assertEqual(x["actual_vendor_AST_reset_calls_executed"],
                         ["policy","preprocessor","postprocessor"])

    def test_one_file_correct_frame_but_not_episode_reset_is_rejected(self):
        r=next(x for x in self.x["all_program_variants"]
               if x["converter_AST_variant"]=="frame0_both"
               and not x["eval_reset_AST_modified"])
        self.assertTrue(r["weak_roundtrip_only_plausible"])
        self.assertTrue(r["actual_cross_component_frame0_contract"])
        self.assertFalse(r["actual_new_epoch_cache_reset_contract"])
        self.assertFalse(r["full_source_semantic_contract"])
        self.assertEqual(r["actual_vendor_AST_reset_calls_executed"],["policy"])

    def test_publisher_actual_reset_contract_verified_independently(self):
        self.assertEqual(self.x["authentic_publisher_actual_reset_statements"],
                         ["policy","preprocessor","postprocessor"])

if __name__=="__main__":
    unittest.main()
