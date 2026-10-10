"""Actual LeRobot #3425 merge-adjacent multi-file bounded program synthesis."""
import unittest
from research.semrepair.verified_multifile_source_synthesis import (
    synthesize_merged_issue_contract,OPERATORS,TEST_CASES
)


class ActualLeRobotMergedMultiFileRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=synthesize_merged_issue_contract()

    def test_real_publisher_merged_source_not_fabricated(self):
        x=self.report
        self.assertIn("/3711",x["publisher_merge"])
        self.assertFalse(x["publisher_upstream_patch_created_by_our_research"])
        self.assertEqual(x["catalog_program_count"],16)
        self.assertEqual(x["per_program_actual_train_source_conditions"],7)
        self.assertEqual(len(x["original_upstream_before_Git_blobs"]),3)
        self.assertEqual(len(x["original_upstream_merged_Git_blobs"]),3)
        self.assertTrue(x["original_merged_reference"]["cross_file_contract_pass"])
        self.assertFalse(x["publisher_actual_full_model_training_executed"])

    def test_only_four_cross_file_operations_together_pass_full_contract(self):
        x=self.report
        self.assertEqual(x["complete_source_contract_candidate_count"],1)
        self.assertEqual(x["minimum_cost_licensed_operator_set"],list(OPERATORS))
        self.assertEqual(x["minimum_cost_over_finite_catalog"],5)
        chosen=next(v for v in x["all_candidate_programs_and_outcomes"]
                    if v["accepted"])
        self.assertEqual(chosen["number_of_source_files_changed"],3)
        self.assertEqual(chosen["source_contract"]["real_train_cases"][0]["actual_pretrained_processor_path"],
                         "published/path")
        self.assertTrue(chosen["source_contract"]["fresh_without_checkpoint_rename_rejected"])

    def test_weak_single_trigger_tests_do_not_certify_full_multifile_semantics(self):
        x=self.report
        self.assertGreater(x["one_source_contract_test_plausible_count"],
                           x["complete_source_contract_candidate_count"])
        wrong=[v for v in x["all_candidate_programs_and_outcomes"]
               if v["source_contract"]["single_trigger_contract_pass"] and
               not v["source_contract"]["cross_file_contract_pass"]]
        self.assertTrue(wrong)
        self.assertTrue(all("fresh_rename_guard" not in v["operators"] for v in wrong))

    def test_no_merged_vendor_permission_claimed(self):
        x=self.report
        self.assertTrue(x["unlicensed_source_patch_rejected"])
        self.assertFalse(x["publisher_upstream_patch_created_by_our_research"])
        self.assertIn("STUBbed processor factory",x["scope"])

    def test_without_cross_file_license_cannot_claim_autofix(self):
        no_guard=synthesize_merged_issue_contract(
            licensed_ops=("preserve_checkpoint","relative_pair","rename_registry"))
        self.assertIsNone(no_guard["minimum_cost_licensed_operator_set"])
        self.assertEqual(no_guard["complete_source_contract_candidate_count"],1)
        no_registry=synthesize_merged_issue_contract(
            licensed_ops=("preserve_checkpoint","relative_pair","fresh_rename_guard"))
        self.assertIsNone(no_registry["minimum_cost_licensed_operator_set"])

    def test_invalid_licensing_safely_rejected(self):
        for ops in [("preserve_checkpoint","preserve_checkpoint"),
                    ("rename_map_nonexistent",),
                    42]:
            with self.subTest(ops=ops),self.assertRaises((TypeError,ValueError)):
                synthesize_merged_issue_contract(licensed_ops=ops)

if __name__=="__main__":
    unittest.main()
