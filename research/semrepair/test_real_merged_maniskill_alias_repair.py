"""Second independent repository MERGED upstream defect, action output parity misses mutation."""
import unittest
from research.semrepair.real_merged_maniskill_alias_repair import (
    run_real_publisher_alias_repair
)


class RealMergedManiSkillAliasRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=run_real_publisher_alias_repair()

    def test_true_external_merged_source_both_commit_blobs(self):
        x=self.r
        self.assertIn("689",x["upstream_publisher_merged_PR"])
        self.assertEqual(x["exact_after_commit"],
                         "84a5026098a5e21c356b93df75937d6b880b7d5e")
        self.assertEqual(x["tested_input_configs"],192)
        self.assertTrue(x["real_publisher_original_and_merged_torch_methods_executed"])
        self.assertTrue(x["source_level_AST_alias_flow_found"])
        self.assertEqual(x["actual_physics_rollouts"],0)

    def test_all_four_source_variants_PASS_output_only_but_two_break_caller_owned_inputs(self):
        v=self.r["aggregate"]
        self.assertEqual(len(v),4)
        self.assertTrue(all(x["action_output_equal_original_publisher_merged_all_192"]
                            for x in v.values()))
        self.assertEqual(v["original"]["violated_caller_input_immutability_high_norm"],96)
        self.assertEqual(v["clone_return_output"]["violated_caller_input_immutability_high_norm"],96)
        self.assertEqual(v["clone_rotation_view"]["violated_caller_input_immutability_high_norm"],0)
        self.assertEqual(v["clone_complete_input"]["violated_caller_input_immutability_high_norm"],0)

    def test_conditional_fault_unseen_in_zero_normalization_branch(self):
        self.assertTrue(all(
            x["violated_caller_input_immutability_low_norm"]==0
            for x in self.r["aggregate"].values()))

    def test_AST_synthesis_minimal_slice_copy_repair_not_full_input_copy(self):
        r=self.r
        self.assertEqual(r["selected_lowest_copy_volume_author_template"],
                         "clone_rotation_view")
        self.assertEqual(r["aggregate"]["clone_rotation_view"]["copy_elements_per_case"],3)
        self.assertEqual(r["aggregate"]["clone_complete_input"]["copy_elements_per_case"],6)
        self.assertTrue(r["candidate_selection_did_not_consult_publisher_after_source"])
        self.assertTrue(r["input_copy_volume_proxy_only_not_memory_profiler"])

    def test_genuine_counterexample_is_input_mutation_not_action_reward(self):
        for row in self.r["raw_cases"]:
            for name,x in row["method_outcomes"].items():
                self.assertLess(x["max_action_output_error_vs_real_merged"],1e-12)
                if name=="original" and row["rotation_norm_gt_one"]:
                    self.assertTrue(x["input_modified"])

if __name__=="__main__":
    unittest.main()
