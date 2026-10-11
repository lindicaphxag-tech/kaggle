"""Natural merged source issues counted ONCE each, not per tensor or test."""
import unittest

from research.semrepair.cross_repository_semantic_repair_matrix import (
    run_small_natural_source_repair_corpus
)


class FourActualPublisherFixesSoftwareRepairCaseStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=run_small_natural_source_repair_corpus()

    def test_two_real_maintainer_repositories_and_four_independent_merged_defects(self):
        x=self.r
        self.assertEqual(x["independently_merged_natural_defects"],4)
        self.assertEqual(x["publisher_repository_count"],2)
        self.assertEqual(x["authored_typed_program_variants_total"],34)
        self.assertTrue(x["upstream_fixes_are_INDEPENDENT_prior_art_not_our_merges"])
        self.assertEqual(x["real_robot_hardware_or_full_neural_VLA_trials"],0)

    def test_each_real_bug_exhibits_cross_boundary_semantic_false_accepts(self):
        for r in self.r["rows"]:
            with self.subTest(upstream=r["independent_upstream_issue"]):
                self.assertGreater(r["weakly_accepted_but_contract_invalid"],0)
                self.assertGreaterEqual(r["complete_cross_boundary_contract_valid"],1)
                self.assertTrue(r["actual_external_MERGED_patch"].startswith("https://github.com/"))
                self.assertTrue(r["publishers_did_NOT_adopt_SemRepair"])

    def test_four_distinct_contract_dimensions_are_measured(self):
        records={r["independent_upstream_issue"]:r for r in self.r["rows"]}
        self.assertIn("provenance",records["LeRobot #3425/#3711"]["contract_semantics"])
        self.assertIn("cache reset",records["LeRobot #4656"]["contract_semantics"])
        self.assertIn("joint permutation",records["LeRobot #4452"]["contract_semantics"])
        self.assertIn("immutability",records["ManiSkill #689"]["contract_semantics"])
        self.assertEqual(records["ManiSkill #689"]["weak_local_test_plausible"],4)
        self.assertEqual(records["ManiSkill #689"]["complete_cross_boundary_contract_valid"],2)
        self.assertEqual(records["LeRobot #4656"]["weak_local_test_plausible"],6)
        self.assertEqual(records["LeRobot #4656"]["complete_cross_boundary_contract_valid"],1)

    def test_no_generic_program_repair_or_independent_model_count_claim(self):
        x=self.r
        self.assertTrue(x["cross_issue_pooled_percentage_is_NOT_generalizable"])
        self.assertTrue(x["all_original_source_runners_are_AST_extracted_with_some_stubbed_interfaces"])
        self.assertEqual(
            x["weak_local_test_false_accepts_total"],
            x["weak_test_plausible_programs_total"]-
            x["full_semantics_valid_programs_total"])

if __name__=="__main__":
    unittest.main()
