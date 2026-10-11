import unittest
from research.semrepair.native_planar_task_repair import (
    run_planar_task_assay, matvec, SWAP, NEGATE, ROT_POS, ROT_NEG,
    N_TARGETS, CASES, frozen_targets
)

class PlanarTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_planar_task_assay()

    def test_matrix_pair_inverse(self):
        vec = (0.31, -0.73)
        self.assertEqual(matvec(SWAP, matvec(SWAP, vec)), vec)
        self.assertEqual(matvec(NEGATE, matvec(NEGATE, vec)), vec)
        self.assertEqual(matvec(ROT_NEG, matvec(ROT_POS, vec)), vec)

    def test_targets_frozen(self):
        self.assertEqual(len(frozen_targets()), N_TARGETS)
        self.assertEqual(len(set(frozen_targets())), N_TARGETS)

    def test_correct_source_patch_matches_reference(self):
        r = self.report["results"]
        for name, _, _, owner in CASES:
            expected = r[f"{name}:source-owner-oracle"]
            self.assertEqual(expected["success"], r["reference:untouched"]["success"])
            self.assertLess(expected["mean_reference_terminal_distance"], 1e-8)

    def test_single_component_changes_task_outcome(self):
        r = self.report["results"]
        self.assertTrue(any(
            r[f"{name}:producer-only"]["success"] < r[f"{name}:source-owner-oracle"]["success"]
            for name, _, _, owner in CASES if owner == "atomic"))

    def test_all_original_trial_records_preserved(self):
        self.assertEqual(len(self.report["rows"]), N_TARGETS * (1 + 5*len(CASES)))
        self.assertEqual(self.report["independent_robot_families"], 1)
        self.assertTrue(self.report["source_owner_oracle_uses_truth"])

if __name__ == "__main__":
    unittest.main()
