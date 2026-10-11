import unittest
from research.semrepair.mujoco_wrong_repair_counterfactual import (
    run_counterfactual, N_EXCITATIONS,
)


class NativeWrongRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_counterfactual()

    def test_paired_counts_and_frozen_command_commitment(self):
        self.assertEqual(self.r["paired_fault_sequence_cells"], 3*N_EXCITATIONS)
        self.assertEqual(len(self.r["command_draw_sha256"]),64)

    def test_bad_source_has_real_physical_error(self):
        self.assertGreater(self.r["min_original_fault_error"], 1e-4)

    def test_unauthorized_patch_indistinguishable_in_this_dynamics(self):
        self.assertEqual(
            self.r["unauthorized_patches_with_matching_nominal_trajectory"],
            3*N_EXCITATIONS)
        self.assertLess(self.r["max_wrong_interface_repair_error"], 2e-8)
        self.assertLess(self.r["max_correct_repair_error"], 2e-8)

    def test_no_semantic_identity_confusion(self):
        self.assertTrue(all(
            x["authorized_change"]!=x["unauthorized_change"]
            and not x["semantic_patch_authority_equal"]
            for x in self.r["results"]
        ))


if __name__ == "__main__":
    unittest.main()
