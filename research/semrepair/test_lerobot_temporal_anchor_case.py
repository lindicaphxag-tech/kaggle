"""Independent real LeRobot source kernel: history-sensitive robot ABI fault."""
import unittest
import torch
from research.semrepair.lerobot_temporal_anchor_case import (
    run_actual_lerobot_anchor_case,load_actual_upstream_function_pair,
    UPSTREAM_COMMIT,UPSTREAM_GIT_BLOB_SHA1,MASK
)

class ExternalLeRobotSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=run_actual_lerobot_anchor_case()

    def test_real_vendor_functions_at_exact_upstream_source_pin(self):
        r=self.result
        self.assertEqual(r["exact_external_git_commit"],UPSTREAM_COMMIT)
        self.assertEqual(r["actual_external_source_git_blob"],UPSTREAM_GIT_BLOB_SHA1)
        self.assertEqual(len(r["verified_external_source_bytes_sha256"]),64)
        self.assertEqual(r["number_of_real_upstream_original_functions_executed"],2)
        self.assertTrue(r["all_comparisons_use_identical_original_upstream_conversion_kernel"])

    def test_true_error_requires_original_hidden_chunk_anchor_under_current_only_information(self):
        r=self.result
        self.assertEqual(r["scenario_worlds"],128)
        self.assertEqual(r["nonindependent_action_rows"],1024)
        self.assertEqual(r["physical_robot_tasks_executed"],0)
        self.assertLess(r["max_fixed_error"],1e-12)
        self.assertGreater(r["min_achievable_current_only_worst_case_error_across_worlds"],0.01)
        self.assertGreater(r["mean_minimax_lower_bound_on_current_only_error"],0.1)
        self.assertAlmostEqual(r["mean_author_injected_stale_reference_error"],
                               r["mean_minimax_lower_bound_on_current_only_error"],places=11)
        self.assertFalse(r["vendor_reported_software_bug"])

    def test_two_hidden_worlds_observationally_identical_but_different_target(self):
        rows=self.result["raw_worlds"]
        for a,b in zip(rows[::2],rows[1::2]):
            self.assertEqual(a["observed_live_state"],b["observed_live_state"])
            self.assertEqual(a["original_relative_chunk"],b["original_relative_chunk"])
            self.assertNotEqual(a["hidden_reference_anchor"],b["hidden_reference_anchor"])
            self.assertAlmostEqual(a["author_buggy_actual_max_error"],
                                   b["author_buggy_actual_max_error"],places=10)

    def test_upstream_gripper_mask_and_relative_roundtrip_without_mocked_tensor_math(self):
        to_rel,to_abs,_=load_actual_upstream_function_pair()
        base=torch.tensor([[.2,-.3,0.,.1,.2,0.,.9]],dtype=torch.float64)
        planned=torch.tensor([[[.8,0.,-.3,.5,0.,.1,.45],
                               [.9,.4,.3,.5,0.,.1,.66]]],dtype=torch.float64)
        relative=to_rel(planned,base,MASK)
        self.assertTrue(torch.allclose(relative[...,6],planned[...,6]))
        real=to_abs(relative,base,MASK)
        torch.testing.assert_close(real,planned,atol=1e-12,rtol=0)

    def test_temporal_annotation_not_claimed_as_public_vendor_fault(self):
        r=self.result
        self.assertTrue(r["fault_is_author_mutation_of_callsite"])
        self.assertIn("CAN tie source repair",r["scope_note"])
        self.assertEqual(r["chunk_steps"],8)

if __name__=="__main__":
    unittest.main()
