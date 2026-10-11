"""Genuine upstream controller software, typed repair grammar and CEGIS limits."""
import unittest
from research.semrepair.active_contract_patch_synthesis import (
    run_constrained_active_source_patch_experiment,
    GRAMMAR,N_FINAL_HELDOUT,
    _full_domain_contract,COUNTEREXAMPLE_BANK
)


class BoundedCounterexampleGuidedPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=run_constrained_active_source_patch_experiment()

    def test_all_original_upstream_source_hashes_and_128_asymmetric_heldout(self):
        r=self.r
        self.assertEqual(len(r["vendor_original_controller_source_SHA256"]),64)
        self.assertEqual(len(r["vendor_original_helper_source_SHA256"]),64)
        self.assertIn("1469",r["vendor_issue"])
        self.assertEqual(r["candidate_count"],7)
        self.assertEqual(r["heldout_distinct_asymmetric_config_action_examples"],N_FINAL_HELDOUT)
        self.assertEqual(r["new_native_robot_task_episodes"],0)

    def test_one_well_chosen_counterexample_eliminates_all_overfit_training_patches(self):
        r=self.r
        self.assertGreaterEqual(r["training_plausible_candidate_count"],5)
        self.assertGreaterEqual(r["training_overfit_wrong_candidate_count"],4)
        self.assertEqual(r["active_probes_selected_count"],1)
        self.assertEqual(r["active_remaining_candidate_count"],1)
        self.assertEqual(r["active_wrong_candidate_count"],0)
        self.assertEqual(r["symbolically_certified_remaining_patch_count"],1)
        self.assertEqual(r["selected_minimal_verified_AST_patch"],"proposed_upper")
        self.assertGreaterEqual(r["active_probe"]["wrong_plausible_eliminated"],4)

    def test_unseen_asymmetric_controller_configs_expose_training_overfitting(self):
        r=self.r
        results={x["patch_id"]:x for x in r["heldout_result_per_candidate"]}
        self.assertEqual(results["proposed_upper"]["cases_passed"],128)
        self.assertEqual(results["proposed_upper"]["cases_failed"],0)
        self.assertLess(results["proposed_upper"]["max_action_error"],1e-12)
        for bad in ("original_negative_lower","negate_lower",
                    "absolute_lower","hardcode_training_upper",
                    "half_span","average_positive_gains"):
            with self.subTest(bad=bad):
                self.assertGreater(results[bad]["cases_failed"],0)
                self.assertFalse(results[bad]["symbolic_contract_true"])

    def test_actual_torch_patch_ast_has_real_semantics_and_no_untrusted_activation(self):
        r=self.r
        self.assertIn("self.config.rot_upper",r["selected_source_executable_AST"])
        self.assertEqual(r["source_permission_decision"],
                         "CANDIDATE_SOURCE_PATCH_REQUIRES_SEPARATE_VENDOR_APPROVAL")
        self.assertFalse(r["real_manufacturer_signer_permission_obtained"])
        self.assertTrue(r["patch_author_generated_not_installed_or_merged"])
        self.assertGreater(r["actual_upstream_torch_kernel_executions"],800)

    def test_symbolic_proof_discriminates_asymmetric_rot_limits_without_sampling(self):
        result={name:_full_domain_contract(expr) for name,expr,c in GRAMMAR}
        self.assertEqual([name for name in result if
                          result[name]["all_domain_symbolic_identity"]],
                         ["proposed_upper"])
        self.assertTrue(all(r["declared_independent_source_domain"].startswith("rot_lower<0")
                            for r in result.values()))

    def test_no_source_permission_refuses_even_if_correct_candidate_exists(self):
        r=run_constrained_active_source_patch_experiment(
            licensed_patch_ids=("hardcode_training_upper","negate_lower"))
        self.assertIsNone(r["selected_minimal_verified_AST_patch"])
        self.assertEqual(r["source_permission_decision"],
                         "REFUSE_NO_AUTHORIZED_CONTRACT_VALID_SOURCE_PATCH")
        self.assertEqual(r["active_wrong_candidate_count"],0)

    def test_invalid_permission_scope_input_fails_closed(self):
        with self.assertRaises(ValueError):
            run_constrained_active_source_patch_experiment(licensed_patch_ids=(123,))

if __name__=="__main__":
    unittest.main()
