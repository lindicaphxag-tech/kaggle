import json,sys,tempfile,unittest
from pathlib import Path
SRC=Path(__file__).resolve().parents[1]/"src"
sys.path.insert(0,str(SRC))
from submission_claim_gate_v2 import audit_text,run,REQUIRED_EVIDENCE_KEYS

def evidence(**updates):
    d={k:False for k in REQUIRED_EVIDENCE_KEYS}
    d.update(updates)
    return d

class ClaimGateTests(unittest.TestCase):
    def test_validation_free_always_blocked(self):
        f=audit_text("Our method is validation-free.",evidence())
        self.assertEqual(len(f),1)
        self.assertEqual(f[0]["rule"],"no_validation_free")

    def test_systematic_claim_needs_multicohort_and_table_integrity(self):
        txt="Default regularisation is systematically mis-specified."
        f=audit_text(txt,evidence())
        self.assertEqual(len(f),1)
        self.assertEqual(set(f[0]["missing_evidence"]),
                         {"three_cohort_full_domain_confirmed",
                          "historical_table_integrity_resolved",
                          "true_predata_frontier_two_cohort_complete"})
        ok=evidence(three_cohort_full_domain_confirmed=True,
                    historical_table_integrity_resolved=True,
                    true_predata_frontier_two_cohort_complete=True)
        self.assertEqual(audit_text(txt,ok),[])

    def test_failed_algorithm_superiority_is_never_reenabled(self):
        alltrue={k:True for k in REQUIRED_EVIDENCE_KEYS}
        f=audit_text("CFCTS outperforms nested tuning.",alltrue)
        self.assertEqual(len(f),1)
        self.assertEqual(f[0]["rule"],"new_algorithm_superiority")

    def test_representation_change_is_prospective_gated(self):
        txt="Regularisation alters the representation conclusion."
        self.assertEqual(len(audit_text(txt,evidence())),1)
        e=evidence(prospective_representation_interaction_confirmed=True)
        self.assertEqual(audit_text(txt,e),[])

    def test_hidden_filterbank_gain_remains_forbidden_after_interaction(self):
        alltrue={k:True for k in REQUIRED_EVIDENCE_KEYS}
        f=audit_text("Regularisation reveals a hidden filter-bank advantage.",alltrue)
        self.assertEqual(len(f),1)
        self.assertEqual(f[0]["rule"],"hidden_filterbank_gain_refuted")

    def test_sota_requires_matched_benchmark_panel(self):
        txt="This is state-of-the-art."
        self.assertEqual(len(audit_text(txt,evidence())),1)
        e=evidence(strong_same_budget_baselines_complete=True,
                   external_benchmark_panel_complete=True)
        self.assertEqual(audit_text(txt,e),[])

    def test_s021_cannot_retroactively_confirm_frontier(self):
        alltrue={k:True for k in REQUIRED_EVIDENCE_KEYS}
        txt="S021-S040 prospectively confirmed the Validity Frontier."
        f=audit_text(txt,alltrue)
        self.assertEqual(len(f),1)
        self.assertEqual(f[0]["rule"],"s021_chronology_overclaim")

    def test_source_discriminability_prospective_claim_is_gated(self):
        txt="The prospective source-discriminability result was positive."
        self.assertEqual(len(audit_text(txt,evidence())),1)
        e=evidence(prospective_source_discriminability_target_confirmed=True)
        self.assertEqual(audit_text(txt,e),[])

    def test_predata_frontier_claim_requires_predata_gate(self):
        txt="True pre-data confirmation supports the Validity Frontier."
        self.assertEqual(len(audit_text(txt,evidence())),1)
        e=evidence(true_predata_frontier_one_cohort_complete=True)
        self.assertEqual(audit_text(txt,e),[])

    def test_mean_tail_benefit_requires_two_cohort_promotion_gate(self):
        txt="The mean-tail-fidelity selector reduces harm."
        self.assertEqual(len(audit_text(txt,evidence())),1)
        one=evidence(mean_tail_fidelity_predata_one_cohort_complete=True)
        self.assertEqual(len(audit_text(txt,one)),1)
        promoted=evidence(mean_tail_fidelity_predata_two_cohort_safe=True)
        self.assertEqual(audit_text(txt,promoted),[])

    def test_external_superiority_requires_frozen_two_cohort_significance(self):
        txt="Both external cohorts confirmed superiority with significant results."
        f=audit_text(txt,evidence(
            true_predata_external_directionally_consistent_two_cohort=True))
        self.assertEqual(len(f),1)
        self.assertEqual(f[0]["rule"],"external_superiority_overclaim")
        e=evidence(true_predata_external_primary_family_significant_two_cohort=True)
        self.assertEqual(audit_text(txt,e),[])

    def test_directional_external_replication_has_own_gate(self):
        txt="The target-effect direction was replicated across external cohorts."
        self.assertEqual(len(audit_text(txt,evidence())),1)
        e=evidence(true_predata_external_directionally_consistent_two_cohort=True)
        self.assertEqual(audit_text(txt,e),[])

    def test_two_cohort_frontier_claim_requires_corrected_analysis(self):
        txt="Both cohorts support a material-fragility Validity Frontier."
        self.assertEqual(len(audit_text(txt,evidence())),1)
        e=evidence(corrected_material_fragility_frontier_two_cohort_complete=True)
        self.assertEqual(audit_text(txt,e),[])

    def test_universal_frontier_delta_never_allowed(self):
        alltrue={k:True for k in REQUIRED_EVIDENCE_KEYS}
        f=audit_text("We identify a single universal delta threshold.",alltrue)
        self.assertEqual(len(f),1)
        self.assertEqual(f[0]["rule"],"universal_frontier_threshold")

    def test_simple_pn_law_never_allowed(self):
        alltrue={k:True for k in REQUIRED_EVIDENCE_KEYS}
        f=audit_text("The mismatch monotonically worsens with p/n.",alltrue)
        self.assertEqual(len(f),1)
        self.assertEqual(f[0]["rule"],"simple_pn_law_refuted")

    def test_lexicographic_superiority_never_allowed(self):
        alltrue={k:True for k in REQUIRED_EVIDENCE_KEYS}
        f=audit_text("The discriminability-first selector outperforms source macro.",alltrue)
        self.assertEqual(len(f),1)
        self.assertEqual(f[0]["rule"],"discriminability_first_superiority")

    def test_missing_evidence_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            ep=Path(td)/"e.json";mp=Path(td)/"m.md"
            ep.write_text("{}");mp.write_text("narrow claim")
            with self.assertRaises(ValueError):
                run(ep,[mp])

    def test_narrow_claim_passes_under_science_hold(self):
        with tempfile.TemporaryDirectory() as td:
            ep=Path(td)/"e.json";mp=Path(td)/"m.md";out=Path(td)/"out.json"
            ep.write_text(json.dumps(evidence()))
            mp.write_text("We test source-only regularisation objective mismatch in the studied tangent-space LDA pipeline.")
            z=run(ep,[mp],out)
            self.assertTrue(z["submission_claims_allowed"])
            self.assertTrue(out.exists())

if __name__=="__main__":unittest.main()
