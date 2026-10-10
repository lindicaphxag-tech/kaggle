from __future__ import annotations
import json,sys,tempfile,unittest
from pathlib import Path
SRC=Path(__file__).resolve().parents[1]/"src"
sys.path.insert(0,str(SRC))
from render_vnext_verified_tables import reject_gain_fields,render

class VerifiedTableTests(unittest.TestCase):
    def fixture(self):
        return {"evidence_sources":{"srcC":{"workflow_run":123,"result_json_sha256":"a"*64}},
          "cohorts":{"C":{"evidence_source_key":"srcC","evidence_tier":"prospective_confirmation",
          "target_information_budget":"Z0","n_participants":3,
          "table_reference_method":"LedoitWolf_LDA",
          "macro_auc":{"LedoitWolf_LDA":.6,"candidate":.7},
          "primary_inference":{"candidate__vs__LedoitWolf_LDA":{
            "mean_delta":.1}}}}}
    def test_delta_derived_from_absolute_auc(self):
        rows,md=render(self.fixture())
        cand=[r for r in rows if r["method"]=="candidate"][0]
        self.assertAlmostEqual(cand["delta_vs_baseline"],.1)
        self.assertIn("+0.1000",md)
    def test_canonical_metadata_without_gain_token_is_allowed(self):
        x=self.fixture();x["manual_derived_effect_fields_forbidden"]=True
        rows,md=render(x)
        self.assertTrue(rows)
        self.assertIn("C",md)

    def test_missing_provenance_key_rejected(self):
        x=self.fixture();del x["cohorts"]["C"]["evidence_source_key"]
        with self.assertRaisesRegex(ValueError,"evidence_source_key"):
            render(x)

    def test_unknown_provenance_key_rejected(self):
        x=self.fixture();x["cohorts"]["C"]["evidence_source_key"]="ghost"
        with self.assertRaisesRegex(ValueError,"evidence_source_key"):
            render(x)

    def test_legacy_source_rejected(self):
        x=self.fixture()
        x["evidence_sources"]["historical_legacy"]={"workflow_run":1}
        x["cohorts"]["C"]["evidence_source_key"]="historical_legacy"
        with self.assertRaisesRegex(ValueError,"Historical/legacy"):
            render(x)

    def test_missing_explicit_table_reference_rejected(self):
        x=self.fixture();del x["cohorts"]["C"]["table_reference_method"]
        with self.assertRaisesRegex(ValueError,"Missing explicit"):
            render(x)

    def test_reference_method_must_exist(self):
        x=self.fixture();x["cohorts"]["C"]["table_reference_method"]="ghost"
        with self.assertRaisesRegex(ValueError,"absent"):
            render(x)

    def test_manual_gain_key_rejected(self):
        x=self.fixture();x["cohorts"]["C"]["Gain"]=.1
        with self.assertRaisesRegex(ValueError,"Manual gain"):
            render(x)
    def test_inconsistent_primary_delta_rejected(self):
        x=self.fixture()
        x["cohorts"]["C"]["primary_inference"]["candidate__vs__LedoitWolf_LDA"]["mean_delta"]=.12
        with self.assertRaisesRegex(ValueError,"Primary delta mismatch"):
            render(x)
    def test_missing_inference_method_rejected(self):
        x=self.fixture()
        x["cohorts"]["C"]["primary_inference"]={"ghost__vs__LedoitWolf_LDA":{"mean_delta":.1}}
        with self.assertRaisesRegex(ValueError,"absent"):
            render(x)

if __name__=="__main__":unittest.main()
