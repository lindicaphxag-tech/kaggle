"""Integration smoke test for the 29-person source-lock-then-target-join runner.

Fakes only the publicly downloadable EEG matrices and the computationally
expensive source selector. Exercises the REAL archive/receipt parser,
scope split, lambda lock, lookup and final AUC arithmetic. No fake result
is ever counted as an EEG outcome or published performance.
"""
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_proper_predictive_posthoc as runner


class FrozenProgramIntegration(unittest.TestCase):
    def test_all_29_lock_and_target_join_with_mocked_source_chooser(self):
        folds=[]
        for sid in range(1,30):
            folds.append({
                "target_subject":sid,
                "source_path":{"feasible_full_lambdas":[0.1,0.2],
                               "lambda_covariance_full":0.1},
                "target_auc_path_by_lambda":{"0.1":0.65,"0.2":0.75},
                "auc":{"source_cov_nll_full_domain_LDA":0.65,
                       "source_macro_full_domain_LDA":0.73}
            })
        receipt={
            "dataset":"Shin2017A","n_subjects":29,
            "target_information_track":"Z0",
            "participant_receipt":[{"participant":sid,"trials":60}
                                   for sid in range(1,30)],
            "folds":folds
        }
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w") as z:
            z.writestr(runner.RESULT,json.dumps(receipt))
        blob=buf.getvalue()
        y=np.tile(np.repeat([0,1],30),29)
        g=np.repeat(np.arange(1,30),60)
        C=np.zeros((1740,2,2))
        def fake_select(X,y,g,allowed):
            self.assertEqual(len(set(g)),28)
            self.assertEqual(list(allowed),[0.1,0.2])
            return .2,{"0.1":1.0,"0.2":.7}
        with tempfile.TemporaryDirectory() as d:
            zpath=Path(d)/"fake.zip";zpath.write_bytes(blob)
            with (patch.object(runner,"ART_SHA",hashlib.sha256(blob).hexdigest()),
                  patch.object(runner.old,"load_dataset",return_value=(C,y,g,None,None,None,None)),
                  patch.object(runner,"select_proper_source_nll",side_effect=fake_select)):
                runner.run(str(zpath),str(Path(d)/"out"))
            base=Path(d)/"out"
            receipt_out=json.loads((base/"SOURCE_CHOICES_FROZEN_BEFORE_TARGET_OUTCOME_JOIN.json").read_text())
            result=json.loads((base/"POSTHOC_PROPER_PREDICTIVE_VS_FROZEN_RESULTS.json").read_text())
            self.assertEqual(len(receipt_out),29)
            self.assertTrue(all(v["original_class_centered_lambda"]==.1 for v in receipt_out))
            self.assertTrue(all(v["new_source_proper_predictive_lambda"]==.2 for v in receipt_out))
            self.assertAlmostEqual(result["mean_source_macro_minus_proper_AUC"],-.02,places=12)
            self.assertAlmostEqual(result["mean_class_centered_minus_proper_AUC"],-.1,places=12)
            self.assertEqual(result["n"],29)
            self.assertTrue(result["no_new_confirmatory_p_value"])


if __name__=="__main__":
    unittest.main()
