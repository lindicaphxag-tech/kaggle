import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[1]/"src"
sys.path.insert(0,str(SRC))
from participant_inference import (
    paired_deltas,signflip_pvalue,bootstrap_mean_ci,bootstrap_mean_direction,
    leave_one_participant_influence,holm,lower_cvar,
    hierarchical_equal_dataset_bootstrap,analyse)

class ParticipantInferenceTests(unittest.TestCase):
    def test_exact_signflip_null_symmetry(self):
        p,mode,draws=signflip_pvalue(np.array([.1,-.1]))
        self.assertEqual(mode,"exact")
        self.assertEqual(draws,4)
        self.assertAlmostEqual(p,1.0)

    def test_exact_signflip_strong_same_direction(self):
        p,mode,draws=signflip_pvalue(np.ones(5)*.1)
        self.assertEqual(draws,32)
        self.assertAlmostEqual(p,2/32)

    def test_holm_monotone_and_ordered(self):
        x=holm([.01,.04,.02])
        self.assertAlmostEqual(x[0],.03)
        self.assertAlmostEqual(x[2],.04)
        self.assertAlmostEqual(x[1],.04)

    def test_bootstrap_is_reproducible(self):
        a=bootstrap_mean_ci(np.array([.01,.02,.03]),seed=7,reps=1000)
        b=bootstrap_mean_ci(np.array([.01,.02,.03]),seed=7,reps=1000)
        self.assertEqual(a,b)

    def test_bootstrap_direction_stability(self):
        z=bootstrap_mean_direction(np.array([.1,.1,.05,.02]),seed=4,reps=500)
        self.assertGreater(z["fraction_mean_gt_zero"],.99)
        self.assertIn("not a posterior",z["interpretation"])

    def test_leave_one_participant_influence(self):
        stable=leave_one_participant_influence(np.array([.05,.04,.03,.02]))
        self.assertTrue(stable["loo_mean_sign_stable"])
        self.assertFalse(stable["any_single_participant_changes_mean_sign"])
        fragile=leave_one_participant_influence(np.array([.30,-.10,-.10,-.10]))
        self.assertTrue(fragile["any_single_participant_changes_mean_sign"])

    def test_tail_risk_metrics(self):
        d=np.array([-.10,-.05,.01,.02,.03,.04,.05,.06,.07,.08])
        c10,k10=lower_cvar(d,.10)
        c25,k25=lower_cvar(d,.25)
        self.assertEqual(k10,1)
        self.assertAlmostEqual(c10,-.10)
        self.assertEqual(k25,3)
        self.assertAlmostEqual(c25,(-.10-.05+.01)/3)

    def test_equal_dataset_weighting(self):
        z=hierarchical_equal_dataset_bootstrap(
            [np.array([.1,.1]),np.array([-.1]*20)],seed=3,reps=500)
        self.assertAlmostEqual(z["equal_dataset_mean_delta"],0.0)
        self.assertEqual(z["datasets"],2)

    def test_extracts_loso_participant_deltas(self):
        data={"folds":[
            {"target_subject":1,"auc":{"A":.5,"B":.6}},
            {"target_subject":2,"auc":{"A":.7,"B":.65}}]}
        d=paired_deltas(data,"A","B")
        np.testing.assert_allclose(d,[.1,-.05])

    def test_analyse_records_overlap_limit(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"x.json"
            p.write_text(json.dumps({"dataset":"D","folds":[
                {"target_subject":1,"auc":{"A":.5,"B":.6}},
                {"target_subject":2,"auc":{"A":.7,"B":.65}}]}))
            out=Path(td)/"out.json"
            z=analyse([p],"A","B",out)
            self.assertFalse(z["shared_loso_training_uncertainty_included"])
            self.assertTrue(z["participant_is_inferential_unit"])
            self.assertTrue(out.exists())

    def test_bad_pvalues_refused(self):
        for bad in ([1.1,.2],[-.1,.2],[]):
            with self.assertRaises(ValueError):holm(bad)

if __name__=="__main__":unittest.main()
