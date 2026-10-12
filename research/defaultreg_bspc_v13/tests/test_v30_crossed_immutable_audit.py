"""Unit-level falsifiers for V30 factorial AUC decomposition and immutable evidence."""
import pathlib
import sys
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"src"))
from verify_v30_crossed_immutable_artifact import contrasts

class CrossedModelSelectorTests(unittest.TestCase):
    def test_full_effect_has_two_equal_factorial_paths(self):
        for values in ((.5,.65,.60,.76),(.8,.8,.9,.9),(.4,.5,.6,.7)):
            c=contrasts(values)
            self.assertAlmostEqual(
                c["paired_full_change"],
                c["lambda_change_old_fit"]+c["refit_effect_purged_lambda"])
            self.assertAlmostEqual(
                c["paired_full_change"],
                c["refit_effect_old_lambda"]+c["lambda_change_purged_fit"])
            self.assertAlmostEqual(
                c["model_by_lambda_interaction"],
                c["lambda_change_purged_fit"]-c["lambda_change_old_fit"])
    def test_same_lambda_with_refit_only(self):
        c=contrasts((.8,.8,.5,.5))
        self.assertEqual(c["lambda_change_old_fit"],0.)
        self.assertEqual(c["lambda_change_purged_fit"],0.)
        self.assertEqual(c["model_by_lambda_interaction"],0.)
        self.assertAlmostEqual(c["paired_full_change"],-.3)
    def test_opposite_selector_effect_by_model(self):
        c=contrasts((.8,.7,.6,.65))
        self.assertLess(c["lambda_change_old_fit"],0.)
        self.assertGreater(c["lambda_change_purged_fit"],0.)
        self.assertGreater(c["model_by_lambda_interaction"],0.)
    def test_not_a_pooled_trial_claim(self):
        c=contrasts((.58,.57,.59,.6))
        self.assertAlmostEqual(c["paired_full_change"],.02)
        self.assertNotEqual(c["refit_effect_old_lambda"],c["paired_full_change"])

if __name__=="__main__":unittest.main()
