"""Regression checks for V16 covariance score estimand audit."""
import unittest
import numpy as np
from covariance_estimand_contract_v16 import compare_source_validation_objectives,_selftest


class CovarianceEstimandTests(unittest.TestCase):
    def test_class_shift_invariance_and_predictive_change(self):
        result=_selftest()
        self.assertTrue(result["class_shift_invariant_centered_score"])
        self.assertTrue(result["class_shift_changes_true_predictive_score"])

    def test_no_validation_target_mean_in_proper_score(self):
        rng=np.random.default_rng(7)
        X=np.vstack([rng.normal(loc=-1,size=(24,3)),rng.normal(loc=1,size=(24,3))])
        y=np.repeat([0,1],24)
        V=np.vstack([rng.normal(loc=-1,size=(12,3)),rng.normal(loc=1,size=(12,3))])
        v=np.repeat([0,1],12)
        base=compare_source_validation_objectives(X,y,V,v,[.01,.1,.5,1.])
        changed=compare_source_validation_objectives(X,y,V+30.,v,[.01,.1,.5,1.])
        np.testing.assert_allclose(base.frozen_class_centered_covariance_score,
                                   changed.frozen_class_centered_covariance_score,rtol=0,atol=1e-12)
        self.assertGreater(np.max(np.abs(np.subtract(base.genuine_label_conditional_predictive_score,
                                   changed.genuine_label_conditional_predictive_score))),100.)

    def test_failed_inputs(self):
        rng=np.random.default_rng(1)
        X=np.vstack([rng.normal(-1,size=(20,2)),rng.normal(1,size=(20,2))])
        y=np.repeat([0,1],20)
        for grid in [[.1,.1],[1.,0.],[float('nan')],[1.1],[.01,-.1]]:
            with self.assertRaises(ValueError):
                compare_source_validation_objectives(X,y,X,y,grid)
        with self.assertRaises(ValueError):
            compare_source_validation_objectives(X,y,X,np.zeros(len(y)),[.1,.2])


if __name__=="__main__":
    unittest.main()
