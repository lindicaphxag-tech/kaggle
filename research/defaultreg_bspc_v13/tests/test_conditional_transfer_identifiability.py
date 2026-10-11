"""Finite synthetic checks for conditional objective safety and impossibility.

Run with:
PYTHONPATH=papers/eeg-default-regularisation/src python -m unittest
  discover -s papers/eeg-default-regularisation/tests
  -p test_conditional_transfer_identifiability.py -v

No EEG outcomes are used; cannot count as cross-cohort validation.
"""
import unittest
import numpy as np
from conditional_transfer_identifiability import (
    conditional_certificate, opposite_gaussian_targets
)


class ConditionalSafetyTests(unittest.TestCase):
    def test_exact_witness_on_500_random_spd_examples(self):
        rng=np.random.default_rng(20261011)
        for _ in range(500):
            k=5
            R=rng.normal(size=(k,k))
            S=R.T@R+.3*np.eye(k)
            wd=rng.normal(size=k)
            wc=rng.normal(size=k)
            d=rng.normal(size=k)
            rho=float(rng.uniform(0,2))
            ans=conditional_certificate(wd,wc,d,S,rho)
            a=wd/np.sqrt(wd@S@wd)-wc/np.sqrt(wc@S@wc)
            witness=np.asarray(ans.worst_contrast)
            self.assertAlmostEqual(float(a@witness),ans.exact_worst_gap,places=9)
            delta=witness-d
            self.assertAlmostEqual(float(delta@np.linalg.solve(S,delta)),rho*rho,places=8)
            L=np.linalg.cholesky(S)
            for j in range(6):
                z=rng.normal(size=k)
                z/=np.linalg.norm(z)
                feasible=d+L@z*rho*rng.random()
                self.assertGreaterEqual(float(a@feasible)+1e-10,ans.exact_worst_gap)

    def test_opposite_target_rankings_same_source_and_covariance(self):
        S=np.diag([2.,1.])
        u=np.array([1.,0.]);v=np.array([0.,1.])
        good,bad=opposite_gaussian_targets(u,v,S)
        self.assertGreater(conditional_certificate(u,v,good,S,0).nominal_margin_gap,0)
        self.assertLess(conditional_certificate(u,v,bad,S,0).nominal_margin_gap,0)

    def test_positive_rescaling_is_auc_invariant(self):
        S=np.diag([2.,1.]);w=np.array([.4,-.8]);d=np.array([.2,.5])
        ans=conditional_certificate(w,5*w,d,S,2.)
        self.assertTrue(ans.invariant_to_mean_shift)
        self.assertAlmostEqual(ans.exact_worst_gap,0.,places=9)
        self.assertFalse(ans.certified_preference)

    def test_non_spd_nonfinite_and_negative_budget_fail_closed(self):
        w=np.array([1.,0.]);d=np.array([.2,.5]);S=np.eye(2)
        for covariance,rad in [(np.array([[1.,3.],[3.,1.]]),.2),
                               (S,-.1),(S,float('nan'))]:
            with self.assertRaises(ValueError):
                conditional_certificate(w,np.array([0.,1.]),d,covariance,rad)


if __name__=="__main__":
    unittest.main()
