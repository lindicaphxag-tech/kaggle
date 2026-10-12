"""V25 adversarial and randomized tests for Bernstein sign-cell method."""
import pathlib
import sys
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from bernstein_rank_cells_v25 import (
    _linear_bernstein_product, spectral_pair_numerator_coefficients,
    bernstein_rank_path,
)


def evaluate(state, X, y, lam):
    e, U, d, mid, nu = state
    scores = (X-mid) @ U @ ((U.T @ d)/((1-lam)*e+lam*nu))
    pos, neg = scores[np.asarray(y)==1], scores[np.asarray(y)==0]
    diff = pos[:,None]-neg[None,:]
    return float(np.mean((diff>0)+.5*(diff==0)))


class BernsteinRankCellTests(unittest.TestCase):
    def test_bernstein_linear_products_reproduce_grid(self):
        rng=np.random.default_rng(25)
        for p in range(0, 7):
            left=rng.uniform(.1,2.,size=p)
            right=rng.uniform(.1,2.,size=p)
            b=_linear_bernstein_product(left,right)
            assert len(b)==p+1
            for t in np.linspace(0,1,17):
                from math import comb
                val=sum(b[k]*comb(p,k)*t**k*(1-t)**(p-k)
                        for k in range(p+1))
                direct=float(np.prod((1-t)*left+t*right))
                self.assertAlmostEqual(val,direct,places=10)

    def test_root_at_half_is_uncertain_with_correct_auc_change(self):
        state=(np.array([1.,5.]),np.eye(2),np.ones(2),np.zeros(2),3.)
        X=np.array([[1.,-2.],[0.,0.]])
        y=[1,0]
        b=spectral_pair_numerator_coefficients(state,X[0]-X[1],.1,.9)
        self.assertLess(float(b[0])*float(b[-1]),0.)
        v=bernstein_rank_path(state,X,y,.1,.9,segments=8)
        self.assertEqual(v["certain_fixed_sign_fraction"],0.)
        self.assertEqual(v["numerator_polynomial_max_degree"],1)
        self.assertAlmostEqual(v["observed_endpoint_auc_low"],1.)
        self.assertAlmostEqual(v["observed_endpoint_auc_high"],0.)
        self.assertEqual(v["observed_endpoint_rank_state_changes"],1)

    def test_one_dim_all_strict_pair_ranks_stable(self):
        state=(np.array([2.]),np.eye(1),np.ones(1),np.zeros(1),3.)
        X=np.array([[2.],[3.],[0.],[-1.]])
        y=[1,1,0,0]
        res=bernstein_rank_path(state,X,y,0.,1.,segments=2)
        self.assertEqual(res["certain_fixed_sign_fraction"],1.)
        self.assertEqual(res["uniform_path_auc_lower_bound"],1.)
        self.assertEqual(res["uniform_path_auc_upper_bound"],1.)

    def test_zero_class_separation_fails_closed_not_false_certified(self):
        state=(np.array([1.,5.]),np.eye(2),np.ones(2),np.zeros(2),3.)
        X=np.zeros((4,2))
        res=bernstein_rank_path(state,X,[1,1,0,0],.1,.9,segments=4)
        self.assertEqual(res["certain_fixed_sign_fraction"],0.)
        self.assertEqual(res["observed_endpoint_auc_low"],.5)
        self.assertLessEqual(res["uniform_path_auc_lower_bound"],.5)
        self.assertGreaterEqual(res["uniform_path_auc_upper_bound"],.5)

    def test_random_spectral_analytic_path_encloses_every_sample(self):
        rng=np.random.default_rng(20261013)
        for p in [2,3,6,9]:
            e=np.exp(rng.normal(size=p))
            U,_=np.linalg.qr(rng.normal(size=(p,p)))
            nu=float(e.mean())
            d=rng.normal(size=p)
            state=(e,U,d,np.zeros(p),nu)
            X=rng.normal(size=(21,p))
            y=np.array([1]*10+[0]*11)
            b=bernstein_rank_path(state,X,y,.03,.97,segments=12,block_size=4)
            self.assertTrue(b["no_target_performance_certificate"])
            self.assertEqual(b["numerator_polynomial_max_degree"],p-1)
            self.assertLessEqual(b["uniform_path_auc_lower_bound"],
                                 b["uniform_path_auc_upper_bound"])
            for lam in np.linspace(.03,.97,151):
                auc=evaluate(state,X,y,lam)
                self.assertLessEqual(auc,
                    b["uniform_path_auc_upper_bound"]+1e-10)
                self.assertGreaterEqual(auc,
                    b["uniform_path_auc_lower_bound"]-1e-10)

    def test_strictly_positive_rescaling_keeps_rank_counts(self):
        state=(np.array([.3,1.5,4.,7.]),np.eye(4),
               np.ones(4),np.zeros(4),3.)
        X=np.array([[2,0,1,-1],[1,2,0,1],
                    [0,1,0,1],[-1,2,-2,0]],float)
        y=[1,1,0,0]
        a=bernstein_rank_path(state,X,y,.01,.99,segments=16)
        b=bernstein_rank_path(state,2.*X+7.,y,.01,.99,segments=16)
        self.assertEqual(a["certain_positive_pairs"],b["certain_positive_pairs"])
        self.assertEqual(a["certain_negative_pairs"],b["certain_negative_pairs"])
        self.assertAlmostEqual(a["observed_endpoint_auc_low"],
                               b["observed_endpoint_auc_low"])

    def test_p_above_safe_range_rejected(self):
        e=np.ones(13)
        state=(e,np.eye(13),np.ones(13),np.zeros(13),1.)
        with self.assertRaises(ValueError):
            bernstein_rank_path(state,np.zeros((2,13)),[1,0],.1,.9)

if __name__ == "__main__":
    unittest.main()
