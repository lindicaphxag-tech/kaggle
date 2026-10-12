"""V24 scientific validity gates for grid-anchored source-only continuum search."""
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from certified_continuous_selector_v23 import (
    certified_source_auc_search, source_person_auc_at_lambda,
)

class GridAnchorIntegrity(unittest.TestCase):
    def setUp(self):
        e = np.array([.3, 1.2, 5.8])
        self.model = ((e, np.eye(3), np.ones(3), np.zeros(3), e.mean()),
                      np.array([[1., -2., 0.], [0., .5, -.4],
                                [-.6, .8, 1.1], [.1, .4, -.2]]),
                      np.array([1, 1, 0, 0]))
        self.grid = [.05,.1,.2,.4,.5,.7,.9,.95]

    def test_anchored_search_cannot_regress_against_finite_grid(self):
        scores = {x: source_person_auc_at_lambda(self.model, x) for x in self.grid}
        r = certified_source_auc_search(
            [self.model], low=.05, high=.95,
            initial_lambdas=self.grid, max_splits=0, tolerance=0.)
        self.assertAlmostEqual(r["best_evaluated_source_macro_auc"], max(scores.values()))
        self.assertEqual(r["initial_anchor_count"], len(self.grid))
        self.assertEqual(r["lambda_evaluations"], len(self.grid))
        self.assertGreaterEqual(
            r["upper_bound_global_continuous_source_macro_auc"],
            r["best_evaluated_source_macro_auc"])

    def test_fair_comparison_uses_same_continuous_interval(self):
        with self.assertRaises(ValueError):
            certified_source_auc_search(
                [self.model], low=.05, high=.95,
                initial_lambdas=[0.0,.5])
        with self.assertRaises(ValueError):
            certified_source_auc_search(
                [self.model], low=.05, high=.95,
                initial_lambdas=[.2,float("nan")])

    def test_tightened_upper_bound_dominates_dense_path(self):
        rng=np.random.default_rng(20261013)
        models=[]
        for person in range(3):
            p=4
            e=np.exp(rng.normal(size=p))
            U,_=np.linalg.qr(rng.normal(size=(p,p)))
            state=(e,U,rng.normal(size=p),np.zeros(p),float(np.mean(e)))
            X=rng.normal(size=(11,p)); y=np.array([1]*5+[0]*6)
            models.append((state,X,y))
        r=certified_source_auc_search(
            models,low=.05,high=.95,
            initial_lambdas=self.grid,max_splits=7,segments=4,tolerance=.015)
        for lam in np.linspace(.05,.95,321):
            actual=np.mean([source_person_auc_at_lambda(m,float(lam)) for m in models])
            self.assertLessEqual(actual,r["upper_bound_global_continuous_source_macro_auc"]+1e-10)
        self.assertGreaterEqual(r["best_evaluated_source_macro_auc"],
                                max(np.mean([source_person_auc_at_lambda(m,l) for m in models])
                                    for l in self.grid)-1e-12)

if __name__ == "__main__":
    unittest.main()
