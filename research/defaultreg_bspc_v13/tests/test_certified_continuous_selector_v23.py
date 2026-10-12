"""Exact source-only continuous shrinkage search regression tests."""
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from certified_continuous_selector_v23 import (
    certified_source_auc_search, source_person_auc_at_lambda
)


class CertifiedSourceSearchTests(unittest.TestCase):
    def test_certifies_constant_source_ranking(self):
        state = (np.array([2.]), np.eye(1), np.array([1.]),
                 np.zeros(1), 3.)
        X = np.array([[3.], [2.], [0.], [-1.]])
        fold = (state, X, np.array([1, 1, 0, 0]))
        r = certified_source_auc_search([fold], low=0., high=1.)
        self.assertEqual(r["selected_lambda_source_macro"], 0.)
        self.assertEqual(r["interval_splits"], 0)
        self.assertEqual(r["best_evaluated_source_macro_auc"], 1.)
        self.assertTrue(r["source_optimality_certificate_met"])
        self.assertAlmostEqual(r["residual_source_optimality_gap_upper_bound"], 0.)

    def test_conditional_bound_covers_dense_oracle_auc(self):
        rng = np.random.default_rng(20261013)
        models = []
        for person in range(3):
            e = np.exp(rng.normal(size=3))
            U, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            state = (e, U, rng.normal(size=3), np.zeros(3), e.mean())
            X = rng.normal(size=(13, 3))
            labels = np.r_[np.ones(6, int), np.zeros(7, int)]
            models.append((state, X, labels))
        r = certified_source_auc_search(
            models, low=.01, high=.99, tolerance=.02, max_splits=6
        )
        grid = np.linspace(.01, .99, 301)
        dense = max(np.mean([
            source_person_auc_at_lambda(model, float(x))
            for model in models
        ]) for x in grid)
        self.assertLessEqual(dense, r["upper_bound_global_continuous_source_macro_auc"]+1e-10)
        self.assertGreaterEqual(r["best_evaluated_source_macro_auc"]+1e-10, 0.)
        self.assertGreaterEqual(r["residual_source_optimality_gap_upper_bound"], 0.)
        if r["source_optimality_certificate_met"]:
            self.assertLessEqual(r["residual_source_optimality_gap_upper_bound"], .02+1e-11)

    def test_budget_exhaustion_fails_closed(self):
        e = np.array([1., 5.])
        state = (e, np.eye(2), np.ones(2), np.zeros(2), 3.)
        X = np.array([[1., -2.], [0., 0.]])
        fold = (state, X, np.array([1, 0]))
        result = certified_source_auc_search(
            [fold], low=.1, high=.9, tolerance=0., max_splits=0
        )
        self.assertEqual(result["interval_splits"], 0)
        self.assertGreaterEqual(
            result["upper_bound_global_continuous_source_macro_auc"],
            result["best_evaluated_source_macro_auc"]
        )
        self.assertFalse(result["unseen_target_performance_bound"])
        self.assertTrue(result["source_only"])

    def test_invalid_budget_fails(self):
        fold = ((np.array([1.]), np.eye(1), np.ones(1),
                 np.zeros(1), 1.), np.array([[0.], [1.]]), [0, 1])
        with self.assertRaises(ValueError):
            certified_source_auc_search([fold], low=.5, high=.5)
        with self.assertRaises(ValueError):
            certified_source_auc_search([fold], low=0., high=1., max_splits=-1)
        with self.assertRaises(ValueError):
            certified_source_auc_search([fold], low=0., high=1., tolerance=float("nan"))

if __name__=="__main__":
    unittest.main()
