"""Synthetic counterexamples and contracts for max/min frontier bias."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frontier_simultaneous_path_sensitivity import simultaneous_path_sensitivity


class SimultaneousPathSensitivityTests(unittest.TestCase):
    def test_constant_population_path_can_show_observed_span(self):
        # All lambda coordinates have the same generating mean (.65).
        # Correlated finite-person estimation noise creates a nonzero
        # observed max-min gap without a true lambda effect.
        rng = np.random.default_rng(1907)
        n, l_count = 14, 405
        lambdas = np.linspace(0., 1., l_count)
        auc = np.clip(.65 + rng.normal(0, .09, (n, l_count)), 0, 1)
        regret = np.tile((lambdas-.2)**2, (n, 1))
        result = simultaneous_path_sensitivity(
            auc, regret, lambdas, reps=1000, seed=11
        )
        self.assertGreater(result["observed_max_auc_span"], 0.02)
        self.assertEqual(result["status"],
                         "POST_OUTCOME_CONDITIONAL_SIMULTANEOUS_SENSITIVITY_NOT_CONFIRMATORY")
        self.assertFalse(result["materiality_certified_under_conditional_reference"])
        self.assertIsNone(result["simultaneous_lower_bound_first_material_regret_budget"])
        self.assertTrue(result["post_outcome_statistic_not_registered"])

    def test_clear_repeatable_slope_survives_supremum_correction(self):
        n, l_count = 16, 31
        lam = np.linspace(0, 1, l_count)
        # Identical paths yield zero bootstrap deviation; a fixed shift
        # in participant baseline leaves the within-path span unchanged.
        auc = np.tile(.60 + .12 * lam, (n, 1))
        regret = np.tile((lam - .3)**2, (n, 1))
        out = simultaneous_path_sensitivity(
            auc, regret, lam, reps=300, seed=2
        )
        self.assertAlmostEqual(out["observed_max_auc_span"], .12)
        self.assertLess(out["full_path_supremum_bootstrap_radius"], 1e-12)
        self.assertTrue(out["materiality_certified_under_conditional_reference"])
        self.assertIsNotNone(
            out["simultaneous_lower_bound_first_material_regret_budget"]
        )

    def test_pathwise_permutation_invariance(self):
        n=10
        lam=np.linspace(0,1,13)
        rng=np.random.default_rng(12)
        A=.65+.01*lam[None,:]+rng.normal(0,.02,(n,len(lam)))
        R=np.tile((lam-.4)**2,(n,1))
        a=simultaneous_path_sensitivity(A,R,lam,reps=300,seed=7)
        b=simultaneous_path_sensitivity(A[::-1],R[::-1],lam,reps=300,seed=7)
        self.assertAlmostEqual(a["observed_max_auc_span"],b["observed_max_auc_span"])
        # The Monte Carlo resampling depends on index order, so a
        # bootstrap quantile need not match at a finite sample count.

    def test_bad_denominators_and_nonfinite_input_fail_closed(self):
        lam=np.linspace(0,1,9)
        A=np.full((5,9),.7)
        R=np.tile((lam-.2)**2,(5,1))
        for badA,badR,badL in [
            (A[:3],R[:3],lam),
            (A[:,:7],R,lam),
            (np.full((5,9),float("nan")),R,lam),
            (A,-R,lam),
            (A,R,lam[::-1]),
        ]:
            with self.assertRaises(ValueError):
                simultaneous_path_sensitivity(
                    badA,badR,badL,reps=200
                )

    def test_return_discloses_shared_source_fit_limit(self):
        lam=np.linspace(0,1,9)
        A=np.tile(.55+.04*lam,(7,1))
        R=np.tile(lam*lam,(7,1))
        x=simultaneous_path_sensitivity(A,R,lam,reps=200)
        self.assertIn("source", " ".join(x["limitations"]).lower())
        self.assertTrue(x["source_regret_path_treated_as_fixed_estimate"])
        self.assertFalse(any(
            "confirmatory evidence" in str(v).lower() for v in x.values()
        ))


    def test_translation_invariant_error_ignores_subject_intercepts(self):
        rng = np.random.default_rng(48)
        n, m = 16, 37
        lam = np.linspace(0., 1., m)
        A = .55 + .07*lam[None, :] + rng.normal(0., .006, (n, m))
        R = np.tile((lam-.17)**2, (n, 1))
        offsets = np.linspace(-.18, .18, n)[:, None]
        a = simultaneous_path_sensitivity(A, R, lam, reps=1000, seed=7)
        b = simultaneous_path_sensitivity(A+offsets, R, lam, reps=1000, seed=7)
        self.assertAlmostEqual(
            a["translation_invariant_oscillation_radius95"],
            b["translation_invariant_oscillation_radius95"], places=12
        )
        self.assertAlmostEqual(
            a["translation_invariant_span_bounds_nominal95"][0],
            b["translation_invariant_span_bounds_nominal95"][0], places=12
        )
        # Whole-path absolute ordinate error is sensitive to subject offsets.
        self.assertGreater(
            b["full_path_supremum_bootstrap_radius"],
            a["full_path_supremum_bootstrap_radius"]
        )

    def test_oscillation_radius_is_bounded_by_twice_sup_radius(self):
        lam=np.linspace(0.,1.,15)
        rng=np.random.default_rng(519)
        A=np.clip(.65+.06*lam[None,:]+rng.normal(0,.03,(11,15)),0,1)
        R=np.tile((lam-.3)**2,(11,1))
        z=simultaneous_path_sensitivity(A,R,lam,reps=600,seed=8)
        self.assertLessEqual(
            z["translation_invariant_oscillation_radius95"],
            2*z["full_path_supremum_bootstrap_radius"]+1e-12
        )
        self.assertEqual(z["status"],
                         "POST_OUTCOME_CONDITIONAL_SIMULTANEOUS_SENSITIVITY_NOT_CONFIRMATORY")

if __name__=="__main__":
    unittest.main()
