"""Regression tests: exact ranking identity, provenance and negative controls."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from source_rank_churn_v22 import pairwise_rank_transition, subject_macro_rank_audit


def brute_auc(y, scores):
    p = np.asarray(scores)[np.asarray(y) == 1]
    n = np.asarray(scores)[np.asarray(y) == 0]
    diff = p[:, None] - n[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


class RankMechanismTests(unittest.TestCase):
    def test_exact_tie_and_full_reversal(self):
        y = [1, 1, 0, 0]
        a = [1., 2., 0., 2.]
        b = [0., 0., 1., 1.]
        d = pairwise_rank_transition(y, a, b, block_size=1)
        self.assertEqual(d["n_pairs"], 4)
        self.assertEqual(sum(map(sum, d["matrix_before_rows_after_columns"])), 4)
        self.assertAlmostEqual(d["auc_before"], brute_auc(y, a))
        self.assertAlmostEqual(d["auc_after"], brute_auc(y, b))
        self.assertAlmostEqual(d["auc_delta_exact"], d["auc_after"] - d["auc_before"])

    def test_scalar_and_translation_invariance(self):
        rng = np.random.default_rng(17)
        y = np.r_[np.ones(36, int), np.zeros(24, int)]
        a = rng.normal(size=60)
        d = pairwise_rank_transition(y, a, 4.1 * a - 78., block_size=7)
        self.assertEqual(d["gross_pair_state_change_fraction"], 0.)
        self.assertAlmostEqual(d["auc_delta_exact"], 0.)

    def test_gross_churn_can_cancel_and_auc_not_move(self):
        y = [1, 1, 0, 0]
        a = [2., 0., 1., 1.]
        b = [0., 2., 1., 1.]
        d = pairwise_rank_transition(y, a, b)
        self.assertEqual(d["gross_pair_state_change_fraction"], 1.)
        self.assertEqual(d["auc_delta_exact"], 0.)
        self.assertEqual(d["strict_pair_reversal_fraction"], 1.)

    def test_negative_scaling_complements_auc(self):
        y = [1, 1, 1, 0, 0]
        a = [0., 2., 3., 1., 2.]
        d = pairwise_rank_transition(y, a, [-x for x in a])
        self.assertAlmostEqual(d["auc_after"], 1 - d["auc_before"])

    def test_randomized_identity_against_direct_auc(self):
        rng = np.random.default_rng(20261012)
        for n in range(3, 36):
            y = np.r_[np.zeros(n, int), np.ones(n + 2, int)]
            a = rng.integers(-3, 4, size=len(y)) / 3
            b = rng.integers(-3, 4, size=len(y)) / 3
            d = pairwise_rank_transition(y, a, b, block_size=n % 5 + 1)
            self.assertAlmostEqual(d["auc_before"], brute_auc(y, a))
            self.assertAlmostEqual(d["auc_after"], brute_auc(y, b))
            self.assertAlmostEqual(d["auc_delta_exact"],
                                   brute_auc(y, b) - brute_auc(y, a))

    def test_macro_not_trial_pooling(self):
        rows = [
            {"participant_id": "s1", "information_scope": "source_validation",
             "source_fit_subject_ids": ["s3", "s4"], "labels": [1, 0],
             "score_before": [1., 0.], "score_after": [0., 1.]},
            {"participant_id": "s2", "information_scope": "source_validation",
             "source_fit_subject_ids": ["s3", "s4"], "labels": [1, 0, 0, 0],
             "score_before": [0., 1., 1., 1.], "score_after": [1., 0., 0., 0.]}
        ]
        r = subject_macro_rank_audit(rows)
        self.assertEqual(r["n_participants"], 2)
        self.assertAlmostEqual(r["macro_auc_delta_exact"], 0.)
        self.assertEqual(r["n_held_participants_harmed"], 1)
        self.assertEqual(r["n_held_participants_improved"], 1)

    def test_leakage_and_pseudoreplication_fail_closed(self):
        row = {"participant_id": "s1", "information_scope": "source_validation",
               "source_fit_subject_ids": ["s2"], "labels": [0, 1],
               "score_before": [0., 1.], "score_after": [1., 0.]}
        with self.assertRaises(ValueError):
            subject_macro_rank_audit([row, row])
        with self.assertRaises(ValueError):
            subject_macro_rank_audit([{**row, "source_fit_subject_ids": ["s1"]}])
        with self.assertRaises(ValueError):
            subject_macro_rank_audit([{**row, "information_scope": "posthoc_target"}])
        with self.assertRaises(ValueError):
            pairwise_rank_transition([0, 0], [1, 2], [2, 1])
        with self.assertRaises(ValueError):
            pairwise_rank_transition([0, 1], [float("nan"), 2], [2, 1])
        with self.assertRaises(ValueError):
            pairwise_rank_transition([0, 1], [0, 1], [1, 0], block_size=0)

    def test_target_audit_is_explicitly_posthoc(self):
        row = {"participant_id": 15, "information_scope": "posthoc_target",
               "source_fit_subject_ids": [1, 2, 3], "labels": [0, 1],
               "score_before": [0, 1], "score_after": [1, 0]}
        d = subject_macro_rank_audit([row], scope="posthoc_target")
        self.assertFalse(d["usable_for_parameter_selection"])
        self.assertTrue(d["no_target_generalisation_guarantee"])


if __name__ == "__main__":
    unittest.main()
