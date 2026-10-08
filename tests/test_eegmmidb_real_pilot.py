"""CPU-only invariants for the prospective real-data EEG motor-imagery pilot."""
import unittest
import numpy as np

from research.eeg.eegmmidb_real_pilot import (
    spectral_features, stress_conditions, evaluate_holdout, SPLITS
)


class ContractTests(unittest.TestCase):
    def test_spectral_shape_and_amplitude(self):
        sfreq = 160
        t = np.arange(480) / sfreq
        a = np.sin(2 * np.pi * 10 * t)
        b = np.sin(2 * np.pi * 20 * t)
        x = np.stack([a, b])[None]
        features = spectral_features(x, sfreq)
        self.assertEqual(features.shape, (1, 2, 2))
        self.assertGreater(features[0, 0, 0], features[0, 0, 1])
        self.assertGreater(features[0, 1, 1], features[0, 1, 0])

    def test_permutation_exact_oracle(self):
        rng = np.random.default_rng(1)
        n = 6
        X = rng.normal(size=(8, n, 2))
        names = tuple('ABCDEF')
        xyz = np.array([[i, 0., 0.] for i in range(n)])
        result, meta = stress_conditions(X, X, names, xyz)
        np.testing.assert_array_equal(result['original'], result['identity_reindexed'])
        self.assertEqual(set(meta['missing_electrodes_by_percentage']), {'10', '25', '40'})

    def test_subject_disjoint(self):
        split = SPLITS['pilot']
        self.assertFalse(set(split['train']) & set(split['test']))
        self.assertFalse(set(split['val']) & set(split['test']))

    def test_no_missing_subject_accepted(self):
        split = SPLITS['pilot']
        ids = split['train'] + split['val'] + split['test']
        rng = np.random.default_rng(2)
        X = rng.normal(size=(len(ids) * 2, 6, 2))
        y = np.tile([0, 1], len(ids))
        subjects = np.repeat(ids, 2)
        xyz = np.column_stack([np.arange(6), np.zeros(6), np.zeros(6)])
        out = evaluate_holdout(X, y, subjects, tuple('ABCDEF'), xyz)
        self.assertEqual(out['n_trials']['test'], 6)
        self.assertEqual(
            out['condition_metrics']['original']['per_trial_prediction'],
            out['condition_metrics']['identity_reindexed']['per_trial_prediction'],
        )
        with self.assertRaisesRegex(ValueError, 'exact preregistered cohort'):
            evaluate_holdout(X[:-2], y[:-2], subjects[:-2], tuple('ABCDEF'), xyz)

    def test_reject_duplicate_names(self):
        X = np.ones((2, 5, 2))
        xyz = np.zeros((5, 3))
        with self.assertRaisesRegex(ValueError, 'named montage'):
            stress_conditions(X, X, tuple('AAAAB'), xyz)


if __name__ == '__main__':
    unittest.main()
