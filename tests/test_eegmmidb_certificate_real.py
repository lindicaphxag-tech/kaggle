import unittest
import numpy as np

from research.eeg.eegmmidb_certificate_real import certify_cohort
from research.eeg.eegmmidb_real_pilot import SPLITS


class RealCertificateTests(unittest.TestCase):
    def test_known_orientation_consistent(self):
        split = SPLITS['pilot']
        ids = split['train'] + split['val'] + split['test']
        rng = np.random.default_rng(2)
        channels = ('C1', 'C3', 'C2', 'C4', 'F3', 'F4')
        X = rng.normal(size=(len(ids) * 4, len(channels), 2))
        y = np.tile([0, 1, 0, 1], len(ids))
        subjects = np.repeat(ids, 4)
        out = certify_cohort(X, y, subjects, channels)
        self.assertEqual(out['false_invariance_certificates'], 0)
        self.assertEqual(out['n_test'], 12)
        self.assertEqual(len(out['per_subject']), 3)

    def test_missing_declared_channel_rejected(self):
        split = SPLITS['pilot']
        ids = split['train'] + split['val'] + split['test']
        X = np.random.default_rng(2).normal(size=(len(ids) * 2, 5, 2))
        y = np.tile([0, 1], len(ids))
        subjects = np.repeat(ids, 2)
        with self.assertRaisesRegex(ValueError, 'Missing predeclared'):
            certify_cohort(
                X, y, subjects, ('C1', 'C3', 'C2', 'F3', 'F4')
            )


if __name__ == '__main__':
    unittest.main()
