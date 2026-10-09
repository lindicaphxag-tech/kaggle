"""Frozen NeuroRVQ feature-drift study contracts; synthetic features only."""
import unittest
import numpy as np
from research.eeg.neurorvq_pretrained_real_eeg_identity import assess

class PretrainedIdentityContracts(unittest.TestCase):
    def test_reordered_known_channels_recover_identical_features(self):
        rng=np.random.default_rng(20)
        z=rng.normal(size=(6,50))
        wrong=z.copy()
        wrong[0,2]+=0.5
        result=assess(z,wrong,z.copy())
        self.assertEqual(result["n_trials"],6)
        self.assertEqual(result["wrong_order_nonidentical_count"],1)
        self.assertEqual(result["restored_max_abs_error"],0.0)

    def test_faulty_known_identity_correction_must_fail(self):
        z=np.ones((4,20))
        restored=z.copy();restored[2,3]+=0.1
        with self.assertRaisesRegex(AssertionError,"failed to recover"):
            assess(z,z,restored)

    def test_nonmatched_representation_rejected(self):
        with self.assertRaisesRegex(ValueError,"matched"):
            assess(np.ones((4,20)),np.ones((4,19)),np.ones((4,20)))

if __name__=="__main__":
    unittest.main()
