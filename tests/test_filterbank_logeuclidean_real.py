"""Contracts for conventional log-Euclidean EEG covariance baseline."""
import unittest
import numpy as np
from research.eeg.filterbank_logeuclidean_real import (
    filter_bank_log_covariance, evaluate, SPLITS
)

class StrongSpatialContracts(unittest.TestCase):
    def test_spd_and_electrode_reindex_exactly_recover(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(7,5,480))
        feat=filter_bank_log_covariance(x)
        self.assertEqual(feat.shape,(7,30))
        self.assertTrue(np.isfinite(feat).all())
        perm=np.array([4,0,3,1,2])
        np.testing.assert_allclose(
            feat,filter_bank_log_covariance(x[:,perm][:,np.argsort(perm)]),
            rtol=0,atol=1e-12,
        )

    def test_zero_power_must_not_look_like_valid_eeg(self):
        with self.assertRaisesRegex(ValueError,"Nonpositive power"):
            filter_bank_log_covariance(np.zeros((2,5,480)))

    def test_subject_split_and_validation_only_hyperparameters(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(32,12))
        subject=np.repeat([1,2,3,4],8)
        y=np.tile([0,1,0,1,0,1,0,1],4)
        result=evaluate(x,y,subject)
        self.assertEqual(result["n_test"],8)
        self.assertEqual(result["subject_splits"]["train"],[1,2])
        self.assertEqual(result["subject_splits"]["test"],[4])
        self.assertEqual(len(result["validation_grid"]),5)
        with self.assertRaisesRegex(ValueError,"Unregistered or missing subjects"):
            evaluate(x[:-8],y[:-8],subject[:-8])

if __name__=="__main__":
    unittest.main()
