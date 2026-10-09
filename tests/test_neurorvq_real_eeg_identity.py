"""Frozen NeuroRVQ feature-drift study contracts; synthetic features only."""
import unittest
import numpy as np
from research.eeg.neurorvq_pretrained_real_eeg_identity import assess, align_channel_token_blocks

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

    def test_channel_major_token_metadata_consistency(self):
        canonical=np.arange(2*4*3*5,dtype=float).reshape(2,4,3,5)
        perm=np.array([1,0,2,3])
        delivered=canonical[:,perm,:,:].reshape(2,-1)
        recovered=align_channel_token_blocks(delivered,perm,n_patches=3)
        np.testing.assert_array_equal(recovered,canonical.reshape(2,-1))

    def test_incomplete_token_metadata_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"geometry"):
            align_channel_token_blocks(np.ones((2,119)),[1,0,2,3],n_patches=3)

    def test_nonmatched_representation_rejected(self):
        with self.assertRaisesRegex(ValueError,"matched"):
            assess(np.ones((4,20)),np.ones((4,19)),np.ones((4,20)))

if __name__=="__main__":
    unittest.main()
