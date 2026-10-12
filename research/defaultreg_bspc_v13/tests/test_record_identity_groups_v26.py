"""V26 EEG recording-level held-person identity leakage prevention tests."""
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from record_identity_groups_v26 import (
    assign_recording_groups, check_disjoint_training
)


class RecordIdentityPurging(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261013)
        self.X = rng.normal(size=(6, 3, 5))
        self.Y = np.array([1, 0, 0, 1, 1, 0])
        self.Z = rng.normal(size=(6, 3, 5))
        self.records = [
            {"subject": 4, "signal": self.X, "labels": self.Y},
            {"subject": 5, "signal": self.X.copy(), "labels": self.Y.copy()},
            {"subject": 47, "signal": self.Z, "labels": self.Y},
            {"subject": 49, "signal": self.Z + 1e-7, "labels": self.Y},
        ]

    def test_original_duplicate_only_when_whole_signal_is_equal(self):
        d, ids = assign_recording_groups(self.records)
        self.assertEqual(d["duplicate_groups"], [[4, 5]])
        self.assertEqual(d["representative_ids"], [4, 47, 49])
        self.assertEqual(d["independent_processed_record_count"], 3)
        self.assertTrue(d["global_target_signal_access_for_grouping"])
        self.assertNotEqual(ids[47], ids[49])

    def test_target_duplicate_in_source_fails_before_any_fit(self):
        d, _ = assign_recording_groups(self.records)
        with self.assertRaises(AssertionError):
            check_disjoint_training(d, 4, [5, 47, 49])
        with self.assertRaises(AssertionError):
            check_disjoint_training(d, 47, [5, 49])
        self.assertTrue(check_disjoint_training(d, 4, [47, 49]))

    def test_repeated_subject_fails(self):
        with self.assertRaises(ValueError):
            assign_recording_groups(self.records + [self.records[0]])

    def test_label_conflict_in_identical_eeg_is_fatal(self):
        altered = dict(self.records[1], labels=1-self.Y)
        with self.assertRaises(ValueError):
            assign_recording_groups([self.records[0], altered])

    def test_nonfinite_signal_fails_closed(self):
        X=self.X.copy()
        X[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            assign_recording_groups([
                {"subject": 1, "signal": X, "labels": self.Y}
            ])

    def test_record_hash_is_exact_not_approximate(self):
        d, _ = assign_recording_groups(self.records)
        self.assertEqual(d["discarded_exact_copies"], 1)
        self.assertNotIn([47, 49], d["duplicate_groups"])

    def test_no_pseudoreplicated_training_source(self):
        d, _ = assign_recording_groups(self.records)
        with self.assertRaises(ValueError):
            check_disjoint_training(d, 4, [47, 47])

if __name__ == "__main__":
    unittest.main()
