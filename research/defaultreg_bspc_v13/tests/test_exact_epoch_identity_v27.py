"""Unit tests for exact partial cross-subject EEG data leakage."""
import pathlib
import sys
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from exact_epoch_identity_v27 import audit_exact_epoch_overlap


def dataset():
    # Four full records are all distinct; the group [1,2,3] forms
    # through *partial* overlap of 1<->2 and 2<->3.
    y = np.array([0, 0, 1, 1])
    a = np.arange(24., dtype=float).reshape(4, 2, 3)
    b = a + 100.
    c = a + 200.
    d = a + 300.
    b[0] = a[0]     # consistent zero-class shared epoch 1/2
    c[2] = b[2]     # consistent one-class shared epoch 2/3
    return [
        {"subject":i, "signal":X, "labels":y.copy()}
        for i, X in enumerate((a, b, c, d), 1)
    ]


class V27EpochOverlapTests(unittest.TestCase):
    def test_partial_overlap_and_transitive_grouping(self):
        out = audit_exact_epoch_overlap(dataset())
        self.assertEqual(out["n_total_trial_instances"], 16)
        self.assertEqual(out["cross_subject_shared_unique_epoch_contents"], 2)
        self.assertEqual(out["n_unique_exact_epoch_contents"], 14)
        self.assertEqual(out["exact_epoch_connected_components"], [[1,2,3],[4]])
        self.assertEqual(out["n_exact_epoch_connected_groups"], 2)
        self.assertEqual(
            {(x["subject_a"],x["subject_b"],x["shared_unique_epoch_contents"])
             for x in out["cross_subject_exact_overlap_edges"]},
            {(1,2,1),(2,3,1)}
        )
        self.assertEqual(out["within_subject_redundant_epoch_instances"], 0)
        self.assertTrue(out["global_EEG_content_access_for_grouping"])

    def test_whole_record_duplicate_counts_all_epochs(self):
        data=dataset()
        data[1]["signal"]=data[0]["signal"].copy()
        out=audit_exact_epoch_overlap(data[:2])
        self.assertEqual(out["exact_epoch_connected_components"], [[1,2]])
        self.assertEqual(out["cross_subject_shared_unique_epoch_contents"], 4)
        self.assertEqual(out["cross_subject_redundant_epoch_instances_lower_bound"], 4)

    def test_mismatched_label_rejected(self):
        data=dataset()
        data[1]["labels"][0]=1 # exact duplicate epoch now opposite label
        with self.assertRaises(ValueError):
            audit_exact_epoch_overlap(data)

    def test_within_record_exact_repetition_is_separate(self):
        data=dataset()
        data[3]["signal"][1] = data[3]["signal"][0]
        out=audit_exact_epoch_overlap(data)
        self.assertEqual(out["within_subject_redundant_epoch_instances"],1)
        self.assertEqual(out["exact_epoch_connected_components"],[[1,2,3],[4]])

    def test_repeated_subject_id_rejected(self):
        data=dataset()
        with self.assertRaises(ValueError):
            audit_exact_epoch_overlap([data[0],data[0]])

    def test_invalid_epoch_or_labels_rejected(self):
        data=dataset()
        data[0]["signal"][0,0,0]=float("nan")
        with self.assertRaises(ValueError):
            audit_exact_epoch_overlap(data)
        data=dataset()
        data[2]["labels"]=np.ones(4,dtype=int)
        with self.assertRaises(ValueError):
            audit_exact_epoch_overlap(data)

    def test_same_content_without_any_cross_subject_duplication(self):
        data=dataset()
        for i in range(1,4):
            data[i]["signal"] += i*1000
        out=audit_exact_epoch_overlap(data)
        self.assertEqual(out["n_exact_epoch_connected_groups"],4)
        self.assertEqual(out["n_subject_pairs_with_exact_overlap"],0)
        self.assertEqual(out["cross_subject_shared_unique_epoch_contents"],0)

if __name__ == "__main__":
    unittest.main()
