"""V28 independent epoch group certificate: adversarial invariants."""
import copy
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from v28_epoch_split_attestation import reconstruct, verify_with_purged


def synthetic():
    groups=[[i] for i in range(1,51)]
    groups[3]=[4,5]
    groups.pop(4)
    return {
        "status":"V27_RETROSPECTIVE_EXACT_EEG_EPOCH_IDENTITY_COMPLETE",
        "global_EEG_content_access_for_grouping":True,
        "subject_trial_counts":{str(i):40 for i in range(1,51)},
        "subject_binary_class_counts":{str(i):[20,20] for i in range(1,51)},
        "n_original_subject_ids":50,
        "n_total_trial_instances":2000,
        "n_unique_exact_epoch_contents":1960,
        "epoch_shape_channels_samples":[29,560],
        "cross_subject_exact_overlap_edges":[{
            "subject_a":4,"subject_b":5,
            "shared_unique_epoch_contents":40,
            "cross_subject_matching_epoch_instance_pairs":40,
            "trials_in_a":40,"trials_in_b":40,
        }],
        "n_exact_epoch_connected_groups":49,
        "exact_epoch_connected_components":groups
    }


class GroupCertificateTests(unittest.TestCase):
    def test_correct_49_record_graph_and_no_duplicate_training(self):
        d=reconstruct(synthetic())
        self.assertEqual(d["n_distinct_epoch_safe_recording_groups"],49)
        self.assertEqual(d["n_exact_epoch_unique_contents"],1960)
        self.assertEqual(d["epoch_components"][3],[4,5])
        fold=next(f for f in d["all_held_group_source_splits"]
                  if f["target_recording_representative"]==4)
        self.assertEqual(fold["n_source_unique_groups"],48)
        self.assertNotIn(5,fold["source_representatives"])
        self.assertNotIn(4,fold["source_representatives"])

    def test_missing_or_spurious_report_edges_are_rejected(self):
        d=synthetic()
        d["cross_subject_exact_overlap_edges"]=[]
        with self.assertRaises(AssertionError):
            reconstruct(d)
        d=synthetic()
        d["cross_subject_exact_overlap_edges"].append(
            copy.deepcopy(d["cross_subject_exact_overlap_edges"][0]))
        with self.assertRaises(AssertionError):
            reconstruct(d)

    def test_partial_overlap_beyond_whole_record_forces_group_merge(self):
        d=synthetic()
        d["cross_subject_exact_overlap_edges"].append({
            "subject_a":5,"subject_b":6,"shared_unique_epoch_contents":1,
            "cross_subject_matching_epoch_instance_pairs":1,
            "trials_in_a":40,"trials_in_b":40
        })
        d["exact_epoch_connected_components"][3]=[4,5,6]
        d["exact_epoch_connected_components"].remove([6])
        d["n_exact_epoch_connected_groups"]=48
        z=reconstruct(d)
        self.assertEqual(z["n_distinct_epoch_safe_recording_groups"],48)
        t=next(x for x in z["all_held_group_source_splits"]
              if x["target_recording_representative"]==4)
        self.assertNotIn(6,t["source_representatives"])

    def test_any_denominator_or_subject_schema_drift_fails_closed(self):
        d=synthetic()
        d["subject_binary_class_counts"]["49"]=[39,1]
        with self.assertRaises(AssertionError):
            reconstruct(d)
        d=synthetic()
        d["n_unique_exact_epoch_contents"]=2001
        with self.assertRaises(AssertionError):
            reconstruct(d)
        d=synthetic()
        d["cross_subject_exact_overlap_edges"][0]["shared_unique_epoch_contents"]=41
        with self.assertRaises(AssertionError):
            reconstruct(d)


if __name__=="__main__":
    unittest.main()
