"""V27 exact-EEG-epoch grouping, independent of classifier predictions.

Any global access to target EEG for duplicate grouping makes this a
RETROSPECTIVE dataset-integrity analysis, never strict unobserved-target Z0.
Hashes detect candidates; candidate identity is confirmed by array equality.
No raw EEG or per-trial hashes are exported in the summary.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import numpy as np


def _trial_digest(arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError("each EEG trial must be a finite channel x sample array")
    h = hashlib.sha256()
    h.update(json.dumps(list(arr.shape)).encode("ascii"))
    h.update(b"\x00")
    h.update(arr.tobytes())
    return h.hexdigest()


def audit_exact_epoch_overlap(records):
    """Analyze all subject-level recordings; output contains no EEG samples.

    records: iterable of {subject, signal [trial, EEG_channel, time], labels}.
    Repeated trial content with discordant labels fails closed.
    Cross-ID edges are established by *any* identical trial, including
    a partial overlap insufficient to duplicate the whole recording.
    """
    records = list(records)
    if not records:
        raise ValueError("empty EEG cohort")
    parent = {}
    # SHA -> {representative_epoch, occurrences=(subject, trial_idx, label)}
    seen = {}
    group_sizes = {}
    subject_labels = {}
    total_trials = 0
    unique_shapes = set()
    for rec in records:
        sid = int(rec["subject"])
        if sid in parent:
            raise ValueError("duplicate subject identifier")
        parent[sid] = sid
        X = np.asarray(rec["signal"], dtype=np.float64)
        labels = np.asarray(rec["labels"])
        if X.ndim != 3 or labels.shape != (len(X),) or not np.isfinite(X).all():
            raise ValueError("EEG signal/label shapes or finite values invalid")
        if len(X) < 2 or not np.isin(labels, [0, 1]).all() or set(labels.tolist()) != {0, 1}:
            raise ValueError("each recording needs both binary classes")
        unique_shapes.add(tuple(X.shape[1:]))
        group_sizes[sid] = int(len(X))
        subject_labels[sid] = [int((labels == 0).sum()), int((labels == 1).sum())]
        total_trials += len(X)
        for trial_idx, (trial, label) in enumerate(zip(X, labels)):
            key = _trial_digest(trial)
            label = int(label)
            if key not in seen:
                seen[key] = [np.array(trial, copy=True), [(sid, trial_idx, label)]]
            else:
                previous, rows = seen[key]
                if not np.array_equal(previous, trial):
                    raise AssertionError("SHA256 collision on different numeric EEG trials")
                if any(lab != label for _, _, lab in rows):
                    raise ValueError("exact same EEG trial has conflicting event labels")
                rows.append((sid, trial_idx, label))
    if len(unique_shapes) != 1:
        raise ValueError("different EEG trial channel or temporal dimensions")

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Count shared unique epoch contents for each ID-pair. A participant
    # with two identical trials is not mistaken for another participant.
    shared_types = defaultdict(int)
    instance_edges = defaultdict(int)
    cross_id_shared_hashes = 0
    within_id_redundant_instances = 0
    across_id_redundant_instances = 0

    for canonical, rows in seen.values():
        by_id = defaultdict(int)
        for sid, index, lab in rows:
            by_id[sid] += 1
        within_id_redundant_instances += sum(n - 1 for n in by_id.values())
        ids = sorted(by_id)
        if len(ids) < 2:
            continue
        cross_id_shared_hashes += 1
        across_id_redundant_instances += sum(by_id.values()) - max(by_id.values())
        for i, sid in enumerate(ids):
            for other in ids[i + 1:]:
                shared_types[(sid, other)] += 1
                instance_edges[(sid, other)] += by_id[sid] * by_id[other]
                union(sid, other)

    components = defaultdict(list)
    for sid in sorted(parent):
        components[root(sid)].append(sid)
    all_groups = sorted([sorted(x) for x in components.values()], key=lambda x: x[0])
    edges = [
        {
            "subject_a": int(a), "subject_b": int(b),
            "shared_unique_epoch_contents": int(types),
            "cross_subject_matching_epoch_instance_pairs": int(instance_edges[a, b]),
            "trials_in_a": group_sizes[a], "trials_in_b": group_sizes[b],
        }
        for (a, b), types in sorted(shared_types.items())
    ]

    return {
        "status": "V27_RETROSPECTIVE_EXACT_EEG_EPOCH_IDENTITY_COMPLETE",
        "no_target_auc_or_model_parameter_read": True,
        "global_EEG_content_access_for_grouping": True,
        "not_prospective_or_strict_Z0": True,
        "comparison_criterion": "exact_numeric_float64_epoch_and_label_consistency",
        "n_original_subject_ids": len(parent),
        "n_total_trial_instances": int(total_trials),
        "n_unique_exact_epoch_contents": len(seen),
        "epoch_shape_channels_samples": list(next(iter(unique_shapes))),
        "subject_trial_counts": {str(k):v for k,v in sorted(group_sizes.items())},
        "subject_binary_class_counts": {str(k):v for k,v in sorted(subject_labels.items())},
        "cross_subject_shared_unique_epoch_contents": int(cross_id_shared_hashes),
        "cross_subject_redundant_epoch_instances_lower_bound": int(across_id_redundant_instances),
        "within_subject_redundant_epoch_instances": int(within_id_redundant_instances),
        "n_subject_pairs_with_exact_overlap": len(edges),
        "cross_subject_exact_overlap_edges": edges,
        "exact_epoch_connected_components": all_groups,
        "n_exact_epoch_connected_groups": len(all_groups),
        "n_original_subject_IDs_in_nontrivial_components": sum(
            len(g) for g in all_groups if len(g) > 1
        ),
        "no_patient_identity_inference": True,
        "absence_of_exact_overlap_does_not_rule_out_near_duplicates": True,
    }
