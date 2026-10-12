"""V26 recording-identity groups built from exact public preprocessed EEG.

GLOBAL preprocessing labels/signals are inspected to determine identities:
this is RETROSPECTIVE dataset-integrity analysis, not strict zero-access
to held person's features. Hash collisions fail closed.
"""
from __future__ import annotations
import hashlib
import numpy as np


def array_digest(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.tobytes()).hexdigest()


def assign_recording_groups(records):
    """Records contain {subject,signal,labels}. No performance-based grouping.

    Return (identity_manifest, representative_ID_list).
    Every group has exactly equal full preprocessed arrays and label vectors;
    a matching waveform with discordant labels is an integrity failure.
    """
    if not records:
        raise ValueError("empty recording cohort")
    signatures = {}
    subjects = set()
    summaries = []
    for rec in records:
        subject = int(rec["subject"])
        if subject in subjects:
            raise ValueError("repeated original subject identifier")
        subjects.add(subject)
        signal = np.asarray(rec["signal"], dtype=np.float64)
        labels = np.asarray(rec["labels"])
        if signal.ndim != 3 or labels.shape != (len(signal),):
            raise ValueError("trial signal/label shape mismatch")
        if not np.isfinite(signal).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("non-finite signal or invalid event labels")
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("source identity record lacks both binary labels")
        signal_sha = array_digest(signal)
        label_sha = array_digest(np.asarray(labels, dtype=np.uint8))
        if signal_sha in signatures:
            old_signal, old_labels, old_label_sha = signatures[signal_sha]
            if not np.array_equal(signal, old_signal):
                raise AssertionError("SHA collision: identical hash for different EEG")
            if not np.array_equal(labels, old_labels):
                raise ValueError("same EEG but conflicting event labels")
            if label_sha != old_label_sha:
                raise AssertionError("label digest mismatch")
        else:
            signatures[signal_sha] = (signal.copy(), labels.copy(), label_sha)
        summaries.append({
            "subject": subject,
            "preprocessed_signal_sha256": signal_sha,
            "binary_label_sha256": label_sha,
            "trials": int(len(labels)),
            "channels": int(signal.shape[1]),
            "samples": int(signal.shape[2]),
        })
    by_signal = {}
    for row in summaries:
        by_signal.setdefault(row["preprocessed_signal_sha256"], []).append(row["subject"])
    groups = sorted([sorted(v) for v in by_signal.values()], key=lambda x: x[0])
    reps = sorted([x[0] for x in groups])
    group_of = {i: group[0] for group in groups for i in group}
    return {
        "status": "EXACT_PROCESSED_EEG_RECORD_IDENTITY_AUDIT",
        "identity_criteria": "float64_contiguous_epoch_array_and_exact_binary_labels",
        "global_target_signal_access_for_grouping": True,
        "not_strict_Z0_selection_or_new_independent_test": True,
        "original_subject_ids": sorted(subjects),
        "independent_processed_record_count": len(groups),
        "discarded_exact_copies": len(summaries) - len(groups),
        "representative_ids": reps,
        "exact_record_groups": groups,
        "duplicate_groups": [g for g in groups if len(g) > 1],
        "original_record_receipts": sorted(summaries, key=lambda x: x["subject"]),
    }, group_of


def check_disjoint_training(record_identity, target_representative, source_representatives):
    groups = record_identity["exact_record_groups"]
    by_subject = {subject: group[0] for group in groups for subject in group}
    t = int(target_representative)
    if t not in by_subject:
        raise ValueError("unknown target identity")
    src = [int(s) for s in source_representatives]
    if len(src) != len(set(src)):
        raise ValueError("duplicated source representative")
    if any(s not in by_subject for s in src):
        raise ValueError("unknown source identity")
    if any(by_subject[s] == by_subject[t] for s in src):
        raise AssertionError("TRAIN/TARGET RECORDING-IDENTITY CONTAMINATION")
    if any(by_subject[s] != s for s in src):
        raise AssertionError("nonrepresentative recording entered source training")
    return True
