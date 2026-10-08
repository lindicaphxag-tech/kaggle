"""Original open-access PhysioNet EEGBCI recordings as *real signal evidence*.

A deliberately separate/source-disjoint held-out test is NOT established:
S001 and S002 are only two independent participants. We create controlled
mislabeling only by numerical reindexing (not actual hardware faults).
Do not upload raw EEG; release aggregate decisions, source hashes and limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from research.electrode_identity.anchor_identity import authorization

SOURCE = "https://physionet.org/files/eegmmidb/1.0.0"
PUBLIC_FILES = (("S001", "R01"), ("S001", "R02"), ("S002", "R01"), ("S002", "R02"))
THRESHOLD = .012


def load_public_record(subject: str, run: str, cache: Path):
    import mne

    base = cache / subject
    base.mkdir(parents=True, exist_ok=True)
    name = subject + run + ".edf"
    file = base / name
    if not file.is_file():
        req = urllib.request.Request(
            f"{SOURCE}/{subject}/{name}",
            headers={"User-Agent": "ResearchReproduction/1.0"},
        )
        with urllib.request.urlopen(req, timeout=90) as stream:
            file.write_bytes(stream.read())
    image_hash = hashlib.sha256(file.read_bytes()).hexdigest()
    raw = mne.io.read_raw_edf(str(file), preload=False, verbose="ERROR")
    rate = float(raw.info["sfreq"])
    if rate != 160.0:
        raise ValueError("Unexpected PhysioNet EEGBCI 160-Hz source")
    eeg_ix = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude=[])
    if len(eeg_ix) != 64:
        raise ValueError(f"Original source expected 64 physical channels: {len(eeg_ix)}")
    name_ix = sorted(
        (raw.ch_names[i].rstrip(".").lower(), int(i)) for i in eeg_ix
    )
    if len(set(name for name, _ in name_ix)) != 64:
        raise ValueError("Duplicate electrode identities in physical source recording")
    x = raw.get_data(picks=[i for _, i in name_ix], start=0, stop=9600)
    if x.shape != (64, 9600):
        raise ValueError("Source baseline is not 60 sec")
    sos = butter(4, [1, 35], btype="bandpass", fs=rate, output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    if not np.isfinite(x).all():
        raise ValueError("Public source EEG contains nonfinite values")
    return x, {"file": name, "sha256": image_hash, "sfreq": rate}


def predeclared_roles():
    candidate = tuple(range(0, 64, 5))[:12]
    anchors = tuple(i for i in range(64) if i not in candidate)[:16]
    if len(candidate) != 12 or len(anchors) != 16 or (set(candidate) & set(anchors)):
        raise ValueError("Predeclared electrode role identity chart is invalid")
    return candidate, anchors


def injected_permutations(n):
    permutations = {"identity": tuple(range(n))}
    settings = {
        "one_swap": ((1, 7),),
        "two_swaps": ((1, 7), (3, 9)),
        "four_swaps": ((0, 8), (1, 7), (2, 10), (3, 9)),
    }
    for key, swaps in settings.items():
        perm = list(range(n))
        for left, right in swaps:
            perm[left], perm[right] = perm[right], perm[left]
        permutations[key] = tuple(perm)
    return permutations


def score_pair(reference, target, protocol, subject):
    ix, anchor_ix = predeclared_roles()
    source, ref_anchors = reference[np.array(ix)], reference[np.array(anchor_ix)]
    dest, dst_anchors = target[np.array(ix)], target[np.array(anchor_ix)]
    def cut(y, lo, hi):
        return y[:, lo * 160 : hi * 160]
    refs = (
        (cut(source, 0, 12), cut(ref_anchors, 0, 12)),
        (cut(source, 12, 24), cut(ref_anchors, 12, 24)),
    )
    if protocol == "same_baseline_time_disjoint":
        targets = (
            (cut(dest, 30, 42), cut(dst_anchors, 30, 42)),
            (cut(dest, 42, 54), cut(dst_anchors, 42, 54)),
        )
    else:
        # R01 eyes-open reference versus R02 eyes-closed target:
        # an actual brain-state shift, not matched waveform.
        targets = (
            (cut(dest, 0, 12), cut(dst_anchors, 0, 12)),
            (cut(dest, 12, 24), cut(dst_anchors, 12, 24)),
        )
    rows = []
    for label, truth in injected_permutations(len(ix)).items():
        p = np.array(truth, int)
        a = authorization(
            refs[0][0], refs[0][1], refs[1][0], refs[1][1],
            targets[0][0][p], targets[0][1],
            targets[1][0][p], targets[1][1],
            minimum_global_margin=THRESHOLD,
        )
        inferred = a.assignment
        rows.append({
            "subject": subject,
            "protocol": protocol,
            "controlled_injection": label,
            "switched_signal_rows": int(np.count_nonzero(p != np.arange(len(p)))),
            "decision": a.status,
            "exact_correct_authorized_repair": (
                a.status == "authorize_repair" and inferred == truth
            ),
            "incorrectly_authorized_repair": (
                a.status == "authorize_repair" and inferred != truth
            ),
            "witnesses_agree": a.witnesses[0].assignment == a.witnesses[1].assignment,
            "minimum_global_assignment_margin": float(min(
                z.alternative_gap_per_channel for z in a.witnesses
            )),
        })
    return rows


def run(cache: Path, output: Path):
    recordings, hashes = {}, {}
    for subject, run_id in PUBLIC_FILES:
        x, proof = load_public_record(subject, run_id, cache)
        recordings[(subject, run_id)] = x
        hashes[subject + run_id] = proof
    results = []
    for subject in ("S001", "S002"):
        opened = recordings[(subject, "R01")]
        closed = recordings[(subject, "R02")]
        results += score_pair(opened, opened, "same_baseline_time_disjoint", subject)
        results += score_pair(opened, closed, "cross_eyes_open_to_closed", subject)
    report = {
        "public_source": SOURCE,
        "source_citation": "Schalk 2009, PhysioNet EEGMMIDB v1.0.0, DOI:10.13026/C28G6P",
        "n_independent_subjects": 2,
        "n_subject_condition_injection_cells": len(results),
        "physical_real_eeg": True,
        "actual_wrong_electrode_hardware_events": False,
        "original_pretrained_neurorvq_executed": False,
        "physically_trusted_anchors_assumed": True,
        "source_reference_recordings_assumed_physically_correct": True,
        "no_target_eeg_task_labels_used": True,
        "two_temporally_separate_witnesses": True,
        "frozen_minimum_global_margin": THRESHOLD,
        "no_tuned_parameters_or_inferred_error_probability": True,
        "safety_or_clinical_claim": False,
        "original_file_hashes": hashes,
        "cases": results,
        "exact_correct_authorizations": sum(r["exact_correct_authorized_repair"] for r in results),
        "wrong_authorizations": sum(r["incorrectly_authorized_repair"] for r in results),
        "refusals": sum(r["decision"] == "refuse" for r in results),
        "false_repair_on_uncorrupted_record": sum(
            r["incorrectly_authorized_repair"] for r in results
            if r["controlled_injection"] == "identity"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in (
        "n_independent_subjects", "n_subject_condition_injection_cells",
        "exact_correct_authorizations", "wrong_authorizations", "refusals",
        "false_repair_on_uncorrupted_record",
    )}, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.cache, args.output)
