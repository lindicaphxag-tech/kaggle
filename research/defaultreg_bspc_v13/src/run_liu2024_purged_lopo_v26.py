#!/usr/bin/env python3
"""V26: retrospective exact-record-identity-purged public Liu2024 LOPO.

Important: the GLOBAL data identity audit necessarily used outer target EEG
signal contents to construct group membership, so this retrospective
reanalysis is NOT strict information-isolated Z0 or a blinded clinical test.

Within fixed recording groups, lambda choices and all models are fitted on
other representatives only, fully SHA-frozen before target label AUC joins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from moabb.datasets import Liu2024

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "defaultreg_proper_predictive"))
import original_frozen_runner as old  # noqa: E402
from run_proper_predictive_posthoc import select_proper_source_nll  # noqa: E402
from record_identity_groups_v26 import (  # noqa: E402
    assign_recording_groups, check_disjoint_training,
)

SUBJECTS = tuple(range(1, 51))
EVENTS = ("left_hand", "right_hand")
SOURCE_MANIFEST = "PURGED_RECORD_IDENTITY_SOURCE_POLICIES_BEFORE_TARGET_AUC.json"
GROUP_MANIFEST = "FULL_COHORT_EXACT_SIGNAL_GROUPS_AFTER_GLOBAL_AUDIT.json"
RESULTS = "PURGED_LIU2024_RECORD_INDEPENDENT_RETROSPECTIVE_RESULTS.json"
PROTOCOL = (ROOT / "research" / "defaultreg_bspc_v13" /
            "V26_RECORD_IDENTITY_PURGED_PRECOMMITTED_RETROSPECTIVE_PROTOCOL.md")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def save_json(obj, path):
    body = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return sha(body)


def load_and_group():
    paradigm = old.MotorImagery(
        n_classes=2, events=list(EVENTS), fmin=8, fmax=30,
        resample=160, baseline=None, reject_by_annotation=False,
    )
    ds = Liu2024()
    raw = []
    C_list, Y_list, G_list = [], [], []
    for subject in SUBJECTS:
        X, events, meta = paradigm.get_data(dataset=ds, subjects=[subject])
        events = np.asarray(events)
        yy = (events == EVENTS[1]).astype(np.uint8)
        if (X.ndim != 3 or X.shape[0] != 40 or X.shape[1] != 29
                or len(events) != 40 or int(yy.sum()) != 20
                or set(events.tolist()) != set(EVENTS)):
            raise RuntimeError("Liu2024 fixed EEG release schema changed")
        if not np.isfinite(X).all():
            raise RuntimeError("nonfinite preprocessed EEG")
        raw.append({"subject": int(subject), "signal": X, "labels": yy})
        C_list.append(old.covariances(X))
        Y_list.append(yy)
        G_list.extend([subject]*40)
        print("V26_RELEASE_EEG_DIGESTED", subject, flush=True)
    identity, identity_of = assign_recording_groups(raw)
    # The original all-50 immutable audit previously found exactly [4,5].
    # New disagreement is a data-version or preprocessing exception.
    if identity["duplicate_groups"] != [[4, 5]] or (
        identity["independent_processed_record_count"] != 49
    ):
        raise AssertionError("V25 all-50 immutable record-identity receipt disagreement")
    assert set(identity["original_subject_ids"]) == set(SUBJECTS)
    del raw
    C, y, g = (np.concatenate(C_list), np.concatenate(Y_list),
               np.asarray(G_list))
    return identity, identity_of, C, y, g


def get_representative_arrays(identity, C, y, g):
    reps = np.asarray(identity["representative_ids"])
    allowed = np.isin(g, reps)
    return C[allowed], y[allowed], g[allowed]


def freeze_group_safe_selection(identity, Cs, ys, gs, output):
    reps = list(map(int, identity["representative_ids"]))
    selected = []
    for target in reps:
        src = gs != target
        source_ids = sorted(map(int, set(gs[src])))
        check_disjoint_training(identity, target, source_ids)
        if len(source_ids) != len(reps)-1:
            raise AssertionError("group-safe source sample count not n_unique-1")
        path = old.source_path(Cs[src], ys[src], gs[src])
        feasible = list(map(float, path["feasible_full_lambdas"]))
        proper, predictive_path = select_proper_source_nll(
            Cs[src], ys[src], gs[src], feasible
        )
        macro = float(path["lambda_source_macro"])
        cent = float(path["lambda_covariance_full"])
        assert macro in feasible and cent in feasible and proper in feasible
        selected.append({
            "target_recording_representative": int(target),
            "source_representative_subjects": source_ids,
            "n_source_unique_recordings": len(source_ids),
            "n_source_trials": int(src.sum()),
            "chosen_lambdas": {
                "source_macro": macro,
                "proper_source_mean_gaussian": float(proper),
                "historical_class_centred": cent,
            },
            "source_grid_macro_auc": float(
                path["criteria_by_lambda"][str(macro)]["source_macro_auc"]
            ),
            "source_held_record_group_disjoint": True,
            "global_record_identity_preinspected_held_signals": True,
            "never_used_held_record_outcome_for_policy_tuning": True,
        })
        print("V26_PURGED_SOURCE_SELECTION_LOCKED", target, flush=True)
    if len(selected) != len(reps):
        raise AssertionError("some recording groups not selected")
    digest = save_json(selected, output / SOURCE_MANIFEST)
    print("V26_SOURCE_CHOICES_SHA_LOCKED_BEFORE_RECORD_TARGET_AUC", digest, flush=True)
    return selected, digest


def evaluate_groups(identity, C, y, g, choice_rows):
    kept = sorted(identity["representative_ids"])
    results = []
    for row in choice_rows:
        target = int(row["target_recording_representative"])
        tr = g != target
        va = g == target
        source_ids = sorted(map(int, set(g[tr])))
        check_disjoint_training(identity, target, source_ids)
        inv = old.reference(C[tr])
        Xtr = old.tangent(C[tr], inv)
        Xva = old.tangent(C[va], inv)
        state = old.spectral_state(Xtr, y[tr])
        scores = {
            k: old.scalar_scores(state, Xva, lam)
            for k, lam in row["chosen_lambdas"].items()
        }
        for kind in ("LW", "OAS"):
            sc, _ = old.analytic_scores(Xtr, y[tr], Xva, kind)
            scores[kind] = sc
        aucs = {k: float(roc_auc_score(y[va], sc)) for k, sc in scores.items()}
        if not all(0. <= x <= 1. for x in aucs.values()):
            raise AssertionError("invalid computed AUC")
        results.append({
            "representative_recording": target,
            "held_trials": int(va.sum()),
            "source_unique_record_count": len(source_ids),
            "auc": aucs,
        })
        print("V26_PURGED_TARGET_RECORD_AUC", target, aucs, flush=True)
    if set(r["representative_recording"] for r in results) != set(kept):
        raise AssertionError("record-independent group denominator mismatch")
    return results


def summarise(identity, rows, digest, group_digest):
    methods = list(rows[0]["auc"])
    auc = {m: np.asarray([p["auc"][m] for p in rows]) for m in methods}
    macro = auc["source_macro"]
    proper = auc["proper_source_mean_gaussian"]
    deltas = macro - proper
    # The same physical EEG identity appearing in two IDs is represented once.
    rng = np.random.default_rng(20261013)
    n = len(rows)
    ix = rng.integers(0, n, (20000, n))
    return {
        "status": "RETROSPECTIVE_LIU2024_SIGNAL_IDENTITY_PURGED_LOPO_COMPLETE",
        "not_new_independent_target_confirmation": True,
        "no_strict_Z0_because_record_identity_used_target_signals": True,
        "source_policy_SHA_before_held_record_auc": digest,
        "global_record_identity_SHA_before_selection": group_digest,
        "n_original_subject_ids": 50,
        "n_exact_processed_recording_groups": n,
        "n_duplicate_ids_discarded": 50-n,
        "total_evaluated_trials": int(sum(p["held_trials"] for p in rows)),
        "frozen_preprocessing": "MOABB-1.7.2 8-30 Hz resample160 29 EEG channels",
        "source_same_processed_record_excluded": True,
        "exact_duplicate_groups": identity["duplicate_groups"],
        "macro_auc_by_method": {m:float(v.mean()) for m,v in auc.items()},
        "source_macro_minus_proper_mean_AUC": float(deltas.mean()),
        "source_macro_minus_proper_descriptive_bootstrap_95":
            np.quantile(deltas[ix].mean(axis=1),[.025,.975]).tolist(),
        "source_macro_minus_proper_harmed_records_GT002":
            int(np.sum(deltas < -.020000000001)),
        "source_macro_minus_proper_improved_records_GT002":
            int(np.sum(deltas > .020000000001)),
        "original_exposed_unpurged_50_ID_source_macro_AUC": .594,
        "original_exposed_unpurged_50_ID_proper_NLL_AUC": .58985,
        "cannot_causally_attribute_effect_to_leakage_alone": True,
        "target_outcomes": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args=parser.parse_args()
    if not PROTOCOL.exists():
        raise RuntimeError("V26 retrospective protocol not frozen")
    args.out.mkdir(parents=True, exist_ok=True)
    identity, identity_map, C, y, g = load_and_group()
    group_sha = save_json(identity, args.out / GROUP_MANIFEST)
    Cunique, yunique, gunique = get_representative_arrays(identity, C, y, g)
    assert len(Cunique)==49*40 and len(set(gunique))==49
    selections, choice_sha=freeze_group_safe_selection(
        identity, Cunique, yunique, gunique, args.out
    )
    # No target ROC-AUC is evaluated until the group-safe source
    # choices have all been frozen and saved.
    results = evaluate_groups(identity, Cunique, yunique, gunique, selections)
    report = summarise(identity, results, choice_sha, group_sha)
    outcome_sha = save_json(report, args.out/RESULTS)
    print(json.dumps({
        "status":report["status"],
        "unique_processed_recordings":report["n_exact_processed_recording_groups"],
        "macro_auc_by_method":report["macro_auc_by_method"],
        "macro_minus_proper":report["source_macro_minus_proper_mean_AUC"],
        "source_manifest_sha256":choice_sha,
        "outcome_report_sha256":outcome_sha,
        "NO_NEW_CONFIRMATORY_Z0_CLAIM":True,
    },indent=2), flush=True)


if __name__=="__main__":
    main()
