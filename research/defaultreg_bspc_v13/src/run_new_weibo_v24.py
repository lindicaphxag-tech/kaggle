#!/usr/bin/env python3
"""V24 new-acquisition Weibo2014 source-locked transfer test.

Protocol: V24_WEIBO2014_PRE_OUTCOME_FIRST_RUN_PROTOCOL.md committed before run.
NO outcome-dependent λ or threshold adaptation. Run all source choices and
SHA freeze BEFORE computing *any* held outer target score or AUC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
from moabb.datasets import Weibo2014
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "defaultreg_proper_predictive"))
import original_frozen_runner as old  # noqa: E402
from run_proper_predictive_posthoc import select_proper_source_nll  # noqa: E402
from certified_continuous_selector_v23 import (  # noqa: E402
    certified_source_auc_search, source_person_auc_at_lambda,
)

PEOPLE = tuple(range(1, 11))
CHANNELS = ("C3", "Cz", "C4")
EVENTS = ("left_hand", "right_hand")
SOURCE_CHOICES = "WEIBO2014_10_PERSON_SOURCE_ONLY_CHOICES_FROZEN_BEFORE_TARGET.json"
SOURCE_RESULTS = "WEIBO2014_10_PERSON_FIRST_RUN_EXTERNAL_TARGET_AUC.json"


def digest(b):
    return hashlib.sha256(b).hexdigest()


def load_public():
    paradigm = old.MotorImagery(
        n_classes=2,
        events=list(EVENTS),
        channels=list(CHANNELS),
        fmin=8, fmax=30, resample=160,
        baseline=None, reject_by_annotation=False,
    )
    data = Weibo2014()
    covs, labels, groups, receipts = [], [], [], []
    for person in PEOPLE:
        X, lab, meta = paradigm.get_data(dataset=data, subjects=[person])
        lab = np.asarray(lab)
        if X.ndim != 3 or X.shape[1] != 3 or set(lab.tolist()) != set(EVENTS):
            raise RuntimeError(f"Weibo2014 locked sensor/event failure: {person}")
        y = (lab == "right_hand").astype(int)
        if np.min(np.bincount(y, minlength=2)) < 20 or not np.isfinite(X).all():
            raise RuntimeError(f"Weibo2014 unavailable class/trial data: {person}")
        covs.append(old.covariances(X))
        labels.append(y)
        groups.extend([person] * len(y))
        receipts.append({
            "person": person, "total_trials": len(y),
            "left_hand": int((y == 0).sum()),
            "right_hand": int((y == 1).sum()),
            "channels": 3, "samples_per_trial": int(X.shape[2]),
        })
        print("WEIBO_LOCKED_ACQUISITION", person, len(y), flush=True)
        del X
    C = np.concatenate(covs)
    y = np.concatenate(labels)
    groups = np.asarray(groups)
    if set(groups) != set(PEOPLE) or C.shape[1:] != (3, 3):
        raise RuntimeError("source EEG participant or sensor denominator mismatch")
    return C, y, groups, receipts


def source_models(C, y, groups):
    models = []
    for held in old._held_groups(sorted(set(map(int, groups)))):
        va = np.isin(groups, held)
        tr = ~va
        inv = old.reference(C[tr])
        Xtr = old.tangent(C[tr], inv)
        Xva = old.tangent(C[va], inv)
        st = old.spectral_state(Xtr, y[tr])
        for p in held:
            keep = groups[va] == p
            yy = y[va][keep]
            if set(yy.tolist()) != {0, 1}:
                raise RuntimeError("source-person validation lacks a class")
            models.append((st, Xva[keep], yy))
    if len(models) != len(set(groups)):
        raise RuntimeError("source-person validation denominator mismatch")
    return models


def select_all(C, y, group):
    frozen = []
    for held in PEOPLE:
        source = group != held
        Cs, ys, gs = C[source], y[source], group[source]
        path = old.source_path(Cs, ys, gs)
        feasible = list(map(float, path["feasible_full_lambdas"]))
        if len(feasible) < 20 or min(feasible) > 0 or max(feasible) < 1:
            raise RuntimeError("complete [0,1] continuum not feasible under fixed model")
        proper, proper_path = select_proper_source_nll(Cs, ys, gs, feasible)
        models = source_models(Cs, ys, gs)
        historical = float(path["lambda_covariance_full"])
        macro = float(path["lambda_source_macro"])
        original_auc = float(path["criteria_by_lambda"][str(macro)]["source_macro_auc"])
        original_reproduced = float(np.mean([
            source_person_auc_at_lambda(m, macro) for m in models
        ]))
        if abs(original_auc-original_reproduced) > 1e-10:
            raise RuntimeError("source path mismatch")
        newer = certified_source_auc_search(
            models, low=0., high=1., tolerance=.02,
            max_splits=5, segments=1, initial_lambdas=feasible,
        )
        if newer["best_evaluated_source_macro_auc"] < original_auc - 1e-10:
            raise RuntimeError("V24 anchored source-search unexpectedly lost original grid")
        frozen.append({
            "held_outer_person": held,
            "source_people": sorted(map(int, set(gs))),
            "selected_lambdas": {
                "source_macro_grid": macro,
                "source_correct_predictive_nll": float(proper),
                "historical_validation_class_centred": historical,
                "v24_anchored_continuous": newer["selected_lambda_source_macro"],
                "fixed_lambda_0_50": .5,
            },
            "source_macro_grid_auc": original_auc,
            "source_continuous_evaluated_auc": newer["best_evaluated_source_macro_auc"],
            "source_continuous_upper_bound": newer["upper_bound_global_continuous_source_macro_auc"],
            "continuous_source_optimality_gap": newer["residual_source_optimality_gap_upper_bound"],
            "continuous_certified_to_002": newer["source_optimality_certificate_met"],
            "source_lambda_evaluations": newer["lambda_evaluations"],
            "source_interval_splits": newer["interval_splits"],
            "source_fit_subjects_exclude_target": True,
            "no_held_target_data_in_regularisation_selection": True,
        })
        print("WEIBO_SOURCE_DECISION_BEFORE_TARGET", held, frozen[-1]["selected_lambdas"], flush=True)
    return frozen


def run(out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = pathlib.Path(__file__).resolve().parents[1] / "V24_WEIBO2014_PRE_OUTCOME_FIRST_RUN_PROTOCOL.md"
    if not protocol.exists():
        raise RuntimeError("protocol not committed")
    proto_hash = digest(protocol.read_bytes())
    C, y, group, receipts = load_public()
    source_choices = select_all(C, y, group)
    if len(source_choices) != 10 or set(r["held_outer_person"] for r in source_choices) != set(PEOPLE):
        raise RuntimeError("not all source-only choices frozen")
    choice_bytes = (json.dumps(source_choices, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (out / SOURCE_CHOICES).write_bytes(choice_bytes)
    choice_hash = digest(choice_bytes)
    print("WEIBO_SOURCE_CHOICES_SHA256_FROZEN_BEFORE_ANY_TARGET_SCORE", choice_hash, flush=True)

    # Held TARGET labels / features first used for prediction AFTER source lock.
    target_records = []
    for item in source_choices:
        held = item["held_outer_person"]
        src = group != held
        test = ~src
        inv = old.reference(C[src])
        state = old.spectral_state(old.tangent(C[src], inv), y[src])
        Xheld = old.tangent(C[test], inv)
        target_scores = {
            k: old.scalar_scores(state, Xheld, lam)
            for k, lam in item["selected_lambdas"].items()
        }
        for analytic in ("LW", "OAS"):
            score, shrink = old.analytic_scores(
                old.tangent(C[src], inv), y[src], Xheld, analytic)
            target_scores[analytic] = score
        target_auc = {
            k: float(roc_auc_score(y[test], v))
            for k, v in target_scores.items()
        }
        target_records.append({
            "held_person": held, "n_trials": int(test.sum()),
            "auc": target_auc,
            "v24_minus_macro": (
                target_auc["v24_anchored_continuous"]-target_auc["source_macro_grid"]
            ),
            "v24_minus_proper": (
                target_auc["v24_anchored_continuous"]-target_auc["source_correct_predictive_nll"]
            ),
        })
        print("WEIBO_FROZEN_TARGET_EVALUATED", held, flush=True)

    methods = list(target_records[0]["auc"])
    means = {k: float(np.mean([p["auc"][k] for p in target_records]))
             for k in methods}
    deltas = np.asarray([p["v24_minus_macro"] for p in target_records], float)
    proper_deltas = np.asarray([p["v24_minus_proper"] for p in target_records], float)
    results = {
        "status": "WEIBO2014_NEW_EXTERNAL_FIRST_RUN_COMPLETE_UNREGISTERED",
        "cohort": "Weibo2014",
        "protocol_sha256": proto_hash,
        "source_lambda_decisions_sha256_before_target": choice_hash,
        "n_outer_people": len(PEOPLE),
        "n_total_trials": int(len(y)),
        "n_channels": 3,
        "n_tangent_features": 6,
        "no_held_target_selected_parameter": True,
        "no_preregistration_or_sota_claim": True,
        "all_people_retained": True,
        "source_selector_ran_before_all_target_scores": True,
        "mean_auc": means,
        "mean_v24_minus_macro": float(deltas.mean()),
        "mean_v24_minus_proper": float(proper_deltas.mean()),
        "harm_gt_002_v24_vs_macro": int((deltas < -.02-1e-12).sum()),
        "improvement_gt_002_v24_vs_macro": int((deltas > .02+1e-12).sum()),
        "n_source_optimality_certificates_closed": sum(
            int(d["continuous_certified_to_002"]) for d in source_choices
        ),
        "participant_receipts": receipts,
        "per_person_target_outcomes": target_records,
        "note": "No independent preregistration; shared source pools, person-level descriptive statistics only."
    }
    (out / SOURCE_RESULTS).write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")
    print("WEIBO_COMPLETE_RESULT_SUMMARY", json.dumps({
        k: results[k] for k in (
            "status", "n_outer_people", "n_total_trials", "mean_auc",
            "mean_v24_minus_macro", "mean_v24_minus_proper",
            "harm_gt_002_v24_vs_macro", "n_source_optimality_certificates_closed")
    }), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=pathlib.Path)
    args = p.parse_args()
    run(args.out)
