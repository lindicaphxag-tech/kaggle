#!/usr/bin/env python3
"""V24 preprotocol-locked entire public 50-patient stroke EEG Z0 evaluation.

Important chronology: the complete 50-person source-only choice manifest
is hashed and written BEFORE ANY outer held-target prediction/AUC.
Full same original tangent-LDA pipeline, precommitted policies/baselines.
This is one new acquisition and NOT clinical deployment evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "defaultreg_bspc_v13" / "src"))
import original_frozen_runner as old
from run_proper_predictive_posthoc import select_proper_source_nll
from run_liu2024_source_only_preoutcome_v24 import get_data, PROTOCOL, SUBJECTS
from certified_continuous_selector_v23 import certified_source_auc_search


def source_fold_models(C, y, g):
    models = []
    for held in old._held_groups(sorted(set(map(int, g)))):
        va = np.isin(g, held)
        tr = ~va
        inv = old.reference(C[tr])
        Xtr = old.tangent(C[tr], inv)
        Xva = old.tangent(C[va], inv)
        state = old.spectral_state(Xtr, y[tr])
        for sid in held:
            member = g[va] == sid
            yy = y[va][member]
            if set(np.unique(yy)) != {0, 1}:
                raise ValueError("source held participant missing class")
            models.append((state, Xva[member], yy))
    if len(models) != len(set(g)):
        raise RuntimeError("source person validation denominator mismatch")
    return models


def freeze_all_source_choices(C, y, g, out):
    """First complete all 50 source-only policies, writing manifest to disk."""
    choices = []
    for sid in SUBJECTS:
        mask = g != sid
        Cs, ys, gs = C[mask], y[mask], g[mask]
        path = old.source_path(Cs, ys, gs)
        grid = path["feasible_full_lambdas"]
        proper, proper_source = select_proper_source_nll(Cs, ys, gs, grid)
        models = source_fold_models(Cs, ys, gs)
        cont = certified_source_auc_search(
            models, low=.05, high=.95, tolerance=.02,
            max_splits=5, segments=1, block_size=32,
        )
        macro = float(path["lambda_source_macro"])
        proposed = float(cont["selected_lambda_source_macro"])
        promoted = bool(cont["source_optimality_certificate_met"])
        policy = proposed if promoted else macro
        picked = {
            "source_macro": macro,
            "historical_class_centred": float(path["lambda_covariance_full"]),
            "proper_source_mean_gaussian": float(proper),
            "v23_fail_closed_continuous": policy,
        }
        if not all(float(v) in grid or k=="v23_fail_closed_continuous"
                   for k,v in picked.items()):
            raise AssertionError("non-original candidate used by baseline")
        if not (0 <= policy <= 1):
            raise AssertionError("invalid frozen algorithm choice")
        choices.append({
            "outer_subject": int(sid),
            "n_source_people": 49,
            "source_people": sorted(map(int, set(gs))),
            "source_trial_count": int(len(ys)),
            "chosen_lambda": picked,
            "v23_continuous_source_result": cont,
            "v23_certified": promoted,
            "v23_fallback_to_original_macro": not promoted,
            "no_outer_target_information_passed_to_selection": True,
            "source_macro_validation_auc": float(path["criteria_by_lambda"][str(macro)]["source_macro_auc"]),
            "source_feasible_lambda_count": int(len(grid)),
        })
        print("SOURCE_LIU2024_SELECTION_BEFORE_TARGET", sid,
              picked, "certified", promoted, flush=True)
    if len(choices) != 50:
        raise AssertionError("50 source-only policies not complete")
    out.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(choices, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    src = out / "FIFTY_SOURCE_ONLY_CHOICES_LOCKED_BEFORE_ANY_TARGET_AUC.json"
    src.write_bytes(body)
    checksum = hashlib.sha256(body).hexdigest()
    (out / "SOURCE_CHOICES_PRETARGET_SHA256.txt").write_text(checksum + "\n")
    print("ALL_50_SOURCE_CHOICES_LOCKED_BEFORE_TARGET_AUC_SHA256",checksum,flush=True)
    return choices, checksum


def evaluate_target_after_freeze(C, y, g, choices):
    outcome = []
    for rec in choices:
        sid = rec["outer_subject"]
        target = g == sid
        source = ~target
        if int(np.sum(target)) != 40 or int(np.sum(source)) != 1960:
            raise RuntimeError("patient or source denominator changed")
        inv = old.reference(C[source])
        Xsource = old.tangent(C[source], inv)
        Xtarget = old.tangent(C[target], inv)
        state = old.spectral_state(Xsource, y[source])
        scores = {
            key: old.scalar_scores(state, Xtarget, float(lam))
            for key, lam in rec["chosen_lambda"].items()
        }
        for name in ("LW", "OAS"):
            sc, shrink = old.analytic_scores(Xsource, y[source], Xtarget, name)
            scores[name] = sc
        aucs = {k: float(roc_auc_score(y[target], v)) for k, v in scores.items()}
        if set(aucs) != set(rec["chosen_lambda"]) | {"LW", "OAS"}:
            raise AssertionError("target method set drift")
        outcome.append({"held_patient": sid, "n_trials": int(sum(target)),
                        "auc": aucs, "source_choice_sha_verified": True})
        print("FROZEN_TARGET_SUBJECT_AUC", sid, aucs, flush=True)
    return outcome


def summarise(outcomes, choices, choice_sha):
    methods = list(outcomes[0]["auc"])
    a = {m: np.asarray([x["auc"][m] for x in outcomes]) for m in methods}
    macro = a["source_macro"]
    proper = a["proper_source_mean_gaussian"]
    proposed = a["v23_fail_closed_continuous"]
    rng = np.random.default_rng(20261013)
    indices = rng.integers(0, len(outcomes), (10000, len(outcomes)))
    def paired(b):
        d = proposed - b
        return {
            "mean_AUC_difference": float(d.mean()),
            "descriptive_paired_subject_bootstrap_95": np.quantile(
                d[indices].mean(1), [.025,.975]
            ).tolist(),
            "harm_GT_0_02_patients": int(np.sum(d < -.020000000001)),
            "improved_GT_0_02_patients": int(np.sum(d > .020000000001)),
        }
    return {
        "status": "EXTERNAL_50_ACUTE_STROKE_Z0_LOPO_COMPLETE_UNREGISTERED",
        "source_choices_SHA256_pre_target_evaluation": choice_sha,
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "dataset": "Liu2024",
        "n_patients": len(outcomes),
        "trials": sum(r["n_trials"] for r in outcomes),
        "n_eeg_channels": 29,
        "mean_target_AUC_by_policy": {m: float(v.mean()) for m,v in a.items()},
        "v23_source_only_certificate_success_folds": int(sum(q["v23_certified"] for q in choices)),
        "v23_fallback_to_macro_folds": int(sum(q["v23_fallback_to_original_macro"] for q in choices)),
        "v23_minus_macro": paired(macro),
        "v23_minus_proper_predictive": paired(proper),
        "target_information_budget": "Z0 strict source-only",
        "independent_patient_inference": "descriptive; shared overlapping sources and single recording cohort",
        "registered": False, "claimed_clinical_validation": False,
        "participants": outcomes,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if not PROTOCOL.exists():
        raise RuntimeError("precommitted clinical protocol is missing")
    C,y,g,patient_receipts=get_data()
    if set(map(int,g)) != set(SUBJECTS) or len(y) != 2000:
        raise RuntimeError("one or more predeclared stroke patients are missing")
    choices, checksum=freeze_all_source_choices(C,y,g,args.out)
    # TARGET LABELS CAN BE USED ONLY AFTER ALL 50 POLICIES WERE WRITTEN.
    outcome=evaluate_target_after_freeze(C,y,g,choices)
    result=summarise(outcome,choices,checksum)
    content=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    (args.out/"FROZEN_AFTER_SOURCE_LOCK_50PATIENT_LOPO_RESULTS.json").write_text(content)
    print(json.dumps({k:v for k,v in result.items() if k!="participants"},indent=2),flush=True)


if __name__ == "__main__":
    main()
