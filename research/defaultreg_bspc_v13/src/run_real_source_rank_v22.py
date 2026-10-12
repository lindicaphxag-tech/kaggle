"""Read-only V22 source-person rank mechanism on the original frozen EEG runner.

Does NOT reselect hyperparameters from target data. Output only includes
SOURCE-validation labels and scores, and contains no held outer target AUC.
This is retrospective diagnostic work: not registered confirmation.

Usage (from repository root):
  python research/defaultreg_bspc_v13/src/run_real_source_rank_v22.py \
      --dataset BNCI2014_002 --outer-person 1 --out /tmp/source-ranks.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "defaultreg_proper_predictive"))
from original_frozen_runner import (  # noqa: E402
    load_dataset, source_path, _held_groups, reference, tangent,
    spectral_state, scalar_scores,
)
from source_rank_churn_v22 import subject_macro_rank_audit  # noqa: E402


def compute(dataset: str, outer_person: int, loaded=None) -> dict:
    if dataset not in ("BNCI2014_002", "Shin2017A"):
        raise ValueError("Only the ORIGINAL FROZEN runner's two cohorts are supported")
    C, y, groups, receipt, spec, n_channels, _ = (
        loaded if loaded is not None else load_dataset(dataset)
    )
    if outer_person not in set(map(int, groups)):
        raise ValueError("outer person not found")
    source_mask = groups != outer_person
    # The target person's features and labels have not been passed to
    # source_path or any inner geometry, score or rank computation.
    Cs, ys, gs = C[source_mask], y[source_mask], groups[source_mask]
    path = source_path(Cs, ys, gs)
    before_lambda = float(path["lambda_covariance_full"])
    after_lambda = float(path["lambda_source_macro"])
    records = []
    for held_people in _held_groups(sorted(set(map(int, gs)))):
        validation = np.isin(gs, held_people)
        training = ~validation
        inv = reference(Cs[training])
        Xtr = tangent(Cs[training], inv)
        Xva = tangent(Cs[validation], inv)
        state = spectral_state(Xtr, ys[training])
        for person in held_people:
            mask = gs[validation] == person
            XX, yy = Xva[mask], ys[validation][mask]
            records.append({
                "participant_id": int(person),
                "information_scope": "source_validation",
                "source_fit_subject_ids": sorted(map(int, set(gs[training]))),
                "labels": yy,
                "score_before": scalar_scores(state, XX, before_lambda),
                "score_after": scalar_scores(state, XX, after_lambda),
            })
    report = subject_macro_rank_audit(records)
    lookup = path["criteria_by_lambda"]
    expected_before = float(lookup[str(float(before_lambda))]["source_macro_auc"])
    expected_after = float(lookup[str(float(after_lambda))]["source_macro_auc"])
    if abs(report["macro_auc_before"] - expected_before) > 1e-10:
        raise AssertionError("source covariance-selected AUC does not match frozen runner")
    if abs(report["macro_auc_after"] - expected_after) > 1e-10:
        raise AssertionError("source macro-selected AUC does not match frozen runner")
    report.update({
        "status": "SOURCE_ONLY_POSTHOC_MECHANISM_NOT_PROSPECTIVE_EFFECT",
        "dataset": dataset,
        "outer_held_person": int(outer_person),
        "source_participant_count": len(set(map(int, gs))),
        "original_source_criterion": "historical_validation_class_centred_covariance",
        "before_lambda": before_lambda,
        "after_lambda": after_lambda,
        "held_target_features_used_to_choose_lambda": False,
        "held_target_labels_used_in_this_diagnostic": False,
        "retraining_and_overlapping_source_pools_not_independent": True,
        "no_target_performance_or_novel_selector_claim": True,
    })
    # Keep only SOURCE-validation scores in the transient local arrays;
    # freeze a deterministic textual report and digest in the output.
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["BNCI2014_002", "Shin2017A"],
                        default="BNCI2014_002")
    parser.add_argument("--outer-person", required=True,
                        help="One participant ID, or all for full source-only LOSO audit")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.outer_person == "all":
        loaded = load_dataset(args.dataset)
        people = sorted(set(map(int, loaded[2])))
        results = [compute(args.dataset, p, loaded) for p in people]
        result = {
            "status": "SOURCE_ONLY_ALL_OUTER_MECHANISM_COMPLETE_POSTHOC",
            "dataset": args.dataset,
            "n_outer_pools": len(results),
            "mean_source_auc_delta_exact": float(np.mean(
                [q["macro_auc_delta_exact"] for q in results])),
            "mean_source_gross_pair_churn": float(np.mean(
                [q["participant_mean_gross_pair_churn"] for q in results])),
            "no_target_outcomes_or_target_tuning": True,
            "source_pool_results": results,
        }
    else:
        result = compute(args.dataset, int(args.outer_person))
    body = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    summary = {
        "status": result["status"],
        "report_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    for key in ("outer_held_person", "macro_auc_delta_exact",
                "participant_mean_gross_pair_churn", "n_outer_pools",
                "mean_source_auc_delta_exact", "mean_source_gross_pair_churn"):
        if key in result:
            summary[key] = result[key]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
