"""V23 conditional source-only continuous shrinkage search on real BNCI2014_002.

IMPORTANT: computes a source-validation optimisation certificate only.
Target data are neither scored nor used, no unseen participant AUC claimed.
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
    load_dataset, source_path, _held_groups, reference, tangent, spectral_state,
)
from certified_continuous_selector_v23 import ( # noqa: E402
    certified_source_auc_search, source_person_auc_at_lambda,
)


def compute(outer, low, high, splits):
    C, y, group, receipt, spec, channels, names = load_dataset("BNCI2014_002")
    if outer not in set(map(int, group)):
        raise ValueError("unknown outer subject")
    source = group != outer
    Cs, ys, gs = C[source], y[source], group[source]
    oldpath = source_path(Cs, ys, gs)
    models = []
    for held_people in _held_groups(sorted(set(map(int, gs)))):
        va = np.isin(gs, held_people)
        tr = ~va
        inv = reference(Cs[tr])
        state = spectral_state(tangent(Cs[tr], inv), ys[tr])
        validation_features = tangent(Cs[va], inv)
        for sid in held_people:
            mask = gs[va] == sid
            models.append((state, validation_features[mask], ys[va][mask]))
    if len(models) != 13:
        raise AssertionError("source held-subject denominator mismatch")
    original_lambda = float(oldpath["lambda_source_macro"])
    original_auc = float(oldpath["criteria_by_lambda"][str(original_lambda)]["source_macro_auc"])
    calculated = float(np.mean([
        source_person_auc_at_lambda(m, original_lambda) for m in models
    ]))
    if abs(original_auc-calculated)>1e-10:
        raise AssertionError("frozen source macro AUC regression failed")
    result = certified_source_auc_search(
        models, low=low, high=high, tolerance=.02,
        max_splits=splits, segments=1,
    )
    result.update({
        "status": "REAL_SOURCE_ONLY_CONTINUOUS_SEARCH_NOT_TARGET_CONFIRMATION",
        "outer_held_person": outer,
        "source_participants": len(models),
        "original_27_grid_source_lambda": original_lambda,
        "original_27_grid_source_macro_auc": original_auc,
        "same_source_validation_model_and_geometry": True,
        "no_outer_target_scores_or_labels_used": True,
        "numerical_optimality_is_NOT_future_target_risk": True,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-person", type=int, default=1)
    parser.add_argument("--low", type=float, default=.05)
    parser.add_argument("--high", type=float, default=.95)
    parser.add_argument("--max-splits", type=int, default=5)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    d = compute(args.outer_person, args.low, args.high, args.max_splits)
    s = json.dumps(d, indent=2, sort_keys=True, allow_nan=False)+"\n"
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(s, encoding="utf-8")
    print(json.dumps({
        "status": d["status"],
        "original_source_grid_AUC": d["original_27_grid_source_macro_auc"],
        "best_continuous_evaluated_source_AUC": d["best_evaluated_source_macro_auc"],
        "continuous_upper_source_AUC": d["upper_bound_global_continuous_source_macro_auc"],
        "source_optimality_gap_upper_bound": d["residual_source_optimality_gap_upper_bound"],
        "certificate_met": d["source_optimality_certificate_met"],
        "splits": d["interval_splits"],
        "receipt_sha256": hashlib.sha256(s.encode()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
