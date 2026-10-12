"""Real BNCI2014_002 continuum rank certificate on original grouped source folds.

These are SOURCE participant trial scores, frozen before any outer TARGET
labels. A confidence certificate for future unseen people is not claimed.

Usage:
 python research/defaultreg_bspc_v13/src/run_real_spectral_certificate_v23.py \
     --outer-person 1 --segments 8 --out /tmp/v23_source_spectral.json
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
from original_frozen_runner import ( # noqa: E402
    load_dataset, source_path, _held_groups, reference, tangent, spectral_state
)
from spectral_path_certificate_v23 import certified_path_rank_stability # noqa: E402


def per_outer(outer_person, loaded, segments=8):
    C, y, groups, receipts, spec, n_channels, channel_names = loaded
    if outer_person not in set(map(int, groups)):
        raise ValueError("unknown outer person")
    source = groups != outer_person
    Cs, ys, gs = C[source], y[source], groups[source]
    path = source_path(Cs, ys, gs)
    la = float(path["lambda_covariance_full"])
    lb = float(path["lambda_source_macro"])
    if la == lb:
        return {"outer_person": int(outer_person), "equal_lambdas": True,
                "source_auc_delta": 0.0, "held_source_people": len(set(gs)),
                "same_policy_no_path": True}
    low, high = min(la, lb), max(la, lb)
    rows = []
    for people in _held_groups(sorted(set(map(int, gs)))):
        validation = np.isin(gs, people)
        training = ~validation
        inv = reference(Cs[training])
        Xtr = tangent(Cs[training], inv)
        Xval = tangent(Cs[validation], inv)
        spectral = spectral_state(Xtr, ys[training])
        for sid in people:
            keep = gs[validation] == sid
            d = certified_path_rank_stability(
                spectral, Xval[keep], ys[validation][keep],
                low, high, segments=segments,
            )
            signed_delta = d["observed_endpoint_auc_delta"] * (
                1 if lb > la else -1
            )
            rows.append({
                "source_person": int(sid),
                "source_train_people": sorted(map(int, set(gs[training]))),
                "auc_delta_lambdaD_minus_lambdaC": signed_delta,
                **d,
            })
    if len(rows) != len(set(map(int, gs))):
        raise AssertionError("source grouped held people were dropped or repeated")
    observed = float(np.mean([r["auc_delta_lambdaD_minus_lambdaC"] for r in rows]))
    c = path["criteria_by_lambda"]
    direct = (float(c[str(float(lb))]["source_macro_auc"]) -
              float(c[str(float(la))]["source_macro_auc"]))
    if abs(observed - direct) > 1e-10:
        raise AssertionError("exact pair ranking disagrees with frozen EEG AUC")
    source_bounds = [r["uniform_path_auc_change_bound"] for r in rows]
    return {
        "outer_person": int(outer_person),
        "held_source_people": len(rows),
        "lambda_historical_class_centered_covariance": la,
        "lambda_source_macro_discrimination": lb,
        "source_auc_delta": observed,
        "held_source_mean_certified_stable_fraction": float(
            np.mean([r["certified_stable_fraction"] for r in rows])
        ),
        "held_source_mean_uniform_path_auc_bound": float(np.mean(source_bounds)),
        "independent_subject_generalisation_bound": False,
        "no_outer_target_data_in_selection_or_certificate": True,
        "source_participants": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-person", type=str, default="1")
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.outer_person != "all":
        requested = [int(args.outer_person)]
    else:
        requested = list(range(1, 15))
    loaded = load_dataset("BNCI2014_002")
    rows = [per_outer(sid, loaded, args.segments) for sid in requested]
    result = {
        "status": "EXECUTED_REAL_SOURCE_SPECTRAL_PATH_NOT_TARGET_VALIDATION",
        "dataset": "BNCI2014_002",
        "n_outer_folds": len(rows),
        "segments": args.segments,
        "mean_source_auc_delta": float(np.mean([r["source_auc_delta"] for r in rows])),
        "mean_certified_stable_source_pair_fraction": float(np.mean([
            r.get("held_source_mean_certified_stable_fraction", 1.0) for r in rows
        ])),
        "no_target_label_or_target_feature_used_in_source_analysis": True,
        "not_prospective_confirmation": True,
        "outer_source_reports": rows,
    }
    body = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "n_outer_folds": len(rows),
        "mean_source_auc_delta": result["mean_source_auc_delta"],
        "mean_certified_stable_source_pair_fraction":
            result["mean_certified_stable_source_pair_fraction"],
        "json_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
