#!/usr/bin/env python3
"""Independent V24 Weibo2014 first-run artifact audit, no fresh EEG processing.

Existing outcomes are EXPOSED: this verifies the frozen artifact and produces
descriptive participant-macro results; it MUST NOT modify selection policy.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import pathlib
import zipfile
import numpy as np

CHOICES = "WEIBO2014_10_PERSON_SOURCE_ONLY_CHOICES_FROZEN_BEFORE_TARGET.json"
OUTCOMES = "WEIBO2014_10_PERSON_FIRST_RUN_EXTERNAL_TARGET_AUC.json"
CHOICES_SHA = "a2fb5b334218a00b988fcbcba5eb71a71bf7848b4ca92b6818243fb933fb6999"
EXPECTED_AUC = {
    "source_macro_grid": .7314052933673468,
    "source_correct_predictive_nll": .7293584183673468,
    "historical_validation_class_centred": .7254888392857144,
    "v24_anchored_continuous": .7314052933673468,
    "fixed_lambda_0_50": .7301492346938775,
    "LW": .7255092474489796,
    "OAS": .7255092474489796,
}


def read_file(z, end):
    candidates = [n for n in z.namelist() if n.endswith(end)]
    if len(candidates) != 1:
        raise AssertionError(f"expected one {end} found {candidates}")
    return z.read(candidates[0])


def audit(zip_path):
    with zipfile.ZipFile(io.BytesIO(pathlib.Path(zip_path).read_bytes())) as z:
        choices_bytes = read_file(z, CHOICES)
        result_bytes = read_file(z, OUTCOMES)
    if hashlib.sha256(choices_bytes).hexdigest() != CHOICES_SHA:
        raise AssertionError("source frozen-choice hash mismatch")
    choices = json.loads(choices_bytes)
    outcomes = json.loads(result_bytes)
    assert outcomes["status"] == "WEIBO2014_NEW_EXTERNAL_FIRST_RUN_COMPLETE_UNREGISTERED"
    assert outcomes["n_outer_people"] == len(choices) == 10
    assert outcomes["n_channels"] == 3 and outcomes["n_tangent_features"] == 6
    assert outcomes["n_total_trials"] == 1580
    assert outcomes["no_held_target_selected_parameter"]
    assert outcomes["source_selector_ran_before_all_target_scores"]
    assert outcomes["source_lambda_decisions_sha256_before_target"] == CHOICES_SHA
    assert outcomes["all_people_retained"]
    seen = set(range(1, 11))
    assert {r["held_outer_person"] for r in choices} == seen
    assert {r["held_person"] for r in outcomes["per_person_target_outcomes"]} == seen
    assert {r["person"] for r in outcomes["participant_receipts"]} == seen
    assert sum(r["total_trials"] for r in outcomes["participant_receipts"]) == 1580
    for r in choices:
        assert set(r["source_people"]) == seen - {r["held_outer_person"]}
        assert r["source_fit_subjects_exclude_target"]
        assert r["no_held_target_data_in_regularisation_selection"]
        assert r["source_continuous_evaluated_auc"] + 1e-10 >= r["source_macro_grid_auc"]
        assert r["source_continuous_upper_bound"] + 1e-10 >= r["source_continuous_evaluated_auc"]
    by = sorted(outcomes["per_person_target_outcomes"], key=lambda x: x["held_person"])
    for p in by:
        assert all(0 <= val <= 1 for val in p["auc"].values())
        assert set(p["auc"]) == set(EXPECTED_AUC)
    means = {key: float(np.mean([p["auc"][key] for p in by])) for key in EXPECTED_AUC}
    for key, expected in EXPECTED_AUC.items():
        assert abs(means[key] - expected) < 1e-12
        assert abs(means[key] - outcomes["mean_auc"][key]) < 1e-12

    rng = np.random.default_rng(20261013)
    pairs = {}
    for method in ("source_macro_grid", "source_correct_predictive_nll",
                   "historical_validation_class_centred", "LW", "OAS"):
        deltas = np.asarray([p["auc"]["v24_anchored_continuous"] -
                             p["auc"][method] for p in by], float)
        ci = np.quantile(deltas[rng.integers(0, 10, (30000, 10))].mean(1), [.025, .975])
        pairs[method] = {
            "mean_auc_difference": float(deltas.mean()),
            "median_auc_difference": float(np.median(deltas)),
            "min_person_difference": float(deltas.min()),
            "max_person_difference": float(deltas.max()),
            "n_harm_more_than_002": int((deltas < -.020000000001).sum()),
            "n_benefit_more_than_002": int((deltas > .020000000001).sum()),
            "bootstrap_95_descriptive_over_10_people": ci.tolist(),
        }
    grid_equal = sum(
        float(r["selected_lambdas"]["v24_anchored_continuous"]) ==
        float(r["selected_lambdas"]["source_macro_grid"])
        for r in choices
    )
    return {
        "status": "PASS_INDEPENDENT_V24_WEIBO_ALL10_FROZEN_ARTIFACT_AUDIT",
        "frozen_source_sha256": CHOICES_SHA,
        "target_outcome_json_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "n_people": 10,
        "trials": 1580,
        "mean_auc": means,
        "n_same_lambda_continuous_vs_grid": grid_equal,
        "n_source_optimality_gaps_below_002": sum(
            int(x["continuous_certified_to_002"]) for x in choices
        ),
        "paired_v24_comparisons": pairs,
        "scope": "Descriptive held-person differences only; in-repository protocol-first unregistered; source pools overlap and bootstrap intervals do not imply confirmatory coverage.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_zip", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    a = parser.parse_args()
    result = audit(a.artifact_zip)
    s = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if a.out:
        a.out.write_text(s)
    print(s)
