"""Descriptive stratified analysis of three frozen EEG study-level effect vectors.

No pooled-trial pseudoreplication. Does not generate a new prospective test.
This was designed AFTER all three target outcomes became visible. Conditional
bootstrap CIs quantify sampled-person variation within these fixed cohorts;
they do NOT quantify sampling additional acquisition datasets, dependent
LOPO source-fit variation, or study-selection uncertainty.

Input: evidence/verified_three_cohort_paired_effect_vectors_v8.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

COHORTS = {
    "EEGMMIDB_S021_S040": 20,
    "BNCI2014_002": 14,
    "BNCI2015_001": 12,
}
ARMS = ("vs_covariance_nll", "vs_ledoitwolf")
EXPECTED_MEANS = {
    "EEGMMIDB_S021_S040": (0.056073153500000014, 0.09197095250000001),
    "BNCI2014_002": (0.017633928571428568, 0.022868303571428588),
    "BNCI2015_001": (0.04328865740740737, 0.04531296296296295),
}


def evaluate(data: dict, *, n_boot: int = 20000, seed: int = 20261010) -> dict:
    """Deterministic equal-dataset descriptive summaries. No p-values."""
    if data.get("schema") != "defaultreg_v8_immutable_paired_effect_vectors_v1":
        raise ValueError("Unrecognized frozen participant-effect source")
    if set(data["cohorts"]) != set(COHORTS):
        raise ValueError("All original three prospective cohorts are mandatory")
    if type(n_boot) is not int or n_boot < 1000 or n_boot > 100000:
        raise ValueError("Invalid fixed participant bootstrap size")
    vectors = {}
    for name, n in COHORTS.items():
        entry = data["cohorts"][name]
        if len(entry["participant_ids"]) != n or len(set(entry["participant_ids"])) != n:
            raise ValueError("Bad heldout participant membership/denominator")
        if set(entry["primary"]) != set(ARMS):
            raise ValueError("Both frozen primary contrasts must be included")
        vectors[name] = {}
        for idx, arm in enumerate(ARMS):
            v = np.asarray(entry["primary"][arm], dtype=np.float64)
            if v.ndim != 1 or len(v) != n or not np.isfinite(v).all() or np.max(np.abs(v)) > 1:
                raise ValueError("Bad paired participant AUC differences")
            if abs(v.mean() - EXPECTED_MEANS[name][idx]) > 2e-9:
                raise ValueError("Archived subject vector disagrees with frozen workflow mean")
            vectors[name][arm] = v
    result = {
        "status": "POST_OUTCOME_DATASET_STRATIFIED_DESCRIPTIVE_ONLY",
        "n_cohorts": 3,
        "n_unique_heldout_participants_total": sum(COHORTS.values()),
        "independent_inference_unit": "heldout participant within fixed dataset",
        "cohort_as_replication_stratum": True,
        "not_a_prospective_meta_analysis": True,
        "bootstrap_is_conditional_on_studied_datasets": True,
        "bootstrap_ignores_outer_shared_source_fit_uncertainty": True,
        "p_value_for_new_aggregate": None,
        "cross_dataset_SOTA_claim_allowed": False,
        "n_bootstrap": n_boot,
        "seed": seed,
        "methods": {},
    }
    for arm in ARMS:
        rng = np.random.default_rng(seed)  # independent reproducible stream for each contrast
        per = {}
        bootstrap_means = []
        for name, n in COHORTS.items():
            v = vectors[name][arm]
            idx = rng.integers(0, n, size=(n_boot, n))
            b = v[idx].mean(axis=1)
            bootstrap_means.append(b)
            per[name] = {
                "n_participants": n,
                "mean_paired_auc_difference": float(v.mean()),
                "median_paired_auc_difference": float(np.median(v)),
                "conditional_person_bootstrap95": [float(z) for z in np.quantile(b, [0.025, 0.975])],
                "improved": int(np.sum(v > 0)),
                "tied": int(np.sum(v == 0)),
                "harmed_more_than_0_02": int(np.sum(v < -0.02)),
                "worst_person": float(v.min()),
                "cvar25_of_paired_effects": float(np.mean(np.sort(v)[:int(np.ceil(0.25*n))])),
            }
        means = np.array([per[k]["mean_paired_auc_difference"] for k in COHORTS])
        meanall = float(np.mean(means))
        equal_dataset_boot = np.mean(np.stack(bootstrap_means), axis=0)
        other = [x for x in COHORTS if x != "EEGMMIDB_S021_S040"]
        external_means = np.array([per[x]["mean_paired_auc_difference"] for x in other])
        external_boot = np.mean(np.stack(bootstrap_means[1:]), axis=0)
        result["methods"][arm] = {
            "per_cohort": per,
            "descriptive_equal_dataset_mean": meanall,
            "conditional_person_bootstrap95_fixed_three_cohorts": [
                float(v) for v in np.quantile(equal_dataset_boot, [0.025, 0.975])
            ],
            "external_only_equal_dataset_mean": float(np.mean(external_means)),
            "conditional_person_bootstrap95_fixed_two_external_cohorts": [
                float(v) for v in np.quantile(external_boot, [0.025, 0.975])
            ],
            "between_dataset_mean_range": [float(means.min()), float(means.max())],
            "between_dataset_mean_sd": float(np.std(means, ddof=1)),
            "leave_one_dataset_out_equal_means": {
                key: float(np.mean([per[k]["mean_paired_auc_difference"] for k in COHORTS if k != key]))
                for key in COHORTS
            },
            "cohorts_with_positive_point_estimate": int(np.sum(means > 0)),
            "cohorts_total": 3,
            "interpretation": (
                "All 3 directions positive DESCRIPTIVELY; intervals condition on "
                "these specific cohorts. No prospective combined p-value, "
                "population-of-datasets CI, universal law or deployed-benefit claim."
            ),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=20000)
    args = ap.parse_args()
    blob = args.input.read_bytes()
    data = json.loads(blob)
    out = evaluate(data, n_boot=args.n_bootstrap)
    out["source_frozen_vectors_sha256"] = hashlib.sha256(blob).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("POST_OUTCOME_THREE_COHORT_DESCRIPTIVE_ONLY")
    for arm, v in out["methods"].items():
        print(arm, v["descriptive_equal_dataset_mean"],
              v["conditional_person_bootstrap95_fixed_three_cohorts"])


if __name__ == "__main__":
    main()
