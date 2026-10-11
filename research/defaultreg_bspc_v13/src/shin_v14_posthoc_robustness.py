#!/usr/bin/env python3
"""Independent POST-OUTCOME sensitivity from original Kaggle Shin2017A ZIP.

No new EEG training, no target-label re-decoding. Held-person outer folds
SHARE source participants: their source-vs-target Spearman p-values are not
independent-sample inference and must never be presented as such.

Usage: python shin_v14_posthoc_robustness.py ORIGINAL_ARTIFACT_ZIP
"""
from __future__ import annotations
import hashlib
import io
import json
import sys
import zipfile

import numpy as np
from scipy.stats import spearmanr, trim_mean

ZIP_SHA = "cda3210d24a8e9c7f3bf2d48490e51ad51f023beb652816a8679088cc3fb1842"
MANIFEST_SHA = "00d1aa5bb912aa7badcdc5f1c9d3f6e690551d4ffee4d6bff234c61c15582e9e"
PAIR_SHA = "5b6af6db1dd09d9729fe3c819fbb9f70316063b9211d3777d1102db158617aee"
JSON_PATH = "defaultreg_frontier_Shin2017A/Shin2017A_frontier_confirmatory.json"


def sign_flip_posthoc(d: np.ndarray, seed: int, n: int = 250000) -> float:
    rng = np.random.default_rng(seed)
    threshold = abs(float(np.mean(d)))
    exceed = 0
    for start in range(0, n, 5000):
        k = min(5000, n - start)
        s = rng.integers(0, 2, size=(k, len(d)), dtype=np.int8) * 2 - 1
        exceed += int(np.count_nonzero(np.abs((s * d).mean(axis=1)) >= threshold - 1e-13))
    return float((exceed + 1) / (n + 1))


def audit(path: str) -> dict:
    data = open(path, "rb").read()
    assert hashlib.sha256(data).hexdigest() == ZIP_SHA, "original ZIP SHA mismatch"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        j = json.loads(archive.read(JSON_PATH))
        assert j["score_manifest_sha256"] == MANIFEST_SHA
        folds = sorted(j["folds"], key=lambda f: int(f["target_subject"]))
        assert len(folds) == 29 and sum(f["target_trials"] for f in folds) == 1740
        x, ycov, ylw = [], [], []
        source_cov_cost = []
        for fold in folds:
            p = fold["source_path"]
            chosen = p["criteria_by_lambda"][str(p["lambda_source_macro"])]
            default = p["criteria_by_lambda"][str(p["lambda_covariance_full"])]
            x.append(chosen["source_macro_auc"] - default["source_macro_auc"])
            source_cov_cost.append(chosen["covariance_mean_nll"] - default["covariance_mean_nll"])
            a = fold["auc"]
            ycov.append(a["source_macro_full_domain_LDA"] - a["source_cov_nll_full_domain_LDA"])
            ylw.append(a["source_macro_full_domain_LDA"] - a["LedoitWolf_LDA"])
        x, ycov, ylw = np.asarray(x), np.asarray(ycov), np.asarray(ylw)
        assert min(x) >= -1e-12 and min(source_cov_cost) >= -1e-12
        # SHA pins exactly the displayed 29-point source gain vs outcome figure.
        pairs = "".join(f"{a:.8f},{b:.8f}\n" for a, b in zip(x, ycov))
        assert hashlib.sha256(pairs.encode()).hexdigest() == PAIR_SHA
        contrasts = {}
        for name, base, y, seed in [
            ("covariance_NLL", "source_cov_nll_full_domain_LDA", ycov, 61011),
            ("LedoitWolf", "LedoitWolf_LDA", ylw, 61012),
        ]:
            original = next(r for r in j["primary_comparisons"] if r["baseline"] == base)
            assert abs(float(y.mean()) - original["mean_delta_auc"]) < 1e-12
            harm = int(np.sum(y < -0.020000000001))
            assert harm == original["harmed_gt_0_02"]
            rng = np.random.default_rng(seed)
            boot = np.mean(y[rng.integers(0, len(y), size=(50000, len(y)))], axis=1)
            leave = [np.mean(np.delete(y, k)) for k in range(len(y))]
            contrasts[name] = {
                "frozen_Holm_p_UNCHANGED": original["holm_p_primary_family"],
                "mean": float(np.mean(y)),
                "median": float(np.median(y)),
                "10pct_trimmed_mean_POSTHOC": float(trim_mean(y, .1)),
                "leave_one_target_subject_out_mean_range_POSTHOC": [float(min(leave)),float(max(leave))],
                "leave_one_target_all_positive": all(v > 0 for v in leave),
                "harmed_gt_0_02_n": harm,
                "participant_bootstrap95_posthoc_conditional_on_source": np.quantile(boot,[.025,.975]).tolist(),
                "MC_signflip_P_POSTHOC_not_primary": sign_flip_posthoc(y,seed),
                "source_advantage_vs_target_effect_spearman_DESCRIPTIVE": float(spearmanr(x,y).statistic),
                "cross_fold_correlation_p_invalid_due_to_overlapping_source_pools": True,
            }
        return {
            "status":"PASS_ARCHIVED_SOURCE_TARGET_POSTHOC_NOT_NEW_PROSPECTIVE_DATA",
            "zip_sha256": ZIP_SHA,
            "figure_29_pairs_sha256": PAIR_SHA,
            "n": 29,
            "source_lambda_disagreement_n": sum(f["source_path"]["lambda_covariance_full"] != f["source_path"]["lambda_source_macro"] for f in folds),
            "source_gain_mean":float(x.mean()),
            "source_covariance_cost_mean":float(np.mean(source_cov_cost)),
            "contrasts":contrasts,
        }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python shin_v14_posthoc_robustness.py ORIGINAL_SHIN_ZIP")
    print(json.dumps(audit(sys.argv[1]), indent=2))
