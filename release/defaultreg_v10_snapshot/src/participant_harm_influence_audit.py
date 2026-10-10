"""POST-OUTCOME robustness description from immutable paired participant effects.

This is an additional descriptive sensitivity analysis, not a prospectively
registered superiority test. Trials are never treated as independent units.
Wilson intervals assume exchangeable participants conditional on the studied
source model and do not cover source-pool or new-dataset uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, median

from three_cohort_conditional_summary import ARMS, COHORTS, EXPECTED_MEANS

SCHEMA = "defaultreg_v8_immutable_paired_effect_vectors_v1"
HARM_THRESHOLD = -0.02


def wilson_reference_interval(k: int, n: int, confidence: float = 0.95) -> list[float]:
    """Two-sided iid-binomial Wilson reference interval, not a source-pool CI."""
    if not (isinstance(k, int) and isinstance(n, int) and 0 <= k <= n and n > 0):
        raise ValueError("Invalid event count or denominator")
    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence level")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    # Wilson's limiting boundaries are exactly 0 and 1 at k=0 and k=n.
    # Enforce them explicitly so harmless platform roundoff cannot turn a
    # fail-closed reproducibility test into a false failure.
    if k == 0:
        lower = 0.0
    if k == n:
        upper = 1.0
    return [lower, upper]


def audit(frozen: dict) -> dict:
    if frozen.get("schema") != SCHEMA or set(frozen.get("cohorts", {})) != set(COHORTS):
        raise ValueError("Immutable evidence schema or cohort denominator changed")
    out = {
        "schema": "defaultreg_postoutcome_harm_influence_v1",
        "status": "POST_OUTCOME_DESCRIPTIVE_NOT_CONFIRMATORY",
        "source_score_family": SCHEMA,
        "harm_definition": "paired target AUC difference < -0.02",
        "effect_unit": "heldout participant (sessions grouped)",
        "interval_assumption": (
            "95% Wilson iid-participant reference interval, conditional on these "
            "fixed datasets and their overlapping source fits; NOT dataset-general "
            "or source-pool uncertainty and NOT a confirmatory safety bound"
        ),
        "multiple_comparison_caution": (
            "Six displayed descriptive contrasts; no new claims of hypothesis testing "
            "or participant safety. These intervals are not simultaneous."
        ),
        "datasets": {},
    }
    for cohort, n in COHORTS.items():
        entry = frozen["cohorts"][cohort]
        ids = entry.get("participant_ids")
        if not isinstance(ids, list) or len(ids) != n or len(set(ids)) != n:
            raise ValueError(f"Invalid immutable participant denominator: {cohort}")
        if set(entry.get("primary", {})) != set(ARMS):
            raise ValueError(f"Missing immutable primary contrast: {cohort}")
        coh = {}
        for i, arm in enumerate(ARMS):
            vector = entry["primary"][arm]
            if not isinstance(vector, list) or len(vector) != n:
                raise ValueError(f"Invalid paired vector: {cohort}/{arm}")
            if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1.0
                   for v in vector):
                raise ValueError(f"Nonfinite/invalid participant AUC delta: {cohort}/{arm}")
            v = [float(x) for x in vector]
            average = sum(v) / n
            if abs(average - EXPECTED_MEANS[cohort][i]) > 2e-9:
                raise ValueError(f"Participant deltas mismatch frozen receipt: {cohort}/{arm}")
            ordered = sorted(v)
            trim = math.floor(0.1 * n)
            trimmed = ordered[trim:n - trim] if trim else ordered
            leave_one_means = [(sum(v) - x) / (n - 1) for x in v]
            harm = sum(x < HARM_THRESHOLD for x in v)
            coh[arm] = {
                "n_participants": n,
                "n_positive": sum(x > 0 for x in v),
                "n_harmed_gt_0_02": harm,
                "harm_fraction": harm / n,
                "harm_wilson95_conditional_reference": wilson_reference_interval(harm, n),
                "mean_effect": average,
                "median_effect": float(median(v)),
                "trimmed_10pct_mean_effect_posthoc": sum(trimmed) / len(trimmed),
                "leave_one_participant_out_mean_range_posthoc": [
                    min(leave_one_means), max(leave_one_means)
                ],
                "leave_one_participant_out_direction_all_positive": min(leave_one_means) > 0,
                "worst_participant_effect": min(v),
                "best_participant_effect": max(v),
            }
        out["datasets"][cohort] = coh
    return out


def markdown_report(out: dict) -> str:
    lines = [
        "# Default Regularisation — post-outcome participant harm and influence audit",
        "",
        "**Evidence tier:** descriptive, after all target outcomes were visible. "
        "Not a new confirmatory analysis.",
        "",
        "Harm = individual paired target AUC difference strictly below -0.02. "
        "Intervals are 95% Wilson iid-participant reference intervals; "
        "they do not capture shared source-model or cross-dataset uncertainty.",
        "",
        "| Dataset | Comparator | N | Harmed | Harm fraction (Wilson 95% reference) | "
        "Mean Δ | Median Δ | 10% trimmed Δ | Delete-one Δ range |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for dataset, contrasts in out["datasets"].items():
        for arm, d in contrasts.items():
            lo, hi = d["harm_wilson95_conditional_reference"]
            min_loo, max_loo = d["leave_one_participant_out_mean_range_posthoc"]
            lines.append(
                f"| {dataset} | {arm} | {d['n_participants']} | {d['n_harmed_gt_0_02']} | "
                f"{d['harm_fraction']:.3f} [{lo:.3f}, {hi:.3f}] | "
                f"{d['mean_effect']:+.4f} | {d['median_effect']:+.4f} | "
                f"{d['trimmed_10pct_mean_effect_posthoc']:+.4f} | "
                f"[{min_loo:+.4f}, {max_loo:+.4f}] |"
            )
    lines.extend([
        "",
        "The delete-one diagnostic only checks influence of one held-out participant; "
        "it cannot substitute for a source-pool re-fit, source-jackknife, or new cohort.",
        "A positive mean does not certify individual safety. A positive external "
        "direction does not override the frozen external Holm tests.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()
    blob = args.input.read_bytes()
    out = audit(json.loads(blob))
    out["immutable_source_sha256"] = hashlib.sha256(blob).hexdigest()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(markdown_report(out))
    print("POST_OUTCOME_HARM_INFLUENCE_AUDIT_PASS", len(out["datasets"]), "datasets")


if __name__ == "__main__":
    main()
