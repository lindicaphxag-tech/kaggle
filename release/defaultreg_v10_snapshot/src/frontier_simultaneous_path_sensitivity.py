"""Post-outcome simultaneous-path sensitivity for the v5 AUC frontier.

Important: the positive percentile-bootstrap lower endpoint of max(AUC)-min(AUC)
is NOT evidence that the *population* path span is materially positive.
Taking max/min across many candidate lambdas introduces selection noise.

V10 uses participant-paired bootstrap whole-path supremum deviations, not
405 separate pointwise intervals. V11 additionally uses a sharper,
translation-invariant path-oscillation error seminorm:

    osc(e) = max_lambda(e(lambda)) - min_lambda(e(lambda))
    sensitivity_lower = max(0, observed_span - q_{1-alpha}(osc(e*))).

For any fixed curves, |span(observed) - span(true)| <= osc(observed - true).
The bootstrap radius is only conditional post-outcome sensitivity and
does not certify unconditional population coverage.

The corresponding materiality budget holds the empirical source-regret axis
fixed, so it is CONDITIONAL on that estimated axis. This is a post-outcome
sensitivity diagnostic, NOT a registered significance test or an unconditional
confidence guarantee. Shared LOSO source fits, repeated source subjects and
new-dataset sampling are not represented by participant resampling.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from v5_material_fragility_frontier import extract_v5_paths


def simultaneous_path_sensitivity(
    auc_rows: np.ndarray,
    regret_rows: np.ndarray,
    lambdas: np.ndarray,
    *,
    tau: float = 0.02,
    reps: int = 10000,
    seed: int = 20261010,
    alpha: float = 0.05,
    batch_size: int = 250,
) -> dict:
    auc = np.asarray(auc_rows, dtype=float)
    regret = np.asarray(regret_rows, dtype=float)
    grid = np.asarray(lambdas, dtype=float)
    if auc.ndim != 2 or auc.shape != regret.shape:
        raise ValueError("Must provide aligned participant x lambda paths")
    n, l_count = auc.shape
    if n < 4 or l_count < 3 or grid.shape != (l_count,):
        raise ValueError("Need >=4 participants and >=3 path points")
    if not (np.isfinite(auc).all() and np.isfinite(regret).all()
            and np.isfinite(grid).all()):
        raise ValueError("Nonfinite path entries")
    if np.any(auc < 0) or np.any(auc > 1) or np.any(regret < -1e-10):
        raise ValueError("Invalid AUC or negative covariance regret")
    if np.any(np.diff(grid) <= 0) or grid[0] < 0 or grid[-1] > 1:
        raise ValueError("Lambda grid must increase within [0,1]")
    if not (isinstance(reps, int) and reps >= 200) or not (
        isinstance(batch_size, int) and batch_size > 0
    ):
        raise ValueError("Insufficient bootstrap repetitions or bad batch size")
    if not (0.0 < tau <= 1.0 and 0.0 < alpha < 1.0):
        raise ValueError("Invalid alpha or materiality threshold")

    point_auc = auc.mean(axis=0)
    point_regret = regret.mean(axis=0)
    shifted = np.maximum(0.0, point_regret - np.min(point_regret))
    point_span = float(np.ptp(point_auc))

    # Bootstrap the ENTIRE paired participant path, with an identical
    # participant index vector at every lambda.
    rng = np.random.default_rng(seed)
    deviations = np.empty(reps, dtype=float)
    oscillations = np.empty(reps, dtype=float)
    for start in range(0, reps, batch_size):
        stop = min(reps, start + batch_size)
        idx = rng.integers(0, n, size=(stop - start, n))
        boot_mean = auc[idx].mean(axis=1)
        error_path = boot_mean - point_auc[None, :]
        deviations[start:stop] = np.max(np.abs(error_path), axis=1)
        oscillations[start:stop] = np.ptp(error_path, axis=1)
    radius = float(np.quantile(deviations, 1.0 - alpha))
    osc_radius = float(np.quantile(oscillations, 1.0 - alpha))
    # If every mean AUC ordinate is within +/- radius, the span differs
    # from the estimated span by at most 2*radius.
    lower = max(0.0, point_span - 2.0 * radius)
    upper = min(1.0, point_span + 2.0 * radius)
    oscillation_lower = max(0.0, point_span - osc_radius)
    oscillation_upper = min(1.0, point_span + osc_radius)

    # The observed source regret axis is held fixed, including its argmin.
    # A tie in regret is admitted jointly by threshold comparison.
    budgets = np.unique(shifted)
    observed = None
    certified = None
    oscillation_budget = None
    for budget in budgets:
        mask = shifted <= budget + 1e-12
        raw_span = float(np.ptp(point_auc[mask]))
        conservative = max(0.0, raw_span - 2.0 * radius)
        invariant_conservative = max(0.0, raw_span - osc_radius)
        if observed is None and raw_span >= tau - 1e-12:
            observed = float(budget)
        if certified is None and conservative >= tau - 1e-12:
            certified = float(budget)
        if oscillation_budget is None and invariant_conservative >= tau - 1e-12:
            oscillation_budget = float(budget)

    return {
        "schema": "defaultreg_postoutcome_simultaneous_path_sensitivity_v1",
        "status": "POST_OUTCOME_CONDITIONAL_SIMULTANEOUS_SENSITIVITY_NOT_CONFIRMATORY",
        "n_participants": n,
        "n_lambdas": l_count,
        "alpha_nominal": alpha,
        "bootstrap_reps": reps,
        "seed": seed,
        "full_path_supremum_bootstrap_radius": radius,
        "observed_max_auc_span": point_span,
        "max_span_sensitivity_bounds_nominal95": [lower, upper],
        "translation_invariant_oscillation_radius95": osc_radius,
        "translation_invariant_span_bounds_nominal95": [oscillation_lower, oscillation_upper],
        "translation_invariant_lower_budget_0_02": oscillation_budget,
        "translation_invariant_materiality_supported_conditional": oscillation_lower >= tau,
        "materiality_auc_threshold": tau,
        "observed_first_material_regret_budget": observed,
        "simultaneous_lower_bound_first_material_regret_budget": certified,
        "materiality_certified_under_conditional_reference": lower >= tau,
        "source_regret_path_treated_as_fixed_estimate": True,
        "post_outcome_statistic_not_registered": True,
        "limitations": [
            "No source-pool refit or uncertainty in the estimated NLL regret axis",
            "No new-dataset sampling uncertainty",
            "Shared overlapping LOSO source fits violate naive independent-participant assumptions",
            "Nominal bootstrap sensitivity only; no population-level significance claim",
            "Target AUC path used only after scores were frozen, never for model selection",
        ],
    }


def analyse_frozen_v5(result: dict, *, reps: int = 10000,
                      seed: int = 20261010) -> dict:
    paths = extract_v5_paths(result)
    out = simultaneous_path_sensitivity(
        paths["auc"],
        paths["regret_axes"]["raw_nll"],
        paths["lambdas"], reps=reps, seed=seed,
    )
    out["dataset"] = paths["dataset"]
    out["subjects"] = list(paths["subjects"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=10000)
    args = ap.parse_args()
    result = json.loads(args.input.read_text())
    out = analyse_frozen_v5(result, reps=args.reps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("FRONTIER_SIMULTANEOUS_POSTOUTCOME_SENSITIVITY", json.dumps({
        key: out[key] for key in (
            "dataset", "observed_max_auc_span",
            "full_path_supremum_bootstrap_radius",
            "max_span_sensitivity_bounds_nominal95",
            "translation_invariant_oscillation_radius95",
            "translation_invariant_span_bounds_nominal95",
            "materiality_certified_under_conditional_reference",
        )
    }))


if __name__ == "__main__":
    main()
