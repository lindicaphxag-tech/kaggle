"""Untouched Regularisation Validity Frontier confirmation.

Datasets:
- Shin2017A: 29 participants, left_hand vs right_hand MI.
- BNCI2014_002: 14 participants, right_hand vs feet MI.

The branch protocol is frozen before target outcomes.

Design:
- outer participant held out;
- Z0 source-only model/geometry/regularisation selection;
- historical FULL_GRID for method selection;
- dense 0.0025 frontier grid + near-zero anchors;
- target score family written and SHA256-hashed before target labels are used
  for AUC evaluation;
- participant is the inferential unit.

This runner is intentionally classical and auditable. It is not a SOTA
architecture benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import moabb
import sklearn
import pyriemann
from moabb.datasets import Shin2017A, BNCI2014_002
from moabb.paradigms import MotorImagery
from pyriemann.classification import MDM
from sklearn.covariance import LedoitWolf, OAS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


SEED = 20261010

FULL_GRID = (
    0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015,
    0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 1.0,
)

NEAR_ZERO = (
    0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075,
    0.01, 0.015, 0.02, 0.03, 0.05,
)
DENSE_GRID = tuple(sorted(set(
    NEAR_ZERO + tuple(round(i * 0.0025, 10) for i in range(401))
)))
UNION_GRID = tuple(sorted(set(FULL_GRID) | set(DENSE_GRID)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dataset_spec(name: str):
    if name == "Shin2017A":
        ds = Shin2017A(
            accept=True,
            motor_imagery=True,
            mental_arithmetic=False,
        )
        return {
            "dataset": ds,
            "subjects": tuple(range(1, 30)),
            "events": ("left_hand", "right_hand"),
            "positive_label": "right_hand",
            "expected_subjects": 29,
        }
    if name == "BNCI2014_002":
        ds = BNCI2014_002()
        return {
            "dataset": ds,
            "subjects": tuple(range(1, 15)),
            "events": ("right_hand", "feet"),
            "positive_label": "feet",
            "expected_subjects": 14,
        }
    raise ValueError(f"Unsupported frozen dataset: {name}")


def covariances(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or not np.isfinite(X).all():
        raise ValueError("Expected finite [trial, channel, time] EEG")
    X = X - X.mean(axis=-1, keepdims=True)
    C = X @ X.transpose(0, 2, 1) / X.shape[-1]
    p = X.shape[1]
    scale = np.trace(C, axis1=1, axis2=2) / p
    C += np.eye(p)[None, :, :] * np.maximum(
        scale[:, None, None] * 1e-6, 1e-20
    )
    return (C + C.transpose(0, 2, 1)) / 2


def reference(C):
    M = np.mean(C, axis=0)
    e, U = np.linalg.eigh((M + M.T) / 2)
    if np.any(~np.isfinite(e)) or np.any(e <= 0):
        raise ValueError("Source tangent reference is not SPD")
    return (U * (e ** -0.5)) @ U.T


def tangent(C, inv):
    Z = np.einsum("ab,nbc,cd->nad", inv, C, inv)
    Z = (Z + Z.transpose(0, 2, 1)) / 2
    e, U = np.linalg.eigh(Z)
    if np.any(~np.isfinite(e)) or np.any(e <= 0):
        raise ValueError("Whitened trial covariance is not SPD")
    L = (U * np.log(e)[:, None, :]) @ U.transpose(0, 2, 1)
    i, j = np.triu_indices(L.shape[1])
    out = L[:, i, j]
    out[:, i != j] *= np.sqrt(2.0)
    return out


def moments(X, y):
    a = X[y == 0]
    b = X[y == 1]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each class needs >=2 rows")
    m0 = a.mean(0)
    m1 = b.mean(0)
    R = np.concatenate((a - m0, b - m1), axis=0)
    S = (R.T @ R) / (len(R) - 2)
    return (S + S.T) / 2, m1 - m0, (m0 + m1) / 2, R


def spectral_state(X, y):
    S, d, mid, _ = moments(X, y)
    e, U = np.linalg.eigh(S)
    if not np.isfinite(e).all():
        raise ValueError("Nonfinite covariance spectrum")
    nu = max(float(np.mean(e)), np.finfo(float).eps)
    return e, U, d, mid, nu


def scalar_scores(state, X, lam):
    e, U, d, mid, nu = state
    r = (1 - lam) * e + lam * nu
    tol = 64 * np.finfo(float).eps * max(
        1.0, nu, float(np.max(np.abs(e)))
    )
    if np.any(~np.isfinite(r)) or np.any(r <= tol):
        raise np.linalg.LinAlgError("Infeasible scalar shrinkage")
    w = U @ ((U.T @ d) / r)
    return (X - mid) @ w


def predictive_nll(state, residual, lam):
    e, U, _, _, nu = state
    r = (1 - lam) * e + lam * nu
    tol = 64 * np.finfo(float).eps * max(
        1.0, nu, float(np.max(np.abs(e)))
    )
    if np.any(~np.isfinite(r)) or np.any(r <= tol):
        raise np.linalg.LinAlgError("Infeasible covariance shrinkage")
    Z = residual @ U
    return float(
        (
            np.sum(np.log(r))
            + np.mean(np.sum((Z * Z) / r, axis=1))
        ) / len(r)
    )


def analytic_scores(Xtr, ytr, Xte, kind):
    _, d, mid, R = moments(Xtr, ytr)
    estimator = (
        LedoitWolf(store_precision=False)
        if kind == "LW"
        else OAS(store_precision=False)
    ).fit(R)
    C = (estimator.covariance_ + estimator.covariance_.T) / 2
    C += np.eye(C.shape[0]) * max(float(np.trace(C)), 1.0) * 1e-12
    w = np.linalg.solve(C, d)
    return (Xte - mid) @ w, float(estimator.shrinkage_)


def _held_groups(people):
    people = list(sorted(map(int, people)))
    folds = [[] for _ in range(5)]
    for rank, sid in enumerate(people):
        folds[rank % 5].append(sid)
    return folds


def source_path(C, y, groups):
    people = tuple(sorted(set(map(int, groups))))
    if len(people) < 8:
        raise ValueError("Need >=8 source participants")

    folds = _held_groups(people)
    store = {
        lam: {"auc": [], "nll": []}
        for lam in UNION_GRID
    }
    feasible = {lam: True for lam in UNION_GRID}
    fold_receipts = []

    for fold_id, held_people in enumerate(folds):
        va = np.isin(groups, held_people)
        tr = ~va
        if not va.any() or not tr.any():
            raise ValueError("Empty grouped source fold")

        inv = reference(C[tr])
        Xtr = tangent(C[tr], inv)
        Xva = tangent(C[va], inv)
        state = spectral_state(Xtr, y[tr])

        by_person = {}
        for sid in held_people:
            mask = groups[va] == sid
            yy = y[va][mask]
            XX = Xva[mask]
            if set(np.unique(yy)) != {0, 1}:
                raise ValueError(f"Source validation participant {sid} lacks class")
            _, _, _, R = moments(XX, yy)
            by_person[sid] = (yy, XX, R)

        fold_receipts.append({
            "fold": fold_id,
            "train_subjects": [
                int(s) for s in people if s not in held_people
            ],
            "validation_subjects": list(map(int, held_people)),
            "train_trials": int(tr.sum()),
            "validation_trials": int(va.sum()),
        })

        for lam in UNION_GRID:
            if not feasible[lam]:
                continue
            aucs = []
            nlls = []
            try:
                for sid in held_people:
                    yy, XX, R = by_person[sid]
                    nlls.append(predictive_nll(state, R, lam))
                    if lam in FULL_GRID:
                        score = scalar_scores(state, XX, lam)
                        aucs.append(float(roc_auc_score(yy, score)))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                feasible[lam] = False
                continue
            if not np.isfinite(nlls).all():
                feasible[lam] = False
                continue
            if lam in FULL_GRID and not np.isfinite(aucs).all():
                feasible[lam] = False
                continue
            if lam in FULL_GRID:
                store[lam]["auc"].extend(aucs)
            store[lam]["nll"].extend(nlls)

    dense_good = [
        lam for lam in DENSE_GRID
        if feasible[lam] and len(store[lam]["nll"]) == len(people)
    ]
    full_good = [
        lam for lam in FULL_GRID
        if lam in dense_good and len(store[lam]["auc"]) == len(people)
    ]
    if len(full_good) < 2 or len(dense_good) < 8:
        raise ValueError("Insufficient feasible full/dense path")

    criteria = {}
    for lam in dense_good:
        n = np.asarray(store[lam]["nll"], dtype=float)
        row = {
            "covariance_mean_nll": float(np.mean(n)),
            "covariance_nll_sd": float(np.std(n, ddof=1)),
            "covariance_nll_se": float(np.std(n, ddof=1) / np.sqrt(len(n))),
        }
        if lam in full_good:
            a = np.asarray(store[lam]["auc"], dtype=float)
            row["source_macro_auc"] = float(np.mean(a))
            row["source_auc_se"] = float(np.std(a, ddof=1) / np.sqrt(len(a)))
        criteria[lam] = row
    lambda_macro = max(
        full_good,
        key=lambda lam: (criteria[lam]["source_macro_auc"], lam),
    )
    lambda_cov = min(
        full_good,
        key=lambda lam: (criteria[lam]["covariance_mean_nll"], lam),
    )

    best_macro = criteria[lambda_macro]["source_macro_auc"]
    macro_margin = 0.01
    discrim_admissible = [
        lam for lam in full_good
        if criteria[lam]["source_macro_auc"] >= best_macro - macro_margin - 1e-15
    ]
    lambda_lex = min(
        discrim_admissible,
        key=lambda lam: (
            criteria[lam]["covariance_mean_nll"],
            abs(lam - lambda_macro),
            lam,
        ),
    )

    dense_cov_lambda = min(
        dense_good,
        key=lambda lam: (criteria[lam]["covariance_mean_nll"], lam),
    )
    dense_cov_min = criteria[dense_cov_lambda]["covariance_mean_nll"]
    dense_regret = {
        str(lam): float(criteria[lam]["covariance_mean_nll"] - dense_cov_min)
        for lam in dense_good
    }

    return {
        "lambda_source_macro": float(lambda_macro),
        "lambda_covariance_full": float(lambda_cov),
        "lambda_lexicographic": float(lambda_lex),
        "lexicographic_macro_margin": macro_margin,
        "lexicographic_admissible_lambdas": list(map(float, discrim_admissible)),
        "dense_covariance_lambda": float(dense_cov_lambda),
        "dense_covariance_regret_by_lambda": dense_regret,
        "criteria_by_lambda": {
            str(lam): criteria[lam] for lam in good
        },
        "feasible_full_lambdas": list(map(float, full_good)),
        "feasible_dense_lambdas": list(map(float, dense_good)),
        "inner_fold_receipts": fold_receipts,
    }


def load_dataset(name):
    spec = dataset_spec(name)
    paradigm = MotorImagery(
        n_classes=2,
        events=list(spec["events"]),
        fmin=8,
        fmax=30,
        resample=160,
        baseline=None,
        reject_by_annotation=False,
    )

    covs = []
    ys = []
    groups = []
    participant_receipt = []
    channel_count = None
    channel_names = None

    for sid in spec["subjects"]:
        X, labels, meta = paradigm.get_data(
            dataset=spec["dataset"],
            subjects=[sid],
        )
        labels = np.asarray(labels)
        if set(labels) != set(spec["events"]):
            raise ValueError(
                f"Unexpected labels for participant {sid}: {set(labels)}"
            )
        y = (labels == spec["positive_label"]).astype(np.int64)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(f"Participant {sid} lacks one class")

        if channel_count is None:
            channel_count = int(X.shape[1])
            channel_names = list(map(str, meta.attrs.get("channels", []))) if hasattr(meta, "attrs") else []
        elif X.shape[1] != channel_count:
            raise ValueError("Channel-count drift across participants")

        C = covariances(X)
        covs.append(C)
        ys.append(y)
        groups.extend([sid] * len(y))

        sessions = (
            sorted(set(map(str, meta["session"])))
            if "session" in meta else []
        )
        runs = (
            sorted(set(map(str, meta["run"])))
            if "run" in meta else []
        )
        participant_receipt.append({
            "participant": int(sid),
            "trials": int(len(y)),
            "class0": int(np.sum(y == 0)),
            "class1": int(np.sum(y == 1)),
            "sessions": sessions,
            "runs": runs,
            "channels": int(X.shape[1]),
            "samples_per_trial": int(X.shape[-1]),
        })
        del X

    C = np.concatenate(covs)
    y = np.concatenate(ys)
    g = np.asarray(groups)

    if set(np.unique(g)) != set(spec["subjects"]):
        raise ValueError("Participant denominator drift")
    if len(spec["subjects"]) != spec["expected_subjects"]:
        raise AssertionError("Frozen denominator mismatch")

    return C, y, g, participant_receipt, spec, channel_count, channel_names


def freeze_target_scores(C_source, y_source, C_target, path, out_file):
    """Generate and write the target score family WITHOUT target labels."""
    inv = reference(C_source)
    Xtr = tangent(C_source, inv)
    Xte = tangent(C_target, inv)
    state = spectral_state(Xtr, y_source)

    lw_scores, lw = analytic_scores(Xtr, y_source, Xte, "LW")
    oas_scores, oas = analytic_scores(Xtr, y_source, Xte, "OAS")

    dense_lambdas = np.asarray(path["feasible_dense_lambdas"], dtype=float)
    dense_scores = np.vstack([
        scalar_scores(state, Xte, float(lam))
        for lam in dense_lambdas
    ])

    full_lookup = {
        float(lam): dense_scores[np.where(np.isclose(dense_lambdas, float(lam), atol=1e-15))[0][0]]
        for lam in path["feasible_full_lambdas"]
        if np.any(np.isclose(dense_lambdas, float(lam), atol=1e-15))
    }

    def fixed(lam):
        lam = float(lam)
        if lam in full_lookup:
            return full_lookup[lam]
        return scalar_scores(state, Xte, lam)

    lr = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=3000
    )
    lr.fit(Xtr, y_source)
    lr_score = lr.decision_function(Xte)

    mdm = MDM(metric="riemann", n_jobs=1)
    mdm.fit(C_source, y_source)
    dist = mdm.transform(C_target)
    if dist.shape[1] != 2:
        raise ValueError("MDM did not return two class distances")
    mdm_score = dist[:, 0] - dist[:, 1]

    np.savez_compressed(
        out_file,
        dense_lambdas=dense_lambdas,
        dense_scores=dense_scores,
        LedoitWolf_LDA=lw_scores,
        OAS_LDA=oas_scores,
        source_macro_full_domain_LDA=fixed(path["lambda_source_macro"]),
        source_cov_nll_full_domain_LDA=fixed(path["lambda_covariance_full"]),
        discriminability_first_lexicographic_LDA=fixed(path["lambda_lexicographic"]),
        TS_LogisticRegression=lr_score,
        Riemannian_MDM=mdm_score,
    )

    return {
        "score_file": str(out_file),
        "score_sha256": sha256_file(out_file),
        "analytic_shrinkage": {
            "LedoitWolf": float(lw),
            "OAS": float(oas),
        },
    }


def evaluate_frozen_scores(score_file, target_y, source_regret_by_lambda):
    """Apply target labels only after the score-family file exists and is hashed."""
    data = np.load(score_file, allow_pickle=False)
    y = np.asarray(target_y)
    methods = (
        "LedoitWolf_LDA",
        "OAS_LDA",
        "source_macro_full_domain_LDA",
        "source_cov_nll_full_domain_LDA",
        "discriminability_first_lexicographic_LDA",
        "TS_LogisticRegression",
        "Riemannian_MDM",
    )
    auc = {
        method: float(roc_auc_score(y, data[method]))
        for method in methods
    }

    lambdas = np.asarray(data["dense_lambdas"], dtype=float)
    score_matrix = np.asarray(data["dense_scores"], dtype=float)
    path_auc = {}
    for i, lam in enumerate(lambdas):
        path_auc[str(float(lam))] = float(
            roc_auc_score(y, score_matrix[i])
        )

    regrets = {
        float(k): float(v)
        for k, v in source_regret_by_lambda.items()
    }
    common = [
        float(lam) for lam in lambdas
        if float(lam) in regrets
    ]
    if len(common) < 8:
        raise ValueError("Too few common dense frontier points")

    budgets = sorted(set(regrets[lam] for lam in common))
    envelope = []
    for budget in budgets:
        allowed = [
            lam for lam in common
            if regrets[lam] <= budget + 1e-15
        ]
        vals = [path_auc[str(float(lam))] for lam in allowed]
        envelope.append({
            "budget": float(budget),
            "n_lambdas": int(len(allowed)),
            "auc_min": float(min(vals)),
            "auc_max": float(max(vals)),
            "auc_span": float(max(vals) - min(vals)),
        })

    max_span = max(row["auc_span"] for row in envelope)

    # Strict subset sensitivity: no interpolation.
    grid_rows = []
    for stride in (1, 2, 4, 8):
        subset = common[::stride]
        if common[-1] not in subset:
            subset.append(common[-1])
        b = sorted(set(regrets[lam] for lam in subset))
        spans = []
        for budget in b:
            allowed = [
                lam for lam in subset
                if regrets[lam] <= budget + 1e-15
            ]
            vals = [path_auc[str(float(lam))] for lam in allowed]
            spans.append(max(vals) - min(vals))
        grid_rows.append({
            "stride": int(stride),
            "candidates": int(len(subset)),
            "max_auc_span": float(max(spans)),
        })

    return {
        "auc": auc,
        "target_auc_path_by_lambda": path_auc,
        "validity_frontier": {
            "envelope_path": envelope,
            "max_auc_span": float(max_span),
            "grid_sensitivity": grid_rows,
            "target_path_used_for_selection": False,
        },
    }


def paired_summary(folds, candidate, baseline):
    d = np.asarray([
        f["auc"][candidate] - f["auc"][baseline]
        for f in folds
    ], dtype=float)
    n = len(d)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(20000, n))
    lo, hi = np.quantile(np.mean(d[idx], axis=1), [0.025, 0.975])

    obs = abs(float(np.mean(d)))
    tol = 64 * np.finfo(float).eps * max(1.0, obs)
    if n <= 20:
        exceed = 0
        total = 1 << n
        for mask in range(total):
            signs = np.ones(n)
            for i in range(n):
                if mask & (1 << i):
                    signs[i] = -1.0
            if abs(float(np.mean(signs * d))) >= obs - tol:
                exceed += 1
        p = float(exceed / total)
        mode = "exact"
        draws = total
    else:
        draws = 200000
        exceed = 0
        done = 0
        while done < draws:
            q = min(10000, draws - done)
            signs = rng.choice((-1.0, 1.0), size=(q, n))
            stats = np.abs(np.mean(signs * d[None, :], axis=1))
            exceed += int(np.sum(stats >= obs - tol))
            done += q
        p = float((exceed + 1) / (draws + 1))
        mode = "monte_carlo"

    return {
        "candidate": candidate,
        "baseline": baseline,
        "n_participants": int(n),
        "mean_delta_auc": float(np.mean(d)),
        "median_delta_auc": float(np.median(d)),
        "q25_delta_auc": float(np.quantile(d, 0.25)),
        "min_delta_auc": float(np.min(d)),
        "max_delta_auc": float(np.max(d)),
        "improved": int(np.sum(d > 0)),
        "tied": int(np.sum(d == 0)),
        "harmed_gt_0_02": int(np.sum(d < -0.02)),
        "participant_bootstrap_ci95": [float(lo), float(hi)],
        "signflip_p_two_sided": p,
        "signflip_mode": mode,
        "signflip_draws": int(draws),
    }


def holm(ps):
    p = np.asarray(ps, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(
            running,
            min(1.0, (len(p) - rank) * float(p[i])),
        )
        out[i] = running
    return list(map(float, out))


def bootstrap_frontier_span(folds):
    spans = np.asarray([
        f["validity_frontier"]["max_auc_span"]
        for f in folds
    ], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(
        0, len(spans), size=(20000, len(spans))
    )
    means = np.mean(spans[idx], axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "participant_mean_max_auc_span": float(np.mean(spans)),
        "participant_median_max_auc_span": float(np.median(spans)),
        "participant_min_max_auc_span": float(np.min(spans)),
        "participant_max_max_auc_span": float(np.max(spans)),
        "participant_bootstrap_mean_ci95": [float(lo), float(hi)],
        "bootstrap_reps": 20000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["Shin2017A", "BNCI2014_002"],
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    score_dir = out / "frozen_scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    C, y, g, participant_receipt, spec, p, channel_names = load_dataset(args.dataset)

    folds = []
    for sid in spec["subjects"]:
        outer = g == sid
        inner = ~outer
        path = source_path(C[inner], y[inner], g[inner])

        score_file = score_dir / f"participant_{sid:03d}_scores.npz"
        freeze_receipt = freeze_target_scores(
            C[inner],
            y[inner],
            C[outer],
            path,
            score_file,
        )

        # Target labels are consumed only after score file + hash exist.
        evaluated = evaluate_frozen_scores(
            score_file,
            y[outer],
            path["dense_covariance_regret_by_lambda"],
        )

        folds.append({
            "target_subject": int(sid),
            "source_subjects": list(map(int, sorted(set(g[inner])))),
            "target_trials": int(outer.sum()),
            "target_label_counts": {
                "class0": int(np.sum(y[outer] == 0)),
                "class1": int(np.sum(y[outer] == 1)),
            },
            "target_information_track": "Z0",
            "target_labels_used_for_selection": False,
            "target_batch_statistics_used": False,
            "score_freeze": freeze_receipt,
            "source_path": path,
            **evaluated,
        })

    methods = tuple(sorted(folds[0]["auc"]))
    if any(tuple(sorted(f["auc"])) != methods for f in folds):
        raise ValueError("Method-set drift across target participants")

    macro = {
        method: float(np.mean([f["auc"][method] for f in folds]))
        for method in methods
    }

    comparisons = [
        paired_summary(
            folds,
            "source_macro_full_domain_LDA",
            "source_cov_nll_full_domain_LDA",
        ),
        paired_summary(
            folds,
            "source_macro_full_domain_LDA",
            "LedoitWolf_LDA",
        ),
    ]
    adjusted = holm([x["signflip_p_two_sided"] for x in comparisons])
    for row, padj in zip(comparisons, adjusted):
        row["holm_p_primary_family"] = float(padj)

    lexicographic = [
        paired_summary(
            folds,
            "discriminability_first_lexicographic_LDA",
            "source_macro_full_domain_LDA",
        ),
        paired_summary(
            folds,
            "discriminability_first_lexicographic_LDA",
            "source_cov_nll_full_domain_LDA",
        ),
    ]

    score_hashes = {
        str(f["target_subject"]): f["score_freeze"]["score_sha256"]
        for f in folds
    }
    score_manifest_sha = hashlib.sha256(
        json.dumps(
            score_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    protocol_file = Path(__file__).resolve().with_name(
        "FROZEN_SHIN_BNCI002_FRONTIER_20261010.md"
    )
    if not protocol_file.exists():
        raise FileNotFoundError("Frozen protocol file missing next to runner")
    protocol_sha256 = sha256_file(protocol_file)

    full_grid_sha256 = hashlib.sha256(
        json.dumps(
            list(FULL_GRID), separators=(",", ":")
        ).encode()
    ).hexdigest()
    dense_grid_sha256 = hashlib.sha256(
        json.dumps(
            list(DENSE_GRID), separators=(",", ":")
        ).encode()
    ).hexdigest()

    target_trial_denominator = int(
        sum(f["target_trials"] for f in folds)
    )
    participant_trial_denominator = int(
        sum(r["trials"] for r in participant_receipt)
    )
    if target_trial_denominator != participant_trial_denominator:
        raise AssertionError(
            "Target trial denominator disagrees with participant receipt"
        )

    code_commit = os.environ.get("GITHUB_SHA", "LOCAL_UNVERSIONED")
    workflow_run_id = os.environ.get("GITHUB_RUN_ID", "LOCAL_UNVERSIONED")

    result = {
        "status": "UNTOUCHED_FRONTIER_CONFIRMATION_EXECUTED",
        "receipt_version": 4,
        "dataset": args.dataset,
        "cohort_role": "confirmatory",
        "code_commit": code_commit,
        "workflow_run_id": workflow_run_id,
        "protocol_file": protocol_file.name,
        "protocol_sha256": protocol_sha256,
        "full_grid_sha256": full_grid_sha256,
        "dense_grid_sha256": dense_grid_sha256,
        "subjects": list(map(int, spec["subjects"])),
        "n_subjects": int(len(spec["subjects"])),
        "events": list(spec["events"]),
        "positive_label": spec["positive_label"],
        "band_hz": [8, 30],
        "resample_hz": 160,
        "dataset_native_task_interval": True,
        "channels_used": int(p),
        "channel_names_if_available": channel_names,
        "feature_dimension": int(p * (p + 1) // 2),
        "target_information_track": "Z0",
        "target_labels_used_before_score_freeze": False,
        "target_batch_statistics_used": False,
        "full_grid": list(FULL_GRID),
        "dense_frontier_grid": list(DENSE_GRID),
        "participant_receipt": participant_receipt,
        "participant_count": int(len(spec["subjects"])),
        "trial_denominator": target_trial_denominator,
        "score_manifest_sha256": score_manifest_sha,
        "mean_auc_by_method": macro,
        "primary_comparisons": comparisons,
        "lexicographic_secondary_comparisons": lexicographic,
        "frontier_summary": bootstrap_frontier_span(folds),
        "folds": folds,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "moabb": moabb.__version__,
            "sklearn": sklearn.__version__,
            "pyriemann": pyriemann.__version__,
        },
        "wall_seconds": round(time.monotonic() - start, 2),
        "claim_boundary": (
            "Untouched confirmation for post-S021 validity-frontier and "
            "lexicographic claims only under this frozen Z0 pipeline."
        ),
    }

    result_file = out / f"{args.dataset}_frontier_confirmatory.json"
    result_file.write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(json.dumps({
        "status": result["status"],
        "dataset": args.dataset,
        "n_subjects": result["n_subjects"],
        "channels": result["channels_used"],
        "feature_dimension": result["feature_dimension"],
        "mean_auc_by_method": macro,
        "primary_comparisons": comparisons,
        "frontier_summary": result["frontier_summary"],
        "trial_denominator": target_trial_denominator,
        "code_commit": code_commit,
        "protocol_sha256": protocol_sha256,
        "score_manifest_sha256": score_manifest_sha,
        "wall_seconds": result["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
