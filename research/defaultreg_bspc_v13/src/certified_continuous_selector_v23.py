"""V23 source-only certified continuous shrinkage search (CCSS).

Maximise participant-macro held-SOURCE ROC-AUC on one scalar LDA shrinkage
interval using spectral interval upper bounds, not an arbitrary finite grid.

IMPORTANT: 'certified' means a conditional bound on the maximum finite
source-validation AUC of the fixed set of source-fitted fold models, NOT
a bound on unseen-subject performance. The result can be inconclusive.
No target person data or model selection on target data are supported.
"""
from __future__ import annotations
import heapq
from typing import Sequence
import numpy as np

from spectral_path_certificate_v23 import certified_path_rank_stability


def source_person_auc_at_lambda(model, lam):
    state, X, y = model
    e, U, d, mid, nu = state
    e, U, d = np.asarray(e), np.asarray(U), np.asarray(d)
    y = np.asarray(y)
    r = (1. - lam) * e + lam * nu
    if np.any(r <= 0):
        raise ValueError("infeasible covariance")
    w = U @ ((U.T @ d) / r)
    scores = (X - mid) @ w
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        raise ValueError("each participant needs positive and negative trials")
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def certified_source_auc_search(
    source_fold_models: Sequence,
    *,
    low: float,
    high: float,
    tolerance: float = .01,
    max_splits: int = 12,
    segments: int = 1,
    block_size: int = 32,
    initial_lambdas: Sequence[float] = (),
):
    """Branch-and-bound max of fixed source-fold macro AUC over [low,high].

    Each model = (source-trained spectral state, held-SOURCE features, labels).
    A numerical upper-bound > best AUC is NOT proof of another better lambda;
    if max_splits exhausts, certificate_met=False, with honest residual.
    """
    if not source_fold_models:
        raise ValueError("need held-source validation fold models")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be nonnegative finite")
    if not isinstance(max_splits, int) or max_splits < 0:
        raise ValueError("max_splits must be nonnegative integer")
    if not (np.isfinite(low) and np.isfinite(high) and 0 <= low < high <= 1):
        raise ValueError("invalid interval")
    anchors = np.asarray(list(initial_lambdas), dtype=float)
    if anchors.ndim != 1 or np.any(~np.isfinite(anchors)) or np.any(anchors < low) or np.any(anchors > high):
        raise ValueError("anchors must be finite and inside the search interval")
    cache = {}

    def macro(lam):
        if lam not in cache:
            cache[lam] = float(np.mean([
                source_person_auc_at_lambda(model, lam)
                for model in source_fold_models
            ]))
        return cache[lam]

    def bound(a, b):
        rows = [
            certified_path_rank_stability(
                model[0], model[1], model[2], a, b,
                segments=segments, block_size=block_size,
            )
            for model in source_fold_models
        ]
        # AUC macro of a fixed pool is the arithmetic mean of subject
        # AUC, so its continuous-path maximum is <= mean of subject upper bounds.
        upper = float(np.mean([r["uniform_path_auc_upper_bound"] for r in rows]))
        ua = macro(a)
        ub = macro(b)
        return min(1., max(ua, ub, upper) + 1e-12)

    scores = {float(lam): macro(float(lam))
              for lam in sorted(set([float(low), float(high)] + anchors.tolist()))}
    best_lam = max(scores, key=lambda la: (scores[la], -la))
    best_auc = scores[best_lam]
    # Anchors are evaluated and *partition* the original interval, so no
    # neglected gap can invalidate the global source-only bound.
    ordered = sorted(scores)
    heap = [(-bound(a, b), a, b)
            for a, b in zip(ordered[:-1], ordered[1:])]
    heapq.heapify(heap)
    initial = max((-x[0] for x in heap), default=best_auc)
    split_count = 0
    while heap and split_count < max_splits:
        top = -heap[0][0]
        if top <= best_auc + tolerance:
            break
        neg_bound, a, b = heapq.heappop(heap)
        midpoint = (a + b) / 2.
        if midpoint == a or midpoint == b:
            heapq.heappush(heap, (neg_bound, a, b))
            break
        score = macro(midpoint)
        scores[midpoint] = score
        if score > best_auc + 1e-14 or (
                abs(score - best_auc) <= 1e-14 and midpoint < best_lam):
            best_auc, best_lam = score, midpoint
        for l, r in ((a, midpoint), (midpoint, b)):
            heapq.heappush(heap, (-bound(l, r), l, r))
        split_count += 1
    upper = max(best_auc, -heap[0][0]) if heap else best_auc
    return {
        "source_only": True,
        "conditional_on_fixed_fitted_inner_models": True,
        "unseen_target_performance_bound": False,
        "interval": [float(low), float(high)],
        "selected_lambda_source_macro": float(best_lam),
        "best_evaluated_source_macro_auc": float(best_auc),
        "upper_bound_global_continuous_source_macro_auc": float(upper),
        "residual_source_optimality_gap_upper_bound": float(max(0, upper-best_auc)),
        "requested_source_optimality_tolerance": float(tolerance),
        "source_optimality_certificate_met": bool(upper - best_auc <= tolerance + 1e-12),
        "source_people": len(source_fold_models),
        "interval_splits": split_count,
        "lambda_evaluations": len(cache),
        "initial_anchor_count": len(ordered),
        "max_splits": max_splits,
        "failed_closed_when_budget_exhausted": bool(
            upper - best_auc > tolerance + 1e-12
        ),
    }
