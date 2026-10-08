"""EEG sensor identity: two-witness, source-anchor conditioned selective repair.

Experimental score consistency is NOT a certified error probability.
Trusted physical anchors and a correctly identified reference recording are
prerequisites; incorrect anchors defeat the inference contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Witness:
    assignment: tuple[int, ...]  # target row -> source physical channel role
    objective: float
    alternative_gap_per_channel: float


@dataclass(frozen=True)
class Decision:
    status: str
    assignment: tuple[int, ...] | None
    witnesses: tuple[Witness, Witness]
    reason: str


def anchor_fingerprint(channels: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Signal Pearson correlations against *trusted physical electrode* traces."""
    x, a = np.asarray(channels, np.float64), np.asarray(anchors, np.float64)
    if x.ndim != 2 or a.ndim != 2 or x.shape[1] != a.shape[1] or x.shape[1] < 32:
        raise ValueError("expected channel + trusted-anchor EEG timeseries")
    if not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError("nonfinite EEG: refuse inference")
    def norm(z):
        z = z - np.mean(z, axis=1, keepdims=True)
        scale = np.sqrt(np.sum(z * z, axis=1, keepdims=True))
        if np.any(scale <= 1e-12):
            raise ValueError("zero-variance physical electrode or anchor")
        return z / scale
    return norm(x) @ norm(a).T


def _best_assignment(source: np.ndarray, target: np.ndarray) -> Witness:
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source/target anchor evidence incompatible")
    n, anchors = source.shape
    if n < 2 or anchors < 2:
        raise ValueError("need multiple channels and trusted anchors")
    costs = ((target[:, None, :] - source[None, :, :]) ** 2).mean(axis=-1)
    rows, cols = linear_sum_assignment(costs)
    ass = np.empty(n, dtype=np.int64)
    ass[rows] = cols
    opt = float(costs[rows, cols].sum())
    # Calculate true *global assignment* alternative, not a greedy row score:
    # forbid each chosen link in turn and resolve the full bijection.
    alternatives = []
    for r in range(n):
        changed = costs.copy()
        changed[r, ass[r]] = np.inf
        alt_rows, alt_cols = linear_sum_assignment(changed)
        alternatives.append(float(changed[alt_rows, alt_cols].sum()))
    margin = max(0., float(min(alternatives) - opt) / n)
    return Witness(tuple(map(int, ass)), opt, margin)


def authorization(
    reference_a: np.ndarray, anchor_reference_a: np.ndarray,
    reference_b: np.ndarray, anchor_reference_b: np.ndarray,
    target_a: np.ndarray, anchor_target_a: np.ndarray,
    target_b: np.ndarray, anchor_target_b: np.ndarray,
    *, minimum_global_margin: float = .012,
) -> Decision:
    """Two separate source/target EEG segment witnesses must agree and be unique.

    The nonzero numeric margin is an exploratory frozen heuristic; not an
    identifiable posterior probability, confidence interval, or safety bound.
    """
    inputs = (
        (reference_a, anchor_reference_a, target_a, anchor_target_a),
        (reference_b, anchor_reference_b, target_b, anchor_target_b),
    )
    witnesses = []
    for ref, ra, tar, ta in inputs:
        witnesses.append(_best_assignment(
            anchor_fingerprint(ref, ra), anchor_fingerprint(tar, ta)))
    w1, w2 = witnesses
    if w1.assignment != w2.assignment:
        return Decision("refuse", None, (w1, w2), "independent EEG segments disagree")
    if min(w1.alternative_gap_per_channel, w2.alternative_gap_per_channel) < minimum_global_margin:
        return Decision("refuse", None, (w1, w2), "source anchors do not distinguish a unique assignment")
    if w1.assignment == tuple(range(len(w1.assignment))):
        return Decision("retain_identity", w1.assignment, (w1, w2), "no repair needed")
    return Decision("authorize_repair", w1.assignment, (w1, w2), "two agreeing globally distinguishable mappings")


def apply_authorized_permutation(target: np.ndarray, decision: Decision) -> np.ndarray:
    x = np.asarray(target)
    if decision.status != "authorize_repair" or decision.assignment is None:
        raise ValueError("no EEG electrode reindex authorized")
    if len(decision.assignment) != x.shape[0]:
        raise ValueError("authorized assignment does not fit data channels")
    return x[np.argsort(np.asarray(decision.assignment, int))]
