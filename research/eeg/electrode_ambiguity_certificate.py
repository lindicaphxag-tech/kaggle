"""Exact linear decision certificate under explicitly bounded electrode-ID ambiguity.

Groups of positions are known to contain exactly the corresponding named
channels, but their within-group assignment may be any bijection. The model
is *linear in channel-wise extracted features*. Hungarian assignment computes
an exact min/max margin over every allowed identity permutation. Authorize a
label only if every allowed assignment gives the same sign.

Known set-membership / assignment optimization, not a claim of a new theorem,
novel EEG classification performance or hardware safety certification.
"""
from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Certificate:
    authorized: bool
    label: int | None
    lower_margin: float
    upper_margin: float
    permutations_examined_implicitly: int
    constraint: str = 'disjoint within-group permutations; known channel multiset'


def _check_partition(n, groups):
    used = set()
    normalized = []
    for g in groups:
        g = tuple(int(j) for j in g)
        if not g or len(set(g)) != len(g) or any(j < 0 or j >= n for j in g):
            raise ValueError('Out-of-range or repeated electrode within group')
        if used.intersection(g):
            raise ValueError('Groups overlap: cannot factorize assignment')
        used.update(g)
        normalized.append(g)
    return normalized


def certified_linear_prediction(features, weights, intercept, groups, *, tol=1e-10):
    """Return tight logit range over disjoint groups of unknown permutations.

    features: [channels, bands], observed log features in uncertain slots.
    weights: [channels, bands], frozen linear classifier raw-space weights.
    groups: disjoint position groups with arbitrarily permuted electrode IDs.

    If these groups do NOT include every physically possible identity mapping,
    the certificate does NOT guarantee correctness for the true montage.
    """
    X = np.asarray(features, dtype=float)
    W = np.asarray(weights, dtype=float)
    if X.ndim != 2 or X.shape != W.shape or not np.isfinite(X).all() or not np.isfinite(W).all():
        raise ValueError('Expected matched finite channel-by-band arrays')
    if not np.isfinite(intercept) or tol < 0:
        raise ValueError('Invalid threshold/intercept')
    norm = _check_partition(len(X), groups)
    known = np.ones(len(X), dtype=bool)
    for g in norm:
        known[np.array(g, dtype=int)] = False
    lower = float(intercept + np.sum(X[known] * W[known]))
    upper = lower
    ncomb = 1
    for g in norm:
        ix = np.array(g, dtype=int)
        # cost[observed row, candidate reference channel].
        cost = X[ix] @ W[ix].T
        i, j = linear_sum_assignment(cost)
        lower += float(cost[i, j].sum())
        i, j = linear_sum_assignment(-cost)
        upper += float(cost[i, j].sum())
        ncomb *= factorial(len(g))
    if lower > upper + max(tol, 1e-8):
        raise RuntimeError('Invalid assignment optimization interval')
    if lower > tol:
        return Certificate(True, 1, lower, upper, ncomb)
    if upper < -tol:
        return Certificate(True, 0, lower, upper, ncomb)
    return Certificate(False, None, lower, upper, ncomb)


def sklearn_pipeline_raw_weights(model, nchannels, nbands=2):
    """Exact raw-feature affine map from fitted StandardScaler+binary LR."""
    scaler = model.named_steps['standardscaler']
    linear = model.named_steps['logisticregression']
    if len(linear.classes_) != 2 or (nchannels * nbands) != len(scaler.mean_):
        raise ValueError('Expected a fitted binary logistic linear model')
    w = (linear.coef_[0] / scaler.scale_).reshape(nchannels, nbands)
    b = float(linear.intercept_[0] - linear.coef_[0] @ (scaler.mean_ / scaler.scale_))
    return w, b, tuple(linear.classes_)
