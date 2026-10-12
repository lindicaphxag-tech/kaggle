"""V23: spectral interval enclosure for pairwise rank stability on a shrinkage path.

For a fixed, source-fitted covariance eigenbasis, source-fitted class contrast
and FIXED held-source trials, scores are rational functions of lambda:
    z_pn(lambda) = sum_i a_pni / ((1-lambda)*e_i + lambda*nu)
This implementation encloses each term on a lambda subinterval, then
reports pairs proven to retain one strict sign across the entire interval.

It certifies ONLY finite, fixed held-source trial rankings conditional
on computed floating-point eigensystems. It gives NO unseen-person
generalisation certificate, confidence interval, or improved selector.
"""
from __future__ import annotations

import numpy as np


def _validate(state, X, y, low, high, segments, block_size):
    if len(state) != 5:
        raise ValueError("state must be (eigenvalues,U,contrast,mid,nu)")
    e, U, d, mid, nu = (np.asarray(state[0], dtype=float),
                        np.asarray(state[1], dtype=float),
                        np.asarray(state[2], dtype=float),
                        np.asarray(state[3], dtype=float), float(state[4]))
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    p = len(e)
    if e.shape != (p,) or U.shape != (p, p) or d.shape != (p,) or mid.shape != (p,):
        raise ValueError("source spectral state dimensions invalid")
    if X.ndim != 2 or X.shape[1] != p or y.shape != (len(X),):
        raise ValueError("features and labels dimensions invalid")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("binary labels with both classes required")
    if not (np.isfinite(X).all() and np.isfinite(e).all()
            and np.isfinite(U).all() and np.isfinite(d).all()
            and np.isfinite(mid).all() and np.isfinite(nu)):
        raise ValueError("nonfinite state or features")
    if not (0 <= low < high <= 1):
        raise ValueError("strict nonempty lambda interval in [0,1] required")
    if not (isinstance(segments, int) and segments > 0 and
            isinstance(block_size, int) and block_size > 0):
        raise ValueError("segments and block_size must be positive integers")
    if nu <= 0 or np.linalg.norm(U.T @ U - np.eye(p), ord=np.inf) > 1e-6:
        raise ValueError("invalid source eigensystem")
    # Use the same feasibility criterion as the original LDA score.
    floor = 64 * np.finfo(float).eps * max(1., nu, float(np.max(np.abs(e))))
    for lam in (low, high):
        r = (1. - lam) * e + lam * nu
        if np.any(r <= floor):
            raise ValueError("path includes a singular/infeasible covariance")
    return e, U, d, nu, X, y


def certified_path_rank_stability(
    state, X, labels, low, high, *, segments=8, block_size=32
):
    """Analytically enclose pair signs on every uniform lambda segment.

    A positive/negative pair is "stable" if the interval bound stays
    strictly above zero on EVERY segment, or strictly below on EVERY
    segment. Pairs close to numerical zero remain UNCERTAIN (fail closed).
    For every lambda in the interval, the finite held-person AUC can
    change from its lower-endpoint AUC by at most uncertain_fraction.
    """
    e, U, d, nu, X, y = _validate(
        state, X, labels, low, high, segments, block_size
    )
    Z = X @ U
    zd = U.T @ d
    pos, neg = Z[y == 1], Z[y == 0]
    grid = np.linspace(low, high, segments + 1)
    denom = np.asarray([(1. - t) * e + t * nu for t in grid])
    total = int(len(pos) * len(neg))
    certain = 0
    changed = 0
    auc_a_sum = 0.
    auc_b_sum = 0.
    remaining = 0

    for k in range(0, len(pos), block_size):
        coeff = (pos[k:k + block_size, None, :]
                 - neg[None, :, :]) * zd[None, None, :]
        base = np.sum(coeff / denom[0], axis=-1)
        end = np.sum(coeff / denom[-1], axis=-1)
        # Guard analytical enclosure from finite-precision cancellation:
        # conservative numerical slack conditional on computed coefficients.
        sure_pos = np.ones(base.shape, dtype=bool)
        sure_neg = np.ones(base.shape, dtype=bool)
        for s in range(segments):
            left_terms = coeff / denom[s]
            right_terms = coeff / denom[s + 1]
            lower = np.minimum(left_terms, right_terms).sum(-1)
            upper = np.maximum(left_terms, right_terms).sum(-1)
            magnitude = np.maximum(
                np.abs(left_terms), np.abs(right_terms)
            ).sum(-1)
            slack = (256 * np.finfo(float).eps *
                     (len(e) + 1) * (1 + magnitude))
            sure_pos &= lower > slack
            sure_neg &= upper < -slack
        stable = sure_pos | sure_neg
        certain += int(stable.sum())
        remaining += int(stable.size)
        first_state, last_state = np.sign(base), np.sign(end)
        changed += int(np.sum(first_state != last_state))
        auc_a_sum += float(np.sum((first_state > 0) + .5 * (first_state == 0)))
        auc_b_sum += float(np.sum((last_state > 0) + .5 * (last_state == 0)))
        if np.any(stable & (first_state != last_state)):
            raise AssertionError("endpoint pair reversal contradicts interval bound")
    if remaining != total:
        raise AssertionError("pair-accounting inconsistency")
    frac = float((total - certain) / total)
    before, after = auc_a_sum / total, auc_b_sum / total
    if abs(after - before) > frac + 1e-12:
        raise AssertionError("finite-person AUC outside analytic enclosure")
    return {
        "source_trial_pair_count": total,
        "lambda_low": float(low),
        "lambda_high": float(high),
        "segments": segments,
        "numerical_scope": "conditional-on-computed-eigensystem-and-conservative-fp-slack",
        "stable_pairs_certified": certain,
        "uncertain_pairs": total - certain,
        "certified_stable_fraction": float(certain / total),
        "uniform_path_auc_change_bound": frac,
        "observed_endpoint_pair_sign_changes": changed,
        "observed_endpoint_auc_low": float(before),
        "observed_endpoint_auc_high": float(after),
        "observed_endpoint_auc_delta": float(after - before),
        "only_fixed_held_source_trial_ranks_certified": True,
        "no_unseen_target_guarantee": True,
    }
