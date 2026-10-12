"""V25 spectral rank-cell analysis via Bernstein polynomial positivity.

Given a FIXED source-fitted shrinkage LDA and held-SOURCE trials,
a pair score contrast is a rational function whose numerator has degree
at most p-1. With strictly positive denominators, the contrast's sign
equals this numerator's sign. Bernstein convex-hull bounds therefore
certify some pair rankings over a *whole continuous lambda interval*.

Mathematical sign certificate is exact in real arithmetic. The NumPy
implementation is conditional on the computed eigensystem and not
formally verified directed-rounding arithmetic. Fail-closed tolerance
reduces fragile claims but is NOT an interval-arithmetic proof for FP
uncertainty. This never guarantees target-person AUC or selection gains.
"""
from __future__ import annotations

import numpy as np
from spectral_path_certificate_v23 import _validate


def _linear_bernstein_product(left, right):
    """Bernstein coefficients of product_j ((1-t)*left_j + t*right_j)."""
    coeff = np.ones(1, dtype=np.float64)
    for a, b in zip(left, right):
        degree = len(coeff) - 1
        nextcoeff = np.zeros(degree + 2, dtype=np.float64)
        k = np.arange(degree + 1)
        nextcoeff[:degree + 1] += (degree + 1 - k) / (degree + 1) * coeff * a
        nextcoeff[1:] += (k + 1) / (degree + 1) * coeff * b
        coeff = nextcoeff
    return coeff


def spectral_pair_numerator_coefficients(state, score_diff, low, high):
    """Return degree-(p-1) Bernstein numerator coefficients for one pair.

    The returned numerator has the sign of the true pair score contrast,
    with a positive (lambda-independent) scale removed. Handles p=1.
    """
    e, U, d, mid, nu = state
    e, U, d = np.asarray(e, float), np.asarray(U, float), np.asarray(d, float)
    v = np.asarray(score_diff, float)
    if v.shape != (len(e),):
        raise ValueError("score_diff must match spectral dimension")
    p = len(e)
    if p < 1 or p > 12:
        raise ValueError("Bernstein analysis limited to p<=12 for FP conditioning")
    if not np.isfinite(v).all() or not 0 <= low < high <= 1:
        raise ValueError("nonfinite pair contrast or invalid lambda bounds")
    ql = (1 - low) * e + low * nu
    qr = (1 - high) * e + high * nu
    if np.any(ql <= 0) or np.any(qr <= 0):
        raise ValueError("singular shrinkage path")
    scale = np.maximum(ql, qr)
    left, right = ql / scale, qr / scale
    pair_coeff = ((v @ U) * (U.T @ d)) / scale
    basis = np.asarray([
        _linear_bernstein_product(np.delete(left, i),
                                  np.delete(right, i))
        for i in range(p)
    ])
    return pair_coeff @ basis


def bernstein_rank_path(
    state, X, labels, low, high, *, segments=8, block_size=32
):
    """Count pair rankings proven sign-invariant on a continuous lambda path.

    This is a different *source-domain* certificate from the V23
    termwise/Taylor enclosure. The uncertified complement may be
    perfectly stable; it is not a count of observed rank reversals.
    """
    e, U, d, nu, X, y = _validate(
        state, X, labels, low, high, segments, block_size)
    p = len(e)
    if p > 12:
        raise ValueError("p>12 spectral numerator polynomial not FP-safe here")
    Z = X @ U
    zd = U.T @ d
    positive, negative = Z[y == 1], Z[y == 0]
    n_pairs = int(len(positive) * len(negative))
    sure_positive_count = 0
    sure_negative_count = 0
    first_credit_sum = 0.0
    last_credit_sum = 0.0
    transitions = 0
    fixed_positive_credit = 0.0
    max_gain = 0.0
    max_loss = 0.0
    path_grid = np.linspace(low, high, segments + 1)
    eps = np.finfo(float).eps

    for start in range(0, len(positive), block_size):
        coeff = (positive[start:start + block_size, None, :]
                 - negative[None, :, :]) * zd[None, None, :]
        sure_pos = np.ones(coeff.shape[:-1], dtype=bool)
        sure_neg = np.ones(coeff.shape[:-1], dtype=bool)
        for left_lam, right_lam in zip(path_grid[:-1], path_grid[1:]):
            ql = (1. - left_lam) * e + left_lam * nu
            qr = (1. - right_lam) * e + right_lam * nu
            scale = np.maximum(ql, qr)
            L, R = ql / scale, qr / scale
            basis = np.stack([
                _linear_bernstein_product(np.delete(L, i),
                                          np.delete(R, i))
                for i in range(p)], axis=0)
            ab = coeff / scale[None, None, :]
            polynomial = np.einsum("...i,ij->...j", ab, basis)
            magnitude = np.einsum(
                "...i,ij->...j", np.abs(ab), np.abs(basis))
            slack = (512. * eps * (p + 1)**2 * (1. + magnitude))
            sure_pos &= np.all(polynomial > slack, axis=-1)
            sure_neg &= np.all(polynomial < -slack, axis=-1)

        den_a = (1-low)*e + low*nu
        den_b = (1-high)*e + high*nu
        first = np.sign(np.sum(coeff/den_a, axis=-1))
        last = np.sign(np.sum(coeff/den_b, axis=-1))
        credit = (first > 0).astype(float) + .5*(first == 0)
        cred_last = (last > 0).astype(float) + .5*(last == 0)
        changed = np.sum(first != last)
        if np.any((sure_pos | sure_neg) & (first != last)):
            raise AssertionError("Bernstein path sign proof contradicts endpoint scores")
        uncertain = ~(sure_pos | sure_neg)
        sure_positive_count += int(np.sum(sure_pos))
        sure_negative_count += int(np.sum(sure_neg))
        first_credit_sum += float(np.sum(credit))
        last_credit_sum += float(np.sum(cred_last))
        transitions += int(changed)
        max_gain += float(np.sum((1-credit)*uncertain))
        max_loss += float(np.sum(credit*uncertain))
        fixed_positive_credit += float(np.sum(sure_pos))

    assert sure_positive_count + sure_negative_count <= n_pairs
    uncertain_n = n_pairs - sure_positive_count - sure_negative_count
    auc_start = first_credit_sum / n_pairs
    auc_end = last_credit_sum / n_pairs
    auc_min = max(0., auc_start - max_loss / n_pairs)
    auc_max = min(1., auc_start + max_gain / n_pairs)
    if not (auc_min-1e-12 <= auc_end <= auc_max+1e-12):
        raise AssertionError("Bernstein AUC path bound excludes endpoint")
    return {
        "source_trial_pair_count": n_pairs,
        "lambda_low": float(low), "lambda_high": float(high),
        "segments": segments,
        "spectral_dimension": p,
        "numerator_polynomial_max_degree": p - 1,
        "max_distinct_sign_changes_per_nondegenerate_pair": p - 1,
        "certain_positive_pairs": sure_positive_count,
        "certain_negative_pairs": sure_negative_count,
        "certain_fixed_sign_fraction": float((n_pairs-uncertain_n)/n_pairs),
        "uncertain_pairs": uncertain_n,
        "uncertain_pair_fraction": float(uncertain_n/n_pairs),
        "observed_endpoint_rank_state_changes": transitions,
        "observed_endpoint_auc_low": float(auc_start),
        "observed_endpoint_auc_high": float(auc_end),
        "uniform_path_auc_lower_bound": float(auc_min),
        "uniform_path_auc_upper_bound": float(auc_max),
        "scope": "source-validation fixed people and fixed fitted fold models",
        "numerical_scope": "conditional-on-computed-eigensystem-not-directed-rounding",
        "no_target_performance_certificate": True,
    }
