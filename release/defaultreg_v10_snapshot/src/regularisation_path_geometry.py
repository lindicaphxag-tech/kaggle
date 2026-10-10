"""Exact geometry of a scalar covariance-shrinkage path.

This module supplies the operator-level mechanism used by Default
Regularisation. It is an analysis/audit tool, NOT a target-driven selector.

For
    A(lambda) = (1-lambda) S + lambda nu I
    w(lambda) = A(lambda)^(-1) d_source
with B = dA/dlambda = nu I - S,

    w' = -A^(-1) B w.

For an evaluation participant with covariance Sigma and class contrast d_eval,
the signed standardized linear-discriminant margin is

    m = (w^T d_eval) / sqrt(w^T Sigma w).

Write
    alpha = (w'^T Sigma w)/(w^T Sigma w)
    r = w' - alpha w.

Then r is Sigma-orthogonal to w and

    m' = (r^T d_eval) / sqrt(w^T Sigma w).

Thus path changes that only rescale w cannot change the standardized margin;
only the Sigma-orthogonal rotation component can do so.

By contrast, Gaussian covariance predictive NLL per dimension,

    C = [log det A + tr(A^(-1) Sigma)] / p,

has derivative

    C' = [tr(A^(-1) B)
          - tr(A^(-1) B A^(-1) Sigma)] / p,

which contains no class-contrast vector. This is the precise mechanism behind
the absence of a general guarantee that covariance-fidelity and discriminative
optima coincide.

Target/evaluation labels may be represented through d_eval when this module is
used post hoc. The module itself must never be called inside target-dependent
model selection.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PathGeometry:
    lambda_value: float
    nu: float
    covariance_nll: float
    covariance_nll_derivative: float
    standardized_margin: float
    standardized_margin_derivative: float
    w_sigma_norm: float
    scale_rate: float
    rotation_sigma_norm: float
    relative_rotation_rate: float
    rotation_dot_eval_contrast: float
    sigma_orthogonality_residual: float
    target_information_used_for_selection: bool = False
    status: str = "ANALYSIS_ONLY_REGULARISATION_PATH_GEOMETRY"


def _sym_spd(M, name):
    X = np.asarray(M, dtype=float)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} must be finite")
    X = (X + X.T) / 2
    e = np.linalg.eigvalsh(X)
    tol = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(e))))
    if np.min(e) <= tol:
        raise ValueError(f"{name} must be positive definite")
    return X


def _vector(x, p, name):
    v = np.asarray(x, dtype=float)
    if v.ndim != 1 or len(v) != p or not np.isfinite(v).all():
        raise ValueError(f"{name} must be finite length-{p} vector")
    return v


def scalar_path_matrix(S, lam):
    S = _sym_spd(S, "S")
    lam = float(lam)
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must lie in [0,1]")
    p = S.shape[0]
    nu = float(np.trace(S) / p)
    A = (1.0 - lam) * S + lam * nu * np.eye(p)
    return (A + A.T) / 2, nu


def covariance_predictive_nll(S, Sigma_eval, lam):
    S = _sym_spd(S, "S")
    Sigma = _sym_spd(Sigma_eval, "Sigma_eval")
    if S.shape != Sigma.shape:
        raise ValueError("S and Sigma_eval dimensions differ")
    A, _ = scalar_path_matrix(S, lam)
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0 or not math.isfinite(float(logdet)):
        raise ValueError("regularised covariance is not SPD")
    return float((logdet + np.trace(np.linalg.solve(A, Sigma))) / len(A))


def standardized_margin(S, d_source, Sigma_eval, d_eval, lam):
    S = _sym_spd(S, "S")
    Sigma = _sym_spd(Sigma_eval, "Sigma_eval")
    if S.shape != Sigma.shape:
        raise ValueError("S and Sigma_eval dimensions differ")
    p = len(S)
    ds = _vector(d_source, p, "d_source")
    de = _vector(d_eval, p, "d_eval")
    A, _ = scalar_path_matrix(S, lam)
    w = np.linalg.solve(A, ds)
    b = float(w @ Sigma @ w)
    if not math.isfinite(b) or b <= 0:
        raise ValueError("non-positive evaluation variance")
    return float(w @ de / math.sqrt(b))


def analyse_path_geometry(S, d_source, Sigma_eval, d_eval, lam):
    S = _sym_spd(S, "S")
    Sigma = _sym_spd(Sigma_eval, "Sigma_eval")
    if S.shape != Sigma.shape:
        raise ValueError("S and Sigma_eval dimensions differ")
    p = len(S)
    ds = _vector(d_source, p, "d_source")
    de = _vector(d_eval, p, "d_eval")
    A, nu = scalar_path_matrix(S, lam)
    B = nu * np.eye(p) - S

    w = np.linalg.solve(A, ds)
    w_prime = -np.linalg.solve(A, B @ w)

    b = float(w @ Sigma @ w)
    if not math.isfinite(b) or b <= 0:
        raise ValueError("non-positive evaluation variance")
    wnorm = math.sqrt(b)

    scale_rate = float(w_prime @ Sigma @ w / b)
    rotation = w_prime - scale_rate * w
    orth = float(rotation @ Sigma @ w)

    rotation_norm_sq = float(rotation @ Sigma @ rotation)
    if rotation_norm_sq < -1e-12:
        raise AssertionError("rotation Sigma-norm became negative")
    rotation_norm = math.sqrt(max(0.0, rotation_norm_sq))

    margin = float(w @ de / wnorm)
    margin_derivative = float(rotation @ de / wnorm)

    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        raise ValueError("regularised covariance is not SPD")
    Ainv_B = np.linalg.solve(A, B)
    Ainv_Sigma = np.linalg.solve(A, Sigma)
    nll = float((logdet + np.trace(Ainv_Sigma)) / p)
    nll_derivative = float(
        (
            np.trace(Ainv_B)
            - np.trace(Ainv_B @ Ainv_Sigma)
        ) / p
    )

    return PathGeometry(
        lambda_value=float(lam),
        nu=float(nu),
        covariance_nll=nll,
        covariance_nll_derivative=nll_derivative,
        standardized_margin=margin,
        standardized_margin_derivative=margin_derivative,
        w_sigma_norm=float(wnorm),
        scale_rate=scale_rate,
        rotation_sigma_norm=float(rotation_norm),
        relative_rotation_rate=float(rotation_norm / wnorm),
        rotation_dot_eval_contrast=float(rotation @ de),
        sigma_orthogonality_residual=orth,
    )


def finite_difference_check(S, d_source, Sigma_eval, d_eval, lam, *, h=1e-6):
    lam = float(lam)
    h = float(h)
    if not 0.0 < h < 1e-2:
        raise ValueError("h must lie in (0, 1e-2)")
    if not h < lam < 1.0 - h:
        raise ValueError("finite-difference check needs lambda away from endpoints")
    exact = analyse_path_geometry(S, d_source, Sigma_eval, d_eval, lam)
    nll_fd = (
        covariance_predictive_nll(S, Sigma_eval, lam + h)
        - covariance_predictive_nll(S, Sigma_eval, lam - h)
    ) / (2 * h)
    margin_fd = (
        standardized_margin(S, d_source, Sigma_eval, d_eval, lam + h)
        - standardized_margin(S, d_source, Sigma_eval, d_eval, lam - h)
    ) / (2 * h)
    return {
        "covariance_nll_derivative_exact": exact.covariance_nll_derivative,
        "covariance_nll_derivative_fd": float(nll_fd),
        "margin_derivative_exact": exact.standardized_margin_derivative,
        "margin_derivative_fd": float(margin_fd),
        "covariance_abs_error": float(abs(exact.covariance_nll_derivative - nll_fd)),
        "margin_abs_error": float(abs(exact.standardized_margin_derivative - margin_fd)),
        "sigma_orthogonality_residual": exact.sigma_orthogonality_residual,
    }


if __name__ == "__main__":
    print("Exact regularisation-path geometry defined; no model selection performed.")
