#!/usr/bin/env python3
"""Conditional exact-rational sample-matrix -> covariance -> tangent bridge.

For exact-rational stored matrix X (n time samples by p channels), fixed
nonnegative ridge r and FIXED exact SPD reference G, assume an externally
justified entrywise sample-input uncertainty |X_true-X| <= eta.

Provides an all-perturbations analytical upper bound on the Euclidean
error of vec_sym(log(G^-1/2 C G^-1/2)); vec_sym uses sqrt(2) for
off-diagonals and is an isometry from Frobenius to Euclidean norm.

Only exact real-operator sensitivity is bounded. NOT a certificate
of floating filtering, data-dependent covariance regularization,
reference estimation, eigendecomposition/logm implementation, MNE
floating-point operations, or independently new EEG classification gain.
The log Lipschitz estimate uses the established SPD resolvent integral.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from fractions import Fraction
from pathlib import Path
import sympy as sp


def _rat(x):
    if isinstance(x, (float, complex)):
        raise TypeError("unbounded binary float is not exact rational")
    r = sp.sympify(x)
    if r.is_Rational is not True:
        raise TypeError("exact rational input required")
    return sp.Rational(r)


def _matrix(A):
    return sp.Matrix(A).applyfunc(_rat)


def _sqrt_upper(q, bits=30):
    """Outward rational bound, never use a floating sqrt."""
    v = Fraction(q)
    if v < 0:
        raise ValueError("negative squared norm")
    scale = 1 << bits
    u = (v.numerator * scale * scale + v.denominator - 1)//v.denominator
    return sp.Rational(math.isqrt(u) + 1, scale)


def _inf_norm(A):
    return max(sum(abs(A[i,j]) for j in range(A.cols))
               for i in range(A.rows))


def _check_spd(A):
    if A.rows != A.cols or A != A.T:
        raise ValueError("matrix must be symmetric square")
    if any(A[:k,:k].det() <= 0 for k in range(1,A.rows+1)):
        raise ValueError("exact SPD gate failed")


def covariance(X, ridge):
    X, ridge = _matrix(X), _rat(ridge)
    if X.rows < 2 or X.cols < 2 or ridge < 0:
        raise ValueError("need n>=2, p>=2 and nonnegative fixed ridge")
    R = X - sp.ones(X.rows,1)*sp.Matrix(
        [sum(X[:,j])/X.rows for j in range(X.cols)]).T
    return R.T*R/(X.rows-1) + ridge*sp.eye(X.cols), R


def certify_bridge(X, G, eta, ridge):
    """Sound for every entrywise eta perturbation and exact fixed-G map.

    Let P denote time centering, deltaX<=sqrt(np)*eta.
    deltaC<= [2||PX||_F deltaX + deltaX^2]/(n-1).
    For C and G SPD: lambda_min(G^-1/2 C G^-1/2)
       >= 1/(||C^-1||_inf ||G||_inf) = a.
    Perturbed whitened covariance deviation norm <=
       ||G^-1||_inf deltaC = b.
    If a-b>0, ||Delta log||_F <= b/(a-b) by the exact SPD
    matrix-log resolvent integral (no numerical log implementation).
    """
    X, G, eta, ridge = _matrix(X), _matrix(G), _rat(eta), _rat(ridge)
    if eta < 0:
        raise ValueError("negative epsilon")
    if G.shape != (X.cols,X.cols):
        raise ValueError("reference dimension")
    _check_spd(G)
    C, R = covariance(X,ridge)
    _check_spd(C)
    n,p = X.shape
    er = _sqrt_upper(n*p)*eta
    deltaC = (2*_sqrt_upper(sum(v*v for v in R))*er + er*er)/(n-1)
    cmin = 1/_inf_norm(C.inv())
    gmax = _inf_norm(G)
    ginv = _inf_norm(G.inv())
    a = cmin/gmax
    b = ginv*deltaC
    m = a-b
    out = {
      "status": "BOUND" if m>0 else "UNKNOWN",
      "n_samples": n, "n_dimensions": p,
      "input_entrywise_eta": str(eta),
      "fixed_ridge": str(ridge),
      "input_error_F_upper": str(er),
      "covariance_error_F_upper": str(deltaC),
      "whitened_nominal_gap_lower": str(a),
      "robust_gap_lower": str(m),
      "reference_G_is_fixed_and_exact_SPD": True,
      "proof_uses_exact_operator_not_float_logm": True,
      "preprocessing_and_reference_roundoff_NOT_certified": True,
      "data_dependent_covariance_regularization_NOT_covered": True,
      "no_EEG_classification_accuracy_gain_claim": True,
    }
    if m>0:
        out["tangent_error_l2_upper"] = str(b/m)
    else:
        out["reason"] = "no guaranteed positive perturbed SPD gap"
    return out


def regression():
    """Exact synthetic inputs; float scipy logm is *only* a diagnostic."""
    import numpy as np
    from scipy.linalg import logm
    rng = random.Random(20261014)
    verified, sampled = 0, 0
    ratio_max = 0.0
    for k in range(24):
        n,p = 12,3
        X = sp.Matrix([[sp.Rational(rng.randint(-14,14),7)
                        for j in range(p)] for i in range(n)])
        G = (sp.diag(1,sp.Rational(3,2),2) if k%2 else
             sp.Matrix([[2,sp.Rational(1,4),0],
                        [sp.Rational(1,4),2,0],[0,0,1]]))
        ridge,eta = sp.Rational(1,5),sp.Rational(1,1000)
        cert = certify_bridge(X,G,eta,ridge)
        if cert["status"]!="BOUND":
            continue
        verified += 1
        C,_ = covariance(X,ridge)
        g = np.array(G,dtype=float)
        eig,U = np.linalg.eigh(g)
        Gi = (U/np.sqrt(eig))@U.T
        def tangent(A):
            return np.asarray(logm(Gi@np.array(A,dtype=float)@Gi).real)
        baseline = tangent(C)
        upper = float(sp.Rational(cert["tangent_error_l2_upper"]))
        for rep in range(2):
            Y = X.copy()
            for i in range(n):
                for j in range(p):
                    Y[i,j] += sp.Rational(rng.choice((-1,0,1)),1000)
            Cy,_ = covariance(Y,ridge)
            difference = np.linalg.norm(tangent(Cy)-baseline,ord="fro")
            assert difference <= upper*(1+1e-9)+1e-13
            ratio_max = max(ratio_max,difference/upper)
            sampled += 1
    assert verified>=20 and sampled>=40
    reject = certify_bridge(
        [[1,0],[0,1],[2,1],[1,2]],sp.eye(2),100,sp.Rational(1,5))
    assert reject["status"]=="UNKNOWN"
    try:
        certify_bridge([[1.0,0],[0,1]],sp.eye(2),0,sp.Rational(1,5))
        raise AssertionError("unbounded floating input accepted")
    except TypeError:
        pass
    try:
        certify_bridge([[1,0],[0,1],[1,2]],sp.diag(1,-1),
                       0,sp.Rational(1,5))
        raise AssertionError("non-SPD reference accepted")
    except ValueError:
        pass
    return {
      "status": "PASS",
      "seed": 20261014,
      "synthetic_exact_rational_matrices": 24,
      "nontrivial_bound_cases": verified,
      "sampled_float_logm_regressions_NOT_formal_proofs": sampled,
      "maximum_observed_to_proven_upper_ratio": ratio_max,
      "large_uncertainty_returns_UNKNOWN": True,
      "no_MNE_IEEE754_full_pipeline_certificate": True,
      "no_new_human_EEG_acquisition": True,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=regression()
    data=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(data)
    print("P4_CONDITIONAL_SAMPLE_TO_TANGENT_BRIDGE",json.dumps(result,sort_keys=True))
    print("SHA256",hashlib.sha256(data).hexdigest())
