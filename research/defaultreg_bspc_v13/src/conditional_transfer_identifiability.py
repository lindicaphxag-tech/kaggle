"""A CONDITIONAL source-only dominance certificate, not a clinical guarantee.

Assume the unseen participant shares a fixed, known positive definite
within-class covariance Sigma with the source model and Gaussian class
conditional distributions. The unknown class-mean contrast satisfies the
declared ellipsoid ||Sigma^(-1/2)(d_target-d_anchor)|| <= rho.
Pairwise AUC ordering equals ordering of m(w,d)=w.d/sqrt(w.Sigma.w).
For a = w_D/||w_D||_Sigma - w_C/||w_C||_Sigma, the exact minimum
margin difference is a.d_anchor - rho*sqrt(a.Sigma.a).

The ellipsoid membership, Gaussian distribution and fixed covariance are
assumptions, NOT verified conclusions about any EEG dataset. The radius
cannot be calibrated to a future target using its features or labels under
strict Z0. Use only as mathematical sensitivity analysis / abstention.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ConditionalCertificate:
    nominal_margin_gap: float
    exact_worst_gap: float
    critical_radius: float | None
    certified_preference: bool
    worst_contrast: list[float]
    invariant_to_mean_shift: bool
    status: str = "CONDITIONAL_GAUSSIAN_MATH_NOT_EEG_SAFETY_OR_PREDICTION"


def conditional_certificate(w_d, w_c, d_anchor, sigma, radius):
    wd,wc,d,S = [np.asarray(x,dtype=float) for x in
                 (w_d,w_c,d_anchor,sigma)]
    rho=float(radius)
    n=d.size
    if n<2 or wd.shape!=(n,) or wc.shape!=(n,) or S.shape!=(n,n):
        raise ValueError("Mismatched directions, anchor or covariance shapes")
    if (not np.isfinite(wd).all() or not np.isfinite(wc).all() or
        not np.isfinite(d).all() or not np.isfinite(S).all() or
        not math.isfinite(rho) or rho<0 or
        not np.allclose(S,S.T,atol=1e-10,rtol=0)):
        raise ValueError("Invalid or nonfinite input")
    e=np.linalg.eigvalsh(S)
    if e.min() <= max(e.max(),1.0)*1e-12:
        raise ValueError("Shared covariance must be sufficiently SPD")
    def norm(w):
        q=float(w@S@w)
        if q<=max(np.linalg.norm(w)**2,1.)*1e-20:
            raise ValueError("Zero discriminant")
        return w/math.sqrt(q)
    a=norm(wd)-norm(wc)
    sensitivity=math.sqrt(max(0.,float(a@S@a)))
    nominal=float(a@d)
    worst=nominal-rho*sensitivity
    invariant=sensitivity<1e-12
    witness=d if invariant else d-rho*(S@a)/sensitivity
    return ConditionalCertificate(
        nominal_margin_gap=nominal,
        exact_worst_gap=worst,
        critical_radius=(float(nominal/sensitivity)
                         if not invariant and nominal>0 else None),
        certified_preference=bool(worst>1e-12),
        worst_contrast=witness.tolist(),
        invariant_to_mean_shift=invariant,
    )


def opposite_gaussian_targets(w_d,w_c,sigma):
    """Two valid target mean contrasts with reversed ranking, same Sigma."""
    S=np.asarray(sigma,dtype=float)
    wd,wc=np.asarray(w_d,dtype=float),np.asarray(w_c,dtype=float)
    if S.shape!=(len(wd),len(wd)) or wc.shape!=wd.shape:
        raise ValueError("Shapes differ")
    # Validation and the common-covariance assumption first.
    zero=np.zeros(len(wd))
    r=conditional_certificate(wd,wc,zero,S,0.)
    if r.invariant_to_mean_shift:
        return zero,zero
    a=wd/math.sqrt(wd@S@wd)-wc/math.sqrt(wc@S@wc)
    return a,-a
