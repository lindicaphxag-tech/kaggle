"""Assumption-aware ABI candidate validation against off-support nonlinear faults.

Three training/check taps do not certify an arbitrary nonlinear action map:
a polynomial can vanish at every queried point and diverge during deployment.
This module (i) verifies finite-sample compatibility with an EXACT deterministic
bounded-error envelope, and (ii) optionally certifies a specified rectangular
deployment domain given an EXTERNALLY VALIDATED Lipschitz bound on the
model residual and a checked full Cartesian grid of taps.

No finite probing without a function-class assumption can certify arbitrary
nonlinear maps. The assumed sensor readings are trusted here. No patch is
installed, signed or authorized by this code.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Sequence

from .observable_interface_factorization import (
    TapSample, PROBE_E1, PROBE_E2, CHECK, I, Mat, Vec, columns, inverse,
    mmul, mv, max_diff, is_identity, nominal_probe,
)

EXTENDED_PROBES: tuple[Vec, ...] = (
    (1.2, 0.), (-1.2, 0.), (0., 1.2), (0., -1.2),
)


def _max_abs_matrix_error(a: Mat, b: Mat) -> float:
    return max(abs(a[i][j]-b[i][j]) for i in range(2) for j in range(2))


def cubic_ambush(u: Vec, *, strength: float, axis: int=0) -> Vec:
    """Nonlinearity identically zero on all THREE frozen legacy probe inputs.

    x roots for 0,.31,.67; y roots for 0,-.27,-.53.
    For high-amplitude commands the cubic perturbation may invert direction.
    A new independent probe at x=+/-1.2 or y=+/-1.2 detects it.
    """
    if not isfinite(strength) or strength < 0 or axis not in (0,1):
        raise ValueError("invalid nonlinear fault")
    x,y=u
    bump=(-strength*x*(x-.31)*(x-.67) if axis==0
          else -strength*y*(y+.27)*(y+.53))
    return (x+bump,y) if axis==0 else (x,y+bump)


def nonlinear_tap(u: Vec, *, strength: float, axis: int=0) -> TapSample:
    """Evaluator delivers numerical taps only; model does not know oracle fault."""
    from .native_planar_task_repair import actual_mujoco_model
    import mujoco
    producer=cubic_ambush(u,strength=strength,axis=axis)
    model=actual_mujoco_model()
    data=mujoco.MjData(model)
    data.ctrl[:]=producer
    mujoco.mj_step(model,data)
    return TapSample(u,producer,(float(data.ctrl[0]),float(data.ctrl[1])))


def _infer_linear_tap_maps(training: Sequence[TapSample]) -> tuple[Mat,Mat,Mat]:
    if len(training)!=2:
        raise ValueError("requires exactly two independent training probes")
    a,b=training
    if any(not isfinite(v) for sample in training
           for v in (sample.requested+sample.producer+sample.controller)):
        raise ValueError("nonfinite training tap")
    U=columns(a.requested,b.requested)
    Uinv=inverse(U)
    P=mmul(columns(a.producer,b.producer),Uinv)
    Q=mmul(columns(a.controller,b.controller),Uinv)
    return P,Q,Uinv


def _one_reading_threshold(u: Vec, Uinv: Mat, eps: float)->float:
    # Each coordinate in BOTH training and validation taps may have bounded
    # additive error <=eps. Requested command vectors are EXACT.
    coeff=mv(Uinv,u)
    return eps*(1.+abs(coeff[0])+abs(coeff[1]))


def validate_numeric_model(training: Sequence[TapSample],
                           validation: Sequence[TapSample], *,
                           tap_noise_bound: float=0.,
                           extra_tolerance: float=1e-9) -> dict:
    """Deterministic finite-point compatibility with the linear *null model*.

    For any true linear P and Q=CP, with independent (not stochastic) tap
    errors |err|<=eps for each coordinate, the fitted predictions at u obey
    |p_hat(u)-p_obs(u)| <= eps*(1+||U^-1*u||_1).
    The same bound holds for controller-output Q fitted directly from taps.
    This does NOT prove the linear hypothesis true if all points pass.
    """
    if not isfinite(tap_noise_bound) or tap_noise_bound<0 or \
       not isfinite(extra_tolerance) or extra_tolerance<0:
        raise ValueError("invalid error budget")
    P,Q,Uinv=_infer_linear_tap_maps(training)
    records=[]
    for row in validation:
        if any(not isfinite(x) for x in row.requested+row.producer+row.controller):
            raise ValueError("nonfinite validation sample")
        threshold=_one_reading_threshold(row.requested,Uinv,tap_noise_bound)+extra_tolerance
        e_p=max_diff(mv(P,row.requested),row.producer)
        e_q=max_diff(mv(Q,row.requested),row.controller)
        records.append({"u":list(row.requested),"producer_residual":e_p,
                        "controller_residual":e_q,"allowed_residual":threshold,
                        "passes":max(e_p,e_q)<=threshold})
    fail=next((r for r in records if not r["passes"]),None)
    return {"decision":"REFUSE_LINEAR_MODEL_FALSIFIED" if fail else "FINITE_PROBES_COMPATIBLE_ONLY",
            "records":records,"falsifying_probe":fail,
            "producer_linear_fit":P,"controller_total_linear_fit":Q,
            "sensors_trusted_assumed":True,
            "global_validity_certified":False,
            "claim_scope":"necessary test for bounded-noise linear null, not sufficient for arbitrary nonlinearity"}


def requested_patch_if_finite_compatible(training: Sequence[TapSample],
                                          validation: Sequence[TapSample], *,
                                          tap_noise_bound:float=0.,
                                          model_identity_margin:float=1e-5) -> dict:
    """Candidate-only gate; refuses ambiguous near-identity configurations.

    For eps>0 classification may still require separate robust source
    provenance/uncertainty checks: this returns candidate, never AUTHORIZED.
    """
    verdict=validate_numeric_model(training,validation,tap_noise_bound=tap_noise_bound)
    if verdict["decision"]!="FINITE_PROBES_COMPATIBLE_ONLY":
        return {"decision":verdict["decision"],"candidate_patch":None,
                "model_validity":verdict}
    P,Q,Uinv=_infer_linear_tap_maps(training)
    try:
        C=mmul(Q,inverse(P))
    except ValueError:
        return {"decision":"REFUSE_UNIDENTIFIABLE_CONTROLLER","candidate_patch":None,
                "model_validity":verdict}
    # The margin is heuristic; there is no claim of a uniform noisy
    # classification guarantee (that needs an explicit singular-value bound).
    pe=is_identity(P,model_identity_margin)
    ce=is_identity(C,model_identity_margin)
    arm="noop" if pe and ce else "controller-only" if pe else \
        "producer-only" if ce else "atomic"
    return {"decision":"PATCH_CANDIDATE_ONLY_NOT_AUTHORIZED",
            "candidate_patch":arm,"model_validity":verdict,
            "no_uniform_noisy_patch_classifier_claimed":True}


def cartesian_cover_radius(
    samples: Sequence[Vec], *, lower:float, upper:float,
)->float:
    """Compute exact L-infinity covering radius of a FULL 2D Cartesian grid.

    Asymmetric domain is square [lower,upper]^2. A rectangular grid with
    missing cross-products is rejected, not guessed to cover the domain.
    """
    if not (isfinite(lower) and isfinite(upper) and lower<upper):
        raise ValueError("invalid deployment square")
    if not samples or any(len(p)!=2 or any(not isfinite(v) for v in p) for p in samples):
        raise ValueError("invalid Cartesian grid")
    grid=set(samples)
    xs=sorted({x for x,y in grid})
    ys=sorted({y for x,y in grid})
    if len(grid)!=len(xs)*len(ys) or any((x,y) not in grid for x in xs for y in ys):
        raise ValueError("incomplete Cartesian grid")
    if any(x<lower or x>upper for x in xs+ys):
        raise ValueError("grid outside the declared deployment domain")
    def radius(vs):
        gaps=[(b-a)/2 for a,b in zip(vs,vs[1:])]
        return max([vs[0]-lower,upper-vs[-1]]+gaps)
    return max(radius(xs),radius(ys))


def conditional_lipschitz_envelope(
    residuals: Sequence[float], samples: Sequence[Vec], *,
    lower:float, upper:float, externally_verified_lipschitz:float|None,
    tolerance:float,
)->dict:
    """Sound conditional bound if an EXTERNAL source proves L-Lipschitz.

    For r(u) max-norm model residual with Lipschitz L under input L-infinity,
    and a complete Cartesian epsilon-net, sup_domain r <= max_grid r + L*rho.
    This is a standard covering argument, not estimated from observed probes.
    """
    rho=cartesian_cover_radius(samples,lower=lower,upper=upper)
    if len(residuals)!=len(samples) or any(not isfinite(v) or v<0 for v in residuals):
        raise ValueError("invalid residual list")
    if not isfinite(tolerance) or tolerance<0:
        raise ValueError("invalid required tolerance")
    if externally_verified_lipschitz is None:
        return {"decision":"REFUSE_NO_EXTERNAL_REGULARITY_BOUND",
                "covered_radius":rho,"certified":False}
    if not isfinite(externally_verified_lipschitz) or externally_verified_lipschitz<0:
        raise ValueError("invalid verified Lipschitz constant")
    upper_residual=max(residuals)+externally_verified_lipschitz*rho
    return {"decision":("CONDITIONALLY_WITHIN_ENVELOPE" if upper_residual<=tolerance
                        else "REFUSE_INSUFFICIENT_ENVELOPE_BOUND"),
            "upper_residual":upper_residual,"covered_radius":rho,
            "certified":upper_residual<=tolerance,
            "conditional_assumption":"EXTERNALLY VALIDATED Lipschitz bound for true residual on declared domain"}


def nonlinear_probe_challenge() -> dict:
    """A falsifier with actual MuJoCo internal-force taps, not a training success report."""
    comparisons=[]
    for axis in (0,1):
        for a in (0.0,.01,.025,.05,.1,.25,.5):
            training=[nonlinear_tap(u,strength=a,axis=axis) for u in (PROBE_E1,PROBE_E2)]
            legacy=validate_numeric_model(training,[nonlinear_tap(CHECK,strength=a,axis=axis)])
            stress=validate_numeric_model(training,
                [nonlinear_tap(u,strength=a,axis=axis)
                 for u in (CHECK,)+EXTENDED_PROBES])
            comparisons.append({"axis":axis,"strength":a,
                                "legacy":legacy["decision"],
                                "deployment_probe":stress["decision"],
                                "max_stress_residual":max(max(
                                    x["producer_residual"],x["controller_residual"])
                                    for x in stress["records"])})
    altered=[x for x in comparisons if x["strength"]>0]
    assert len(altered)==12
    assert all(x["legacy"]=="FINITE_PROBES_COMPATIBLE_ONLY" for x in altered)
    assert all(x["deployment_probe"]=="REFUSE_LINEAR_MODEL_FALSIFIED" for x in altered)
    return {"protocol":"semrepair-native-mujoco-out-of-support-nonlinear-falsification-v1",
            "independent_manufacturer_families":0,
            "constructed_fault_families":2,
            "dependent_parameter_cases":len(altered),
            "legacy_false_linear_acceptances":sum(
                x["legacy"]=="FINITE_PROBES_COMPATIBLE_ONLY" for x in altered),
            "extra_probe_detected_nonlinear":sum(
                x["deployment_probe"]=="REFUSE_LINEAR_MODEL_FALSIFIED" for x in altered),
            "rows":comparisons,"nominal_probe_count":3,"extended_probe_count":7,
            "claim_scope":"adversarially selected cubic polynomial roots deliberately pass the old taps; "
                          "greater probe cost, bounded crafted family, no independent generalization or true source authentication"}
