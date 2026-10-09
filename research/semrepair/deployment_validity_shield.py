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


def finite_probe_interpolation_counterexample(
    probe_inputs: Sequence[Vec], *, evaluation: Vec=(2.,.4),
    coefficient: float=.1,
) -> dict:
    """Construct a nonlinear interface *identical* at EVERY selected probe.

    For any finite set of commands, let roots be its unique x coordinates and
    perturb producer x by coefficient * product(x-root). This polynomial is zero
    at every observed input. Off-support divergence requires an evaluation x
    coordinate absent from those roots. Therefore finite taps alone can NEVER
    prove an arbitrary nonlinear ABI globally correct. Elementary polynomial
    interpolation, not a new theorem.
    """
    if not probe_inputs or not isfinite(coefficient) or coefficient==0 or \
       any(len(u)!=2 or not all(isfinite(z) for z in u) for u in probe_inputs):
        raise ValueError("invalid finite nonempty probe list")
    if len(evaluation)!=2 or not all(isfinite(z) for z in evaluation):
        raise ValueError("invalid held-out command")
    roots=sorted({u[0] for u in probe_inputs})
    if evaluation[0] in roots:
        raise ValueError("evaluation x coordinate must be unqueried")
    def phi(u:Vec)->Vec:
        product=coefficient
        for root in roots:
            product*=u[0]-root
        return (u[0]+product,u[1])
    max_training=max(max_diff(phi(u),u) for u in probe_inputs)
    difference=max_diff(phi(evaluation),evaluation)
    if max_training>1e-10 or difference<=0:
        raise AssertionError("constructive alias or divergence invalid")
    return {
        "distinct_x_roots":roots,"probed_commands":len(probe_inputs),
        "maximum_training_discrepancy":max_training,
        "unqueried_command":list(evaluation),
        "unqueried_discrepancy":difference,
        "lesson":"any fixed finite tap set may be fooled by an unconstrained nonlinear source",
        "novel_mathematical_priority":False,
        "claim_scope":"symbolic finite-probe impossibility under unconstrained function class",
    }


def _interval_max_norm(center:Mat, target:Mat, radii_by_column:Vec) -> tuple[float,float]:
    """Rigorous lower/upper bounds on max_ij |true_ij-target_ij|."""
    lower=0.
    upper=0.
    for i in range(2):
        for j in range(2):
            deviation=abs(center[i][j]-target[i][j])
            rad=radii_by_column[j]
            lower=max(lower,max(0.,deviation-rad))
            upper=max(upper,deviation+rad)
    return lower,upper


def _certified_invertible_2x2(P:Mat, rad:Vec)->dict:
    """Bound determinant error using independent component interval radii.

    For each matrix column j, |Delta_ij|<=rad[j]. For det=ad-bc
    the conservative error budget is sum of first-order terms plus two
    cross-products. If det_hat beats this, every compatible matrix is
    invertible (but no condition-number guarantee is claimed).
    """
    a,b=P[0];c,d=P[1];r0,r1=rad
    estimate=a*d-b*c
    uncertainty=(abs(a)+abs(c))*r1+(abs(b)+abs(d))*r0+2*r0*r1
    return {"estimate":estimate,"maximum_determinant_error":uncertainty,
            "certified":abs(estimate)>uncertainty}


def bounded_noise_patch_contract(
    training: Sequence[TapSample],validation: Sequence[TapSample], *,
    per_coordinate_error_bound:float, declared_effect_tolerance:float=0.025,
) -> dict:
    """Conditional classification of the *observable* two-interface map.

    Unlike heuristic exact-matrix classification, this certifies whether
    P-I and Q-P have max-entry norm <=tau or >tau for EVERY admissible
    linear model consistent with the noisy training readings.

    Q maps upstream requests to controller output. Q-P = (C-I)P.
    The downstream 'effect mismatch' metric is NOT ||C-I|| and is meaningful
    as an identity test only when P is invertible, which we separately check.
    This is NOT a source-bound legal permission to patch code.
    """
    if (not isfinite(per_coordinate_error_bound) or per_coordinate_error_bound<0
        or not isfinite(declared_effect_tolerance) or declared_effect_tolerance<=0):
        raise ValueError("invalid error or map margin")
    check=validate_numeric_model(
        training,validation,tap_noise_bound=per_coordinate_error_bound)
    if check["decision"]!="FINITE_PROBES_COMPATIBLE_ONLY":
        return {"decision":check["decision"],"candidate_patch":None}
    P,Q,Uinv=_infer_linear_tap_maps(training)
    eps=per_coordinate_error_bound
    # Since each 2-vector column measurement error is bounded coordinatewise
    # by eps, the columns of E@U^-1 have scalar radius:
    column_radii=tuple(eps*(abs(Uinv[0][j])+abs(Uinv[1][j]))
                       for j in range(2))
    invertibility=_certified_invertible_2x2(P,column_radii)
    if not invertibility["certified"]:
        return {"decision":"REFUSE_NOISY_RANK_AMBIGUITY","candidate_patch":None,
                "producer_determinant_bound":invertibility}
    producer_interval=_interval_max_norm(P,I,column_radii)
    D=tuple(tuple(Q[i][j]-P[i][j] for j in range(2)) for i in range(2))
    downstream_interval=_interval_max_norm(
        D,((0.,0.),(0.,0.)),
        tuple(2*r for r in column_radii))
    def classify(interval):
        low,high=interval
        return "HEALTHY" if high <= declared_effect_tolerance else \
               "FAULT" if low > declared_effect_tolerance else "UNKNOWN"
    upstream=classify(producer_interval)
    downstream=classify(downstream_interval)
    if "UNKNOWN" in (upstream,downstream):
        return {"decision":"REFUSE_INSUFFICIENT_PRECISION","candidate_patch":None,
                "producer_contract":upstream,"downstream_effect_contract":downstream,
                "producer_interval":producer_interval,
                "downstream_effect_interval":downstream_interval}
    patch=("noop" if upstream=="HEALTHY" and downstream=="HEALTHY" else
           "producer-only" if upstream=="FAULT" and downstream=="HEALTHY" else
           "controller-only" if upstream=="HEALTHY" else "atomic")
    return {
        "decision":"NOISY_LINEAR_PATCH_CANDIDATE_NOT_AUTHORIZED",
        "candidate_patch":patch,"producer_contract":upstream,
        "downstream_effect_contract":downstream,
        "producer_interval":producer_interval,
        "downstream_effect_interval":downstream_interval,
        "producer_determinant_bound":invertibility,
        "effect_tolerance":declared_effect_tolerance,
        "error_bound":per_coordinate_error_bound,
        "global_nonlinear_validity_certified":False,
        "independent_source_verified":False,
        "proof_scope":"deterministic coordinatewise tap noise and assumed globally linear ABI; "
                      "every admissible map agrees on the two observable tolerance decisions",
    }


def matched_seven_probe_baseline_challenge() -> dict:
    """Fair readout-count control: predeclared random-style vs extremal probes.

    Same two fitting taps, same fixed old holdout and four extra actual native
    simulator taps for BOTH methods. Their performance may tie. It is invalid
    to call the seven-versus-three comparison an algorithmic gain at fixed cost.
    """
    RANDOM_FOUR: tuple[Vec,...] = (
        (-.43,.78),(.81,-.91),(-1.03,.22),(.16,1.06),
    )
    n=0
    counts={"extremal_7":0,"generic_7":0,"historical_3":0}
    for axis in (0,1):
        for strength in (.01,.025,.05,.1,.25,.5):
            n+=1
            fit=[nonlinear_tap(u,strength=strength,axis=axis)
                 for u in (PROBE_E1,PROBE_E2)]
            for name,extra in (("historical_3",()),("extremal_7",EXTENDED_PROBES),
                               ("generic_7",RANDOM_FOUR)):
                validation=[nonlinear_tap(u,strength=strength,axis=axis)
                            for u in (CHECK,)+extra]
                outcome=validate_numeric_model(fit,validation)
                counts[name]+=outcome["decision"]=="REFUSE_LINEAR_MODEL_FALSIFIED"
    assert counts["historical_3"]==0
    return {
        "constructed_nonlinear_parameter_cases":n,
        "detection_counts":counts,
        "probe_budgets":{"historical_3":3,"extremal_7":7,"generic_7":7},
        "generic_probe_coordinates":[list(v) for v in RANDOM_FOUR],
        "strong_baseline_scope":"same real MuJoCo numeric readouts, exactly same seven probes total",
        "no_algorithmic_superiority_claim":"extremal against generic equal-budget comparator has to be tested, never assumed",
        "no_independent_test_family":True,
    }


CORNER_PROBES: tuple[Vec,...] = (
    (1.2,1.2),(-1.2,-1.2),(1.2,-1.2),(-1.2,1.2),
)


def cross_axis_alias_tap(u:Vec, *, coefficient:float)->TapSample:
    """Native MuJoCo readout of a two-axis interaction invisible on axis probes.

    Each legacy or axis-only probe satisfies either x=0, y=0, x=.31
    or y=-.27, causing the polynomial cross-interaction to vanish.
    Corner probes have BOTH axes nonzero and break that alias.
    """
    if not isfinite(coefficient) or coefficient<0:
        raise ValueError("invalid interaction strength")
    import mujoco
    from .native_planar_task_repair import actual_mujoco_model
    x,y=u
    p=(x+coefficient*x*y*(x-.31)*(y+.27),y)
    model=actual_mujoco_model()
    data=mujoco.MjData(model)
    data.ctrl[:]=p
    mujoco.mj_step(model,data)
    return TapSample(u,p,(float(data.ctrl[0]),float(data.ctrl[1])))


def cross_axis_equal_cost_falsification()->dict:
    """A SECOND fault family attacks the previously added axial probe layout.

    Demonstrates the limitation *before* upgrading probe coverage; it does
    not imply optimized active probing against independent physical sources.
    """
    generic=((- .43,.78),(.81,-.91),(-1.03,.22),(.16,1.06))
    results=[]
    for strength in (.01,.025,.05,.1,.25,.5):
        train=[cross_axis_alias_tap(u,coefficient=strength)
               for u in (PROBE_E1,PROBE_E2)]
        row={"strength":strength}
        for label,extra in (("three",()),("axial_seven",EXTENDED_PROBES),
                            ("generic_seven",generic),("corner_seven",CORNER_PROBES)):
            validation=[cross_axis_alias_tap(u,coefficient=strength)
                        for u in (CHECK,)+extra]
            row[label]=validate_numeric_model(train,validation)["decision"]
        results.append(row)
    detected={name:sum(row[name]=="REFUSE_LINEAR_MODEL_FALSIFIED" for row in results)
              for name in ("three","axial_seven","generic_seven","corner_seven")}
    assert detected=={"three":0,"axial_seven":0,"generic_seven":6,"corner_seven":6}
    return {"protocol":"cross-axis-off-support-nonlinear-interface-first-score",
            "constructed_conditions":len(results),
            "detection_counts":detected,
            "costs":{"three":3,"axial_seven":7,"generic_seven":7,"corner_seven":7},
            "rows":results,
            "limitation":"author-generated polynomial vanishes at all original and axis-only probes, "
                         "one MuJoCo 2DOF plant and no learned policy; corner probes TIE a generic same-budget baseline",
    }
