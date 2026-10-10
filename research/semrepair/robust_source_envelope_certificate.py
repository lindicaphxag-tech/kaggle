"""Robust 2-channel safety contract for uncertain authorized source repairs.

Problem: A signed *nominal* config can be geometrically correct and yet
unsafe under legitimate measured actuator calibration drift. For a known
small elementwise coefficient error box |ΔP_ij|<=dP_ij and
|ΔC_ij|<=dC_ij, the repaired actual stage is

       v=(C+ΔC) clip((P+ΔP)u,[-L,L]^2).

For all u in [-R,R]^2, certify jointly

 (a) internal clip NEVER activates in the claimed command envelope, and
 (b) ||v-u||_infinity <= epsilon.

Sufficient worst-case inequalities (under the stated *exogenous* uncertainty
bounds) are:

  R*max_i Σ_j (|P_ij|+dP_ij) <= L,
  R*max_i Σ_j { |(CP-I)_ij|
                    + Σ_k [|C_ik| dP_kj
                           + dC_ik |P_kj|
                           + dC_ik dP_kj]} <= epsilon.

We deliberately use interval bounds rather than a Gaussian confidence score,
a learned safety head or an unverified risk estimator. This gives NO safety
guarantee if declared coefficient bounds are wrong, if the controller is
nonlinear/time varying, a source tap is forged, or there is unmodelled delay.

Two permitted outcomes: certified conditional on trustworthy bounds, or
REFUSE. No paper shall count REFUSE as false negative recovery failure.

This is algebraic interval arithmetic, NOT novel control theory. Research
novelty must come from actual authenticated, timed, successful closed-loop
fault repair and outside-simulator validation.
"""
from __future__ import annotations

from math import isfinite
from typing import Sequence

I=((1.0,0.0),(0.0,1.0))


def _mat2(value,name:str,*,nonnegative:bool=False):
    if not isinstance(value,(tuple,list)) or len(value)!=2:
        raise ValueError(f"{name}: matrix dimension must be 2x2")
    out=[]
    for row in value:
        if not isinstance(row,(tuple,list)) or len(row)!=2:
            raise ValueError(f"{name}: matrix dimension must be 2x2")
        items=[]
        for v in row:
            if type(v) not in (int,float) or not isfinite(v):
                raise ValueError(f"{name}: finite entries required")
            if nonnegative and v<0:
                raise ValueError(f"{name}: uncertainty bounds must be nonnegative")
            items.append(float(v))
        out.append(tuple(items))
    return tuple(out)


def _multiply(A,B):
    return tuple(tuple(sum(A[i][k]*B[k][j] for k in range(2))
                       for j in range(2)) for i in range(2))


def robust_repair_certificate(*, producer,controller,
                              producer_abs_error_bounds,
                              controller_abs_error_bounds,
                              full_command_radius:float,
                              internal_clip_limit:float,
                              allowed_max_output_error:float,
                              numeric_margin:float=1e-12)->dict:
    """Fail closed if the specified physical uncertainty ENVELOPE is invalid."""
    P=_mat2(producer,"producer")
    C=_mat2(controller,"controller")
    dP=_mat2(producer_abs_error_bounds,"producer_abs_error_bounds",nonnegative=True)
    dC=_mat2(controller_abs_error_bounds,"controller_abs_error_bounds",nonnegative=True)
    for name,x,strict in (
        ("full_command_radius",full_command_radius,True),
        ("internal_clip_limit",internal_clip_limit,True),
        ("allowed_max_output_error",allowed_max_output_error,False),
        ("numeric_margin",numeric_margin,False)
    ):
        if type(x) not in (int,float) or not isfinite(x) or \
           (strict and x<=0) or (not strict and x<0):
            raise ValueError(f"{name}: invalid contract bound")
    R=float(full_command_radius)
    clip=float(internal_clip_limit)
    epsilon=float(allowed_max_output_error)
    m=float(numeric_margin)

    # This worst-case input amplitude is exact for each uncertain row
    # under independently signed coefficient error intervals.
    max_amplitude=R*max(
        sum(abs(P[i][j])+dP[i][j] for j in range(2))
        for i in range(2))
    nominal=_multiply(C,P)
    error_matrix=tuple(tuple(
        abs(nominal[i][j]-I[i][j])+
        sum(abs(C[i][k])*dP[k][j]+
            dC[i][k]*abs(P[k][j])+
            dC[i][k]*dP[k][j]
            for k in range(2))
        for j in range(2)) for i in range(2))
    worst_output_error=R*max(
        sum(error_matrix[i][j] for j in range(2)) for i in range(2))
    no_clipping=max_amplitude+m<=clip
    acceptable_output=worst_output_error+m<=epsilon
    return {
        "decision":"CONDITIONAL_ROBUST_CERTIFIED" if no_clipping and acceptable_output
                   else "REFUSE_UNPROVEN_UNCERTAIN_INTERFACE_ENVELOPE",
        "certified":bool(no_clipping and acceptable_output),
        "robust_internal_clip_abs_input_bound":max_amplitude,
        "robust_infinity_output_error_bound":worst_output_error,
        "no_internal_clipping_for_any_allowed_source_drift":no_clipping,
        "all_command_errors_within_declared_tolerance":acceptable_output,
        "uncertainty_assumed_exogenous_and_valid":True,
        "certificate_is_conditional_not_external_measurement":True,
        "claim":"for all u in [-R,R]^2 and all independent coefficient perturbations within the declared elementwise intervals, provided the underlying source maps remain exactly linear and time invariant",
        "limitations":"does not authenticate bound provenance, does not cover hidden backlash, sensor corruption, model changes, delay or task success"
    }


def _main():
    import json
    certs={
        "exact_signed_identity":robust_repair_certificate(
            producer=I,controller=I,
            producer_abs_error_bounds=((0.,0.),(0.,0.)),
            controller_abs_error_bounds=((0.,0.),(0.,0.)),
            full_command_radius=2.,internal_clip_limit=2.001,
            allowed_max_output_error=1e-10),
        "minor_legitimate_coefficient_drift":robust_repair_certificate(
            producer=I,controller=I,
            producer_abs_error_bounds=((.001,.001),(.001,.001)),
            controller_abs_error_bounds=((.001,.001),(.001,.001)),
            full_command_radius=2.,internal_clip_limit=2.01,
            allowed_max_output_error=.01),
        "insufficient_headroom":robust_repair_certificate(
            producer=I,controller=I,
            producer_abs_error_bounds=((.001,.001),(.001,.001)),
            controller_abs_error_bounds=((.001,.001),(.001,.001)),
            full_command_radius=2.,internal_clip_limit=2.0,
            allowed_max_output_error=.01),
    }
    print(json.dumps(certs,sort_keys=True,indent=2))


if __name__=="__main__":
    _main()
