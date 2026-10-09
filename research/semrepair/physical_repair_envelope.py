"""Exact static actuator envelope verification for two counterfactual repairs.

A proof about an actuator map is NOT a proof about actual end-effector safety.
Assumptions: one translational DOF, positive motor gear, positive software scale,
symmetric hard control clipping, positive joint mass, identical normalized
damping; acceleration is (gear*clip(scale*u) - damping*v)/mass.

The difference of these actuator maps is piecewise linear in u. Its absolute
maximum on [-U,U] is attained at an endpoint or saturation breakpoint. We
exhaustively evaluate that FINITE set; no arbitrary dense sampling tolerance.
The separate source binding remains mandatory for permission to patch code.
"""
from __future__ import annotations

from math import isfinite
from typing import NamedTuple


class MotorMap(NamedTuple):
    software_scale: float
    gear: float
    mass: float
    damping: float
    control_limit: float


def _validate(m: MotorMap) -> None:
    if any(not isfinite(x) or x <= 0 for x in
           (m.software_scale, m.gear, m.mass, m.damping, m.control_limit)):
        raise ValueError("positive finite actuator and plant parameters required")


def normalized_drive(m: MotorMap, command: float) -> float:
    _validate(m)
    if not isfinite(command):
        raise ValueError("nonfinite command")
    clipped=max(-m.control_limit,min(m.control_limit,m.software_scale*command))
    return m.gear*clipped/m.mass


def exact_control_envelope_gap(reference: MotorMap, candidate: MotorMap,
                               maximum_abs_command: float) -> dict:
    """The exact sup_u |drive_ref(u)-drive_candidate(u)|, |u|<=U.

    Requires equal damping / mass so the velocity term cancels. For any U,
    all possible piecewise-linear kinks are exhausted analytically.
    """
    _validate(reference);_validate(candidate)
    if not isfinite(maximum_abs_command) or maximum_abs_command < 0:
        raise ValueError("nonnegative finite requested envelope")
    if not abs(reference.damping/reference.mass-candidate.damping/candidate.mass) < 1e-12:
        raise ValueError("different normalized damping requires velocity envelope")
    cap_ref=reference.control_limit/reference.software_scale
    cap_alt=candidate.control_limit/candidate.software_scale
    positive={0.0,float(maximum_abs_command)}
    positive.update(x for x in (cap_ref,cap_alt) if x < maximum_abs_command)
    points=sorted(set(positive)|{-x for x in positive})
    effects=[(abs(normalized_drive(reference,u)-normalized_drive(candidate,u)),u)
             for u in points]
    maximum,witness=max(effects,key=lambda item:(item[0],-abs(item[1])))
    return {"maximum_normalized_drive_error":maximum,
            "witness_command":witness,
            "tested_all_piecewise_linear_breakpoints":points,
            "max_abs_command":maximum_abs_command,
            "claim_scope":"exact static normalized motor map plus equal damping/mass; no end-to-end hardware safety"}


def source_and_envelope_authority(*, trusted_source_patch: str | None,
                                  proposed_patch: str, reference: MotorMap,
                                  proposed_motor: MotorMap,
                                  maximum_abs_command: float,
                                  allowed_normalized_drive_error: float=1e-10) -> dict:
    """Executable fail-closed joint guard, does NOT itself install a code patch."""
    if not isinstance(proposed_patch,str) or not proposed_patch:
        raise ValueError("nonempty patch identity required")
    if not isfinite(allowed_normalized_drive_error) or allowed_normalized_drive_error<0:
        raise ValueError("invalid drive error budget")
    result=exact_control_envelope_gap(reference,proposed_motor,maximum_abs_command)
    if trusted_source_patch is None:
        decision="REFUSE_UNVERIFIED_SOURCE"
    elif trusted_source_patch != proposed_patch:
        decision="REFUSE_WRONG_SOURCE_PATCH"
    elif result["maximum_normalized_drive_error"] > allowed_normalized_drive_error:
        decision="REFUSE_OUTSIDE_REPAIR_ENVELOPE"
    else:
        decision="PERMIT_CONDITIONAL_PHYSICAL_AND_SOURCE_GATE"
    return {"decision":decision,"patch_id":proposed_patch,
            "deployment_control_envelope":maximum_abs_command,
            "drive_error_bound":result["maximum_normalized_drive_error"],
            "witness":result["witness_command"],
            "not_a_robot_safety_or_patch_execution_certificate":True}
