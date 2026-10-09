"""Native MuJoCo counterexample: nominally compensating patches can diverge at actuation limits.

A locally incorrect repair can be output-equivalent under development inputs
and yet fail on frozen high-command transfer, even within the *same* real
MuJoCo plant. It is a source-bound mechanism assay, not robot fleet evidence,
not a learned VLA, and not a new controller design.

The fixed simulator uses motor ctrlrange[-10,10], physics dt 0.005s and 15
steps per command; no image rendering or GPU required.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

from .mujoco_native_authority import WORLDS, corrected, rollout, max_error
from .mujoco_wrong_repair_counterfactual import unauthorized_compensating_repair

ACTUATOR_CONTROL_LIMIT = 10.0
LOW_COMMANDS = (0.2, -0.7, 0.1, 0.75, -0.25, 0.0)
HIGH_COMMANDS = (13.0, -13.0, 8.0, -8.0, 12.0, -12.0)
STRESS_AMPLITUDES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                     4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 13.0, 15.0)
ERROR_MATCH_TOL = 2e-8
ERROR_DIVERGE_TOL = 1e-5


def effective_motor_force(*, producer_scale: float, gear: float,
                          command: float, ctrl_limit: float = ACTUATOR_CONTROL_LIMIT) -> float:
    """Closed-form saturated motor map; independently checked against MuJoCo."""
    if not all(math.isfinite(x) and x > 0 for x in
               (producer_scale, gear, ctrl_limit)):
        raise ValueError("positive finite parameters required")
    if not math.isfinite(command):
        raise ValueError("nonfinite command")
    return gear * max(-ctrl_limit, min(ctrl_limit, producer_scale * command))


def exact_equivalence_domain(s1: float, g1: float, s2: float, g2: float, limit: float) -> dict:
    """Certified symmetric nonsaturated equivalence interval for maps with s1*g1 == s2*g2.

    At |u| <= min(L/s1, L/s2), both control maps are linear and equal.
    For unequal s1 and s2, at least one map saturates earlier, so immediately
    beyond that boundary they differ until (possibly) other regime changes.
    The interval is a sound *lower bound* on equality, not a safety certificate.
    """
    if any(not math.isfinite(x) or x <= 0 for x in (s1, g1, s2, g2, limit)):
        raise ValueError("positive finite parameters required")
    if not math.isclose(s1*g1, s2*g2, abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("nominal effect scales not equal")
    bound = min(limit/s1, limit/s2)
    return {"guaranteed_equivalent_control_abs_le":bound,
            "different_software_scaling":not math.isclose(s1,s2),
            "assumptions":"scalar motor gear times clipped control, positive producer scales and gear; no downstream nonlinear plant guarantee"}


def probe_actual_model(world, commands) -> dict:
    ref = next(w for w in WORLDS if w.name == "reference")
    right = corrected(world)
    wrong = unauthorized_compensating_repair(world)
    ref_trace = rollout(ref, commands)
    true_trace = rollout(right, commands)
    wrong_trace = rollout(wrong, commands)
    external_right = max_error(ref_trace["external"], true_trace["external"])
    external_wrong = max_error(ref_trace["external"], wrong_trace["external"])
    motor_right = max_error(ref_trace["transmission_force"], true_trace["transmission_force"])
    motor_wrong = max_error(ref_trace["transmission_force"], wrong_trace["transmission_force"])
    domain = exact_equivalence_domain(right.software_scale, right.gear,
                                      wrong.software_scale, wrong.gear,
                                      ACTUATOR_CONTROL_LIMIT)
    return {"world":world.name, "commands":tuple(commands),
            "canonical_vs_authorized_external_error": external_right,
            "canonical_vs_wrong_patch_external_error": external_wrong,
            "canonical_vs_authorized_motor_force_error": motor_right,
            "canonical_vs_wrong_patch_motor_force_error": motor_wrong,
            "safe_nominal_equal_input_range":domain["guaranteed_equivalent_control_abs_le"],
            "in_range":max(map(abs,commands))<=domain["guaranteed_equivalent_control_abs_le"]}


def frozen_saturation_challenge() -> dict:
    """Predeclared low-amplitude vs stress transfer, all 3 original fault worlds."""
    fault_worlds = tuple(w for w in WORLDS if w.name!="reference")
    rows=[]
    for world in fault_worlds:
        low=probe_actual_model(world,LOW_COMMANDS)
        high=probe_actual_model(world,HIGH_COMMANDS)
        if low["canonical_vs_wrong_patch_external_error"] > ERROR_MATCH_TOL:
            raise AssertionError("development compensation no longer equivalent")
        if low["canonical_vs_authorized_external_error"] > ERROR_MATCH_TOL:
            raise AssertionError("approved patch fails development closure")
        if high["canonical_vs_authorized_external_error"] > ERROR_MATCH_TOL:
            raise AssertionError("approved patch fails transfer closure")
        if high["canonical_vs_wrong_patch_motor_force_error"] <= ERROR_DIVERGE_TOL:
            raise AssertionError("saturation mechanism does not break wrong-patch force equality")
        rows.append({"world":world.name,"low":low,"stress":high})
    amplitude_sweep=[]
    for world in fault_worlds:
        for amplitude in STRESS_AMPLITUDES:
            result=probe_actual_model(world,(amplitude,-amplitude))
            amplitude_sweep.append({
                "world":world.name, "magnitude":amplitude,
                "in_certified_equal_command_range":result["in_range"],
                "true_patch_force_error":result["canonical_vs_authorized_motor_force_error"],
                "wrong_patch_force_error":result["canonical_vs_wrong_patch_motor_force_error"],
                "true_patch_trajectory_error":result["canonical_vs_authorized_external_error"],
                "wrong_patch_trajectory_error":result["canonical_vs_wrong_patch_external_error"]})
    frozen_input={"development":LOW_COMMANDS,"stress":HIGH_COMMANDS,"sweep":STRESS_AMPLITUDES}
    return {
        "protocol":"source-bound-repair-saturation-transfer-mujoco-v1",
        "input_sha256":sha256(json.dumps(frozen_input,sort_keys=True).encode()).hexdigest(),
        "world_count":len(fault_worlds), "independent_robot_families":1,
        "development_amplitude_max":max(map(abs,LOW_COMMANDS)),
        "transfer_amplitude_max":max(map(abs,HIGH_COMMANDS)),
        "matched_development_wrong_patch_cells":sum(
            r["low"]["canonical_vs_wrong_patch_external_error"]<ERROR_MATCH_TOL for r in rows),
        "stress_wrong_patch_force_divergence_cells":sum(
            r["stress"]["canonical_vs_wrong_patch_motor_force_error"]>ERROR_DIVERGE_TOL for r in rows),
        "stress_wrong_patch_trajectory_divergence_cells":sum(
            r["stress"]["canonical_vs_wrong_patch_external_error"]>ERROR_DIVERGE_TOL for r in rows),
        "stress_true_patch_trajectory_closure_cells":sum(
            r["stress"]["canonical_vs_authorized_external_error"]<ERROR_MATCH_TOL for r in rows),
        "rows":rows, "amplitude_sweep":amplitude_sweep,
        "limits":"three engineered faults in one 1-DOF MuJoCo actuator; stress inputs exceed development support, no learned robot policy, no external adoption; offline wrong-patch replays only",
    }


if __name__=="__main__":
    print(json.dumps(frozen_saturation_challenge(), indent=2,sort_keys=True))
