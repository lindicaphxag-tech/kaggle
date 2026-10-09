"""Open-catalog SemRepair assay: actual MuJoCo dynamics and source identity.

Contrast finite-known-world authorization, unknown actuator dynamics, and
observationally identical implementation packages with different authority.

The source manifest is supplied as an *externally trusted condition*, not
inferred from sensor values, and there is NO cryptographic attestation
implementation here. An unknown vendor may always copy a known package ID
if that condition is not independently enforced. No robot-task success claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Callable
import json

import mujoco

from .mujoco_native_authority import (
    World, WORLDS, CONTROL_HOLD_STEPS, MEASUREMENT_HALF_WIDTH, PRECISION,
    rollout, signature, max_error,
)
from .native_measured_blind_replay import HELD_EXCITATIONS, CHANNELS

TOLERANCE = 2 * MEASUREMENT_HALF_WIDTH


@dataclass(frozen=True)
class Unseen:
    id: str
    actuator_family: str
    source_package: str
    physical: World | None
    position_kp: float | None = None


UNSEEN = (
    Unseen(
        "foreign-identical-gain", "motor", "other-vendor/repackaged-controller",
        World("foreign-identical-gain", 1., 2., 1., .2,
              "unknown-vendor:repair-requires-review"),
    ),
    Unseen(
        "foreign-mass-reparameterized", "motor", "new-actuator/inertia-gear",
        World("foreign-mass-reparameterized", 1., 2.5, 1.25, .25,
              "unknown-build:gear:2.5-to-1.25"),
    ),
    Unseen(
        "foreign-position-kp3", "position", "vendor-c/position-servo-kp3",
        None, 3.,
    ),
    Unseen(
        "foreign-position-kp12", "position", "vendor-c/position-servo-kp12",
        None, 12.,
    ),
)


def _source_digest(package: str) -> str:
    return sha256(("source-build-v1\n" + package).encode()).hexdigest()


def known_source_digest(world: World) -> str:
    return _source_digest("catalog/"+world.name)


def _position_rollout(commands: tuple[float,...], kp: float) -> dict:
    if kp <= 0:
        raise ValueError("position gain must be positive")
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco model="unseen_position_actuator">
      <option timestep="0.005" gravity="0 0 0" integrator="Euler"/>
      <worldbody><body name="slider">
        <joint name="slide" type="slide" axis="1 0 0" damping="0.2"
               limited="true" range="-4 4"/>
        <geom name="mass" type="sphere" size="0.05" mass="1.0"/>
      </body></worldbody>
      <actuator><position name="servo" joint="slide" kp="{kp}"
                   ctrllimited="true" ctrlrange="-10 10"/></actuator>
    </mujoco>""")
    data = mujoco.MjData(model)
    external, controller, force = [], [], []
    for command in commands:
        for _ in range(CONTROL_HOLD_STEPS):
            data.ctrl[0] = command
            mujoco.mj_step(model, data)
            external.append((float(data.qpos[0]),float(data.qvel[0])))
            controller.append(float(data.ctrl[0]))
            force.append(float(data.qfrc_actuator[0]))
    return {
        "external":tuple(external),
        "producer_to_controller_ctrl":tuple(controller),
        "transmission_force":tuple(force),
        "mujoco":mujoco.__version__,
    }


def unseen_rollout(case: Unseen, commands: tuple[float,...]) -> dict:
    if case.actuator_family=="position":
        return _position_rollout(commands,case.position_kp)
    if case.actuator_family=="motor" and case.physical is not None:
        return rollout(case.physical,commands)
    raise ValueError("unrecognized simulator family")


def decide_from_sensor_traces(
    *, measured: dict, templates: dict[str,dict],
    source_digest: str | None=None,
    enforce_source_provenance: bool=False
) -> dict:
    """No input here identifies true physical world or correct repair.

    A fixed catalog signature filter can reject observations outside every
    candidate, but cannot possibly distinguish in-catalog from out-of-catalog
    systems that have exactly identical observations. Source provenance can
    change this only if a trusted external verifier really supplied it.
    """
    hypotheses={w.name for w in WORLDS}
    authority={w.name:w.authority for w in WORLDS}
    calls=[]
    if enforce_source_provenance:
        if source_digest is None:
            return {"answer":"REFUSE","reason":"no_verified_source", "taps":calls}
        hypotheses={w for w in hypotheses
                    if known_source_digest(next(k for k in WORLDS if k.name==w))
                    == source_digest}
        if not hypotheses:
            return {"answer":"REFUSE","reason":"source_not_in_catalog",
                    "taps":calls}
    for channel,_,_ in CHANNELS:
        if len({authority[w] for w in hypotheses})==1:
            break
        reference=signature(measured[channel])
        hypotheses={w for w in hypotheses
                    if max_error(signature(templates[w][channel]),reference)
                    <= TOLERANCE}
        calls.append(channel)
        if not hypotheses:
            return {"answer":"REFUSE","reason":"out_of_model_observation",
                    "taps":calls}
    possible={authority[w] for w in hypotheses}
    if len(possible)!=1:
        return {"answer":"REFUSE","reason":"unidentifiable_authority",
                "taps":calls}
    return {"answer":next(iter(possible)),"reason":"finite_catalog_match",
            "taps":calls}


def study() -> dict:
    rows=[]
    for wave,commands in enumerate(HELD_EXCITATIONS):
        catalog={w.name:rollout(w,commands) for w in WORLDS}
        for w in WORLDS:
            observation=catalog[w.name]
            baseline=decide_from_sensor_traces(measured=observation,templates=catalog)
            sourced=decide_from_sensor_traces(
                measured=observation,templates=catalog,
                source_digest=known_source_digest(w),
                enforce_source_provenance=True)
            if baseline["answer"]!=w.authority or sourced["answer"]!=w.authority:
                raise AssertionError("existing legitimate finite catalog regressed")
        for case in UNSEEN:
            observation=unseen_rollout(case,commands)
            no_source=decide_from_sensor_traces(
                measured=observation,templates=catalog)
            verified=decide_from_sensor_traces(
                measured=observation,templates=catalog,
                source_digest=_source_digest(case.source_package),
                enforce_source_provenance=True)
            # All foreign source packages MUST be rejected if provenance is
            # externally verified and no matching hash is on the allowlist.
            if verified["answer"]!="REFUSE":
                raise AssertionError("source whitelist admitted foreign source")
            rows.append({
                "source_id":case.id,
                "wave":wave,
                "actuator_family":case.actuator_family,
                "without_source_attestation":no_source,
                "with_assumed_trusted_manifest":verified,
                "physically_stepped":True,
                "source_attestation_is_external_assumption":True,
            })
    indistinguishable=[
        r for r in rows
        if r["source_id"]=="foreign-identical-gain"
    ]
    if any(r["without_source_attestation"]["answer"]!="gear:2-to-1"
           for r in indistinguishable):
        raise AssertionError("identical-physics witness was not truly aliased")
    return {
        "protocol":"open-catalog-native-mujoco-source-semantic-gap-v1",
        "native_mujoco_version":mujoco.__version__,
        "catalog_worlds":len(WORLDS),
        "new_world_families":sorted({x.actuator_family for x in UNSEEN}),
        "previously_unused_worlds":len(UNSEEN),
        "excitation_sequences":len(HELD_EXCITATIONS),
        "open_catalog_condition_cells":len(rows),
        "without_source_assumption_authorized_unknown":sum(
            r["without_source_attestation"]["answer"]!="REFUSE" for r in rows),
        "without_source_assumption_rejected_unknown":sum(
            r["without_source_attestation"]["answer"]=="REFUSE" for r in rows),
        "with_trusted_source_rejected_unknown":sum(
            r["with_assumed_trusted_manifest"]["answer"]=="REFUSE" for r in rows),
        "identical_physics_different_repair_wrong_without_source":len(indistinguishable),
        "cases":rows,
        "proof_scope":"sensor-equivalent worlds need different repair: no sensor-only method can always authorize correctly and retain coverage",
        "critical_limitation":"trusted package digest is assumed and provided by oracle, not implemented authentication; 4 author-built variants/3 dependent waveforms, not real new robot fleet or task recovery",
    }


if __name__=="__main__":
    print(json.dumps(study(),indent=2,sort_keys=True))
