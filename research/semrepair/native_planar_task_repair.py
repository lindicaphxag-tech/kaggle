"""Actual native MuJoCo 2-DOF closed-loop tracking under compensating ABI faults.

This is a *frozen classical PD feedback controller*, NOT learned VLA policy.
A two-axis slider/point-mass plant is simulated in MuJoCo; on each control
step the same PD command is transformed by a faulty producer then faulty
controller interface. Singleton patches and atomic patches modify those
concrete separate mappings. The scenario construction is first-party, with
one hand-specified task family and 16 deterministic source targets.

Claims are limited to simulator execution and source-bound patch interaction,
not unseen robot models, learned-policy improvement or hardware safety.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
from typing import NamedTuple

import mujoco

Matrix = tuple[tuple[int,int],tuple[int,int]]
IDENTITY: Matrix=((1,0),(0,1))
SWAP: Matrix=((0,1),(1,0))
NEGATE: Matrix=((-1,0),(0,-1))
ROT_POS: Matrix=((0,-1),(1,0))
ROT_NEG: Matrix=((0,1),(-1,0))

CONTROLLER_KP=12.0
CONTROLLER_KD=4.0
CONTROL_MAX=5.0
DT=0.01
STEPS=400
POSITION_TOL=0.10
SEED=20261010
N_TARGETS=16
CASES=(
    # A single incorrect software boundary should be repaired at that owner.
    ("producer-swap-only", SWAP, IDENTITY, "producer-only"),
    ("controller-sign-only", IDENTITY, NEGATE, "controller-only"),
    # Each pair composes to physical identity and is *not* safe to half-fix.
    ("compensated-swap", SWAP, SWAP, "atomic"),
    ("compensated-rotation", ROT_POS, ROT_NEG, "atomic"),
)
ARMS=("untouched","producer-only","controller-only","atomic","source-owner-oracle")


def matvec(m:Matrix, vector:tuple[float,float]) -> tuple[float,float]:
    return (m[0][0]*vector[0]+m[0][1]*vector[1],
            m[1][0]*vector[0]+m[1][1]*vector[1])


def effective_abi(producer:Matrix, controller:Matrix, repair:str):
    if repair in ("producer-only","atomic"):
        producer=IDENTITY
    if repair in ("controller-only","atomic"):
        controller=IDENTITY
    return producer,controller


def actual_mujoco_model():
    # The two translational DOFs belong to one rigid planar slider body;
    # dynamics, actuator clipping and joint limits are run by native MuJoCo.
    xml=f"""
    <mujoco model="semrepair_2d_pointmass_interface">
      <option timestep="{DT}" gravity="0 0 0" integrator="Euler"/>
      <worldbody>
        <body name="point" pos="0 0 0">
          <joint name="x" type="slide" axis="1 0 0" damping="0.25"
                 limited="true" range="-2 2"/>
          <joint name="y" type="slide" axis="0 1 0" damping="0.25"
                 limited="true" range="-2 2"/>
          <geom name="payload" type="sphere" size="0.04" mass="1.0"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="drive_x" joint="x" gear="1" ctrllimited="true"
               ctrlrange="-5 5"/>
        <motor name="drive_y" joint="y" gear="1" ctrllimited="true"
               ctrlrange="-5 5"/>
      </actuator>
    </mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def frozen_targets() -> tuple[tuple[float,float],...]:
    rng=random.Random(SEED)
    targets=[]
    for _ in range(N_TARGETS):
        # Prevent trivially already-solved targets and exact x/y symmetry.
        x=round(rng.uniform(-0.95,0.95),6)
        y=round(rng.uniform(-0.95,0.95),6)
        if abs(x)<0.25:x=0.32 if x>=0 else -0.32
        if abs(y)<0.25:y=-0.37 if y>=0 else 0.37
        targets.append((x,y))
    return tuple(targets)


def simulate(model, target:tuple[float,float],producer:Matrix,controller:Matrix,repair:str)->dict:
    data=mujoco.MjData(model)
    eff_p,eff_c=effective_abi(producer,controller,repair)
    total_drive=0.
    for _ in range(STEPS):
        state=(float(data.qpos[0]),float(data.qpos[1]))
        vel=(float(data.qvel[0]),float(data.qvel[1]))
        policy_cmd=tuple(max(-CONTROL_MAX,min(CONTROL_MAX,
                     CONTROLLER_KP*(target[i]-state[i])-CONTROLLER_KD*vel[i]))
                     for i in range(2))
        actual=matvec(eff_c,matvec(eff_p,policy_cmd))
        data.ctrl[:]=actual
        total_drive+=sum(abs(a) for a in actual)
        mujoco.mj_step(model,data)
    terminal=(float(data.qpos[0]),float(data.qpos[1]))
    distance=math.dist(target,terminal)
    return {"target":list(target),"end":list(terminal),
            "distance":distance,"success":distance<POSITION_TOL,
            "command_l1":total_drive,
            "steps":STEPS}


def run_planar_task_assay()->dict:
    model=actual_mujoco_model()
    goals=frozen_targets()
    records=[]
    for target_idx,target in enumerate(goals):
        reference=simulate(model,target,IDENTITY,IDENTITY,"untouched")
        records.append({"case":"reference","target_idx":target_idx,
                        "arm":"untouched","result":reference,
                        "reference_terminal_error":0.0})
        for name,producer,controller,authority in CASES:
            for arm in ARMS:
                patch=(authority if arm=="source-owner-oracle" else arm)
                outcome=simulate(model,target,producer,controller,patch)
                records.append({"case":name,"target_idx":target_idx,"arm":arm,
                    "executed_patch":patch,"result":outcome,
                    "reference_terminal_error":math.dist(
                        outcome["end"],reference["end"])})
    # Sanity checks concern exact mechanism, not hand-tuned SOTA rates.
    for name, _, _, owner in CASES:
        matched=[r for r in records if r["case"]==name and r["arm"]=="source-owner-oracle"]
        assert len(matched)==N_TARGETS
        if max(r["reference_terminal_error"] for r in matched)>1e-8:
            raise AssertionError(f"correct source repair fails to reproduce reference {name}")
    summary={}
    for name in ("reference",)+tuple(c[0] for c in CASES):
        arms=("untouched",) if name=="reference" else ARMS
        for arm in arms:
            rows=[r for r in records if r["case"]==name and r["arm"]==arm]
            summary[f"{name}:{arm}"]={
                "success":sum(x["result"]["success"] for x in rows),
                "trials":len(rows),
                "mean_goal_distance":sum(x["result"]["distance"] for x in rows)/len(rows),
                "mean_reference_terminal_distance":sum(x["reference_terminal_error"] for x in rows)/len(rows)}
    if summary["reference:untouched"]["success"]<N_TARGETS-2:
        raise AssertionError("fixed PD policy is too weak even for reference")
    if not any(summary[f"{name}:untouched"]["success"]<
               summary[f"{name}:source-owner-oracle"]["success"]
               for name,_,_,owner in CASES if owner!="atomic"):
        raise AssertionError("no task-recovery signal from single-source repair")
    if not any(summary[f"{name}:producer-only"]["success"]<
               summary[f"{name}:source-owner-oracle"]["success"]
               for name,_,_,owner in CASES if owner=="atomic"):
        raise AssertionError("no single-patch regression on compensating fault")
    return {"protocol":"semrepair-native-planar-closedloop-mujoco-v1",
            "mujoCo_version":mujoco.__version__,
            "controller":"frozen classical 2D joint-space PD (not VLA)",
            "simulator":"MuJoCo coupled 2-DOF translation body; real timestep integration",
            "task":"terminal 2D target tracking", "position_tolerance":POSITION_TOL,
            "target_seed":SEED,"target_count":N_TARGETS,
            "target_sha256":sha256(json.dumps(goals).encode()).hexdigest(),
            "independent_robot_families":1,"fault_worlds":len(CASES),
            "source_owner_oracle_uses_truth":True,
            "results":summary,"rows":records,
            "limitations":"author-designed task and worlds; same policy/plant for all trials; 16 correlated targets, no independent source authentication and no learned policy or hardware test"}


if __name__=="__main__":
    print(json.dumps(run_planar_task_assay(),indent=2,sort_keys=True))
