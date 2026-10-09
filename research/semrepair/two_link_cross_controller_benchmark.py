"""Cross-plant and cross-controller native MuJoCo interface repair benchmark.

An articulated TWO-HINGE robot is deliberately different from the original
two-slider point mass. Frozen controllers: joint-space PD (has inverse-kinematic
joint target) and Cartesian Jacobian-transpose impedance (receives only XY
end-effector goal). No learned VLA is used. Same continuous target and physics
seeds are held fixed across matched fault/repair arms.

A matched BLACKBOX inverse-composition baseline gets exactly the same two
pre-execution action/tap probes as our source-aware linear factorization.
The source-aware repair is a candidate in a numerical ABI simulator; no
software artifact was independently authenticated or installed.

Do not count target trials as independent robot or source families.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from math import cos, sin, dist, isfinite
from pathlib import Path
from fractions import Fraction
import json
import random

import mujoco
import numpy as np

from .observable_interface_factorization import (
    TapSample, PROBE_E1, PROBE_E2, CHECK,
    identify_repair_from_numeric_taps, inverse, mmul, mv, Mat, I,
)

SIM_STEPS=480
DT=.01
SEED=20261010
N_GOALS=16
GOAL_TOL=.07
INIT=(.25,.7)
LINKS=(.45,.45)
ARM_NAMES=("untouched","producer-only","controller-only","atomic",
           "numeric-source-candidate","matched-blackbox-inverse",
           "oracle-source-owner")
CASES=(
    ("reference",I,I,"noop"),
    ("producer-swap",((0.,1.),(1.,0.)),I,"producer-only"),
    ("controller-sign",I,((-1.,0.),(0.,-1.)),"controller-only"),
    ("compensated-swap",((0.,1.),(1.,0.)),((0.,1.),(1.,0.)),"atomic"),
    ("compensated-rotation",((0.,-1.),(1.,0.)),((0.,1.),(-1.,0.)),"atomic"),
    ("unlisted-shear",((1.,.25),(0.,1.)),((1.,-.25),(0.,1.)),"atomic"),
)
CONTROLLERS=("joint-space-pd","cartesian-jacobian")


def make_model():
    xml=f"""
    <mujoco model="semrepair-two-link-articulated">
      <compiler angle="radian"/>
      <option timestep="{DT}" gravity="0 0 0" integrator="Euler"/>
      <worldbody>
        <body name="upper-arm" pos="0 0 0">
          <joint name="shoulder" type="hinge" axis="0 0 1" damping=".55"
                 limited="true" range="-2.8 2.8"/>
          <geom name="upper" type="capsule" fromto="0 0 0 .45 0 0"
                size=".035" mass=".85"/>
          <body name="forearm" pos=".45 0 0">
            <joint name="elbow" type="hinge" axis="0 0 1" damping=".45"
                   limited="true" range="-2.6 2.6"/>
            <geom name="lower" type="capsule" fromto="0 0 0 .45 0 0"
                  size=".029" mass=".55"/>
            <site name="tip" pos=".45 0 0" size=".025"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="shoulder_motor" joint="shoulder" gear="1"
               ctrllimited="true" ctrlrange="-12 12"/>
        <motor name="elbow_motor" joint="elbow" gear="1"
               ctrllimited="true" ctrlrange="-12 12"/>
      </actuator>
    </mujoco>"""
    model=mujoco.MjModel.from_xml_string(xml)
    assert model.nv==2 and model.nu==2
    # MJCF defaults to angle=degree. The solver targets are in radians.
    # Without this guard [-2.8,2.8] becomes merely +/-0.0489 rad,
    # making a healthy controller fail due to hidden limit violation.
    if not (model.jnt_range[0][0]<-2.5 and model.jnt_range[0][1]>2.5 and
            model.jnt_range[1][0]<-2.4 and model.jnt_range[1][1]>2.4):
        raise AssertionError("MuJoCo joint limit units must be radians")
    return model


def frozen_targets()->tuple[tuple[float,float],...]:
    rng=random.Random(SEED)
    return tuple((round(rng.uniform(-.65,.65),6),
                  round(rng.uniform(.47,1.22),6))
                 for _ in range(N_GOALS))


def tip_xy(q:tuple[float,float]) -> tuple[float,float]:
    a,b=q
    return (LINKS[0]*cos(a)+LINKS[1]*cos(a+b),
            LINKS[0]*sin(a)+LINKS[1]*sin(a+b))


def _policy(model,data,controller:str,target_angles,tip_site)->tuple[float,float]:
    q=np.array(data.qpos[:2],dtype=float)
    v=np.array(data.qvel[:2],dtype=float)
    if controller=="joint-space-pd":
        goal=np.asarray(target_angles)
        cmd=32.*(goal-q)-7.*v
    elif controller=="cartesian-jacobian":
        point=np.array(data.site_xpos[tip_site][:2],dtype=float)
        goal=np.array(tip_xy(target_angles),dtype=float)
        jp=np.zeros((3,model.nv))
        jr=np.zeros((3,model.nv))
        mujoco.mj_jacSite(model,data,jp,jr,tip_site)
        J=jp[:2,:]
        error=goal-point
        cmd=J.T@(58.*error-11.*(J@v))-1.8*v
    else:
        raise ValueError("unknown controller")
    return tuple(np.clip(cmd,-10.,10.))


def numeric_actuator_probe(model,producer:Mat,controller:Mat,request)->TapSample:
    """Step THIS articulated MuJoCo, not the old point-mass fixture."""
    data=mujoco.MjData(model)
    data.qpos[:2]=INIT
    mujoco.mj_forward(model,data)
    sent=mv(producer,request)
    applied=mv(controller,sent)
    data.ctrl[:]=applied
    mujoco.mj_step(model,data)
    force=tuple(float(x) for x in data.qfrc_actuator[:2])
    measured=tuple(float(x) for x in data.ctrl[:2])
    if max(abs(force[i]-measured[i]) for i in (0,1))>1e-10:
        raise AssertionError("source-to-actuator probe not modeled faithfully")
    return TapSample(request,sent,measured)


def _source_patch(producer,controller,arm,oracle_patch,candidate):
    if arm=="oracle-source-owner":
        arm=oracle_patch
    elif arm=="numeric-source-candidate":
        arm=candidate
    if arm in ("atomic","producer-only"):
        producer=I
    if arm in ("atomic","controller-only"):
        controller=I
    return producer,controller


def run_episode(model, *, controller_policy, goal_angles, producer:Mat,
                controller:Mat, arm:str, oracle_patch:str,
                candidate:str, blackbox_inverse:Mat|None)->dict:
    data=mujoco.MjData(model)
    data.qpos[:2]=INIT
    mujoco.mj_forward(model,data)
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"tip")
    pp,cc=_source_patch(producer,controller,arm,oracle_patch,candidate)
    total_abs_ctrl=0.
    total_saturated=0
    for step in range(SIM_STEPS):
        u=_policy(model,data,controller_policy,goal_angles,site)
        if arm=="matched-blackbox-inverse":
            assert blackbox_inverse is not None
            u=mv(blackbox_inverse,u)
        sent=mv(pp,u)
        actual=mv(cc,sent)
        data.ctrl[:]=actual
        total_abs_ctrl+=sum(abs(a) for a in actual)
        total_saturated+=int(any(abs(v)>=12.-1e-8 for v in actual))
        mujoco.mj_step(model,data)
    tip=tuple(float(z) for z in data.site_xpos[site][:2])
    goal=tip_xy(goal_angles)
    err=dist(tip,goal)
    return {"success":bool(err<GOAL_TOL),"terminal_error":err,
            "final_joint_velocity":tuple(float(x) for x in data.qvel[:2]),
            "tip":tip, "goal":goal,
            "command_abs_sum":total_abs_ctrl,
            "physical_saturated_steps":total_saturated}


def run_cross_controller_benchmark()->dict:
    model=make_model()
    goals=frozen_targets()
    records=[]
    diagnostic_rows=[]
    for name,p,c,oracle_patch in CASES:
        train=[numeric_actuator_probe(model,p,c,x) for x in (PROBE_E1,PROBE_E2)]
        val=numeric_actuator_probe(model,p,c,CHECK)
        diagnostic=identify_repair_from_numeric_taps(train,val)
        candidate=diagnostic.get("candidate_patch")
        if not candidate:
            raise AssertionError("frozen linear probe should recover a candidate")
        B=mmul(c,p)
        inverse_B=inverse(B)
        diagnostic_rows.append({
            "case":name,"source_candidate":candidate,"source_oracle_for_score":oracle_patch,
            "source_candidate_matches_oracle":candidate==oracle_patch,
            "blackbox_inverse_matrix":inverse_B,
            "heldout_max_tap_error":max(diagnostic["heldout_producer_error"],
                                         diagnostic["heldout_controller_error"])})
        for policy in CONTROLLERS:
            for i,goal in enumerate(goals):
                for arm in ARM_NAMES:
                    result=run_episode(model,controller_policy=policy,goal_angles=goal,
                          producer=p,controller=c,arm=arm,oracle_patch=oracle_patch,
                          candidate=candidate,blackbox_inverse=inverse_B)
                    records.append({"case":name,"policy":policy,
                                    "goal_index":i,"arm":arm,"result":result})
    stats={}
    for case,_,_,_ in CASES:
        for policy in CONTROLLERS:
            for arm in ARM_NAMES:
                vals=[r["result"] for r in records if r["case"]==case
                      and r["policy"]==policy and r["arm"]==arm]
                stats[f"{case}|{policy}|{arm}"]={
                    "goals":len(vals),"successes":sum(x["success"] for x in vals),
                    "mean_terminal_error":sum(x["terminal_error"] for x in vals)/len(vals),
                    "mean_control_magnitude":sum(x["command_abs_sum"] for x in vals)/len(vals),
                    "mean_saturated_timesteps":sum(x["physical_saturated_steps"] for x in vals)/len(vals)}
    # No fabricated positive outcome is enforced: failed transfers are saved.
    if len(records)!=len(CASES)*len(CONTROLLERS)*N_GOALS*len(ARM_NAMES):
        raise AssertionError("missing scenario or baseline")
    if not all(r["source_candidate_matches_oracle"] for r in diagnostic_rows):
        raise AssertionError("noiseless linear factorization failed")
    return {
        "protocol":"semrepair-two-link-cross-controller-mujoco-v1",
        "simulator":"native MuJoCo 3.3.7 two hinge bodies, two link inertial capsules",
        "mujoco_version":mujoco.__version__,
        "policy_families":CONTROLLERS,
        "source_model_families":1,
        "mechanically_distinct_from_prior_pointmass":True,
        "learned_policy_count":0,"independent_real_robots":0,
        "true_software_patch_installation":False,
        "diagnoser_sees_oracle_source_labels":False,
        "source_provenance_authentication":False,
        "matched_blackbox_composition_receives_same_training_probes":True,
        "number_of_interventions":len(ARM_NAMES),
        "goal_count_per_case_controller":N_GOALS,
        "scenario_hash":sha256(json.dumps({
            "seed":SEED,"goals":goals,"cases":CASES,"controllers":CONTROLLERS,
            "steps":SIM_STEPS,"dt":DT},sort_keys=True).encode()).hexdigest(),
        "diagnosis":diagnostic_rows,"summary":stats,"raw":records,
        "limitation":"both controllers authored in this project, one physical articulated robot model, "
                     "same predeclared transformation family and honest low-amplitude taps; "
                     "no pretrained VLA, independent manufacturer, or real patch execution"
    }


if __name__=="__main__":
    print(json.dumps(run_cross_controller_benchmark(),sort_keys=True,indent=2))
