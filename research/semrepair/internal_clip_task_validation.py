"""Native two-link CLOSED-LOOP task experiment with unreachable black-box torques.

Unlike earlier linear mappings without an internal control clip, the faulty
interface is C clip(P u,-9,9), with C P = identity only for unsaturated u.
Source-bound atomic repair changes P,C to identity and recovers ALL nominal
torques within source actuator limit 9. A BLACKBOX policy cannot command
unreachable C[-9,9]^2 even with full oracle knowledge of C, P and the target
torque. We compare the stronger per-step *optimal Euclidean actuator torque
projection* to source recovery, NOT only a naive matrix inverse.

This oracle optimizes instantaneous torque tracking, not global future task
success. All examples belong to one authored MuJoCo plant and two hand-coded
policies; no independently pretrained VLA or hardware software patch.
"""
from __future__ import annotations
from hashlib import sha256
from math import dist
import json

import mujoco
import numpy as np

from .two_link_cross_controller_benchmark import (
    make_model,INIT,DT,SIM_STEPS,frozen_targets,tip_xy,_policy,CONTROLLERS,GOAL_TOL
)
from .internal_saturation_reachability import (
    PRODUCER,CONTROLLER,LIMIT,exact_blackbox_torque_oracle
)
from .observable_interface_factorization import I,mmul,mv

SOURCES=("reference","corrupted-untouched","linear-blackbox-inverse",
         "omniscient-one-step-blackbox-oracle","source-bound-atomic")
MAX_POLICY_TORQUE=8.


def _simulate(model, policy, goal, arm)->dict:
    data=mujoco.MjData(model)
    data.qpos[:2]=INIT
    mujoco.mj_forward(model,data)
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"tip")
    trace=[]
    hit=0
    for step in range(SIM_STEPS):
        requested=_policy(model,data,policy,goal,site)
        # Common frozen policy interface contract: requested motor torque
        # is limited to the safe source range [-8,8], strictly inside L=9.
        u=tuple(np.clip(requested,-MAX_POLICY_TORQUE,MAX_POLICY_TORQUE))
        if arm in ("reference","source-bound-atomic"):
            actual=u
        elif arm in ("corrupted-untouched","linear-blackbox-inverse"):
            # P and C compensate without clipping, so the blackbox matrix
            # inverse is identity; source clipping destroys that identity.
            intermediate=mv(PRODUCER,u)
            clipped=tuple(np.clip(intermediate,-LIMIT,LIMIT))
            actual=mv(CONTROLLER,clipped)
        elif arm=="omniscient-one-step-blackbox-oracle":
            actual=exact_blackbox_torque_oracle(u)["closest_actuator_torque"]
        else:
            raise ValueError("undeclared arm")
        data.ctrl[:]=actual
        mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)
        fingertip=tuple(float(x) for x in data.site_xpos[site][:2])
        task_error=dist(fingertip,tip_xy(goal))
        tau_gap=dist(actual,u)
        hit+=int(task_error<GOAL_TOL)
        trace.append((step,task_error,tau_gap))
    final_error=trace[-1][1]
    return {"success":final_error<GOAL_TOL,
            "final_error":final_error,
            "mean_task_error":sum(x[1] for x in trace)/len(trace),
            "mean_ideal_torque_gap":sum(x[2] for x in trace)/len(trace),
            "max_ideal_torque_gap":max(x[2] for x in trace),
            "steps_in_goal_tolerance":hit,
            "trajectory":trace}


def run_clipped_task_assay()->dict:
    if mmul(CONTROLLER,PRODUCER)!=I:
        raise AssertionError("compensating source maps are not identity")
    model=make_model()
    targets=frozen_targets()
    rows=[]
    summary={}
    for ctrl in CONTROLLERS:
        for j,goal in enumerate(targets):
            for arm in SOURCES:
                r=_simulate(model,ctrl,goal,arm)
                rows.append({"controller":ctrl,"target_id":j,"arm":arm,"result":r})
        for arm in SOURCES:
            vals=[x["result"] for x in rows if x["controller"]==ctrl and x["arm"]==arm]
            summary[f"{ctrl}|{arm}"]={
                "goals":len(vals),
                "successes":sum(x["success"] for x in vals),
                "mean_final_error":sum(x["final_error"] for x in vals)/len(vals),
                "mean_integrated_task_error":sum(x["mean_task_error"] for x in vals)/len(vals),
                "mean_torque_tracking_error":sum(x["mean_ideal_torque_gap"] for x in vals)/len(vals),
                "max_torque_tracking_error":max(x["max_ideal_torque_gap"] for x in vals)}
    assert len(rows)==len(targets)*len(CONTROLLERS)*len(SOURCES)
    assert all(summary[f"{ctrl}|source-bound-atomic"]["mean_torque_tracking_error"]==0.
               for ctrl in CONTROLLERS)
    assert all(summary[f"{ctrl}|reference"]["mean_torque_tracking_error"]==0.
               for ctrl in CONTROLLERS)
    return {
        "protocol":"semrepair-native-articulated-closedloop-clipped-abi-v1",
        "mujoco_version":mujoco.__version__,
        "physical_families":1,"controller_families":CONTROLLERS,
        "pretrained_vla_count":0,"independent_hardware":0,
        "task_frozen_goals":len(targets),
        "interface_source_clip":LIMIT,"policy_torque_envelope":MAX_POLICY_TORQUE,
        "same_policy_and_goal_per_arm":True,
        "blackbox_comparator":"omniscient Euclidean nearest feasible motor torque at every step; unbounded input and known source map",
        "blackbox_not_global_task_optimum":True,
        "actual_source_patch_installed":False,
        "scenario_sha256":sha256(json.dumps({
            "targets":targets,"seed":"20261010","P":PRODUCER,"C":CONTROLLER,
            "L":LIMIT,"max_policy":MAX_POLICY_TORQUE,"steps":SIM_STEPS,
            "policies":CONTROLLERS},sort_keys=True).encode()).hexdigest(),
        "summary":summary,"rows":rows,
        "limitation":"one manually specified articulated arm with two manually written controllers; "
                     "source repair is a simulator mapping swap and oracle blackbox is per-step torque-optimal, "
                     "not globally optimal for terminal reach success"}
    }


if __name__=="__main__":
    print(json.dumps(run_clipped_task_assay(),indent=2,sort_keys=True))
