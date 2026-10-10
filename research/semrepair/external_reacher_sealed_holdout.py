"""Sealed external Reacher-v5 pretrained SAC fault-SUPPORT study.

This protocol was frozen AFTER observing that the pilot 16 seeds with
k=0.8, L=1.0 yielded exactly zero fault impact. The new 32 seeds DO NOT
overlap those pilot seeds, and all predeclared fault magnitudes and controls
are reported, including zero and negative outcomes. Do not pool across
severities as if independently sourced robot families.

Official Farama/Minari pretrained expert and Gymnasium Reacher-v5 only;
no training, fake software signer or external hardware use. Physical test
plant and rewards belong to the official environment. The defect is an
authored two-component action ABI before official actuator clipping.

A numerical 3-tap source candidate is inferred in the UNSATURATED
low-action region without passing it the ground-truth fault name. It
does not authenticate any source bytes or prove the mappings globally
linear under clipping. The "oracle-source-atomic" arm is an upper bound.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from .frozen_external_reacher_sac import (
    ENV_NAME,HF_REPO,HF_REVISION,HF_FILENAME,
    frozen_external_checkpoint,execute_frozen_episode
)
from .strong_clipped_actuator_oracle import (
    output_reachability_polygons,optimal_clipped_output
)
from .observable_interface_factorization import (
    mv,mmul,I,TapSample,identify_repair_from_numeric_taps
)

HOLDOUT_SEEDS=tuple(range(23011,23043))
FAULTS=(
    ("pilot-severity",.8,1.),
    ("large-shear",2.,1.),
    ("very-large-shear",3.5,1.),
    ("shear-plus-extra-clip",2.,.4),
    ("large-shear-plus-extra-clip",3.5,.4),
)
ARMS=("official-healthy","fault-unrepaired","matched-blackbox-inverse",
      "exact-all-clip-blackbox-oracle","blind-source-patch-candidate",
      "oracle-source-atomic-upper-bound")
EPISODE_STEPS=50
TIP_DISTANCE=.075


def _matrices(k):
    P=((1.,float(k)),(0.,1.))
    C=((1.,-float(k)),(0.,1.))
    if mmul(C,P)!=I:
        raise AssertionError("compensating preclip transforms must cancel")
    return P,C


def identify_blind_fault(P,C,L):
    # Probe maximum required preclip magnitude is analytically bounded by
    # 0.02*(1+k) <=0.09 at maximum k=3.5; all L >=0.4.
    probes=((.02,0.),(0.,-.02))
    hold=(.012,-.013)
    def observe(u):
        p=mv(P,u)
        inter=np.clip(p,-L,L)
        t=mv(C,(float(inter[0]),float(inter[1])))
        return TapSample(u,p,t)
    result=identify_repair_from_numeric_taps(
        [observe(u) for u in probes],observe(hold))
    return result


def _action(ideal,arm,P,C,L,polygons):
    a=(float(ideal[0]),float(ideal[1]))
    if arm in ("official-healthy","blind-source-patch-candidate",
               "oracle-source-atomic-upper-bound"):
        return np.asarray(a,dtype=np.float32)
    if arm in ("fault-unrepaired","matched-blackbox-inverse"):
        # CP=I before clipping; the best simple exact matrix inverse is I,
        # so the two declared arms honestly coincide in this construction.
        tmp=mv(P,a)
        inter=np.clip(tmp,-L,L)
        out=mv(C,(float(inter[0]),float(inter[1])))
        return np.asarray(out,dtype=np.float32)
    if arm=="exact-all-clip-blackbox-oracle":
        out=optimal_clipped_output(a,polygons)
        return np.asarray(out["action"],dtype=np.float32)
    raise ValueError("undeclared arm")


def _episode(env,policy,seed,arm,P,C,L,polygons):
    obs,_=env.reset(seed=seed)
    ret=0.
    dists=[]
    action_gaps=[]
    inner_clip_events=0
    for step in range(EPISODE_STEPS):
        ideal,_=policy.predict(obs,deterministic=True)
        requested=(float(ideal[0]),float(ideal[1]))
        inner=mv(P,requested)
        inner_clip_events+=int(max(abs(v) for v in inner)>L+1e-10)
        commanded=_action(ideal,arm,P,C,L,polygons)
        actual=np.clip(commanded,env.action_space.low,env.action_space.high)
        obs,reward,terminated,truncated,info=env.step(actual)
        ret+=float(reward)
        distance=float(-info["reward_dist"])
        if distance<0 or not isfinite(distance):
            raise AssertionError("official Reacher distance reward inconsistency")
        dists.append(distance)
        action_gaps.append(float(np.linalg.norm(np.asarray(ideal)-actual)))
        if terminated or truncated:
            break
    if len(dists)!=EPISODE_STEPS:
        raise AssertionError("unexpected task horizon")
    return {"seed":seed,"arm":arm,
            "return":ret,"mean_tip_distance":sum(dists)/len(dists),
            "final_tip_distance":dists[-1],"min_tip_distance":min(dists),
            "ever_hit_0075":min(dists)<=TIP_DISTANCE,
            "mean_action_gap":sum(action_gaps)/len(action_gaps),
            "inner_clip_events_when_policy_ideal_passes_producer":inner_clip_events,
            "steps":len(dists)}


def run_sealed_external_holdout():
    policy,checkpoint_sha=frozen_external_checkpoint()
    all_rows=[]
    summary={}
    with gym.make(ENV_NAME) as env:
        for fault,k,L in FAULTS:
            P,C=_matrices(k)
            candidate=identify_blind_fault(P,C,L)
            # This check is an invariant of the manufactured linear core,
            # NOT a permission to run a trusted source patch.
            if candidate.get("candidate_patch")!="atomic":
                raise AssertionError("blind candidate fails known linear core")
            regions=output_reachability_polygons(C,inner_bound=L,outer_bound=1.)
            for seed in HOLDOUT_SEEDS:
                for arm in ARMS:
                    r=_episode(env,policy,seed,arm,P,C,L,regions)
                    r["fault_family"]=fault
                    r["fault_shear"]=k
                    r["internal_source_clip"]=L
                    all_rows.append(r)
            for arm in ARMS:
                cohort=[x for x in all_rows if x["fault_family"]==fault
                        and x["arm"]==arm]
                summary[f"{fault}|{arm}"]={
                    "episodes":len(cohort),
                    "average_official_return":sum(x["return"] for x in cohort)/len(cohort),
                    "average_mean_tip_distance":sum(x["mean_tip_distance"] for x in cohort)/len(cohort),
                    "average_final_tip_distance":sum(x["final_tip_distance"] for x in cohort)/len(cohort),
                    "ever_hit_0075_count":sum(x["ever_hit_0075"] for x in cohort),
                    "average_mean_policy_action_gap":sum(x["mean_action_gap"] for x in cohort)/len(cohort),
                    "total_inner_clip_events":sum(x["inner_clip_events_when_policy_ideal_passes_producer"] for x in cohort)}
    if len(all_rows)!=len(HOLDOUT_SEEDS)*len(FAULTS)*len(ARMS):
        raise AssertionError("missing preregistered arms or seeds")
    for fault,_,_ in FAULTS:
        for seed in HOLDOUT_SEEDS:
            ref=next(x for x in all_rows if (x["fault_family"],x["seed"],x["arm"])==
                     (fault,seed,"official-healthy"))
            for other in ("oracle-source-atomic-upper-bound",
                          "blind-source-patch-candidate"):
                row=next(x for x in all_rows if (x["fault_family"],x["seed"],x["arm"])==
                         (fault,seed,other))
                if abs(ref["return"]-row["return"])>1e-8:
                    raise AssertionError("frozen source restoration differs from healthy")
    paired_effects={}
    for fault,_,_ in FAULTS:
        def delta(arm):
            return [
                next(x["return"] for x in all_rows if x["fault_family"]==fault and
                     x["seed"]==seed and x["arm"]=="blind-source-patch-candidate")-
                next(x["return"] for x in all_rows if x["fault_family"]==fault and
                     x["seed"]==seed and x["arm"]==arm)
                for seed in HOLDOUT_SEEDS]
        deltas=delta("exact-all-clip-blackbox-oracle")
        paired_effects[fault]={
            "mean_reward_improvement_source_vs_exact_oracle":sum(deltas)/len(deltas),
            "source_better_seed_count":sum(d>1e-8 for d in deltas),
            "source_worse_seed_count":sum(d< -1e-8 for d in deltas),
            "tied_seed_count":sum(abs(d)<=1e-8 for d in deltas),
            "all_paired_reward_differences":deltas}
    return {
        "protocol":"semrepair-external-sb3-reacher-frozen-postpilot-holdout-v1",
        "post_pilot_predeclared_not_originally_prospective":True,
        "original_pilot_seeds":list(range(1001,1013)),
        "holdout_seeds":list(HOLDOUT_SEEDS),
        "independent_seed_count":len(HOLDOUT_SEEDS),
        "physical_robot_families":1,
        "external_pretrained_policy_count":1,
        "publisher":HF_REPO,"publisher_commit":HF_REVISION,
        "checkpoint_sha256":checkpoint_sha,
        "fault_grid":FAULTS,"arms":ARMS,
        "per_step_exact_blackbox_clipped_action_oracle":True,
        "oracle_not_globally_optimal_future_control":True,
        "source_trust_cryptographically_validated":False,
        "actual_software_patch_installed":False,
        "policy_retrained":False,
        "scenario_sha256":sha256(json.dumps({
            "seeds":HOLDOUT_SEEDS,"faults":FAULTS,"arms":ARMS,
            "steps":EPISODE_STEPS,"commit":HF_REVISION},sort_keys=True).encode()).hexdigest(),
        "summary":summary,"paired":paired_effects,"raw_rows":all_rows,
        "claim_scope":"postpilot fixed grid, official Reacher-v5 and frozen published SAC; "
                      "only one external robot family, engineered interface mappings, no source trust or autonomous patch install",
    }


if __name__=="__main__":
    print(json.dumps(run_sealed_external_holdout(),indent=2,sort_keys=True))
