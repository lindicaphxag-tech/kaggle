"""Second physical ROBOT BODY, independently pretrained external SAC Pusher-v5.

Official Farama Gymnasium Pusher-v5 is 7-actuator object pushing with 23-D
state observations and an ACTUAL [-2,2]^7 control box, not the 2-actuator
Reacher-v5 goal reaching experiment. We use exact pinned Farama-Minari SAC
checkpoint bytes and zero author policy training.

A predeclared 2x2 reciprocal shear on SHOULDER controls (0,1) plus internal
componentwise clipping, leaving the other 5 control channels untouched, is
an AUTHOR-ENGINEERED software interface fault (not a discovered vendor bug).
All 5 fault configurations, 32 heldout environmental reset seeds, and 6
paired intervention arms are retained. The severities are the SAME relative
action-box ratios as the Reacher experiment, with the actuator bound scaled
from 1 to 2 as physically required by official Pusher.

Strong FULL KNOWLEDGE blackbox comparator computes EXACT per-timestep 2D
Euclidean nearest achievable action AFTER BOTH internal software clipping
and official Pusher action clipping; the unchanged other 5 actuators match
the requested value. This is not global horizon-optimal reward.

Three tiny internal numeric taps identify source P and C without giving the
diagnoser symbolic fault labels. The source repair result is a candidate/
upper bound under assumed trustworthy numeric tap provenance; no real source
issuer attestation or installed patch is demonstrated.

The most important outcome may be NEGATIVE: manuscript must retain all
transfer failures and any reward increase attributable to reduced control
penalties rather than improved pushing distance.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from math import isfinite
import json
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from huggingface_hub import hf_hub_download

from .strong_clipped_actuator_oracle import (
    output_reachability_polygons,optimal_clipped_output
)
from .observable_interface_factorization import (
    mv,mmul,I,TapSample,identify_repair_from_numeric_taps
)

ENV="Pusher-v5"
POLICY_REPO="farama-minari/Pusher-v5-SAC-expert"
POLICY_COMMIT="3e103602f3fba3500ee6f21d762c0ff3e94892ff"
POLICY_FILE="pusher-v5-SAC-expert.zip"
PUBLISHED_LFS_SHA256="6a2a51ddc1f2539aaa779bb7390c3022a0e42effc096962ee2bdaa1fb74f37b3"
FIRST_TWO_ACTIONS=(0,1)
ENV_LIMIT=2.
SCENARIOS=(
    ("mild-control-pair",.8,2.),
    ("shear-2",2.,2.),
    ("shear-3point5",3.5,2.),
    ("shear-2-plus-clip",2.,.8),
    ("shear-3point5-plus-clip",3.5,.8),
)
SEEDS=tuple(range(23011,23043))
ARMS=(
    "official-healthy",
    "untouched-compensating-source-fault",
    "composite-inverse-blackbox",
    "exact-two-clip-full-information-oracle",
    "blind-numeric-source-candidate",
    "source-known-atomic-upper-bound",
)
N_STEPS=100

def _matrices(k):
    P=((1.,float(k)),(0.,1.))
    C=((1.,-float(k)),(0.,1.))
    if mmul(C,P)!=I:
        raise AssertionError("compensating no-clip map should be identity")
    return P,C

def load_external_pusher_checkpoint():
    f=hf_hub_download(repo_id=POLICY_REPO,filename=POLICY_FILE,revision=POLICY_COMMIT)
    recorded=sha256(Path(f).read_bytes()).hexdigest()
    if recorded!=PUBLISHED_LFS_SHA256:
        raise RuntimeError(f"external publisher checkpoint identity changed: {recorded}")
    actor=SAC.load(f,device="cpu")
    if tuple(actor.observation_space.shape)!=(23,) or \
       tuple(actor.action_space.shape)!=(7,):
        raise ValueError("published Pusher SAC dimensions differ from official interface")
    return actor,recorded

def blind_producer_controller_numeric_taps(k:float,L:float):
    P,C=_matrices(k)
    def tap(u):
        produced=mv(P,u)
        inner=np.clip(produced,-L,L)
        downstream=mv(C,(float(inner[0]),float(inner[1])))
        return TapSample(u,produced,downstream)
    result=identify_repair_from_numeric_taps(
        [tap((.02,0.)),tap((0.,-.02))],tap((.012,-.013)))
    return result

def altered_action(ideal, arm:str, P,C,L,reachable):
    request=np.asarray(ideal,dtype=np.float32)
    if request.shape!=(7,):
        raise ValueError("official Pusher-v5 action must have 7 channels")
    if arm not in ARMS:
        raise ValueError("invalid method")
    physical=np.array(request,dtype=np.float32)
    first=(float(request[0]),float(request[1]))
    if arm in ("official-healthy","blind-numeric-source-candidate",
               "source-known-atomic-upper-bound"):
        return physical
    if arm=="exact-two-clip-full-information-oracle":
        corrected=optimal_clipped_output(first,reachable)["action"]
    else:
        # CP=I before internal clipping, so exact matrix-inverse baseline
        # upstream correction is identity in this fixed compensated family.
        intermediate=mv(P,first)
        clipped=np.clip(intermediate,-L,L)
        corrected=mv(C,(float(clipped[0]),float(clipped[1])))
    physical[:2]=corrected
    return physical

def _step_pusher(env,actor,seed,arm,P,C,L,reachable):
    observation,_=env.reset(seed=seed)
    rewards=[]
    object_dists=[]
    contact_dists=[]
    control_rewards=[]
    action_errors=[]
    actuator_clip_count=0
    for t in range(N_STEPS):
        ideal,_=actor.predict(observation,deterministic=True)
        inner=mv(P,(float(ideal[0]),float(ideal[1])))
        actuator_clip_count+=int(max(abs(z) for z in inner)>L+1e-10)
        injected=altered_action(ideal,arm,P,C,L,reachable)
        delivered=np.clip(injected,-ENV_LIMIT,ENV_LIMIT).astype(np.float32)
        observation,reward,done,truncated,info=env.step(delivered)
        rewards.append(float(reward))
        # Respect OFFICIAL reward term semantics: object-to-target distance
        # and fingertip-to-object distance, NOT Reacher-style fingertip-to-target.
        object_distance=-float(info["reward_dist"])
        # Official reward_near already includes its default weight 0.5.
        # Do not mislabel a weighted penalty as an unweighted distance.
        fingertip_distance=-float(info["reward_near"])/.5
        control_reward=float(info["reward_ctrl"])
        if abs(float(reward)-(float(info["reward_dist"])+
                              float(info["reward_near"])+control_reward))>1e-6:
            raise AssertionError("official Pusher reward decomposition invalid")
        control_rewards.append(control_reward)
        if object_distance<0 or fingertip_distance<0:
            raise AssertionError("official Pusher reward term outside distance domain")
        object_dists.append(object_distance)
        contact_dists.append(fingertip_distance)
        action_errors.append(float(np.linalg.norm(np.asarray(ideal)-delivered)))
        if done or truncated:
            break
    if len(rewards)!=N_STEPS:
        raise AssertionError("unexpected Pusher episode horizon")
    return {
        "seed":seed,"arm":arm,
        "steps":len(rewards),"official_episode_return":sum(rewards),
        "mean_object_goal_distance":sum(object_dists)/len(object_dists),
        "final_object_goal_distance":object_dists[-1],
        "minimum_object_goal_distance":min(object_dists),
        "mean_fingertip_object_distance":sum(contact_dists)/len(contact_dists),
        "mean_ideal_action_error":sum(action_errors)/len(action_errors),
        "mean_official_control_reward":sum(control_rewards)/len(control_rewards),
        "inner_clip_events_at_current_policy_states":actuator_clip_count,
    }

def run_external_pusher():
    policy,weight_sha=load_external_pusher_checkpoint()
    rows=[]
    per_arm={}
    with gym.make(ENV) as env:
        if env.action_space.shape!=(7,) or \
           not np.allclose(env.action_space.high,ENV_LIMIT) or \
           not np.allclose(env.action_space.low,-ENV_LIMIT):
            raise ValueError("official Pusher actuator boundary changed")
        for scenario,k,L in SCENARIOS:
            P,C=_matrices(k)
            blind=blind_producer_controller_numeric_taps(k,L)
            if blind["candidate_patch"]!="atomic":
                raise AssertionError("two honest numeric ABI probes failed source component classification")
            polygons=output_reachability_polygons(C,inner_bound=L,outer_bound=ENV_LIMIT)
            for seed in SEEDS:
                for arm in ARMS:
                    row=_step_pusher(env,policy,seed,arm,P,C,L,polygons)
                    row["scenario"]=scenario
                    row["shear"]=k
                    row["intermediate_clip"]=L
                    rows.append(row)
            for arm in ARMS:
                v=[r for r in rows if r["scenario"]==scenario and r["arm"]==arm]
                per_arm[scenario+"|"+arm]={
                    "n":len(v),
                    "mean_official_return":sum(r["official_episode_return"] for r in v)/len(v),
                    "mean_object_goal_distance":sum(r["mean_object_goal_distance"] for r in v)/len(v),
                    "mean_final_object_goal_distance":sum(r["final_object_goal_distance"] for r in v)/len(v),
                    "mean_fingertip_object_distance":sum(r["mean_fingertip_object_distance"] for r in v)/len(v),
                    "mean_action_error":sum(r["mean_ideal_action_error"] for r in v)/len(v),
                    "mean_official_control_reward":sum(r["mean_official_control_reward"] for r in v)/len(v),
                    "total_inner_clip_events":sum(r["inner_clip_events_at_current_policy_states"] for r in v),
                }
    if len(rows)!=len(SEEDS)*len(SCENARIOS)*len(ARMS):
        raise AssertionError("missing predeclared sample")
    diffs={}
    for scenario,_,_ in SCENARIOS:
        delta=[]; object_delta=[]
        for seed in SEEDS:
            ours=next(r for r in rows if (r["scenario"],r["seed"],r["arm"])==
                      (scenario,seed,"blind-numeric-source-candidate"))
            oracle=next(r for r in rows if (r["scenario"],r["seed"],r["arm"])==
                      (scenario,seed,"exact-two-clip-full-information-oracle"))
            reference=next(r for r in rows if (r["scenario"],r["seed"],r["arm"])==
                      (scenario,seed,"official-healthy"))
            if abs(ours["official_episode_return"]-reference["official_episode_return"])>1e-7:
                raise AssertionError("correct candidate no longer recreates healthy external Pusher")
            delta.append(ours["official_episode_return"]-oracle["official_episode_return"])
            object_delta.append(oracle["mean_object_goal_distance"]-
                                ours["mean_object_goal_distance"])
        diffs[scenario]={
            "paired_original_reward_source_minus_oracle":sum(delta)/len(delta),
            "source_wins":sum(d>1e-8 for d in delta),
            "source_losses":sum(d<-1e-8 for d in delta),
            "ties":sum(abs(d)<=1e-8 for d in delta),
            "paired_object_goal_distance_reduction":sum(object_delta)/len(object_delta),
            "paired_reward_rows":delta,
            "paired_distance_rows":object_delta,
        }
    return {
        "protocol":"semrepair-cross-robot-farama-pusher-v5-expert-v1",
        "official_env":ENV,
        "native_action_dimension":7,
        "native_observation_dimension":23,
        "external_policy_repo":POLICY_REPO,
        "external_policy_full_commit":POLICY_COMMIT,
        "external_policy_verified_sha256":weight_sha,
        "policy_trained_by_authors":False,
        "distinct_physical_robot_from_Reacher_v5":True,
        "count_new_external_robot_models":1,
        "new_pretrained_policy_count":1,
        "fault_is_author_created":True,
        "source_attestation_and_real_patch_installation":False,
        "oracle_exact_one_step_physical_action_projection_after_two_clips":True,
        "oracle_not_full_horizon_reward_optimal":True,
        "faults_same_dimensionless_range_as_Reacher_scaled_to_Pusher_action_box":True,
        "identical_frozen_seeds_across_interventions":True,
        "seed_count":len(SEEDS),
        "fixed_fault_grid":SCENARIOS,
        "scenario_sha256":sha256(json.dumps({
            "robot":ENV,"policy_commit":POLICY_COMMIT,
            "faults":SCENARIOS,"seeds":SEEDS,"arms":ARMS,
            "steps":N_STEPS,"axes":FIRST_TWO_ACTIONS},sort_keys=True).encode()).hexdigest(),
        "per_arm":per_arm,"paired":diffs,"raw_rows":rows,
        "limitations":"a single predeclared 2-axis engineered software defect within a 7-axis official robot; "
                      "no VLA, no secure source provenance, no full-horizon MPC comparison, and sample IDs are initial-state seeds not distinct robot manufacturers",
    }

if __name__=="__main__":
    print(json.dumps(run_external_pusher(),sort_keys=True,indent=2))
