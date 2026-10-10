"""SEQUENTIAL second pretrained policy on SAME 7-axis official Pusher robot.

After the external expert run was inspected, we lock in a second published
Farama/Minari SAC-medium checkpoint. Faults, source identification, control
channels, action oracle, seeds and official reward decomposition ALL REUSE
the expert Pusher protocol unchanged. This is *two checkpoints on one model*,
not another robot, independent publisher, or preregistered full study.
All five severity outcomes remain visible, including harm and ties.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from huggingface_hub import hf_hub_download

from .external_pusher_transfer import (
    ENV,ENV_LIMIT,SCENARIOS,SEEDS,ARMS,
    _matrices,_step_pusher,blind_producer_controller_numeric_taps
)
from .strong_clipped_actuator_oracle import output_reachability_polygons

PUBLISHER_REPO="farama-minari/Pusher-v5-SAC-medium"
PUBLISHER_COMMIT="d7d16ee5dfaa1f9e43df96cbdd6846cead1dabb6"
PUBLISHER_FILE="pusher-v5-SAC-medium.zip"
PUBLISHER_FILE_SHA256="6ad4f083db0713cb5c6ec649cbcff50c5205e25c78756aed89f9921d75b1d2b8"


def load_second_pusher_policy():
    checkpoint=hf_hub_download(repo_id=PUBLISHER_REPO,
                               filename=PUBLISHER_FILE,revision=PUBLISHER_COMMIT)
    raw=sha256(Path(checkpoint).read_bytes()).hexdigest()
    if raw!=PUBLISHER_FILE_SHA256:
        raise RuntimeError("published Pusher-medium checkpoint did not match pinned LFS bytes")
    agent=SAC.load(checkpoint,device="cpu")
    if agent.action_space.shape!=(7,) or agent.observation_space.shape!=(23,):
        raise ValueError("second SAC Pusher policy model interface changed")
    return agent,raw


def run_second_pusher_policy():
    actor,weight_sha=load_second_pusher_policy()
    rows=[]
    summary={}
    with gym.make(ENV) as env:
        if env.action_space.shape!=(7,) or \
           not np.allclose(env.action_space.high,ENV_LIMIT):
            raise ValueError("official Pusher actuator bounds are not preserved")
        for fault,k,L in SCENARIOS:
            P,C=_matrices(k)
            candidate=blind_producer_controller_numeric_taps(k,L)
            if candidate["candidate_patch"]!="atomic":
                raise AssertionError("numeric source inference failed under original frozen inputs")
            reachable=output_reachability_polygons(
                C,inner_bound=L,outer_bound=ENV_LIMIT)
            for seed in SEEDS:
                for arm in ARMS:
                    r=_step_pusher(env,actor,seed,arm,P,C,L,reachable)
                    r["scenario"]=fault
                    rows.append(r)
            for arm in ARMS:
                values=[r for r in rows if r["scenario"]==fault and r["arm"]==arm]
                summary[fault+"|"+arm]={
                    "episodes":len(values),
                    "mean_official_reward":sum(r["official_episode_return"] for r in values)/len(values),
                    "mean_object_goal_distance":sum(r["mean_object_goal_distance"] for r in values)/len(values),
                    "mean_fingertip_object_distance":sum(r["mean_fingertip_object_distance"] for r in values)/len(values),
                    "mean_official_control_reward":sum(r["mean_official_control_reward"] for r in values)/len(values)}
    if len(rows)!=len(SCENARIOS)*len(SEEDS)*len(ARMS):
        raise AssertionError("missing original frozen benchmark cases")
    paired={}
    for fault,_,_ in SCENARIOS:
        reward=[]
        actual_object_distance=[]
        for seed in SEEDS:
            selected={r["arm"]:r for r in rows if r["scenario"]==fault and r["seed"]==seed}
            source=selected["blind-numeric-source-candidate"]
            base=selected["exact-two-clip-full-information-oracle"]
            healthy=selected["official-healthy"]
            if abs(source["official_episode_return"]-healthy["official_episode_return"])>1e-7:
                raise AssertionError("source candidate no longer recreates healthy second Pusher actor")
            reward.append(source["official_episode_return"]-base["official_episode_return"])
            actual_object_distance.append(base["mean_object_goal_distance"]-
                                          source["mean_object_goal_distance"])
        paired[fault]={
            "source_minus_exact_instant_action_oracle_reward":sum(reward)/len(reward),
            "source_reduces_actual_object_goal_distance_m":sum(actual_object_distance)/len(actual_object_distance),
            "source_better_reward_seeds":sum(x>1e-8 for x in reward),
            "source_worse_reward_seeds":sum(x< -1e-8 for x in reward),
            "source_tied_reward_seeds":sum(abs(x)<=1e-8 for x in reward),
            "paired_reward_rows":reward,
        }
    return {
        "protocol":"semrepair-second-pusher-SAC-checkpoint-replication-v1",
        "official_env":ENV,"action_dimension":7,
        "source_author":PUBLISHER_REPO,
        "source_model_commit":PUBLISHER_COMMIT,
        "verified_weight_sha256":weight_sha,
        "policy_training_by_authors":0,
        "same_seeds_as_original_pusher_expert":True,
        "faults_and_controls_frozen_unchanged":True,
        "post_first_pusher_expert_outcome_sequential_replication":True,
        "new_physical_robot_count":0,
        "true_source_signatures_or_patch_installs":False,
        "exact_inner_and_outer_physical_action_projection_oracle":True,
        "not_globally_optimal_future_reward_oracle":True,
        "n_paired_episodes":len(rows),
        "summary":summary,"paired":paired,"raw_rows":rows,
        "limitation":"new actor checkpoint on the identical official 7-axis Pusher task, all faults authored; no VLA or independently verified source patch installation",
    }


if __name__=="__main__":
    print(json.dumps(run_second_pusher_policy(),indent=2,sort_keys=True))
