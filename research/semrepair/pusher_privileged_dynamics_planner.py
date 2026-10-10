"""Fully known MuJoCo dynamics / original reward one-step rollout black-box oracle.

Unlike the prior mathematical closest-action oracle, this comparator reads the
EXACT official Pusher-v5 MuJoCo qpos/qvel/actuator state, knows the plant and
true generated source fault and the official reward weights. For every step,
it rolls out candidate PHYSICALLY REACHABLE actions in a cloned MjData,
scores original object-goal / fingertip-near / control reward, and executes
the best feasible action in the real frozen original Pusher-v5 environment.

This is a stronger-information black-box baseline on the actual TASK reward,
not just action tracking. It is still only one-step receding horizon and a
finite discrete action candidate library, NOT a globally optimal multi-step
MPC, full dynamic programming, or proof of worst-case task success.

The task/seed panel was devised after observing Pusher expert results: an
exploratory red-team falsifier, NEVER independent preregistered efficacy.
All performance losses and failures must be displayed.
"""
from __future__ import annotations
from hashlib import sha256
import json
import math
from itertools import product
import numpy as np
import gymnasium as gym
import mujoco

from .external_pusher_transfer import (
    load_external_pusher_checkpoint,ENV,ENV_LIMIT,
    _matrices,SCENARIOS
)
from .strong_clipped_actuator_oracle import (
    output_reachability_polygons,optimal_clipped_output,mm
)

SEEDS=tuple(range(23011,23027))  # 16 ALREADY-SEEN Pusher reset identities
SEVERE=("shear-3point5-plus-clip",3.5,.8)
N_STEPS=100
ORACLE_ACTION_LIBRARY_SPEC="exact action projection plus clipped feasible source grid z in {-L,-L/2,0,L/2,L}^2"
REWARD_DIST_WEIGHT=1.0
REWARD_NEAR_WEIGHT=.5
REWARD_CTRL_WEIGHT=.1


class PrivilegedOneStepModelPlanner:
    def __init__(self,model,frame_skip:int,C,limit:float):
        self.model=model
        self.frame_skip=frame_skip
        self.C=C
        self.limit=limit
        self.source_pairs=[(x*limit,y*limit) for x,y in product(
            (-1.,-.5,0.,.5,1.),repeat=2)]
        self.scratch=mujoco.MjData(model)
        self.state_spec=mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.state=np.empty(mujoco.mj_stateSize(model,self.state_spec),dtype=np.float64)
        self.fingertip=model.body("tips_arm").id
        self.obj=model.body("object").id
        self.goal=model.body("goal").id
        self.reward_evaluations=0

    def native_reward_of_action(self,env_source_data,action):
        # mj_getState/mj_setState copies actual physics state from the TRUE
        # current official environment into an isolated MjData. No baseline
        # interacts with the real task to evaluate its unselected candidates.
        mujoco.mj_getState(self.model,env_source_data,self.state,self.state_spec)
        mujoco.mj_setState(self.model,self.scratch,self.state,self.state_spec)
        mujoco.mj_forward(self.model,self.scratch)
        self.scratch.ctrl[:]=np.asarray(action,dtype=np.float64)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model,self.scratch)
        mujoco.mj_forward(self.model,self.scratch)
        obj=self.scratch.xpos[self.obj]
        tip=self.scratch.xpos[self.fingertip]
        goal=self.scratch.xpos[self.goal]
        score=(-np.linalg.norm(obj-goal)*REWARD_DIST_WEIGHT
               -np.linalg.norm(obj-tip)*REWARD_NEAR_WEIGHT
               -np.sum(np.square(action))*REWARD_CTRL_WEIGHT)
        self.reward_evaluations+=1
        return float(score)

    def choose_feasible_action(self,env_source_data,ideal):
        ideal=np.asarray(ideal,dtype=np.float64)
        if ideal.shape!=(7,):
            raise ValueError("requires Pusher full seven torque channels")
        regions=output_reachability_polygons(
            self.C,inner_bound=self.limit,outer_bound=ENV_LIMIT)
        nearest=optimal_clipped_output((ideal[0],ideal[1]),regions)["action"]
        proposals={tuple(round(float(z),10) for z in nearest)}
        for u in self.source_pairs:
            raw=mm(self.C,u)
            result=(float(np.clip(raw[0],-ENV_LIMIT,ENV_LIMIT)),
                    float(np.clip(raw[1],-ENV_LIMIT,ENV_LIMIT)))
            proposals.add(tuple(round(float(z),10) for z in result))
        baseline=None
        for proposal in sorted(proposals):
            actual=ideal.copy()
            actual[:2]=proposal
            actual=np.clip(actual,-ENV_LIMIT,ENV_LIMIT)
            score=self.native_reward_of_action(env_source_data,actual)
            # Include Euclidean tracking as tie break ONLY when official
            # native original immediate rewards tie to precision 1e-11.
            criterion=(round(score,11), -float(np.linalg.norm(actual-ideal)))
            if baseline is None or criterion>baseline[0]:
                baseline=(criterion,actual,score)
        if baseline is None:raise AssertionError("empty oracle action search")
        return baseline[1],baseline[2]


def _rollout(actor,env,seed,policy_arm,planner):
    obs,_=env.reset(seed=seed)
    rewards=[]
    distances=[]
    estimated_reward_errors=[]
    for i in range(N_STEPS):
        ideal,_=actor.predict(obs,deterministic=True)
        if policy_arm=="healthy":
            actual=np.asarray(ideal,dtype=np.float64)
            predicted=None
        elif policy_arm=="globally-instant-action-optimal":
            poly=output_reachability_polygons(
                planner.C,inner_bound=planner.limit,outer_bound=ENV_LIMIT)
            first=optimal_clipped_output((float(ideal[0]),float(ideal[1])),poly)["action"]
            actual=np.asarray(ideal,dtype=np.float64).copy()
            actual[:2]=first
            predicted=None
        elif policy_arm=="privileged-one-step-native-task-planner":
            actual,predicted=planner.choose_feasible_action(env.unwrapped.data,ideal)
        else:raise ValueError("invalid oracle arm")
        actual=np.clip(actual,env.action_space.low,env.action_space.high)
        obs,reward,done,truncated,info=env.step(actual)
        actual_reward=float(reward)
        rewards.append(actual_reward)
        distances.append(-float(info["reward_dist"]))
        if predicted is not None:
            error=abs(predicted-actual_reward)
            estimated_reward_errors.append(error)
            if error>1e-7:
                raise AssertionError(f"privileged native task model diverged from true reward: {error}")
        if done or truncated:break
    if len(rewards)!=N_STEPS:
        raise AssertionError("official Pusher full task length changed")
    return {
        "seed":seed,"arm":policy_arm,"steps":len(rewards),
        "original_return":sum(rewards),
        "mean_actual_object_goal_distance":sum(distances)/len(distances),
        "max_native_simulator_reward_prediction_error":max(estimated_reward_errors,default=0.),
        "per_step_original_reward":rewards,
        "per_step_object_goal_distance":distances}


def run_privileged_task_reward_baseline():
    actor,weight_sha=load_external_pusher_checkpoint()
    name,k,L=SEVERE
    _,C=_matrices(k)
    rows=[]
    with gym.make(ENV) as env:
        planner=PrivilegedOneStepModelPlanner(
            env.unwrapped.model,env.unwrapped.frame_skip,C,L)
        for seed in SEEDS:
            for arm in ("healthy","globally-instant-action-optimal",
                        "privileged-one-step-native-task-planner"):
                rows.append(_rollout(actor,env,seed,arm,planner))
    report={}
    for arm in ("healthy","globally-instant-action-optimal",
                "privileged-one-step-native-task-planner"):
        x=[row for row in rows if row["arm"]==arm]
        report[arm]={
            "n":len(x),
            "mean_official_return":sum(t["original_return"] for t in x)/len(x),
            "mean_actual_object_goal_distance":sum(
                t["mean_actual_object_goal_distance"] for t in x)/len(x)}
    paired={}
    for b in ("globally-instant-action-optimal",
              "privileged-one-step-native-task-planner"):
        difference=[]
        distance=[]
        for seed in SEEDS:
            healthy=next(r for r in rows if r["seed"]==seed and r["arm"]=="healthy")
            other=next(r for r in rows if r["seed"]==seed and r["arm"]==b)
            difference.append(healthy["original_return"]-other["original_return"])
            distance.append(other["mean_actual_object_goal_distance"]-
                            healthy["mean_actual_object_goal_distance"])
        paired[b]={
            "mean_source_advantage_official_return":sum(difference)/len(difference),
            "source_reward_wins":sum(d>1e-8 for d in difference),
            "source_reward_losses":sum(d< -1e-8 for d in difference),
            "reward_ties":sum(abs(d)<=1e-8 for d in difference),
            "mean_source_object_goal_distance_reduction":sum(distance)/len(distance),
            "paired_reward_differences":difference,
            "paired_distance_differences":distance}
    return {
        "protocol":"official-Pusher-v5-privileged-native-true-reward-one-step-blackbox-v1",
        "external_checkpoint_sha256":weight_sha,
        "robot":ENV,
        "prior_pusher_experiment_was_already_inspected":True,
        "distinct_new_heldout_seed_count":0,
        "source_model_clipping":L,"source_model_shear":k,
        "full_information_actual_native_MuJoCo_state":True,
        "full_information_actual_native_MuJoCo_plant":True,
        "original_official_reward_optimized_per_step":True,
        "globally_optimal_future_task":False,
        "finite_reachable_action_grid_not_continuous_optimum":True,
        "oracle_model_reward_evaluations":planner.reward_evaluations,
        "real_task_record_count":len(rows),
        "scenario_sha256":sha256(json.dumps({
            "seeds":SEEDS,"policy_model_sha":weight_sha,
            "scenario":SEVERE,"candidate_actions":ORACLE_ACTION_LIBRARY_SPEC,
            "step_horizon":1,"env":ENV},sort_keys=True).encode()).hexdigest(),
        "results":report,"paired":paired,"rows":rows,
        "limitation":"one-step oracle has exact native model and original task reward but evaluates finite candidate actions; true best global-horizon planning and natural faults remain untested",
    }


if __name__=="__main__":
    print(json.dumps(run_privileged_task_reward_baseline(),indent=2,sort_keys=True))
