"""Frozen-policy, TRUE native Pusher task-reward 3-step black-box MPC.

Research-grade red-team comparison for SemRepair. Explicit strengths:
  - exact official Pusher-v5 MuJoCo model and LIVE full integration state;
  - exact authored clipped source failure (k=3.5, inner L=0.8);
  - no limits on learning the producer/controller maps and no diagnosis cost;
  - branch-by-branch actor recomputation using the *frozen external SAC*;
  - original contact-rich task reward at EACH predicted official timestep;
  - an exactly feasible discrete library plus global instantaneous action
    projection. No source module patch allowed, blackbox only.

Weaknesses stated up front:
  - H=3 is NOT globally full-episode reward optimal;
  - first-step search grid finite; tail uses exact action projection rather
    than optimizing every possible future control;
  - tested on 16 already-inspected same Pusher reset seeds;
  - model rollouts are oracle privileged and counted explicitly;
  - not pretrained VLA, hardware, or independent manufacturer validation.

This aims to FALSIFY the claim that source repair is superior to all possible
black-box recovery policies. We claim only superiority to the measured named
baselines when/if actually demonstrated on original official task metrics.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import product
import json
import numpy as np
import gymnasium as gym
import mujoco

from .external_pusher_transfer import (
    ENV,ENV_LIMIT,load_external_pusher_checkpoint,_matrices
)
from .strong_clipped_actuator_oracle import (
    output_reachability_polygons,optimal_clipped_output,mm
)
from .pusher_privileged_dynamics_planner import (
    PrivilegedOneStepModelPlanner,REWARD_NEAR_WEIGHT,
    REWARD_CTRL_WEIGHT,REWARD_DIST_WEIGHT
)

SEEDS=tuple(range(23011,23027))  # previously inspected 16 seed identities
FAULT_LABEL="shear-3point5-plus-clip"
FAULT_SHEAR=3.5
FAULT_INNER_CLIP=.8
STEPS=100
HORIZON=3
# Nine physically admissible endpoints plus exact Euclidean projection.
FUTURE_CONTROL_GRID=(-1.,0.,1.)
ARM_NAMES=("official-healthy","exact-instantaneous-action-oracle",
           "privileged-native-three-step-tail-MPC")


class TrueNativeThreeStepMPC:
    def __init__(self,model,frame_skip,actor,controller,inner_clip):
        self.model=model
        self.skip=frame_skip
        self.actor=actor
        self.C=controller
        self.clip=inner_clip
        self.scratch=mujoco.MjData(model)
        self.spec=mujoco.mjtState.mjSTATE_INTEGRATION
        self.stored=np.empty(mujoco.mj_stateSize(model,self.spec),dtype=np.float64)
        self.tip=model.body("tips_arm").id
        self.obj=model.body("object").id
        self.goal=model.body("goal").id
        self.polygons=output_reachability_polygons(
            controller,inner_bound=inner_clip,outer_bound=ENV_LIMIT)
        self.rollout_evaluations=0
        self.physics_integration_steps=0
        self.policy_tail_predictions=0

    def _action_after_internal_adapter(self,inner):
        raw=mm(self.C,inner)
        return tuple(float(np.clip(v,-ENV_LIMIT,ENV_LIMIT)) for v in raw)

    def _proposals(self,ideal):
        exact=optimal_clipped_output(
            (float(ideal[0]),float(ideal[1])),self.polygons)["action"]
        out={tuple(round(float(z),12) for z in exact)}
        for u,v in product(FUTURE_CONTROL_GRID,repeat=2):
            pair=self._action_after_internal_adapter((u*self.clip,v*self.clip))
            out.add(tuple(round(float(z),12) for z in pair))
        return sorted(out)

    def _native_actor_observation(self):
        d=self.scratch
        # Official Pusher-v5 _get_obs order, not a hand-designed surrogate:
        # qpos[0:7], qvel[0:7], tip_xyz, object_xyz, goal_xyz.
        x=np.concatenate((d.qpos[:7],d.qvel[:7],
                          d.xpos[self.tip],d.xpos[self.obj],d.xpos[self.goal]))
        if x.shape!=(23,):
            raise AssertionError("official external Pusher 23D state mismatch")
        return x

    def _advance_and_native_reward(self,actual):
        d=self.scratch
        d.ctrl[:]=actual
        for _ in range(self.skip):
            mujoco.mj_step(self.model,d)
            self.physics_integration_steps+=1
        # Official Gym Pusher _get_rew is called WITHOUT extra mj_forward,
        # and uses body COM fields as left by the final mj_step.
        obj=d.xpos[self.obj]
        tip=d.xpos[self.tip]
        goal=d.xpos[self.goal]
        reward=(-np.linalg.norm(obj-goal)*REWARD_DIST_WEIGHT
                -np.linalg.norm(obj-tip)*REWARD_NEAR_WEIGHT
                -np.square(actual).sum()*REWARD_CTRL_WEIGHT)
        return float(reward)

    def branch_value(self,source_data,ideal,first_pair,*,horizon=HORIZON):
        if horizon<1 or horizon>HORIZON:
            raise ValueError("horizon must be within frozen 1..3")
        mujoco.mj_getState(self.model,source_data,self.stored,self.spec)
        mujoco.mj_setState(self.model,self.scratch,self.stored,self.spec)
        step_action=np.asarray(ideal,dtype=np.float64).copy()
        step_action[:2]=first_pair
        step_action=np.clip(step_action,-ENV_LIMIT,ENV_LIMIT)
        score=self._advance_and_native_reward(step_action)
        self.rollout_evaluations+=1
        for _ in range(horizon-1):
            simulated_observation=self._native_actor_observation()
            desired,_=self.actor.predict(simulated_observation,deterministic=True)
            self.policy_tail_predictions+=1
            projected=optimal_clipped_output(
                (float(desired[0]),float(desired[1])),self.polygons)["action"]
            followup=np.asarray(desired,dtype=np.float64).copy()
            followup[:2]=projected
            followup=np.clip(followup,-ENV_LIMIT,ENV_LIMIT)
            score+=self._advance_and_native_reward(followup)
        return score

    def choose(self,source_data,ideal):
        proposals=self._proposals(ideal)
        selected=None
        for pair in proposals:
            val=self.branch_value(source_data,ideal,pair)
            tracking=float(np.linalg.norm(np.asarray(pair)-np.asarray(ideal)[:2]))
            criterion=(round(val,10),-tracking)
            if selected is None or criterion>selected[0]:
                selected=(criterion,pair,val)
        if selected is None:
            raise AssertionError("true task-model planner had zero feasible controls")
        action=np.asarray(ideal,dtype=np.float64).copy()
        action[:2]=selected[1]
        return np.clip(action,-ENV_LIMIT,ENV_LIMIT),selected[2],len(proposals)


def _episode(env,actor,planner,seed,arm):
    obs,_=env.reset(seed=seed)
    total_reward=0.
    distances=[]
    first_step_accuracy=[]
    candidate_counts=[]
    for t in range(STEPS):
        desired,_=actor.predict(obs,deterministic=True)
        if arm=="official-healthy":
            chosen=np.asarray(desired,dtype=np.float64)
            predicted=None
        elif arm=="exact-instantaneous-action-oracle":
            projected=optimal_clipped_output(
                (float(desired[0]),float(desired[1])),planner.polygons)["action"]
            chosen=np.asarray(desired,dtype=np.float64).copy()
            chosen[:2]=projected
            predicted=None
        elif arm=="privileged-native-three-step-tail-MPC":
            chosen,value,candidates=planner.choose(env.unwrapped.data,desired)
            # Validate one-step reward separately on the selected action.
            single=planner.branch_value(
                env.unwrapped.data,desired,tuple(float(x) for x in chosen[:2]),
                horizon=1)
            candidate_counts.append(candidates)
            predicted=single
        else:
            raise ValueError("unknown arm")
        action=np.clip(chosen,env.action_space.low,env.action_space.high)
        obs,r,done,truncated,info=env.step(action)
        total_reward+=float(r)
        distances.append(-float(info["reward_dist"]))
        if predicted is not None:
            discrepancy=abs(float(r)-predicted)
            first_step_accuracy.append(discrepancy)
            if discrepancy>1e-7:
                raise AssertionError(f"native 3-step planner differs from real official reward by {discrepancy}")
        if done or truncated:
            break
    if len(distances)!=STEPS:
        raise AssertionError("official Pusher task length is not 100")
    return {
        "seed":seed,"arm":arm,"n_steps":len(distances),
        "official_episode_return":total_reward,
        "mean_physical_object_goal_distance":sum(distances)/len(distances),
        "max_selected_action_reward_prediction_error":max(first_step_accuracy,default=0.),
        "mean_candidate_count":sum(candidate_counts)/len(candidate_counts) if candidate_counts else 0,
        "per_timestep_original_reward_validated":len(first_step_accuracy),
    }


def run_external_pusher_three_step_mpc():
    actor,sha=load_external_pusher_checkpoint()
    _,C=_matrices(FAULT_SHEAR)
    rows=[]
    with gym.make(ENV) as env:
        planner=TrueNativeThreeStepMPC(env.unwrapped.model,
                                       env.unwrapped.frame_skip,
                                       actor,C,FAULT_INNER_CLIP)
        for seed in SEEDS:
            for arm in ARM_NAMES:
                rows.append(_episode(env,actor,planner,seed,arm))
    if len(rows)!=len(SEEDS)*len(ARM_NAMES):
        raise AssertionError("incomplete three-arm panel")
    metrics={}
    for arm in ARM_NAMES:
        data=[r for r in rows if r["arm"]==arm]
        metrics[arm]={
            "episodes":len(data),
            "mean_original_reward":sum(r["official_episode_return"] for r in data)/len(data),
            "mean_object_goal_distance":sum(r["mean_physical_object_goal_distance"] for r in data)/len(data),
            "max_true_reward_prediction_error":max(
                r["max_selected_action_reward_prediction_error"] for r in data)}
    def paired(arm):
        effect=[]
        distance=[]
        for seed in SEEDS:
            healthy=next(x for x in rows if x["seed"]==seed and x["arm"]=="official-healthy")
            baseline=next(x for x in rows if x["seed"]==seed and x["arm"]==arm)
            effect.append(healthy["official_episode_return"]-
                          baseline["official_episode_return"])
            distance.append(baseline["mean_physical_object_goal_distance"]-
                            healthy["mean_physical_object_goal_distance"])
        return {
            "mean_source_reward_advantage":sum(effect)/len(effect),
            "source_reward_wins":sum(x>1e-8 for x in effect),
            "source_reward_losses":sum(x<-1e-8 for x in effect),
            "source_reward_ties":sum(abs(x)<=1e-8 for x in effect),
            "source_object_goal_distance_reduction":sum(distance)/len(distance),
            "all_source_minus_oracle_reward_differences":effect}
    return {
        "protocol":"pusher-privileged-true-native-three-step-frozen-policy-tail-mpc-v1",
        "official_robot":ENV,"checkpoint_sha256":sha,
        "actor_is_frozen_third_party_SAC":True,
        "source_repair_arm_is_author_implemented_healthy_reference":True,
        "old_seen_seed_count":len(SEEDS),"new_holdout_seeds":0,
        "independent_hardware_manufacturers":0,
        "fault_source_shear":FAULT_SHEAR,
        "fault_source_internal_clip":FAULT_INNER_CLIP,
        "horizon_physical_steps":HORIZON,
        "fully_known_native_physics":True,
        "fully_known_hidden_interface":True,
        "optimizes_exact_original_official_reward":True,
        "globally_optimal_for_100_step_reward":False,
        "first_step_candidate_grid_is_finite":True,
        "tail_policy_is_frozen_SAC_with_exact_one_step_action_projection":True,
        "real_native_reward_candidate_rollouts":planner.rollout_evaluations,
        "native_mujoco_internal_integration_steps":planner.physics_integration_steps,
        "tail_actor_predictions":planner.policy_tail_predictions,
        "scenario_sha256":sha256(json.dumps({
            "seeds":SEEDS,"fault":(FAULT_SHEAR,FAULT_INNER_CLIP),
            "horizon":HORIZON,"arms":ARM_NAMES,
            "proposals":FUTURE_CONTROL_GRID,
            "policy_sha256":sha},sort_keys=True).encode()).hexdigest(),
        "summaries":metrics,
        "paired_vs_healthy":{
            "exact-instantaneous-action-oracle":paired("exact-instantaneous-action-oracle"),
            "privileged-native-three-step-tail-MPC":paired("privileged-native-three-step-tail-MPC")},
        "raw_rows":rows,
        "limitations":"retrospective 16 seeds / one severe authored fault / finite exact-physics 3-step lookahead, NOT unrestricted full-horizon blackbox optimum"
    }


if __name__=="__main__":
    print(json.dumps(run_external_pusher_three_step_mpc(),indent=2,sort_keys=True))
