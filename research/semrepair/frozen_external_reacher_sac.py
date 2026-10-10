"""External frozen SAC policy on Farama Gymnasium's official Reacher-v5.

Unlike first-party PD controllers, this loads the exact published Farama/Minari
SAC expert checkpoint, verifies external repository commit, and runs original
Reacher-v5 rewards. Test seeds and faults are predeclared. The policy is NEVER
trained, fine-tuned, adapted or given hidden interface configuration labels.

Fault: a two-stage action adapter t=C clip(P*u,-1,1), where CP=I before
intermediate clipping. A correct atomic source repair removes both wrong
components; all arms then enter official Reacher's actuator [-1,1] interface.
The external *policy* and robot mechanics are unmodified. Candidate repairs
are source-known upper bounds: source provenance is NOT authenticated or
automatically inferred on this external model.

An oracle pointwise inner-action projector minimizes the Euclidean action
mismatch before final Reacher clipping, with exact convex parallelogram
projection. It is NOT globally optimal for reward, and the outer clip can
change the exact feasible geometry; do not claim this is the global best
black-box action adapter. All methods are evaluated at the same rewards.

If external artifact is missing, the job FAILS rather than silently
substituting a locally trained or randomly initialized model.
"""
from __future__ import annotations
from hashlib import sha256
import json
from math import dist
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from huggingface_hub import hf_hub_download
from .internal_saturation_reachability import exact_blackbox_torque_oracle
from .strong_clipped_actuator_oracle import (
    output_reachability_polygons,optimal_clipped_output
)
from .observable_interface_factorization import mv, mmul, I

# The official Farama repository commit ID, not 'main' / changing latest.
HF_REPO="farama-minari/Reacher-v5-SAC-expert"
HF_REVISION="ad8fadfbe3ab40ce9c1bae5b33641041056b3090"
HF_FILENAME="reacher-v5-sac-expert.zip"
ENV_NAME="Reacher-v5"
ENV_SEEDS=(1001,1012,1023,1034,1045,1056,1067,1078,1089,1090,1101,1112,
           1123,1134,1145,1156)
FAULT_MAGNITUDE=.8
INNER_CONTROL_LIMIT=1.0
P=((1.,FAULT_MAGNITUDE),(0.,1.))
C=((1.,-FAULT_MAGNITUDE),(0.,1.))
ARMS=("healthy","compensating-fault","matched-linear-inverse",
      "full-information-inner-geometry-projection",
      "exact-two-clip-action-oracle","correct-source-atomic-upper-bound")
GOAL_DISTANCE_THRESHOLD=.075
EXACT_REACHABILITY_POLYGONS=output_reachability_polygons(
    C,inner_bound=INNER_CONTROL_LIMIT,outer_bound=1.)



def frozen_external_checkpoint():
    checkpoint=hf_hub_download(repo_id=HF_REPO,filename=HF_FILENAME,revision=HF_REVISION)
    raw_sha=sha256(Path(checkpoint).read_bytes()).hexdigest()
    policy=SAC.load(checkpoint,device="cpu")
    if tuple(policy.observation_space.shape)!=(10,) or \
       tuple(policy.action_space.shape)!=(2,):
        raise ValueError("upstream checkpoint observation/action contract mismatch")
    return policy,raw_sha


def _transform(action:np.ndarray,arm:str)->np.ndarray:
    ideal=(float(action[0]),float(action[1]))
    if arm in ("healthy","correct-source-atomic-upper-bound"):
        return np.asarray(ideal,dtype=np.float32)
    if arm=="matched-linear-inverse":
        # The matched blackbox gets exact same source CP product (identity).
        upstream=ideal
        # CP=I before clipping; it must still pass via the nonrepairable
        # inner clipped actuator interface.
        inner=np.clip(mv(P,upstream),-INNER_CONTROL_LIMIT,INNER_CONTROL_LIMIT)
        actual=mv(C,tuple(float(x) for x in inner))
    elif arm=="compensating-fault":
        inner=np.clip(mv(P,ideal),-INNER_CONTROL_LIMIT,INNER_CONTROL_LIMIT)
        actual=mv(C,tuple(float(x) for x in inner))
    elif arm=="full-information-inner-geometry-projection":
        actual=exact_blackbox_torque_oracle(ideal,controller=C,
                                             limit=INNER_CONTROL_LIMIT)["closest_actuator_torque"]
    elif arm=="exact-two-clip-action-oracle":
        actual=optimal_clipped_output(ideal,EXACT_REACHABILITY_POLYGONS)["action"]
    else:
        raise ValueError("unknown intervention")
    return np.asarray(actual,dtype=np.float32)


def execute_frozen_episode(env,policy,seed:int,arm:str)->dict:
    state,info=env.reset(seed=seed)
    frozen_actions=[]
    rewards=[]
    distal=[]
    deviations=[]
    for index in range(50):
        proposed,_=policy.predict(state,deterministic=True)
        delivered=_transform(proposed,arm)
        # Physical Reacher-v5 clips to [-1,1]; preserve actual posted torque
        applied=np.clip(delivered,env.action_space.low,env.action_space.high)
        next_state,reward,terminated,truncated,information=env.step(applied)
        # Reacher-v5 includes relative fingertip-target vector in last 3
        # components. Compute actual Euclidean distal error directly from
        # the underlying official MuJoCo x-position sites instead.
        unwrapped=env.unwrapped
        tip=np.asarray(unwrapped.get_body_com("fingertip"),dtype=float)
        target=np.asarray(unwrapped.get_body_com("target"),dtype=float)
        ee_error=float(np.linalg.norm(tip-target))
        if abs(ee_error + float(information["reward_dist"]))>1e-7:
            raise AssertionError("tip/target geometry inconsistent with official reward")
        distal.append(ee_error)
        rewards.append(float(reward))
        deviations.append(float(np.linalg.norm(np.asarray(proposed)-applied)))
        frozen_actions.append([float(x) for x in applied])
        state=next_state
        if terminated or truncated:
            break
    if len(rewards)!=50:
        raise RuntimeError("unexpected native Reacher time limit/truncation")
    return {"seed":seed,"arm":arm,"steps":len(rewards),
            "episode_return":sum(rewards),
            "mean_tip_target_distance":sum(distal)/len(distal),
            "final_tip_target_distance":distal[-1],
            "min_tip_target_distance":min(distal),
            "ever_reached_0075":bool(min(distal)<=GOAL_DISTANCE_THRESHOLD),
            "mean_policy_action_error":sum(deviations)/len(deviations),
            "peak_policy_action_error":max(deviations),
            "rewards":rewards,"tip_distances":distal,
            "delivered_actions":frozen_actions}


def run_official_reacher_external_policy()->dict:
    # Reacher-v5 is third-party Farama's official model, NOT our custom
    # two-link physics model. The SAC expert is a downloaded frozen artifact.
    if mmul(C,P)!=I:
        raise AssertionError("fault must compensate only before clipping")
    policy,model_sha=frozen_external_checkpoint()
    all_rows=[]
    with gym.make(ENV_NAME) as env:
        for seed in ENV_SEEDS:
            for arm in ARMS:
                all_rows.append(execute_frozen_episode(env,policy,seed,arm))
    results={}
    for arm in ARMS:
        entries=[row for row in all_rows if row["arm"]==arm]
        results[arm]={
            "episodes":len(entries),
            "mean_original_return":sum(r["episode_return"] for r in entries)/len(entries),
            "mean_tip_target_distance":sum(r["mean_tip_target_distance"] for r in entries)/len(entries),
            "mean_final_tip_target_distance":sum(r["final_tip_target_distance"] for r in entries)/len(entries),
            "ever_reached_0075_count":sum(r["ever_reached_0075"] for r in entries),
            "mean_policy_action_error":sum(r["mean_policy_action_error"] for r in entries)/len(entries)}
    for seed in ENV_SEEDS:
        ref=next(r for r in all_rows if r["seed"]==seed and r["arm"]=="healthy")
        fixed=next(r for r in all_rows if r["seed"]==seed and r["arm"]=="correct-source-atomic-upper-bound")
        if ref["delivered_actions"]!=fixed["delivered_actions"] or \
           abs(ref["episode_return"]-fixed["episode_return"])>1e-10:
            raise AssertionError("source-correct arm no longer exactly matches healthy model")
    return {
        "protocol":"semrepair-external-frozen-sb3-sac-reacher-v5-v1",
        "third_party_environment":ENV_NAME,
        "third_party_policy_repo":HF_REPO,
        "third_party_policy_commit":HF_REVISION,
        "checkpoint_sha256":model_sha,
        "gymnasium_version":gym.__version__,
        "policy_frozen":True,"policy_training_steps_by_authors":0,
        "source_model_groups":1,
        "model_independent_pretrained":True,
        "seed_count":len(ENV_SEEDS),
        "action_fault":"two-part source ABI inverse shear with internal clip [-1,1] applied in adapter outside official model",
        "same_initial_seed_across_arms":True,
        "source_fix_is_oracle_upper_bound_not_authenticated":True,
        "blackbox_torque_oracle_not_global_task_optimum":True,
        "exact_double_clip_oracle_is_globally_optimal_in_instantaneous_action_error":True,
        "exact_double_clip_oracle_not_global_future_reward_optimum":True,
        "blackbox_inner_projection_ignores_outer_clipping_geometry":True,
        "stronger_two_clip_oracle_included":True,
        "original_official_reward_preserved":True,
        "scenario_sha256":sha256(json.dumps({
            "seeds":ENV_SEEDS,"policy_commit":HF_REVISION,"arms":ARMS,
            "P":P,"C":C,"control_limit":INNER_CONTROL_LIMIT},sort_keys=True).encode()).hexdigest(),
        "summary":results,"rows":all_rows,
        "limitations":"one official Reacher environment and one published SAC checkpoint; fixed engineered interface bug, no runtime code patch or authenticated origin, no independent source family; no claims of optimal task controller",
    }


if __name__=="__main__":
    print(json.dumps(run_official_reacher_external_policy(),indent=2,sort_keys=True))
