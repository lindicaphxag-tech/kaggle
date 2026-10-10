"""Second independently published frozen SAC checkpoint, identical Reacher protocol.

No hyperparameter or fault retuning from expert heldout results. The SAME
32 goals and five fixed fault levels are evaluated against a different
publisher-provided SAC-medium checkpoint. This is cross-checkpoint *within*
one official robot environment, not cross-embodiment or a second ML algorithm.

The expert results were already observed before this validation was added,
so this is an explicit sequential replication, not a blindly preregistered
multi-policy design.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json
import gymnasium as gym
from stable_baselines3 import SAC
from huggingface_hub import hf_hub_download
from .external_reacher_sealed_holdout import (
    FAULTS,HOLDOUT_SEEDS,ARMS,ENV_NAME,_matrices,
    identify_blind_fault,_episode
)
from .strong_clipped_actuator_oracle import output_reachability_polygons

PUBLISHER_REPO="farama-minari/Reacher-v5-SAC-medium"
PUBLISHER_SHA="7366bc42655652e36705974ec234a6456d695222"
PUBLISHER_FILE="reacher-v5-sac-medium.zip"


def load_external_medium():
    file=hf_hub_download(repo_id=PUBLISHER_REPO,filename=PUBLISHER_FILE,
                         revision=PUBLISHER_SHA)
    checksum=sha256(Path(file).read_bytes()).hexdigest()
    policy=SAC.load(file,device="cpu")
    if tuple(policy.observation_space.shape)!=(10,) or \
       tuple(policy.action_space.shape)!=(2,):
        raise ValueError("external SAC-medium input/output mismatch")
    return policy,checksum


def run_medium_checkpoint_replication():
    policy,model_hash=load_external_medium()
    data=[]
    reports={}
    with gym.make(ENV_NAME) as env:
        for name,k,L in FAULTS:
            P,C=_matrices(k)
            candidate=identify_blind_fault(P,C,L)
            if candidate["candidate_patch"]!="atomic":
                raise AssertionError("candidate ABI changes across model checkpoints")
            polygons=output_reachability_polygons(C,inner_bound=L,outer_bound=1.)
            for seed in HOLDOUT_SEEDS:
                for arm in ARMS:
                    row=_episode(env,policy,seed,arm,P,C,L,polygons)
                    row["fault_family"]=name
                    data.append(row)
            for arm in ARMS:
                values=[r for r in data if r["fault_family"]==name and r["arm"]==arm]
                reports[f"{name}|{arm}"]={
                    "episodes":len(values),
                    "mean_original_reward":sum(r["return"] for r in values)/len(values),
                    "mean_tip_distance":sum(r["mean_tip_distance"] for r in values)/len(values),
                    "hit_count":sum(r["ever_hit_0075"] for r in values),
                    "counterfactual_clip_events_in_actor_states":sum(
                        r["inner_clip_events_when_policy_ideal_passes_producer"] for r in values)}
    if len(data)!=len(HOLDOUT_SEEDS)*len(FAULTS)*len(ARMS):
        raise AssertionError("incomplete peer checkpoint replication")
    result={}
    for name,k,L in FAULTS:
        reference=reports[f"{name}|official-healthy"]
        repaired=reports[f"{name}|blind-source-patch-candidate"]
        if abs(reference["mean_original_reward"]-repaired["mean_original_reward"])>1e-10:
            raise AssertionError("source candidate did not reproduce healthy external policy")
        differences=[]
        for seed in HOLDOUT_SEEDS:
            source=next(r["return"] for r in data if r["seed"]==seed and
                        r["fault_family"]==name and r["arm"]=="blind-source-patch-candidate")
            oracle=next(r["return"] for r in data if r["seed"]==seed and
                        r["fault_family"]==name and r["arm"]=="exact-all-clip-blackbox-oracle")
            differences.append(source-oracle)
        result[name]={
            "source_minus_pointwise_optimal_blackbox_mean_return":sum(differences)/len(differences),
            "positive_seed_count":sum(d>1e-8 for d in differences),
            "negative_seed_count":sum(d< -1e-8 for d in differences),
            "ties":sum(abs(d)<=1e-8 for d in differences),
            "paired_differences":differences}
    return {
        "protocol":"external-medium-sac-checkpoint-replication-v1",
        "second_pretrained_publisher":PUBLISHER_REPO,
        "checkpoint_revision":PUBLISHER_SHA,
        "downloaded_checkpoint_sha256":model_hash,
        "policy_training_steps_by_authors":0,
        "same_faults_seeds_and_baselines_as_expert":True,
        "independent_robot_embodiments_added":0,
        "independent_policy_checkpoints_added":1,
        "not_blinded_to_expert_results":True,
        "fixed_original_official_reward":True,
        "total_episode_count":len(data),
        "summary":reports,"paired":result,"rows":data,
        "limitation":"same Reacher-v5 mechanical model, two SAC models from same publisher, source authenticity and actual patch installation unimplemented",
    }


if __name__=="__main__":
    print(json.dumps(run_medium_checkpoint_replication(),indent=2,sort_keys=True))
