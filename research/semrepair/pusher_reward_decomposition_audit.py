"""Independent from-executed-artifact Pusher-v5 task improvement audit.

Unlike old two-axis Reacher success, Pusher contains THREE weighted original
reward components; the reviewer MUST know whether an apparent advantage
reflects actual OBJECT movement to goal, fingertip approach, or simply
avoiding energy penalty. This script starts ONLY from raw official environment
per-episode records; no rerunning environments, refitting severities or
dropping negative cases.

Two original endpoints, official cumulative reward and object-to-goal
distance, across five predeclared severities: 10 simultaneous contrasts.
We Bonferroni-adjust exact two-sided paired sign tests across all TEN.
The seed bootstrap is descriptive conditional on one official robot model.
"""
from __future__ import annotations
from hashlib import sha256
from math import comb,isfinite
from random import Random
import json
import sys

SEVERITIES=("mild-control-pair","shear-2","shear-3point5",
            "shear-2-plus-clip","shear-3point5-plus-clip")
SEEDS=tuple(range(23011,23043))
METHOD="blind-numeric-source-candidate"
ORACLE="exact-two-clip-full-information-oracle"
HEALTHY="official-healthy"
CONTROL_WEIGHT=.1
NEAR_WEIGHT=.5
GOAL_WEIGHT=1.
T=100

def exact_sign_test(effects,epsilon=1e-8):
    wins=sum(x>epsilon for x in effects)
    losses=sum(x< -epsilon for x in effects)
    tied=len(effects)-wins-losses
    n=wins+losses
    if n==0:return {"p":1.,"wins":wins,"losses":losses,"ties":tied}
    k=min(wins,losses)
    p=min(1.,2.*sum(comb(n,i) for i in range(k+1))/(2**n))
    return {"p":p,"wins":wins,"losses":losses,"ties":tied}

def bootstrap_paired(effects,*,replicates=4000):
    rng=Random(20261010)
    n=len(effects)
    means=[]
    for i in range(replicates):
        means.append(sum(effects[rng.randrange(n)] for _ in range(n))/n)
    means.sort()
    return [means[int(.025*replicates)],means[int(.975*replicates)-1]]

def audit_pusher(full):
    rows=full["raw_rows"]
    if len(rows)!=32*5*6:
        raise ValueError("all 960 paired original task outcomes required")
    if full.get("official_env")!="Pusher-v5" or full.get("native_action_dimension")!=7:
        raise ValueError("cross-robot evidence contract mismatch")
    if set(a[0] for a in full["fixed_fault_grid"])!=set(SEVERITIES):
        raise ValueError("missing or altered predeclared fault severity")
    groups={}
    for entry in rows:
        key=(entry["scenario"],entry["seed"],entry["arm"])
        if key in groups:raise ValueError("same world/seed/intervention appears twice")
        if not all(isfinite(entry[x]) for x in (
            "official_episode_return","mean_object_goal_distance",
            "mean_fingertip_object_distance","mean_official_control_reward")):
            raise ValueError("invalid original reward decomposition")
        if entry.get("steps")!=T:
            raise ValueError("truncated or reweighted Pusher task episode")
        groups[key]=entry
    expected={(scenario,seed,arm)
              for scenario in SEVERITIES
              for seed in SEEDS
              for arm in full["per_arm"] if False}
    # We also insist each of all six actual actions is present.
    declared=tuple(full.get("scenario_arms",(
        "official-healthy","untouched-compensating-source-fault",
        "composite-inverse-blackbox","exact-two-clip-full-information-oracle",
        "blind-numeric-source-candidate","source-known-atomic-upper-bound")))
    if len(declared)!=6 or len(set(declared))!=6:
        raise ValueError("six intervention identities required")
    must={(scenario,seed,arm) for scenario in SEVERITIES for seed in SEEDS for arm in declared}
    if set(groups)!=must:
        raise ValueError("incomplete original Pusher fault/seed/arm inventory")

    reports={}
    for scenario in SEVERITIES:
        gain=[]
        better_object=[]
        contact_gain=[]
        energy_gain=[]
        for seed in SEEDS:
            source=groups[(scenario,seed,METHOD)]
            oracle=groups[(scenario,seed,ORACLE)]
            ref=groups[(scenario,seed,HEALTHY)]
            if abs(source["official_episode_return"]-ref["official_episode_return"])>1e-7:
                raise ValueError("source candidate does not recreate healthy external task")
            gain.append(source["official_episode_return"]-oracle["official_episode_return"])
            better_object.append(oracle["mean_object_goal_distance"]-
                                 source["mean_object_goal_distance"])
            contact_gain.append(oracle["mean_fingertip_object_distance"]-
                                source["mean_fingertip_object_distance"])
            energy_gain.append(source["mean_official_control_reward"]-
                               oracle["mean_official_control_reward"])
            expected_return_delta=T*(GOAL_WEIGHT*better_object[-1]+
                                    NEAR_WEIGHT*contact_gain[-1]+energy_gain[-1])
            if abs(gain[-1]-expected_return_delta)>5e-5:
                raise ValueError("official Pusher return gain does not equal declared reward components")
        metrics={}
        for label,arr in (("official_return_gain",gain),
                          ("actual_object_goal_distance_reduction",better_object)):
            p=exact_sign_test(arr)
            metrics[label]={
                "mean":sum(arr)/len(arr),"bootstrap_95pct_descriptive":bootstrap_paired(arr),
                "two_sided_exact_sign_p":p["p"],
                "ten_contrast_bonferroni_p":min(1.,10*p["p"]),
                "positive_seed_count":p["wins"],
                "negative_seed_count":p["losses"],"tied_seed_count":p["ties"]}
        reports[scenario]={
            "paired_seed_n":len(SEEDS),
            "endpoints":metrics,
            "mean_contact_distance_reduction":sum(contact_gain)/len(contact_gain),
            "mean_weighted_control_reward_gain_per_step":sum(energy_gain)/len(energy_gain),
            "net_official_reward_components_close_to_exactly_reported":True,
        }
    return {
        "protocol":"recompute-seven-actuator-pusher-fivefault-original-reward-attribution-v1",
        "physical_models":1,"independent_state_reset_seeds":32,
        "familywise_sign_test_contrasts":10,
        "two_joint_control_source_fault_authored_not_external":True,
        "significance_not_independent_robot_manufacturers":True,
        "raw_rows_sha256":sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        "results":reports,
    }

if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("Usage: -m module original-pusher-raw.json")
    with open(sys.argv[1],encoding="utf-8") as f:raw=json.load(f)
    print(json.dumps(audit_pusher(raw),indent=2,sort_keys=True))
