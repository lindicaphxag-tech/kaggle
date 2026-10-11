"""Patient-style *episode-seed paired* auditing for two real published SAC policies.

This is a descriptive heldout paired design, NOT an independent device-level
trial: five severity arms on the SAME seeds are correlated and two policies
from the SAME publisher share the same environment. Multiple comparisons
are handled conservatively, using a five-contrast Bonferroni correction.

The paired bootstrap over 32 fixed pseudorandom environment resets is a
conditional sensitivity summary, NOT a distribution-free safety certificate.
Never report per-step timesteps as independent examples.
"""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
from math import comb,isfinite
from random import Random
import json
import sys

AUDIT_SEED=20261010
BOOTSTRAP_REPLICATES=4000


def exact_two_sided_sign_p(positive:int,negative:int)->float:
    if positive<0 or negative<0:
        raise ValueError("invalid sign counts")
    n=positive+negative
    if n==0:return 1.0
    k=min(positive,negative)
    return min(1.,2.*sum(comb(n,j) for j in range(k+1))/(2**n))


def paired_bootstrap_ci(values:list[float])->tuple[float,float]:
    if not values or any(not isfinite(x) for x in values):
        raise ValueError("invalid paired reward differences")
    rng=Random(AUDIT_SEED)
    n=len(values)
    avg=[]
    for _ in range(BOOTSTRAP_REPLICATES):
        avg.append(sum(values[rng.randrange(n)] for _ in range(n))/n)
    avg.sort()
    return (avg[int(.025*BOOTSTRAP_REPLICATES)],
            avg[int(.975*BOOTSTRAP_REPLICATES)-1])


def audit_seeds_and_paired_effects(data:dict)->dict:
    rows=data.get("raw_rows",data.get("rows"))
    if not rows:
        raise ValueError("missing raw paired trial outcomes")
    source_name="blind-source-patch-candidate"
    comparator="exact-all-clip-blackbox-oracle"
    expected_seeds=set(data.get("holdout_seeds",range(23011,23043)))
    if len(expected_seeds)!=32:
        raise ValueError("expected 32 independent reset identities")
    severity=sorted({r["fault_family"] for r in rows})
    if len(severity)!=5:
        raise ValueError("must retain ALL five predeclared fault magnitudes")
    bykey={}
    for row in rows:
        key=(row["fault_family"],row["seed"],row["arm"])
        if key in bykey:raise ValueError("duplicate seed, fault, intervention")
        if not all(isfinite(row[k]) for k in ("return","mean_tip_distance","final_tip_distance")):
            raise ValueError("nonfinite native task observation")
        bykey[key]=row
    group_reports={}
    for world in severity:
        scores=[]
        distances=[]
        signs={"positive":0,"negative":0,"ties":0}
        for seed in sorted(expected_seeds):
            a=bykey.get((world,seed,source_name))
            b=bykey.get((world,seed,comparator))
            if a is None or b is None:
                raise ValueError("missing source-oracle paired physical trial")
            d=a["return"]-b["return"]
            scores.append(d)
            distances.append(b["mean_tip_distance"]-a["mean_tip_distance"])
            signs["positive"]+=int(d>1e-8)
            signs["negative"]+=int(d< -1e-8)
            signs["ties"]+=int(abs(d)<=1e-8)
        if len(scores)!=32:raise AssertionError("wrong per-severity denominator")
        lo,hi=paired_bootstrap_ci(scores)
        p=exact_two_sided_sign_p(signs["positive"],signs["negative"])
        group_reports[world]={
            "paired_reset_unit":"seed; not separate robot/physical body",
            "reset_count":len(scores),
            "mean_official_reward_gain":sum(scores)/len(scores),
            "source_better":signs["positive"],
            "source_worse":signs["negative"],
            "exact_ties":signs["ties"],
            "two_sided_sign_p_excluding_ties":p,
            "bonferroni_adjusted_p_for_five_fixed_levels":min(1.,5*p),
            "paired_seed_bootstrap_95pct_descriptive":(lo,hi),
            "mean_tip_distance_reduction":sum(distances)/len(distances)}
    return {
        "protocol":"external-frozen-reacher-seed-paired-five-level-audit-v1",
        "source_model_checkpoint_sha256":data.get("checkpoint_sha256",
                                                 data.get("downloaded_checkpoint_sha256")),
        "comparisons_by_severity":group_reports,
        "familywise_comparisons":5,
        "no_independent_robot_or_controller_family_generalization":True,
        "bootstrap_not_iid_robot_certification":True,
        "oracle_only_instantaneous_action_optimal":True,
        "raw_sha256":sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}


if __name__=="__main__":
    if len(sys.argv)!=2:
        raise SystemExit("Usage: python -m research.semrepair.paired_external_sac_audit RAW.json")
    with open(sys.argv[1],encoding="utf-8") as f:raw=json.load(f)
    print(json.dumps(audit_seeds_and_paired_effects(raw),indent=2,sort_keys=True))
