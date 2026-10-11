"""ALL source-authority policies on 64 pairs of original LeRobot code.

One example pair has two hidden history worlds with same relative chunk and
current observation. NO fake independent robotics samples: 64 author-
constructed pairs, two worlds each, 8 correlated chunk steps, original
frozen publisher conversion kernel used for ALL real actions.

Compared policies with equal information:
- always use current anchor: known max error = pair Chebyshev radius;
- radius-aware current only: REFUSE if cannot meet strict 0.01 tolerance;
- trusted source history: execute correct original LeRobot function;
- trusted externally cached history: SAME exact performance as source,
  CHEAPER under declared test costs (counterexample to source necessity);
- stale cache, missing source trust: REFUSE;
- tolerance >= exact worst-case radius: current-only is sufficient.

This is a software semantics certificate study, not VLA simulator success.
"""
from __future__ import annotations
import json
import random
import torch
from .lerobot_temporal_anchor_case import (
    load_actual_upstream_function_pair,WORLD_PAIRS,CHUNK_HORIZON,
    ACTION_CHANNELS,MASK,UPSTREAM_COMMIT,UPSTREAM_GIT_BLOB_SHA1
)
from .temporal_evidence_repair_operator import (
    select_information_authority,execute_original_lerobot_conversion
)

TOLERANCE=.01
COST_SOURCE=1.
COST_CACHE=.1
COST_CURRENT=0.


def run_temporal_source_evidence_benchmark():
    _,to_abs,external_source_sha=load_actual_upstream_function_pair()
    rng=random.Random(20261010)
    outputs=[]
    for pair in range(WORLD_PAIRS):
        current=[rng.uniform(-.4,.4) for _ in range(ACTION_CHANNELS)]
        offset=[rng.uniform(.04,.28)*rng.choice((-1,1))
                for _ in range(ACTION_CHANNELS)]
        offset[-1]=0.
        chunk=[[rng.uniform(-.3,.3) for _ in range(ACTION_CHANNELS)]
               for _ in range(CHUNK_HORIZON)]
        alternatives=[]
        for sign in (-1,+1):
            a=[current[i]+sign*offset[i] for i in range(ACTION_CHANNELS)]
            alternatives.append(torch.tensor([a],dtype=torch.float64))
        now=torch.tensor([current],dtype=torch.float64)
        relative=torch.tensor([chunk],dtype=torch.float64)
        # The expected absolute commands here are ACTUAL upstream function
        # outputs. No reference to hand-written q_rel + anchor formulas.
        radius=max(abs(x) for x in offset[:6])
        assert radius > TOLERANCE
        for branch,actual in enumerate(alternatives):
            desired=to_abs(relative,actual,MASK)
            current_only=to_abs(relative,now,MASK)
            err_current=float((current_only-desired).abs().amax())
            # Real historical cache is independently provisioned. If a
            # blackbox retained chunk-start observations, it can tie source.
            cache_dec,cache_ref=select_information_authority(
                relative=relative,current=now,mask=MASK,
                possible_historical_anchors=alternatives,
                allowable_max_action_error=TOLERANCE,
                historical_cache=actual,cache_verified=True,
                cache_epoch=pair,chunk_epoch=pair,
                source_anchor=actual,source_anchor_verified=True,
                source_epoch=pair,
                cost_cache=COST_CACHE,cost_source=COST_SOURCE)
            cache=execute_original_lerobot_conversion(to_abs,relative,cache_dec,
                                                       cache_ref,MASK)
            source_dec,source_ref=select_information_authority(
                relative=relative,current=now,mask=MASK,
                possible_historical_anchors=alternatives,
                allowable_max_action_error=TOLERANCE,
                cache_verified=False,
                source_anchor=actual,source_anchor_verified=True,
                source_epoch=pair,chunk_epoch=pair,
                cost_cache=COST_CACHE,cost_source=COST_SOURCE)
            source=execute_original_lerobot_conversion(to_abs,relative,source_dec,
                                                        source_ref,MASK)
            noauth,_=select_information_authority(
                relative=relative,current=now,mask=MASK,
                possible_historical_anchors=alternatives,
                allowable_max_action_error=TOLERANCE,
                cache_verified=False,source_anchor_verified=False,
                chunk_epoch=pair)
            stale,_=select_information_authority(
                relative=relative,current=now,mask=MASK,
                possible_historical_anchors=alternatives,
                allowable_max_action_error=TOLERANCE,
                historical_cache=actual,cache_verified=True,
                cache_epoch=pair+1,source_anchor_verified=False,
                chunk_epoch=pair)
            generous,guess=select_information_authority(
                relative=relative,current=now,mask=MASK,
                possible_historical_anchors=alternatives,
                allowable_max_action_error=radius+1e-10,
                historical_cache=actual,cache_verified=True,
                cache_epoch=pair,source_anchor=actual,
                source_anchor_verified=True,source_epoch=pair,
                chunk_epoch=pair)
            approx=execute_original_lerobot_conversion(to_abs,relative,generous,guess,MASK)
            outputs.append({
                "pair":pair,"world_branch":-1 if branch==0 else 1,
                "true_minimax_worst_error":radius,
                "current_only_error":err_current,
                "historical_cache_recovery_error":float((cache-desired).abs().amax()),
                "authorized_source_recovery_error":float((source-desired).abs().amax()),
                "generous_tolerance_current_only_error":float((approx-desired).abs().amax()),
                "historical_cache_operation":cache_dec.operation,
                "source_operation":source_dec.operation,
                "no_credential_action_decision":noauth.outcome,
                "stale_history_action_decision":stale.outcome,
                "generous_tolerance_operation":generous.operation,
                "cached_history_cost":cache_dec.monetary_or_permission_cost,
                "source_history_cost":source_dec.monetary_or_permission_cost,
                "source_file_was_actually_modified":source_dec.source_file_modified,
                "independent_real_robot_execution":False,
            })
    if len(outputs)!=2*WORLD_PAIRS:
        raise AssertionError("missing pair branch")
    if any(x["authorized_source_recovery_error"]>1e-12 or
           x["historical_cache_recovery_error"]>1e-12 for x in outputs):
        raise AssertionError("source and same-information blackbox must both be exact")
    if any(x["no_credential_action_decision"]!="REFUSE_UNCERTIFIED_CURRENT_STATE" or
           x["stale_history_action_decision"]!="REFUSE_UNCERTIFIED_CURRENT_STATE" for x in outputs):
        raise AssertionError("faulty current-only action incorrectly authorized")
    if any(x["generous_tolerance_operation"]!="CURRENT_ONLY_MINIMAX" for x in outputs):
        raise AssertionError("cheap current-only action should pass loose tolerance")
    return {
        "protocol":"source-authority-temporal-evidence-four-ablation-20261010",
        "independent_real_publisher_repo":"huggingface/lerobot",
        "verified_git_commit":UPSTREAM_COMMIT,
        "verified_source_blob_sha1":UPSTREAM_GIT_BLOB_SHA1,
        "verified_source_bytes_sha256":external_source_sha,
        "origin_of_fault":"author-injected stale temporal chunk anchor callsite",
        "original_vendor_bug_reported":False,
        "source_bytes_actually_modified":False,
        "real_physical_robot_evaluations":0,
        "scenario_worlds":len(outputs),
        "pair_count":WORLD_PAIRS,
        "chunk_steps":CHUNK_HORIZON,
        "correlated_action_outputs_not_independent_trials":len(outputs)*CHUNK_HORIZON,
        "strict_tolerance":TOLERANCE,
        "mean_current_only_max_error":sum(x["current_only_error"] for x in outputs)/len(outputs),
        "mean_minimax_theory_lower_bound":sum(x["true_minimax_worst_error"] for x in outputs)/len(outputs),
        "max_authorized_source_error":max(x["authorized_source_recovery_error"] for x in outputs),
        "max_history_enabled_blackbox_error":max(x["historical_cache_recovery_error"] for x in outputs),
        "untrusted_action_refusals":sum(x["no_credential_action_decision"]==
                                         "REFUSE_UNCERTIFIED_CURRENT_STATE" for x in outputs),
        "stale_epoch_refusals":sum(x["stale_history_action_decision"]==
                                   "REFUSE_UNCERTIFIED_CURRENT_STATE" for x in outputs),
        "history_enabled_blackbox_matches_source":all(
            abs(x["historical_cache_recovery_error"]-x["authorized_source_recovery_error"])<1e-12
            for x in outputs),
        "source_repair_is_not_necessary_when_valid_cache_available":True,
        "minimum_declared_cost_source_only":COST_SOURCE,
        "minimum_declared_cost_with_valid_cache":COST_CACHE,
        "strict_unauthorized_false_action_count":0,
        "rows":outputs,
        "limitations":"synthetic two-world information problem using real unchanged vendor functions, costs author supplied, credentials externally assumed and no source patch installed",
    }


if __name__=="__main__":
    print(json.dumps(run_temporal_source_evidence_benchmark(),indent=2,sort_keys=True))
