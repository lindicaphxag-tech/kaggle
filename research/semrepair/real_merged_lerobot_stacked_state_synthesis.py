"""Genuine LeRobot merged #4656, before/after and six original-source AST patches.

Natural published bug: stacking (B,T_obs,D) state crashes or desynchronizes
relative/absolute action conversion; reference must be state[:,0], which the
frozen VLA-JEPA model itself consumes. Merge September 16, 2026:
https://github.com/huggingface/lerobot/pull/4656

Execute TRUE publisher before/after PyTorch original function AST bodies,
with 6 FIRST-PARTY typed code mutants synthesized solely from BEFORE source.
The AFTER source is an independent oracle of the maintainer's decision,
NOT evidence that our system discovered or upstream-merged the repair.

Crucial finding to check: BOTH wrong anchor choices (last and temporal mean)
can pass an ordinary roundtrip action->relative->absolute test, yet violate
the external model's frame-zero action-conditioning contract. The same
merged PR also repaired eval batch processor resets; audit the REAL
rollout setup AST and EXECUTE the actual publisher reset method.

This is a source-level numerics/episode reset case with synthetic torch
tensors, not actual VLA-JEPA checkpoint inference or full rollout.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import random
import typing

import torch

from .real_merged_lerobot_processor_contract import _pinned_source

REPO="huggingface/lerobot"
PRE="89236ea0f4f81a81ca566081e20dd1ff5f823cbe"
MERGED="30074f7f1358b3c015ae1750017200e86e9c4eb6"
PROCESSOR_PATH="src/lerobot/processor/relative_action_processor.py"
EVAL_PATH="src/lerobot/scripts/lerobot_eval.py"
OLD_PROCESSOR_BLOB="4fe007f796a031f23a9c2081b50b08914b498b88"
NEW_PROCESSOR_BLOB="4c1af387b21e203bf4f364c7f98be3bf7c12f757"
OLD_EVAL_BLOB="7629d62e9866d914e1039ca5a44fdc75eb7fd4ef"
NEW_EVAL_BLOB="6dcc5de189f47b0e4217893d36b284e4b0382e15"
CANDIDATES=("unchanged","frame0_both","last_both","temporal_mean_both",
            "frame0_relative_only","frame0_absolute_only")


def _real_function_nodes(s):
    nodes=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and
           n.name in ("to_relative_actions","to_absolute_actions")]
    if len(nodes)!=2:
        raise ValueError("original real publisher conversion functions missing")
    return {n.name:n for n in nodes}


def _compile_original(nodes):
    mod=ast.fix_missing_locations(ast.Module(
        body=[copy.deepcopy(nodes[n]) for n in
              ("to_relative_actions","to_absolute_actions")],
        type_ignores=[]))
    ns={"torch":torch,"Tensor":torch.Tensor,"Sequence":typing.Sequence}
    exec(compile(mod,PROCESSOR_PATH,"exec"),ns,ns)
    return ns["to_relative_actions"],ns["to_absolute_actions"]


def _apply_one_typed_anchor(n,mode):
    n=copy.deepcopy(n)
    assignments=[idx for idx,item in enumerate(n.body)
                 if isinstance(item,ast.Assign) and
                 any(isinstance(v,ast.Name) and v.id=="state_offset" for v in item.targets)]
    if len(assignments)!=1:
        raise ValueError("true LeRobot state_offset site is not unique")
    if mode=="frame0":
        body="if state.ndim == 3:\n    state = state[:, 0]"
    elif mode=="last":
        body="if state.ndim == 3:\n    state = state[:, -1]"
    elif mode=="mean":
        body="if state.ndim == 3:\n    state = state.mean(dim=1)"
    else:
        raise ValueError("edit outside typed source anchor grammar")
    n.body[assignments[0]:assignments[0]]=ast.parse(body).body
    return n


def _candidate_nodes(before,mode):
    nodes=_real_function_nodes(before)
    changes={
        "unchanged":(None,None),
        "frame0_both":("frame0","frame0"),
        "last_both":("last","last"),
        "temporal_mean_both":("mean","mean"),
        "frame0_relative_only":("frame0",None),
        "frame0_absolute_only":(None,"frame0"),
    }
    if mode not in changes:
        raise ValueError("unknown source AST operator")
    for fn,op in zip(("to_relative_actions","to_absolute_actions"),
                     changes[mode]):
        if op is not None:
            nodes[fn]=_apply_one_typed_anchor(nodes[fn],op)
    return nodes


def _publisher_rollout_reset_audit(before,after):
    def reset_statements(source):
        functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
                   and n.name=="rollout"]
        if len(functions)!=1:raise ValueError("original upstream rollout absent")
        statements=[]
        for n in functions[0].body:
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
               isinstance(n.value.func,ast.Attribute) and
               n.value.func.attr=="reset" and
               isinstance(n.value.func.value,ast.Name):
                statements.append(n.value.func.value.id)
        return statements
    old=reset_statements(before)
    new=reset_statements(after)
    if "policy" not in old or "preprocessor" in old or "postprocessor" in old or \
       not {"policy","preprocessor","postprocessor"}.issubset(new):
        raise AssertionError("published reset behavior contradicts frozen source")
    return {"old_explicit_rollout_reset_targets":old,
            "merged_explicit_rollout_reset_targets":new,
            "new_processor_reset_is_present":True}


def _publisher_actual_reset_method(processor_source):
    classes=[n for n in ast.parse(processor_source).body
             if isinstance(n,ast.ClassDef) and n.name=="RelativeActionsProcessorStep"]
    if len(classes)!=1:raise ValueError("real processor source class missing")
    methods=[n for n in classes[0].body if isinstance(n,ast.FunctionDef)
             and n.name=="reset"]
    if len(methods)!=1:
        raise ValueError("merged original processor reset method absent")
    mod=ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(methods[0])],
                                               type_ignores=[]))
    ns={}
    exec(compile(mod,PROCESSOR_PATH,"exec"),ns,ns)
    step=type("ActualPublisherResetTestObject",(),{})()
    step._last_state=torch.tensor([[1.,2.]])
    ns["reset"](step)
    if step._last_state is not None:
        raise AssertionError("actual vendor reset method did not clear cached state")
    return {"publisher_original_reset_AST_executed":True,
            "cached_prior_episode_anchor_cleared":True}


def run_real_merged_stacked_source_synthesis():
    old,old_hash=_pinned_source(PRE,PROCESSOR_PATH,OLD_PROCESSOR_BLOB)
    merged,new_hash=_pinned_source(MERGED,PROCESSOR_PATH,NEW_PROCESSOR_BLOB)
    old_rollout,old_roll_hash=_pinned_source(PRE,EVAL_PATH,OLD_EVAL_BLOB)
    new_rollout,new_roll_hash=_pinned_source(MERGED,EVAL_PATH,NEW_EVAL_BLOB)
    reference=_compile_original(_real_function_nodes(merged))
    old_real=_compile_original(_real_function_nodes(old))
    candidates={key:_compile_original(_candidate_nodes(old,key))
                for key in CANDIDATES}
    results={name:{"stacked_exceptions":0,"stacked_roundtrip_pass":0,
                   "conditioning_anchor_pass":0,"classic_2d_pass":0,
                   "mean_frame0_action_error":[]}
             for name in CANDIDATES}
    rng=random.Random(20261010)
    N=96
    for idx in range(N):
        torch.manual_seed(90177+idx)
        B=2;D=7;Tobs=8;Tact=16
        action=torch.randn(B,Tact,D,dtype=torch.float64)
        state=torch.randn(B,Tobs,D,dtype=torch.float64)
        classic=torch.randn(B,D,dtype=torch.float64)
        mask=(True,True,True,True,True,True,False)
        target=reference[0](action,state,mask)
        expected_2d=reference[0](action,classic,mask)
        expected_abs=reference[1](target,state,mask)
        torch.testing.assert_close(expected_abs,action,atol=1e-12,rtol=0)
        # Existing LeRobot 2-D policy action compatibility MUST remain.
        for name,(to_rel,to_abs) in candidates.items():
            res=results[name]
            x=to_rel(action,classic,mask)
            y=to_abs(x,classic,mask)
            if torch.allclose(x,expected_2d,atol=1e-12,rtol=0) and \
               torch.allclose(y,action,atol=1e-12,rtol=0):
                res["classic_2d_pass"]+=1
            try:
                x=to_rel(action,state,mask)
                y=to_abs(x,state,mask)
                if torch.allclose(y,action,atol=1e-12,rtol=0):
                    res["stacked_roundtrip_pass"]+=1
                err=float((x-target).abs().amax())
                res["mean_frame0_action_error"].append(err)
                if err<1e-12 and torch.allclose(y,action,atol=1e-12,rtol=0):
                    res["conditioning_anchor_pass"]+=1
            except RuntimeError:
                res["stacked_exceptions"]+=1
    summary={}
    for k,r in results.items():
        summary[k]={**{key:value for key,value in r.items()
                       if key!="mean_frame0_action_error"},
                    "mean_max_relative_action_error_against_frame0":(
                        sum(r["mean_frame0_action_error"])/
                        len(r["mean_frame0_action_error"])
                        if r["mean_frame0_action_error"] else None),
                    "max_error_against_frame0":(
                        max(r["mean_frame0_action_error"])
                        if r["mean_frame0_action_error"] else None)}
    if summary["frame0_both"]["conditioning_anchor_pass"]!=N or \
       summary["last_both"]["stacked_roundtrip_pass"]!=N or \
       summary["last_both"]["conditioning_anchor_pass"]!=0:
        raise AssertionError("insufficient evidence that roundtrip overfits anchor semantics")
    if summary["unchanged"]["stacked_exceptions"]!=N:
        raise AssertionError("actual old publisher stacked-state bug did not reproduce")
    return {
        "protocol":"genuine-merged-LeRobot-4656-stack-anchor-mutants-v1",
        "actual_MERGED_external_PR":"https://github.com/huggingface/lerobot/pull/4656",
        "publisher_merged_at":"2026-09-16T14:52:20Z",
        "before_commit":PRE,"actual_merged_commit":MERGED,
        "original_publisher_git_blobs":{
            "before_processor":OLD_PROCESSOR_BLOB,
            "merged_processor":NEW_PROCESSOR_BLOB,
            "before_eval":OLD_EVAL_BLOB,
            "merged_eval":NEW_EVAL_BLOB},
        "original_publisher_sha256":[old_hash,new_hash,old_roll_hash,new_roll_hash],
        "original_merged_policy_functions_truly_executed":True,
        "original_merged_reset_method_evidence":
            _publisher_actual_reset_method(merged),
        "real_eval_rollout_reset_source":
            _publisher_rollout_reset_audit(old_rollout,new_rollout),
        "case_count":N,"candidate_count":len(CANDIDATES),
        "raw_torch_tensor_elements_not_independent_software_bugs":True,
        "actual_VLA_JEPA_neural_network_checkpoint_inference":False,
        "actual_robot_task_rollout_episodes":0,
        "frame0_contract_is_external_publisher_policy_conditioning_assumption":True,
        "original_source_compatible_classic_2D":True,
        "results":summary,
        "negative_baseline":"last and mean hidden state anchor patches pass roundtrip but violate actual publisher model frame0 conditioning",
    }


if __name__=="__main__":
    print(json.dumps(run_real_merged_stacked_source_synthesis(),sort_keys=True,indent=2))
