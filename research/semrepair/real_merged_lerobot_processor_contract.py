"""Examine a GENUINE MERGED external LeRobot multi-file program repair.

Independent upstream report:
  https://github.com/huggingface/lerobot/issues/3425

Upstream accepted after multiple alternative UNMERGED patch proposals:
  https://github.com/huggingface/lerobot/pull/3711
  merged 2026-06-03; merge commit:
  19fe315971079e116b917ed4d482bfbf3c93bbbe

This audit runs the ACTUAL original Python source's processor-selection
statements, before vs after, in isolated narrow Python execution contexts;
all robotics checkpoint/GPU objects are REPLACED BY RECORDED STUBS.
It verifies byte/Git-blob identity, original control flow, and preservation
of source/config processor provenance; no model checkpoint is downloaded,
no training run is executed and vendor source is not modified.

Unlike an arm-level action error, the failure is an inter-component
PROCESSOR PROVENANCE invariant:
  relative=true + pretrained checkpoint -> keep checkpoint's paired
  pretrained processor configuration and add explicit version-compatible
  relative/absolute and rename overrides, NOT silently rebuild a fresh
  processor pipeline that discards checkpoint-specific transforms.

This is a stronger JSS/IST real-world "candidate patch overfitting" case,
NOT an independent accepted review of our method.
"""
from __future__ import annotations

import ast
from hashlib import sha1,sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json
import textwrap
import urllib.request

REPO="huggingface/lerobot"
BEFORE_COMMIT="906b585826c0e34235a347554abfb94e0d65484b"
AFTER_COMMIT="19fe315971079e116b917ed4d482bfbf3c93bbbe"
SCRIPT_PATH="src/lerobot/scripts/lerobot_train.py"
BEFORE_SOURCE_BLOB="463668eb25fefbf1814cd9bc0a855e06ffd93093"
AFTER_SOURCE_BLOB="4ddef3105bb5ac5164dd08c5e629f78834bbd5e0"
UPSTREAM_ISSUE="https://github.com/huggingface/lerobot/issues/3425"
UPSTREAM_MERGED_PR="https://github.com/huggingface/lerobot/pull/3711"
N_SEMANTIC_CASES=6


def _pinned_source(commit:str,path:str,expected_blob:str):
    url=f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}"
    with urllib.request.urlopen(url,timeout=45) as r:
        b=r.read()
    if len(b)<1000 or len(b)>150000:
        raise ValueError("invalid or missing exact upstream publisher source")
    git_sha=sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
    if git_sha!=expected_blob:
        raise ValueError("publisher content does not match pinned immutable git blob")
    return b.decode("utf-8"),sha256(b).hexdigest()


def _extract_original_executable_setup(src):
    """Use original AST statements verbatim; no recreated if predicates."""
    parsed=ast.parse(src,filename=SCRIPT_PATH)
    fns=[n for n in parsed.body if isinstance(n,ast.FunctionDef)
         and n.name=="train"]
    if len(fns)!=1:
        raise ValueError("original LeRobot train function not unique")
    body=fns[0].body
    starts=[i for i,n in enumerate(body)
            if isinstance(n,ast.Assign) and len(n.targets)==1 and
            isinstance(n.targets[0],ast.Name) and n.targets[0].id=="active_cfg"]
    if len(starts)!=1:
        raise ValueError("original policy/processor provenance block not located")
    start=starts[0]
    ends=[i for i in range(start+1,len(body))
          if isinstance(body[i],ast.If) and
          isinstance(body[i].test,ast.Name) and
          body[i].test.id=="is_main_process"]
    if not ends:
        raise ValueError("original processor factory invocation boundary not found")
    stop=ends[0]
    stmts=body[start:stop]
    # Contract: this block MUST call the original maker (no fake rewritten
    # conditional code). All expensive environment code lies outside it.
    if not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and
               n.func.id=="make_pre_post_processors"
               for s in stmts for n in ast.walk(s)):
        raise ValueError("original factory call not contained in extracted module")
    func=ast.FunctionDef(
        name="actual_publisher_processor_setup",
        args=ast.arguments(posonlyargs=[],args=[
            ast.arg(arg=x) for x in
            ("cfg","dataset","policy","device","make_pre_post_processors",
             "make_reward_pre_post_processors","logging")],
            vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),
        body=[*stmts,ast.Return(value=ast.Tuple(elts=[
            ast.Name(id="processor_pretrained_path",ctx=ast.Load()),
            ast.Name(id="processor_kwargs",ctx=ast.Load())
        ],ctx=ast.Load()))],
        decorator_list=[],returns=None,type_comment=None)
    mod=ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[]))
    namespace={}
    exec(compile(mod,SCRIPT_PATH,"exec"),namespace,namespace)
    return namespace["actual_publisher_processor_setup"],len(stmts)


def _execute_real_pre_and_post_source(fn,*,pretrained_path,relative,rename_map,
                                      resume=False,reward_model=False):
    active=SimpleNamespace(
        pretrained_path=pretrained_path,use_relative_actions=relative,
        relative_exclude_joints=["gripper"],action_feature_names=["action"])
    cfg=SimpleNamespace(
        trainable_config=active,resume=resume,
        is_reward_model_training=reward_model,
        rename_map=rename_map,
        reward_model=None,policy=SimpleNamespace(type="pi05"))
    dataset=SimpleNamespace(meta=SimpleNamespace(
        stats={"observation.images.front":{"mean":0.5,"std":0.2}}))
    policy=SimpleNamespace(config=SimpleNamespace(
        input_features={"observation.images.left_wrist_0_rgb":"camera"},
        output_features={"action":"action"},
        normalization_mapping={"VISUAL":"IDENTITY"}))
    device=SimpleNamespace(type="cpu")
    interactions=[]
    def maker(**kwargs):
        interactions.append(kwargs)
        return "publisher-preprocessor","publisher-postprocessor"
    def reward_maker(*args,**kwargs):
        interactions.append({"reward":True,**kwargs})
        return "reward-preprocessor","reward-postprocessor"
    import logging
    final_path,proc_kwargs=fn(cfg,dataset,policy,device,maker,
                              reward_maker,logging)
    if len(interactions)!=1:
        raise AssertionError("original factory called incorrectly")
    # Original maker arguments are recorded without carrying non-serializable
    # stub namespace objects into immutable JSON evidence.
    incoming={k:(dict(vars(v)) if isinstance(v,SimpleNamespace) else v)
              for k,v in interactions[0].items()}
    return {
        "controller_checkpoint_path_requested":pretrained_path,
        "processor_checkpoint_path_actually_used":final_path,
        "checkpoint_provenance_preserved":bool(pretrained_path) and
                                          pretrained_path==final_path,
        "relative_requested":bool(relative),"resume":bool(resume),
        "is_reward_model_training":bool(reward_model),
        "rename_requested":bool(rename_map),
        "actual_original_factory_kwargs":incoming,
        "original_processor_config_kwargs":proc_kwargs,
    }


def _model_invariant(out,*,required_named_observations=True,
                     need_relative_pipeline=True):
    """Conservative simulated *consumer contract*, not pretrained inference."""
    if not out["controller_checkpoint_path_requested"]:
        return {"decision":"SCRATCH_NO_CHECKPOINT_CONTRACT","valid":True}
    if not out["checkpoint_provenance_preserved"]:
        return {"decision":"FAIL_LOST_PRETRAINED_PROCESSOR_PROVENANCE",
                "valid":False}
    kw=out["actual_original_factory_kwargs"]
    pre=kw.get("preprocessor_overrides") or {}
    post=kw.get("postprocessor_overrides") or {}
    if required_named_observations and out["rename_requested"]:
        renamed=pre.get("rename_observations_processor",{}).get("rename_map")
        if not renamed:
            return {"decision":"FAIL_MISSING_RENAME_OVERRIDES","valid":False}
    if out["relative_requested"] and need_relative_pipeline:
        rel=pre.get("relative_actions_processor",{})
        abs_=post.get("absolute_actions_processor",{})
        if not rel.get("enabled") or not abs_.get("enabled"):
            return {"decision":"FAIL_MISSING_RELATIVE_ABSOLUTE_PROCESSOR_OVERRIDES",
                    "valid":False}
    return {"decision":"PASS_RECORDED_PROCESSOR_CONTRACT","valid":True}


def audit_actual_merged_lerobot_source():
    before,before_hash=_pinned_source(BEFORE_COMMIT,SCRIPT_PATH,BEFORE_SOURCE_BLOB)
    after,after_hash=_pinned_source(AFTER_COMMIT,SCRIPT_PATH,AFTER_SOURCE_BLOB)
    src_fns={};statement_counts={}
    for name,source in (("pre_merge",before),("maintainer_merged",after)):
        fn,n=_extract_original_executable_setup(source)
        src_fns[name]=fn
        statement_counts[name]=n
    cases=[
        # Main reported original bug: relative training with mapped images.
        ("pretrained_relative_with_renames",
         dict(pretrained_path="external/PI05-checkpoint",relative=True,
              rename_map={"raw.camera":"observation.images.left_wrist_0_rgb"})),
        ("pretrained_relative_no_renames",
         dict(pretrained_path="external/PI05-checkpoint",relative=True,
              rename_map=None)),
        ("pretrained_absolute_with_renames",
         dict(pretrained_path="external/PI05-checkpoint",relative=False,
              rename_map={"raw.camera":"observation.images.left_wrist_0_rgb"})),
        ("fresh_relative_no_renames",
         dict(pretrained_path=None,relative=True,rename_map=None)),
        ("pretrained_resume_relative_with_renames",
         dict(pretrained_path="external/PI05-checkpoint",relative=True,
              rename_map={"raw.camera":"observation.images.left_wrist_0_rgb"},
              resume=True)),
        ("fresh_absolute_no_renames",
         dict(pretrained_path=None,relative=False,rename_map=None)),
    ]
    records=[]
    for label,args in cases:
        older=_execute_real_pre_and_post_source(src_fns["pre_merge"],**args)
        newer=_execute_real_pre_and_post_source(src_fns["maintainer_merged"],**args)
        records.append({
            "scenario":label,"is_full_model_training":False,
            "before":older,"after":newer,
            "old_software_contract":_model_invariant(older),
            "new_software_contract":_model_invariant(newer),
        })
    reported=records[0]
    if reported["old_software_contract"]["valid"] or not \
       reported["new_software_contract"]["valid"]:
        raise AssertionError("source-level bug/fix contract contradicts genuine merged PR")
    return {
        "protocol":"real-HF-LeRobot-3425-before-after-merged-3711-processor-provenance-v1",
        "publisher_repository":REPO,
        "public_issue":UPSTREAM_ISSUE,
        "actual_MERGED_publisher_PR":UPSTREAM_MERGED_PR,
        "merge_confirmed_date":"2026-06-03",
        "exact_before_git_commit":BEFORE_COMMIT,
        "exact_after_merge_commit":AFTER_COMMIT,
        "verified_before_actual_source_git_blob":BEFORE_SOURCE_BLOB,
        "verified_after_actual_source_git_blob":AFTER_SOURCE_BLOB,
        "before_actual_source_sha256":before_hash,
        "after_actual_source_sha256":after_hash,
        "source_AST_statements_executed":statement_counts,
        "frozen_original_publisher_train_setup_source_ran":True,
        "programmatically_reimplemented_factory":True,
        "actual_pretrained_model_loaded":False,
        "actual_robot_task_or_training_steps_run":0,
        "scenario_count":len(records),
        "merger_reported_bug_before_pass":reported["old_software_contract"]["valid"],
        "merger_reported_bug_after_pass":reported["new_software_contract"]["valid"],
        "summary_passed_scenarios_before":sum(x["old_software_contract"]["valid"]
                                               for x in records),
        "summary_passed_scenarios_after":sum(x["new_software_contract"]["valid"]
                                              for x in records),
        "scenarios":records,
        "limitations":"AST-extracted original control-flow blocks with mocked factory/checkpoints, not full VLA training; synthetic contract assertions informed by actual merged PR, one bug family",
    }


if __name__=="__main__":
    print(json.dumps(audit_actual_merged_lerobot_source(),
                     indent=2,sort_keys=True))
