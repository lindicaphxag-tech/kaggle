"""Genuine LeRobot 2026-09-22 MERGED sync/RTC compensating layout defect.

This executes ORIGINAL external LeRobot SyncInferenceEngine.get_action,
make_robot_action, and final aligned context/guard AST, frozen to before
and after Git commits of real upstream PR #4452.

A single policy output tensor is ordered according to the trained
checkpoint. Original get_action maps through dataset feature-name order
and then reorders, introducing a net permutation when dataset and
checkpoint action names differ. Correct merged source returns policy
tensor unmodified and aligns BOTH state and action feature names at
context construction. This is a real source-level regression test, NOT
a live dual-robot VLA rollout, ROS middleware test or our upstream PR.

The author-created stub policy emits distinct per-joint fixed actions;
all named vendor source computations are AST-extracted from genuine
published source files. Equal-information input remapping may compensate
a permutation under fixed known joint orders, so success is not proof of
fundamental blackbox action irrecoverability.
"""
from __future__ import annotations
import ast
from contextlib import nullcontext
from copy import copy
import json
import random
from types import SimpleNamespace
import typing
import torch

from .real_merged_lerobot_processor_contract import _pinned_source

PRE="697ceebd7696fa66f5e831f573c475cb9fe3367a"
MERGED="0120909ed1e1df860217fefa78755015b0315744"
SYNC="src/lerobot/rollout/inference/sync.py"
CONTEXT="src/lerobot/rollout/context.py"
POLICY_UTIL="src/lerobot/policies/utils.py"
OLD_SYNC="30631b7939cdd68595c7140f86924746f649226a"
NEW_SYNC="7ca61466a1ab1c10ac4d843692d0beb7dec55d3c"
OLD_CONTEXT="7fc3c0d7e402af745a1a6dfcf64da0aab1b0162a"
NEW_CONTEXT="e097c0146145f011022ed41efb5f0caf3bf5a8c5"
PUBLISHER_POLICY_HELPER="63d6b724375ed96b183640da4022ed4b3b3e62ce"


def _exact_node(s,func_name,class_name=None):
    tree=ast.parse(s)
    body=tree.body
    if class_name:
        classes=[n for n in body if isinstance(n,ast.ClassDef)
                 and n.name==class_name]
        if len(classes)!=1:raise ValueError("publisher class identity changed")
        body=classes[0].body
    nodes=[n for n in body if isinstance(n,ast.FunctionDef)
           and n.name==func_name]
    if len(nodes)!=1:raise ValueError("publisher function source absent")
    return nodes[0]


def _compile_vendor(node,namespace):
    tree=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    env=dict(namespace)
    exec(compile(tree,"pinned-real-publisher-source.py","exec"),env,env)
    return env[node.name]


def _true_helpers(source,old_context,new_context):
    make_node=_exact_node(source,"make_robot_action")
    maker=_compile_vendor(make_node,{
        "ACTION":"action","PolicyAction":torch.Tensor,"RobotAction":dict,
        "torch":torch,
    })
    guard=_compile_vendor(
        _exact_node(new_context,"_assert_state_matches_action_order"),
        {"OBS_STATE":"observation.state"})
    align=_compile_vendor(_exact_node(new_context,"_align_to_checkpoint_order"),
                          {"logger":SimpleNamespace(warning=lambda *a,**kw:None)})
    return maker,guard,align


def _actual_sync_method(src,maker):
    node=_exact_node(src,"get_action","SyncInferenceEngine")
    return _compile_vendor(node,{
        "torch":torch,"copy":copy,"nullcontext":nullcontext,
        "logger":SimpleNamespace(info=lambda *a,**kw:None),
        "prepare_observation_for_inference":lambda o,*args:o,
        "make_robot_action":maker,
    })


def _fake_operator(policy_action,checkpoint_order,hardware_order):
    """All actual source methods run; only expensive policy is stubbed."""
    return SimpleNamespace(
        _device=torch.device("cpu"),
        _policy=SimpleNamespace(
            config=SimpleNamespace(use_amp=False),
            select_action=lambda obs:policy_action.unsqueeze(0),
            drop_queued_actions=lambda:None),
        _robot_type="two-mirrored-arms",
        _preprocessor=lambda x:x,_postprocessor=lambda x:x,
        _take_task=lambda:("test",False),
        _set_dispatched_task=lambda name:None,
        _dataset_features={"action":{"names":hardware_order}},
        _ordered_action_keys=checkpoint_order,
    )


def run_real_merged_compensating_order_issue():
    old_s,old_sha=_pinned_source(PRE,SYNC,OLD_SYNC)
    new_s,new_sha=_pinned_source(MERGED,SYNC,NEW_SYNC)
    old_c,old_csha=_pinned_source(PRE,CONTEXT,OLD_CONTEXT)
    new_c,new_csha=_pinned_source(MERGED,CONTEXT,NEW_CONTEXT)
    helper,hsha=_pinned_source(PRE,POLICY_UTIL,PUBLISHER_POLICY_HELPER)
    maker,guard,align=_true_helpers(helper,old_c,new_c)
    old_method=_actual_sync_method(old_s,maker)
    fixed_method=_actual_sync_method(new_s,maker)
    left=[f"left_{i}.pos" for i in range(4)]
    right=[f"right_{i}.pos" for i in range(4)]
    hardware=left+right
    checkpoint=right+left
    raw=[]
    rng=random.Random(20261010)
    for seed in range(96):
        # Every component numerical value identifies its own policy joint.
        action=torch.tensor([
            (i+1)*10+rng.uniform(-.025,.025) for i in range(8)],
            dtype=torch.float64)
        obj=_fake_operator(action,checkpoint,hardware)
        before=old_method(obj,{"observation.state":"stub"})
        after=fixed_method(obj,{"observation.state":"stub"})
        ctrl_old=tuple(float(v) for v in before)
        ctrl_new=tuple(float(v) for v in after)
        reference=tuple(float(v) for v in action)
        err_old=max(abs(a-b) for a,b in zip(ctrl_old,reference))
        err_new=max(abs(a-b) for a,b in zip(ctrl_new,reference))
        raw.append({"case":seed,
                    "actual_original_sync_max_abs_error":err_old,
                    "actual_merged_sync_max_abs_error":err_new,
                    "before_first_joint_action":ctrl_old[0],
                    "after_first_joint_action":ctrl_new[0]})
    if not all(x["actual_original_sync_max_abs_error"]>20 and
               x["actual_merged_sync_max_abs_error"]<1e-12 for x in raw):
        raise AssertionError("did not reproduce the genuine upstream order bug")
    old_name_layout={name:float for name in hardware}
    new_state=align(old_name_layout,checkpoint,what="state")
    new_action=align(old_name_layout,checkpoint,what="action")
    if list(new_state)!=checkpoint or list(new_action)!=checkpoint:
        raise AssertionError("real source context did not align named layouts")
    rejected=False
    try:
        guard({"observation.state":{"names":hardware}},checkpoint)
    except ValueError:
        rejected=True
    if not rejected:
        raise AssertionError("real source order guard did not fail closed")
    # Legitimate different scalar state set is not a broken permutation.
    guard({"observation.state":{"names":hardware+["base.vel"]}},checkpoint)
    aligned_ok=True
    guard({"observation.state":{"names":checkpoint}},checkpoint)
    # Negative known-hardware order controls: if layouts are already the
    # same, old source is not automatically worse than merged source.
    sample=torch.tensor([float(i+1) for i in range(8)],dtype=torch.float64)
    old_hardware=_fake_operator(sample,hardware,hardware)
    old_healthy=old_method(old_hardware,{"observation.state":"stub"})
    if not torch.allclose(old_healthy,sample,atol=0,rtol=0):
        raise AssertionError("false-positive source repair in already aligned state")
    return {
        "protocol":"real-merged-LeRobot-sync-RTC-action-name-compensation-4452-v1",
        "real_independently_MERGED_upstream_PR":"https://github.com/huggingface/lerobot/pull/4452",
        "real_merger_date":"2026-09-22",
        "before_commit":PRE,"external_merge_commit":MERGED,
        "Git_blobs":[OLD_SYNC,NEW_SYNC,OLD_CONTEXT,NEW_CONTEXT,PUBLISHER_POLICY_HELPER],
        "Git_content_SHA256":[old_sha,new_sha,old_csha,new_csha,hsha],
        "real_vendor_sync_implementation_before_and_after_executed":True,
        "real_vendor_make_robot_action_implementation_executed":True,
        "real_merged_context_alignment_and_fail_closed_guard_executed":True,
        "correct_automatic_state_action_name_reordering":aligned_ok,
        "true_state_action_permutation_refused":rejected,
        "different_state_sensor_feature_set_not_refused":True,
        "no_robot_or_pretrained_policy_rollout_executed":True,
        "distinct_source_regression_samples":len(raw),
        "baseline_original_mean_action_error":sum(
            x["actual_original_sync_max_abs_error"] for x in raw)/len(raw),
        "source_corrected_max_action_error":max(
            x["actual_merged_sync_max_abs_error"] for x in raw),
        "already_aligned_original_remains_correct":True,
        "raw_cases":raw,
        "limitation":"true original publisher AST, stubbed policy actions not real bimanual robot hardware or VLA",
    }


if __name__=="__main__":
    print(json.dumps(run_real_merged_compensating_order_issue(),indent=2,sort_keys=True))
