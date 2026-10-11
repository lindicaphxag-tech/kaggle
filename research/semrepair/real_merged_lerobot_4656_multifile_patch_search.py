"""Real upstream MERGED LeRobot #4656: jointly repair temporal frame and eval cache.

Actual publisher commit 30074f7f1358b3c015ae1750017200e86e9c4eb6
and base 89236ea0f4f81a81ca566081e20dd1ff5f823cbe, original Git
blob SHA1 checked. The two-file fix must:
(1) choose state[:,0] for BOTH relative/absolute action functions; and
(2) call preprocessor.reset() and postprocessor.reset() at new eval batch.

An output-only action->relative->absolute roundtrip may accept "last"
or temporal mean anchors, even though the actual VLA-JEPA policy consumes
frame ZERO. A single-chunk input/output test may also pass when the reset
path leaks a cached prior-batch anchor. We enumerate 6 typed source
function AST variations x 2 typed original EVAL reset AST edits = 12
executable source programs, select based on independent frame and epoch
contracts and declared per-edit cost. All mutation grammar terms were
defined after publisher source merge, so this is RETROSPECTIVE template
case analysis, not blinded automated discovery of the upstream fix.
Original publisher full VLA checkpoint / robot evaluation not run.
"""
from __future__ import annotations
import ast
import copy
import json
from types import SimpleNamespace
from .real_merged_lerobot_processor_contract import _pinned_source
from .real_merged_lerobot_stacked_state_synthesis import (
    PRE,MERGED,PROCESSOR_PATH,EVAL_PATH,OLD_PROCESSOR_BLOB,
    NEW_PROCESSOR_BLOB,OLD_EVAL_BLOB,NEW_EVAL_BLOB,
    CANDIDATES,run_real_merged_stacked_source_synthesis,
    _publisher_actual_reset_method
)

N=96


def _rollout_reset_ast(s):
    tree=ast.parse(s)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)
           and n.name=="rollout"]
    if len(funcs)!=1:raise ValueError("real vendor rollout procedure changed")
    reset=[n for n in funcs[0].body if isinstance(n,ast.Expr)
           and isinstance(n.value,ast.Call)
           and isinstance(n.value.func,ast.Attribute)
           and isinstance(n.value.func.value,ast.Name)
           and n.value.func.value.id=="policy"
           and n.value.func.attr=="reset"]
    if len(reset)!=1:raise ValueError("real vendor baseline policy reset site absent")
    return reset[0]


def _mutate_actual_rollout_reset(src,repair):
    if not isinstance(repair,bool):
        raise ValueError("typed real rollout reset edit required")
    source=src
    node=_rollout_reset_ast(src)
    if repair:
        lines=source.splitlines(keepends=True)
        addition="    preprocessor.reset()\n    postprocessor.reset()\n"
        source="".join(lines[:node.end_lineno])+addition+\
               "".join(lines[node.end_lineno:])
        ast.parse(source)
    return source


def _exec_real_rollout_source_reset_segment(source):
    tree=ast.parse(source)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)
           and n.name=="rollout"]
    if len(funcs)!=1:raise ValueError("true publisher rollout func missing")
    body=funcs[0].body
    selected=[]
    for node in body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and \
           isinstance(node.value.func,ast.Attribute) and \
           node.value.func.attr=="reset" and \
           isinstance(node.value.func.value,ast.Name) and \
           node.value.func.value.id in {"policy","preprocessor","postprocessor"}:
            selected.append(copy.deepcopy(node))
    fn=ast.FunctionDef(name="vendor_original_rollout_resets",
                       args=ast.arguments(posonlyargs=[],args=[
                           ast.arg(arg=x) for x in
                           ("policy","preprocessor","postprocessor")],
                           vararg=None,kwonlyargs=[],kw_defaults=[],
                           kwarg=None,defaults=[]),
                       body=selected or [ast.Pass()],decorator_list=[],
                       returns=None,type_comment=None)
    env={}
    tree=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    exec(compile(tree,EVAL_PATH,"exec"),env,env)
    log=[]
    class Tracer:
        def __init__(self,name):self.name=name
        def reset(self):log.append(self.name)
    env["vendor_original_rollout_resets"](
        Tracer("policy"),Tracer("preprocessor"),Tracer("postprocessor"))
    return log


def run_real_4656_joint_temporal_source_repairs():
    # Independent already source-byte-checked original Torch 96-case
    # executions; this runs real source BEFORE lookup of MERGED reference.
    numerical=run_real_merged_stacked_source_synthesis()
    old_eval,old_sha=_pinned_source(PRE,EVAL_PATH,OLD_EVAL_BLOB)
    original_processor,processor_sha=_pinned_source(
        PRE,PROCESSOR_PATH,OLD_PROCESSOR_BLOB)
    variants=[]
    for name in CANDIDATES:
        action=numerical["results"][name]
        for repair_reset in (False,True):
            edited=_mutate_actual_rollout_reset(old_eval,repair_reset)
            actual_reset=_exec_real_rollout_source_reset_segment(edited)
            reset_contract=(actual_reset==["policy","preprocessor",
                                           "postprocessor"])
            frame_contract=(action["conditioning_anchor_pass"]==N and
                            action["stacked_exceptions"]==0 and
                            action["classic_2d_pass"]==N)
            # Difference from test-only: roundtrip does not prove model
            # input-frame source semantics or cross-episode reset.
            weak_output_only=(action["stacked_roundtrip_pass"]==N)
            full=frame_contract and reset_contract
            variants.append({
                "converter_AST_variant":name,
                "eval_reset_AST_modified":repair_reset,
                "actual_vendor_AST_reset_calls_executed":actual_reset,
                "weak_roundtrip_only_plausible":weak_output_only,
                "actual_cross_component_frame0_contract":frame_contract,
                "actual_new_epoch_cache_reset_contract":reset_contract,
                "full_source_semantic_contract":full,
                "declared_source_edit_cost":(0 if name=="unchanged"
                                             else (1 if name.endswith("_only")
                                                   else 2))+
                                            (2 if repair_reset else 0),
                "source_files_modified":int(name!="unchanged")+
                                         int(repair_reset),
            })
    wins=[x for x in variants if x["full_source_semantic_contract"]]
    winner=min(wins,key=lambda x:(
        x["declared_source_edit_cost"],
        x["source_files_modified"],
        x["converter_AST_variant"]
    )) if wins else None
    # Publisher reference AFTER candidates are constructed; never use
    # publisher merged bytes to synthesize an author candidate.
    merged_eval,_=_pinned_source(MERGED,EVAL_PATH,NEW_EVAL_BLOB)
    publisher_reset=_exec_real_rollout_source_reset_segment(merged_eval)
    if publisher_reset!=["policy","preprocessor","postprocessor"]:
        raise AssertionError("authentic upstream merged reset semantics unexpected")
    if winner is None or winner["converter_AST_variant"]!="frame0_both" or \
       not winner["eval_reset_AST_modified"] or len(wins)!=1:
        raise AssertionError("true multi-file source contract not uniquely repaired")
    return {
        "protocol":"LeRobot-genuine-merged-4656-temporal-two-source-file-12-mutant-search-v1",
        "genuinely_publisher_MERGED_PR":"https://github.com/huggingface/lerobot/pull/4656",
        "author_claims_upstream_patch_was_ours":False,
        "original_before_source_git_blobs":[OLD_PROCESSOR_BLOB,OLD_EVAL_BLOB],
        "publisher_merged_source_git_blobs":[NEW_PROCESSOR_BLOB,NEW_EVAL_BLOB],
        "actual_source_evaluator_executes_original_AST":True,
        "actual_VLA_policy_network_loaded":False,
        "native_robot_task_trials":0,
        "independent_num_source_cases_per_candidate":N,
        "typed_two_file_candidate_program_count":len(variants),
        "single_roundtrip_test_plausible_program_count":sum(
            x["weak_roundtrip_only_plausible"] for x in variants),
        "full_crosscomponent_semantics_valid_count":len(wins),
        "weak_plausible_but_semantically_invalid_program_count":sum(
            x["weak_roundtrip_only_plausible"] and
            not x["full_source_semantic_contract"] for x in variants),
        "minimum_cost_full_semantics_candidate":winner,
        "authentic_publisher_actual_reset_statements":publisher_reset,
        "original_before_eval_sha256":old_sha,
        "original_before_processor_sha256":processor_sha,
        "all_program_variants":variants,
        "limits":"typed candidate grammar and real source resets; no full model inference; cost units asserted, not measured runtime",
    }


if __name__=="__main__":
    print(json.dumps(run_real_4656_joint_temporal_source_repairs(),
                     indent=2,sort_keys=True))
