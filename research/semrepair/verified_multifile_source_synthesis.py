"""Finite multi-file repair search on TRUE frozen LeRobot #3425 source.

Search 16 combinations of four author-specified source edit operators,
executing actual old-source train AST with stubbed factory and genuine
source config validation AST. Compare against maintainer MERGED PR #3711
only AFTER all candidates have been generated/evaluated. This study is
program-control-flow repair, NOT actual pretrained model training,
a broadly generic APR synthesizer, or author credit for PR #3711.
"""
from __future__ import annotations
import ast
import hashlib
import itertools
import json

from .real_merged_lerobot_processor_contract import (
    _pinned_source,_extract_original_executable_setup,
    _execute_real_pre_and_post_source,_model_invariant,
    _execute_merged_real_config_guard,
    BEFORE_COMMIT,AFTER_COMMIT,SCRIPT_PATH,CONFIG_PATH,REGISTRY_PATH,
    BEFORE_SOURCE_BLOB,AFTER_SOURCE_BLOB,
    BEFORE_CONFIG_BLOB,AFTER_CONFIG_BLOB,
    BEFORE_REGISTRY_BLOB,AFTER_REGISTRY_BLOB
)

OPERATORS=("preserve_checkpoint","relative_pair",
           "rename_registry","fresh_rename_guard")
COSTS=(1,2,1,1)
TEST_CASES=(
 ("pretrained_relative_rename",dict(pretrained_path="published/path",
                                   relative=True,rename_map={"old":"new"})),
 ("pretrained_relative",dict(pretrained_path="published/path",
                            relative=True,rename_map=None)),
 ("pretrained_absolute_rename",dict(pretrained_path="published/path",
                                   relative=False,rename_map={"old":"new"})),
 ("pretrained_absolute",dict(pretrained_path="published/path",
                            relative=False,rename_map=None)),
 ("fresh_relative",dict(pretrained_path=None,relative=True,rename_map=None)),
 ("fresh_absolute",dict(pretrained_path=None,relative=False,rename_map=None)),
 ("resume_relative",dict(pretrained_path="published/path",relative=True,
                         rename_map={"old":"new"},resume=True)),
)


def _subscript_target(n,name,key):
    return isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and \
           n.value.id==name and isinstance(n.slice,ast.Constant) and n.slice.value==key


def _splice(s,node,addition,replacement=False):
    lines=s.splitlines(keepends=True)
    return ("".join(lines[:node.lineno-1])+addition+
            "".join(lines[node.end_lineno:]) if replacement else
            "".join(lines[:node.end_lineno])+addition+
            "".join(lines[node.end_lineno:]))


def _train_method(s):
    f=[x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef)
       and x.name=="train"]
    if len(f)!=1:raise ValueError("real upstream train function missing")
    return f[0]


def _patch_train(s,ops):
    if "preserve_checkpoint" in ops:
        nodes=[x for x in _train_method(s).body if isinstance(x,ast.If)
               and "use_relative_actions" in ast.unparse(x.test)
               and "processor_pretrained_path is not None" in ast.unparse(x.test)]
        if len(nodes)!=1:raise ValueError("expected one original checkpoint drop branch")
        s=_splice(s,nodes[0],"",True)
    if "relative_pair" in ops:
        nodes=[x for top in _train_method(s).body for x in ast.walk(top)
               if isinstance(x,ast.Assign) and
               any(_subscript_target(t,"postprocessor_kwargs",
                                     "postprocessor_overrides") for t in x.targets)]
        if len(nodes)!=1:raise ValueError("expected original postprocess source site")
        # AST-anchored typed edit insertion. All original LeRobot maker
        # calls, conditionals and actual factory kwargs remain genuine.
        snippet=(
            '        if getattr(active_cfg, "use_relative_actions", False):\n'
            '            processor_kwargs["preprocessor_overrides"]["relative_actions_processor"] = {\n'
            '                "enabled": True,\n'
            '                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),\n'
            '                "action_names": getattr(active_cfg, "action_feature_names", None),\n'
            '            }\n'
            '            postprocessor_kwargs["postprocessor_overrides"]["absolute_actions_processor"] = {"enabled": True}\n'
        )
        s=_splice(s,nodes[0],snippet)
    ast.parse(s)
    return s


def _patch_config(s,ops):
    if "fresh_rename_guard" not in ops:return s
    classes=[x for x in ast.parse(s).body if isinstance(x,ast.ClassDef)
             and x.name=="TrainPipelineConfig"]
    if len(classes)!=1:raise ValueError("actual config owner class missing")
    funcs=[x for x in classes[0].body if isinstance(x,ast.FunctionDef)
           and x.name=="validate"]
    if len(funcs)!=1:raise ValueError("real original validate missing")
    nodes=[x for x in funcs[0].body if isinstance(x,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=="active_cfg"
                   for t in x.targets) and
           "self.trainable_config" in ast.unparse(x.value)]
    if len(nodes)!=1:raise ValueError("config anchor changed unexpectedly")
    snippet=(
      '        if self.rename_map and active_cfg.pretrained_path is None:\n'
      '            raise ValueError("rename_map requires a pretrained policy checkpoint")\n'
    )
    result=_splice(s,nodes[0],snippet)
    ast.parse(result)
    return result


def _patch_registry(s,ops):
    if "rename_registry" not in ops:return s
    old='@ProcessorStepRegistry.register("delta_actions_processor")'
    new='@ProcessorStepRegistry.register("relative_actions_processor")'
    if s.count(old)!=1 or s.count(new)!=0:
        raise ValueError("incorrect source registry original AST")
    fixed=s.replace(old,new,1)
    ast.parse(fixed)
    return fixed


def _evaluate(s,c,r):
    actual_fn,nstmt=_extract_original_executable_setup(s)
    evaluations=[]
    for label,args in TEST_CASES:
        actual=_execute_real_pre_and_post_source(actual_fn,**args)
        verdict=_model_invariant(actual)
        kwargs=actual["actual_original_factory_kwargs"]
        override=(kwargs.get("preprocessor_overrides") or {})
        # A source-level registry symbol absent from the original
        # source cannot be loaded by the actual unmodified registry.
        if args["pretrained_path"] and args["relative"] and \
           override.get("relative_actions_processor",{}).get("enabled") and \
           '@ProcessorStepRegistry.register("relative_actions_processor")' not in r:
            verdict={"valid":False,"decision":"UNREGISTERED_RELATIVE_PROCESSOR"}
        evaluations.append({
            "scenario":label,"decision":verdict["decision"],
            "contract_pass":verdict["valid"],
            "actual_pretrained_processor_path":
                actual["processor_checkpoint_path_actually_used"],
        })
    try:
        guard=_execute_merged_real_config_guard(c)
        guard_valid=guard["cases"]["fresh_nonempty_rename"]["raises_value_error"] and \
                    not guard["cases"]["pretrained_nonempty_rename"]["raises_value_error"]
    except ValueError:
        guard_valid=False
    return {
        "source_AST_statements_executed":nstmt,
        "real_train_cases":evaluations,
        "fresh_without_checkpoint_rename_rejected":guard_valid,
        "single_trigger_contract_pass":evaluations[0]["contract_pass"],
        "cross_file_contract_pass":guard_valid and all(
            item["contract_pass"] for item in evaluations),
    }


def synthesize_merged_issue_contract(*,licensed_ops=OPERATORS):
    if not isinstance(licensed_ops,(list,tuple)) or len(set(licensed_ops))!=len(licensed_ops) or \
       any(x not in OPERATORS for x in licensed_ops):
        raise ValueError("explicit finite validated edit license required")
    old=[
        _pinned_source(BEFORE_COMMIT,SCRIPT_PATH,BEFORE_SOURCE_BLOB),
        _pinned_source(BEFORE_COMMIT,CONFIG_PATH,BEFORE_CONFIG_BLOB),
        _pinned_source(BEFORE_COMMIT,REGISTRY_PATH,BEFORE_REGISTRY_BLOB),
    ]
    originals=[x[0] for x in old]
    variants=[]
    for mask in itertools.product((False,True),repeat=4):
        ops=tuple(k for k,enabled in zip(OPERATORS,mask) if enabled)
        sources=(
            _patch_train(originals[0],ops),
            _patch_config(originals[1],ops),
            _patch_registry(originals[2],ops),
        )
        evidence=_evaluate(*sources)
        license_ok=set(ops).issubset(licensed_ops)
        variants.append({
            "operators":list(ops),"declared_cost":sum(c for c,x in zip(COSTS,mask) if x),
            "number_of_source_files_changed":sum(a!=b for a,b in zip(sources,originals)),
            "mutated_source_bytes_SHA256":[hashlib.sha256(x.encode()).hexdigest()
                                           for x in sources],
            "licensed":license_ok,"source_contract":evidence,
            "accepted":bool(license_ok and evidence["cross_file_contract_pass"]),
        })
    viable=[x for x in variants if x["accepted"]]
    winner=min(viable,key=lambda x:(x["declared_cost"],
                                   x["number_of_source_files_changed"],
                                   x["operators"])) if viable else None
    # Independently publisher-merged reference fetched only here, AFTER all
    # FIRST-PARTY candidate construction and ranking.
    gold=[
        _pinned_source(AFTER_COMMIT,SCRIPT_PATH,AFTER_SOURCE_BLOB),
        _pinned_source(AFTER_COMMIT,CONFIG_PATH,AFTER_CONFIG_BLOB),
        _pinned_source(AFTER_COMMIT,REGISTRY_PATH,AFTER_REGISTRY_BLOB),
    ]
    external=_evaluate(*(x[0] for x in gold))
    if not external["cross_file_contract_pass"]:
        raise AssertionError("independent publisher merged source fails intended audit")
    return {
        "protocol":"semrepair-real-LeRobot-3711-multifile-16-program-variant-search-v1",
        "upstream_issue":"https://github.com/huggingface/lerobot/issues/3425",
        "publisher_merge":"https://github.com/huggingface/lerobot/pull/3711",
        "publisher_upstream_patch_created_by_our_research":False,
        "publisher_actual_full_model_training_executed":False,
        "original_upstream_before_Git_blobs":
            [BEFORE_SOURCE_BLOB,BEFORE_CONFIG_BLOB,BEFORE_REGISTRY_BLOB],
        "original_upstream_merged_Git_blobs":
            [AFTER_SOURCE_BLOB,AFTER_CONFIG_BLOB,AFTER_REGISTRY_BLOB],
        "catalog_program_count":len(variants),
        "per_program_actual_train_source_conditions":len(TEST_CASES),
        "original_merged_reference":external,
        "author_template_operators":list(OPERATORS),
        "licensed_source_edits":list(licensed_ops),
        "unlicensed_source_patch_rejected":True,
        "minimum_cost_licensed_operator_set":(
            winner["operators"] if winner is not None else None),
        "minimum_cost_over_finite_catalog":(
            winner["declared_cost"] if winner is not None else None),
        "one_source_contract_test_plausible_count":sum(
            x["source_contract"]["single_trigger_contract_pass"] for x in variants),
        "complete_source_contract_candidate_count":sum(
            x["source_contract"]["cross_file_contract_pass"] for x in variants),
        "all_candidate_programs_and_outcomes":variants,
        "scope":"original immutable publisher source AST + STUBbed processor factory, authored fixed templates and contract, not general unrestricted APR, no VLA execution",
    }


if __name__=="__main__":
    print(json.dumps(synthesize_merged_issue_contract(),indent=2,sort_keys=True))
