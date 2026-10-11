"""Real independently MERGED ManiSkill #689 hidden input-alias repair.

Publisher issue: https://github.com/mani-skill/ManiSkill/pull/689
Actual publisher merge on 2024-11-08:
  base 6d3e8b5446eea74cf13251b7f28e9cbfc03013ed
  merged 84a5026098a5e21c356b93df75937d6b880b7d5e

Before source: rot_action = action[:, 3:]; then rot_action[mask] = ...
That slice is an ALIAS of caller-owned input memory. Under norm>1 it
SILENTLY MUTATES caller state while producing a returned command identical
to the actual merged code. An action-output-only repair oracle misses this.

Byte-exact original ManiSkill publisher Python AST controller and original
gym_utils.clip_and_scale_action are compiled and executed with Torch. We
generate three first-party AST mutations from the BEFORE code:
 - clone just the rotation slice (publisher's actual 1-node fix);
 - clone the complete original input tensor;
 - clone just the returned output (plausible but INEFFECTIVE).

The algorithm recognizes only ONE local typed alias/write grammar, not
whole-program alias analysis, and compares to the genuine publisher merged
fix only after candidates are constructed. Actual isolated source methods
are run; no real SAPIEN/PhysX task, robotics reward or vendor signature.
"""
from __future__ import annotations
import ast
import copy
from hashlib import sha1,sha256
from random import Random
from types import SimpleNamespace
import json
import urllib.request
import torch

REPO="mani-skill/ManiSkill"
PRE="6d3e8b5446eea74cf13251b7f28e9cbfc03013ed"
AFTER="84a5026098a5e21c356b93df75937d6b880b7d5e"
CONTROLLER="mani_skill/agents/controllers/pd_ee_pose.py"
UTILS="mani_skill/utils/gym_utils.py"
OLD_CONTROLLER_SHA1="006273444929f4321e732b8ed7376990ff889655"
MERGED_CONTROLLER_SHA1="4b599c176e8f9ee7d207fe7a5ad7be031fff7db8"
UTILS_SHA1="9df64461b6d760ec26a0c04b4ed2ce95256faa1d"
CASES=96
VARIANTS=("original","clone_rotation_view","clone_complete_input",
          "clone_return_output")


def _pinned(commit,path,expected):
    url="https://raw.githubusercontent.com/"+REPO+"/"+commit+"/"+path
    with urllib.request.urlopen(url,timeout=40) as handle:
        b=handle.read()
    if not b or len(b)>200000:
        raise ValueError("missing or oversized frozen vendor source")
    digest=sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
    if digest!=expected:
        raise ValueError("real published source Git blob identity mismatch")
    return b.decode(),sha256(b).hexdigest()


def _actual_controller_method(source):
    cls=[x for x in ast.parse(source).body if isinstance(x,ast.ClassDef)
         and x.name=="PDEEPoseController"]
    if len(cls)!=1:raise ValueError("true publisher controller missing")
    f=[x for x in cls[0].body if isinstance(x,ast.FunctionDef)
       and x.name=="_clip_and_scale_action"]
    if len(f)!=1:raise ValueError("original publisher action source missing")
    return copy.deepcopy(f[0])


def _actual_helper(source):
    f=[x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef)
       and x.name=="clip_and_scale_action"]
    if len(f)!=1:raise ValueError("real publisher gym helper missing")
    env={"torch":torch}
    mod=ast.fix_missing_locations(ast.Module(body=[f[0]],type_ignores=[]))
    exec(compile(mod,UTILS,"exec"),env,env)
    return env["clip_and_scale_action"]


def _source_alias_obligations(method):
    aliases=[]
    for node in ast.walk(method):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and \
           isinstance(node.targets[0],ast.Name) and \
           isinstance(node.value,ast.Subscript) and \
           isinstance(node.value.value,ast.Name) and \
           node.value.value.id=="action":
            name=node.targets[0].id
            writes=[w for w in ast.walk(method)
                    if isinstance(w,ast.Assign) and any(
                       isinstance(t,ast.Subscript) and
                       isinstance(t.value,ast.Name) and t.value.id==name
                       for t in w.targets)]
            if writes:
                aliases.append((node,name,len(writes)))
    return aliases


def _mutate(old,variant):
    if variant not in VARIANTS:
        raise ValueError("not an approved typed AST repair operator")
    fn=copy.deepcopy(old)
    if variant=="clone_rotation_view":
        hits=_source_alias_obligations(fn)
        if len(hits)!=1 or hits[0][1]!="rot_action":
            raise ValueError("source alias flow not localized uniquely")
        node=hits[0][0]
        node.value=ast.Call(
            func=ast.Attribute(value=node.value,attr="clone",ctx=ast.Load()),
            args=[],keywords=[])
    elif variant=="clone_complete_input":
        fn.body[:0]=ast.parse("action = action.clone()").body
    elif variant=="clone_return_output":
        ret=[x for x in fn.body if isinstance(x,ast.Return)]
        if len(ret)!=1:raise ValueError("original return statement missing")
        ret[0].value=ast.Call(
            func=ast.Attribute(value=ret[0].value,attr="clone",ctx=ast.Load()),
            args=[],keywords=[])
    return ast.fix_missing_locations(fn)


def _compile(method,helper):
    env={"torch":torch,"gym_utils":SimpleNamespace(
         clip_and_scale_action=helper)}
    mod=ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(method)],
                                               type_ignores=[]))
    exec(compile(mod,CONTROLLER,"exec"),env,env)
    return env["_clip_and_scale_action"]


def _owner():
    return SimpleNamespace(
        config=SimpleNamespace(rot_lower=-.1,rot_upper=.1),
        action_space_low=torch.tensor([-.1]*6,dtype=torch.float64),
        action_space_high=torch.tensor([.1]*6,dtype=torch.float64),
    )


def run_real_publisher_alias_repair():
    old,old_sha=_pinned(PRE,CONTROLLER,OLD_CONTROLLER_SHA1)
    true_fix,fix_sha=_pinned(AFTER,CONTROLLER,MERGED_CONTROLLER_SHA1)
    util,util_sha=_pinned(PRE,UTILS,UTILS_SHA1)
    helper=_actual_helper(util)
    original=_actual_controller_method(old)
    actual_merged=_actual_controller_method(true_fix)
    assert len(_source_alias_obligations(original))==1
    candidates={k:_compile(_mutate(original,k),helper) for k in VARIANTS}
    publisher=_compile(actual_merged,helper)
    rng=Random(20261010)
    records=[]
    for i in range(CASES*2):
        norm_gt_one=i<CASES
        positions=[rng.uniform(-.7,.7) for _ in range(3)]
        rotation=[rng.choice((-1.,1.))*rng.uniform(.72,.98)
                  for _ in range(3)] if norm_gt_one else [
                      rng.uniform(-.25,.25) for _ in range(3)]
        source=torch.tensor([positions+rotation],dtype=torch.float64)
        expected_source=source.clone()
        ground=publisher(_owner(),source.clone())
        results={}
        for k,f in candidates.items():
            caller=source.clone()
            returned=f(_owner(),caller)
            results[k]={
                "input_modified":not bool(torch.equal(caller,expected_source)),
                "max_action_output_error_vs_real_merged":
                    float((returned-ground).abs().max()),
                "actual_action_input_elements_cloned":
                    3 if k=="clone_rotation_view" else
                    6 if k=="clone_complete_input" else 0,
            }
        records.append({"case":i,"rotation_norm_gt_one":norm_gt_one,
                        "method_outcomes":results})
    aggregate={}
    for name in VARIANTS:
        v=[r["method_outcomes"][name] for r in records]
        aggregate[name]={
            "action_output_equal_original_publisher_merged_all_192":
                all(r["max_action_output_error_vs_real_merged"]<1e-12 for r in v),
            "violated_caller_input_immutability_high_norm":
                sum(v[i]["input_modified"] for i in range(CASES)),
            "violated_caller_input_immutability_low_norm":
                sum(v[i]["input_modified"] for i in range(CASES,len(v))),
            "copy_elements_per_case":
                v[0]["actual_action_input_elements_cloned"],
        }
    eligible=[name for name,x in aggregate.items()
              if x["action_output_equal_original_publisher_merged_all_192"]
              and x["violated_caller_input_immutability_high_norm"]==0
              and x["violated_caller_input_immutability_low_norm"]==0]
    winner=min(eligible,key=lambda name:(
        aggregate[name]["copy_elements_per_case"],name)) if eligible else None
    if winner!="clone_rotation_view" or \
       aggregate["original"]["violated_caller_input_immutability_high_norm"]!=CASES:
        raise AssertionError("original source alias failure or minimal typed repair not reproduced")
    return {
        "protocol":"ManiSkill-real-upstream-MERGED-689-source-input-immutability-v1",
        "upstream_publisher_merged_PR":
            "https://github.com/mani-skill/ManiSkill/pull/689",
        "publisher_merged_date":"2024-11-08",
        "exact_before_commit":PRE,"exact_after_commit":AFTER,
        "controller_before_git_blob":OLD_CONTROLLER_SHA1,
        "controller_after_git_blob":MERGED_CONTROLLER_SHA1,
        "original_utils_git_blob":UTILS_SHA1,
        "actual_source_sha256":[old_sha,fix_sha,util_sha],
        "real_publisher_original_and_merged_torch_methods_executed":True,
        "actual_physics_rollouts":0,
        "author_created_mutant_candidates":list(VARIANTS),
        "candidate_selection_did_not_consult_publisher_after_source":True,
        "source_level_AST_alias_flow_found":True,
        "tested_input_configs":len(records),
        "selected_lowest_copy_volume_author_template":winner,
        "aggregate":aggregate,
        "input_copy_volume_proxy_only_not_memory_profiler":True,
        "raw_cases":records,
        "limitation":"one known source alias/load-store grammar, two source commits; no generic points-to analysis, source ownership verification or robot task success",
    }


if __name__=="__main__":
    print(json.dumps(run_real_publisher_alias_repair(),sort_keys=True,indent=2))
