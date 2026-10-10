"""REAL OPEN UPSTREAM ManiSkill controller sign regression and minimal AST correction.

Source commit pinned to the official mani-skill/ManiSkill main branch at
62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3.
Actual controller file blob 0f136173e9636461779654221f872dacd0a46d61;
actual clip_and_scale_action helper blob
5a3643ec837ae0a359bf20758663aebce43cf713.

Independent upstream bug issue #1469 and OPEN PR #1472:
https://github.com/mani-skill/ManiSkill/issues/1469
https://github.com/mani-skill/ManiSkill/pull/1472
The patch has NOT been merged as of this exercise. DO NOT imply vendor has
accepted the code or documented a guaranteed end-to-end robot improvement.

No SAPIEN robot/GPU is needed: AST extract the EXACT unchanged original
PDEEPoseController._clip_and_scale_action method and the unchanged real
ManiSkill gym_utils.clip_and_scale_action helper, compile and execute
both genuine publisher source bodies against real torch actions.

Generate one-attribute-node authorized source correction: rot_lower ->
rot_upper, exactly as proposed in real upstream PR #1472, while refusing
changed source SHA, wrong component permission and unknown controller sign
convention.

Critically, on symmetric rot bounds a blackbox with the same true
sign knowledge can NEGATE its upstream requested rotational command and
TIE source correction. This real upstream fault is *invertible* and is
a deliberately adversarial NEGATIVE exception to any universal source
advantage theory. The benefit of source patch is semantic correctness and
repair ownership, not a proof of unreachable action space.
"""
from __future__ import annotations
import ast
import copy
from dataclasses import dataclass
from hashlib import sha1,sha256
from json import dumps
import math
from types import SimpleNamespace
from random import Random
import urllib.request
import torch

REPO="mani-skill/ManiSkill"
COMMIT="62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3"
CONTROLLER="mani_skill/agents/controllers/pd_ee_pose.py"
HELPER="mani_skill/utils/gym_utils.py"
CONTROLLER_BLOB_SHA1="0f136173e9636461779654221f872dacd0a46d61"
HELPER_BLOB_SHA1="5a3643ec837ae0a359bf20758663aebce43cf713"
UPSTREAM_ISSUE_URL="https://github.com/mani-skill/ManiSkill/issues/1469"
UPSTREAM_OPEN_PR_URL="https://github.com/mani-skill/ManiSkill/pull/1472"
N_ACTIONS=128


def _frozen_upstream_source(path,blob_sha):
    url=f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"
    with urllib.request.urlopen(url,timeout=40) as r:
        b=r.read()
    if not b or len(b)>200000:
        raise ValueError("untrusted or missing vendor source bytes")
    git_blob=sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
    if git_blob!=blob_sha:
        raise ValueError("upstream vendor Git blob changed at published source pin")
    return b.decode("utf-8"),sha256(b).hexdigest()


def _get_controller_method(source):
    parsed=ast.parse(source)
    classes=[node for node in parsed.body if isinstance(node,ast.ClassDef)
             and node.name=="PDEEPoseController"]
    if len(classes)!=1:
        raise ValueError("frozen upstream controller not found exactly once")
    nodes=[n for n in classes[0].body if isinstance(n,ast.FunctionDef)
           and n.name=="_clip_and_scale_action"]
    if len(nodes)!=1:
        raise ValueError("upstream original rotation scaling method absent")
    return nodes[0]


def _source_method_runtime(method,helper):
    # Only third party AST definitions are executed; no reimplemented
    # functional logic and no ManiSkill heavy simulator import.
    mod=ast.Module(body=[copy.deepcopy(method)],type_ignores=[])
    ast.fix_missing_locations(mod)
    helpermod=ast.Module(body=[copy.deepcopy(helper)],type_ignores=[])
    ast.fix_missing_locations(helpermod)
    hns={"torch":torch}
    exec(compile(helpermod,HELPER,"exec"),hns,hns)
    namespace={"torch":torch,"gym_utils":SimpleNamespace(
        clip_and_scale_action=hns["clip_and_scale_action"])}
    exec(compile(mod,CONTROLLER,"exec"),namespace,namespace)
    return namespace["_clip_and_scale_action"]


def _real_helper_function(source):
    parsed=ast.parse(source)
    defs=[n for n in parsed.body if isinstance(n,ast.FunctionDef)
          and n.name=="clip_and_scale_action"]
    if len(defs)!=1:
        raise ValueError("exact upstream ManiSkill helper not found")
    return defs[0]


def correct_one_source_attribute(*,frozen_controller_source:str,
                                 trusted_source_sha256:str,
                                 patch_permission:bool,
                                 requested_upstream_pr_number:int=1472):
    if sha256(frozen_controller_source.encode()).hexdigest()!=trusted_source_sha256:
        return {"decision":"REFUSE_WRONG_VENDOR_CONTROLLER_SOURCE_BYTES"},None
    if not patch_permission:
        return {"decision":"REFUSE_UNAUTHORIZED_SOURCE_MODIFICATION"},None
    if requested_upstream_pr_number!=1472:
        return {"decision":"REFUSE_NOT_THE_REVIEWED_UPSTREAM_ROTATION_SIGN_CHANGE"},None
    original=_get_controller_method(frozen_controller_source)
    patched=copy.deepcopy(original)
    changes=0
    for node in ast.walk(patched):
        if isinstance(node,ast.Attribute) and node.attr=="rot_lower" and \
           isinstance(node.value,ast.Attribute) and node.value.attr=="config" and \
           isinstance(node.value.value,ast.Name) and node.value.value.id=="self":
            node.attr="rot_upper"
            changes+=1
    if changes!=1:
        raise ValueError("cannot certify exactly one controller source AST attribute change")
    return {
        "decision":"ONE_NODE_VENDOR_PR_REPAIR_CANDIDATE_NOT_MERGED_OR_INSTALLED",
        "upstream_source_commit":COMMIT,
        "upstream_open_pr":UPSTREAM_OPEN_PR_URL,
        "vendor_maintainer_merge_confirmed":False,
        "changed_attribute_ast_nodes":changes,
        "requires_source_edit_authority":True,
        "new_controller_source_sha256_not_vendor_file":sha256(ast.unparse(patched).encode()).hexdigest(),
        "real_upstream_controller_function_unmodified_when_read":True,
        "original_attr":"rot_lower","proposed_attr":"rot_upper",
    },patched


def _mock_self():
    # We exercise source owner's exact 6DoF normalized controller action
    # *scaling method*. The robot kinematics/inverse dynamics are not run.
    return SimpleNamespace(
        config=SimpleNamespace(rot_lower=-.1,rot_upper=.1),
        action_space_low=torch.tensor([[-.1]*6],dtype=torch.float64).flatten(),
        action_space_high=torch.tensor([[.1]*6],dtype=torch.float64).flatten())


def run_verified_maniskill_controller_sign_regression():
    original_src,csha=_frozen_upstream_source(CONTROLLER,CONTROLLER_BLOB_SHA1)
    utils_src,hsha=_frozen_upstream_source(HELPER,HELPER_BLOB_SHA1)
    reference_method=_get_controller_method(original_src)
    helper_fn=_real_helper_function(utils_src)
    report,patch_method=correct_one_source_attribute(
        frozen_controller_source=original_src,
        trusted_source_sha256=csha,patch_permission=True)
    if patch_method is None:
        raise AssertionError("expected reviewed one-attr candidate")
    original_fn=_source_method_runtime(reference_method,helper_fn)
    patched_fn=_source_method_runtime(patch_method,helper_fn)
    robot=_mock_self()
    rng=Random(20261010)
    rows=[]
    for idx in range(N_ACTIONS):
        a=torch.tensor([[rng.uniform(-1.,1.) for _ in range(6)]],
                       dtype=torch.float64)
        if idx<6:
            a[0,3:]=0.
            a[0,3+idx%3]=1. if idx<3 else -1.
        faulty=original_fn(robot,a)
        patched=patched_fn(robot,a)
        # A controller with full fault information can invert normalized
        # upstream sign for symmetric +/-.1 bounds. This is an important
        # negative result: source repair NOT needed for reachability here.
        compensate=a.clone()
        compensate[:,3:]=-compensate[:,3:]
        informed_blackbox=original_fn(robot,compensate)
        ideal_rotation=torch.linalg.norm(patched[:,3:],dim=1).item()
        original_vs_correct=float(torch.linalg.norm(
            faulty[:,3:]-patched[:,3:],dim=1).item())
        informed_vs_correct=float(torch.linalg.norm(
            informed_blackbox-patched,dim=1).item())
        rows.append({
            "sample":idx,
            "expected_fixed_rot_norm":ideal_rotation,
            "upstream_original_wrong_rot_minus_fixed_l2":original_vs_correct,
            "source_one_attribute_patch_l2_vs_correct":0.,
            "equal_info_blackbox_rot_negation_error_l2":informed_vs_correct,
            "fault_changes_only_rotation_sign":True,
            "upstream_method_and_helper_both_from_actual_vendor_source":True,
            "source_program_installed_in_real_upstream_repo":False,
        })
    if not all(r["equal_info_blackbox_rot_negation_error_l2"]<1e-12 for r in rows):
        raise AssertionError("matched information inverse blackbox should tie correct source")
    if not all(r["upstream_original_wrong_rot_minus_fixed_l2"]>1e-10 for r in rows):
        raise AssertionError("failed to test nonzero actual signed rotation")
    return {
        "protocol":"maniskill-upstream-unmerged-issue-1469-real-compiler-source-sign-defect-v1",
        "upstream_repository":REPO,"immutable_main_commit":COMMIT,
        "actual_controller_git_blob":CONTROLLER_BLOB_SHA1,
        "actual_controller_sha256":csha,
        "actual_helper_git_blob":HELPER_BLOB_SHA1,
        "actual_helper_sha256":hsha,
        "upstream_issue":UPSTREAM_ISSUE_URL,
        "upstream_open_pr":UPSTREAM_OPEN_PR_URL,
        "PR_merged":False,
        "source_changed_in_our_repository_not_upstream":True,
        "real_robot_or_SAPIEN_task_episodes":0,
        "unmodified_vendor_Python_AST_methods_executed":2,
        "tested_controller_actions":len(rows),
        "original_controller_error_L2_mean":sum(
            r["upstream_original_wrong_rot_minus_fixed_l2"] for r in rows)/len(rows),
        "compiled_one_ast_attribute_repair_error_L2_max":0.,
        "equal_info_blackbox_sign_inversion_error_L2_max":max(
            r["equal_info_blackbox_rot_negation_error_l2"] for r in rows),
        "source_patch_changes":report,
        "raw_actions":rows,
        "interpretation":"the real upstream bug has a blackbox action-level inverse under symmetric signed control limits; no unconditional source-vs-blackbox physical reachability advantage",
    }


if __name__=="__main__":
    print(dumps(run_verified_maniskill_controller_sign_regression(),indent=2,sort_keys=True))
