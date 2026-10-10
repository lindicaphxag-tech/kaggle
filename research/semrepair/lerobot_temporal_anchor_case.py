"""Independent software-source case study using REAL, unchanged LeRobot code.

Upstream HF LeRobot commit and actual file SHA1 git blob (not arbitrary text
retyped from docs) are pinned. Download source code at EXACT commit, extract
only its original to_relative_actions / to_absolute_actions AST function
definitions, and execute them with REAL torch tensors. No need to import
all LeRobot robotics dependencies or a physical robot.

Source file:
 huggingface/lerobot @ b9cb121cb7d4e3e26ec5c906d08dda68d151ba52
 src/lerobot/processor/relative_action_processor.py
 GIT BLOB = 3405402904cf15ca18227b3a3fe006d7936f9d53

This is a HISTORICAL ANCHOR-LIFETIME contract regression injected at the
callsite BY US, NOT an assertion that this upstream LeRobot version is buggy:
  - correct: inverse relative conversion uses chunk-anchored state S_0;
  - mutation: its caller overwrites the anchor with live state S_t;
  - repair: restore original pinned reference S_0 until chunk exhausted.

The healthy and simulated buggy callers invoke EXACTLY the same unmodified
upstream function (no mock arithmetic kernel). A black-box controller with
only (current state, relative chunk) cannot distinguish two worlds with
different hidden historical anchors; we provide a powerful *minimax*
current-only midpoint competitor. This is information-set conditional:
a black-box that can store the ORIGINAL observation history could simply
recover the source anchor and tie the repaired implementation.

The case tests an independent TIME-DEPENDENT stateful interface semantic,
not a two-stage clipping shear or an actual leaked vendor bug. It is a pure
software conformance experiment, NOT a robot task success experiment.
"""
from __future__ import annotations

import ast
from hashlib import sha1,sha256
import json
import math
import random
import urllib.request

import torch

UPSTREAM_REPO="huggingface/lerobot"
UPSTREAM_COMMIT="b9cb121cb7d4e3e26ec5c906d08dda68d151ba52"
UPSTREAM_PATH="src/lerobot/processor/relative_action_processor.py"
UPSTREAM_GIT_BLOB_SHA1="3405402904cf15ca18227b3a3fe006d7936f9d53"
UPSTREAM_RAW_URL=(
    "https://raw.githubusercontent.com/"+UPSTREAM_REPO+"/"+
    UPSTREAM_COMMIT+"/"+UPSTREAM_PATH
)
WORLD_PAIRS=64
CHUNK_HORIZON=8
ACTION_CHANNELS=7
MASK=[True,True,True,True,True,True,False] # gripper kept absolute
SOURCE_BUG_AUTHOR_INJECTED=True


def load_actual_upstream_function_pair():
    with urllib.request.urlopen(UPSTREAM_RAW_URL,timeout=40) as resp:
        raw=resp.read()
    if not raw or len(raw)>200000:
        raise ValueError("publisher source artifact empty or unexpectedly huge")
    envelope=b"blob "+str(len(raw)).encode()+b"\0"+raw
    sha=sha1(envelope).hexdigest()
    if sha!=UPSTREAM_GIT_BLOB_SHA1:
        raise ValueError("frozen actual upstream Git blob does not match pin")
    module=ast.parse(raw.decode("utf-8"),filename=UPSTREAM_PATH)
    funcs=[node for node in module.body
           if isinstance(node,ast.FunctionDef) and node.name in
           ("to_relative_actions","to_absolute_actions")]
    if len(funcs)!=2:
        raise ValueError("publisher functions absent or duplicated")
    clean=ast.Module(body=funcs,type_ignores=[])
    ast.fix_missing_locations(clean)
    namespace={"torch":torch,"Tensor":torch.Tensor,"Sequence":__import__("typing").Sequence}
    exec(compile(clean,UPSTREAM_PATH,"exec"),namespace,namespace)
    return namespace["to_relative_actions"],namespace["to_absolute_actions"],sha256(raw).hexdigest()


def run_actual_lerobot_anchor_case():
    to_rel,to_abs,original_source_sha256=load_actual_upstream_function_pair()
    rng=random.Random(20261010)
    worlds=[]
    for pair in range(WORLD_PAIRS):
        # Both possible worlds have the SAME observable (relative prediction
        # chunk, current state) but deliberately different hidden historical
        # anchors. This is an informational red-team construction, not a
        # sampled pair of actual robot simulator trajectories.
        current=[rng.uniform(-.4,.4) for _ in range(ACTION_CHANNELS)]
        offset=[rng.uniform(.04,.28)*rng.choice((-1,1))
                for _ in range(ACTION_CHANNELS)]
        offset[-1]=0.
        chunk=[[rng.uniform(-.3,.3) for _ in range(ACTION_CHANNELS)]
                for _ in range(CHUNK_HORIZON)]
        for branch in (-1,+1):
            anchor=[current[i]+branch*offset[i]
                    for i in range(ACTION_CHANNELS)]
            # from actual policy action semantics, not invented processor
            # code: dataset absolute actions reconstructed from q_rel+anchor.
            relative=torch.tensor([chunk],dtype=torch.float64)
            base=torch.tensor([anchor],dtype=torch.float64)
            absolute=to_abs(relative,base,MASK)
            re_rel=to_rel(absolute,base,MASK)
            if not torch.allclose(re_rel,relative,atol=1e-12,rtol=0):
                raise AssertionError("original upstream round-trip relative conversion changed")
            # We inject a caller source regression that wrongly passes the
            # live state to a correct original external software function.
            now=torch.tensor([current],dtype=torch.float64)
            faulty=to_abs(relative,now,MASK)
            # "minimax blackbox" knows the TWO hidden anchors are centered
            # at the same current state. Its best possible symmetric
            # max-error choice per coordinate equals this midpoint. It is
            # NOT optimal against a blackbox with access to full historical
            # state / stateful reconstruction.
            strong_current_only_minimax=to_abs(relative,now,MASK)
            fixed=to_abs(relative,base,MASK)
            healthy_max=max(abs(float(z)) for z in
                            (fixed-absolute).flatten())
            bad_max=max(abs(float(z)) for z in
                        (faulty-absolute).flatten())
            minimax_max=max(abs(float(z)) for z in
                            (strong_current_only_minimax-absolute).flatten())
            if healthy_max>1e-12:
                raise AssertionError("correct cached anchor not roundtripping")
            worlds.append({
                "world_pair":pair,
                "hidden_historical_anchor_branch":branch,
                "chunk_timesteps":CHUNK_HORIZON,
                "observed_live_state":current,
                "hidden_reference_anchor":anchor,
                "original_relative_chunk":chunk,
                "author_buggy_actual_max_error":bad_max,
                "strong_minimax_current_only_max_error":minimax_max,
                "repaired_real_upstream_max_error":healthy_max,
                "masked_non_gripper_coords":6,
                "original_source_functions_invoked_with_real_torch":True,
            })
    if len(worlds)!=2*WORLD_PAIRS:raise AssertionError("expected both worlds")
    # A well-informed but CURRENT-ONLY deterministic blackbox receives
    # identical inputs for each pair, hence MUST return the same output.
    # Desired outputs differ by 2*offset and require positive worst-case
    # deviation >= max_i |offset_i|. It can match the bound at midpoint.
    bounds=[]
    for pair in range(WORLD_PAIRS):
        a,b=worlds[2*pair:2*pair+2]
        if a["observed_live_state"]!=b["observed_live_state"] or \
           a["original_relative_chunk"]!=b["original_relative_chunk"]:
            raise AssertionError("different observable current-only inputs")
        radius=max(abs(a["hidden_reference_anchor"][i]-
                       b["hidden_reference_anchor"][i])/2
                   for i in range(6))
        assert abs(a["strong_minimax_current_only_max_error"]-radius)<1e-11
        assert abs(b["strong_minimax_current_only_max_error"]-radius)<1e-11
        bounds.append(radius)
    return {
        "protocol":"frozen-upstream-lerobot-temporal-anchor-info-lower-bound-v1",
        "official_external_repo":UPSTREAM_REPO,
        "exact_external_git_commit":UPSTREAM_COMMIT,
        "actual_external_source_git_blob":UPSTREAM_GIT_BLOB_SHA1,
        "verified_external_source_bytes_sha256":original_source_sha256,
        "upstream_source_file":UPSTREAM_PATH,
        "vendor_reported_software_bug":False,
        "fault_is_author_mutation_of_callsite":True,
        "number_of_real_upstream_original_functions_executed":2,
        "torch_backend_original_software":True,
        "all_comparisons_use_identical_original_upstream_conversion_kernel":True,
        "physical_robot_tasks_executed":0,
        "action_dimensions":ACTION_CHANNELS,
        "original_action_channels_converted":6,
        "chunk_steps":CHUNK_HORIZON,
        "world_pairs":WORLD_PAIRS,
        "scenario_worlds":len(worlds),
        "nonindependent_action_rows":len(worlds)*CHUNK_HORIZON,
        "max_fixed_error":max(r["repaired_real_upstream_max_error"] for r in worlds),
        "min_achievable_current_only_worst_case_error_across_worlds":min(bounds),
        "mean_minimax_lower_bound_on_current_only_error":sum(bounds)/len(bounds),
        "mean_author_injected_stale_reference_error":sum(
            x["author_buggy_actual_max_error"] for x in worlds)/len(worlds),
        "lower_bound_per_pair":bounds,
        "raw_worlds":worlds,
        "scope_note":"conditional information-set impossibility: blackbox with historical anchor access CAN tie source repair; NOT vendor bug or actual robot success",
    }


if __name__=="__main__":
    print(json.dumps(run_actual_lerobot_anchor_case(),indent=2,sort_keys=True))
