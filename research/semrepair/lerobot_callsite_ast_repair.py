"""First-party one-AST-node callsite repair invoking REAL pinned LeRobot code.

This is NOT a reported upstream bug. An author-written wrapper changes the
state reference from persistent chunk-start state to live state. The repair
synthesizer changes exactly ONE argument AST Name, only with SOURCE WRITE
AUTHORITY and a valid chunk-epoch anchor. Both compiled BEFORE/AFTER call the
unchanged hash-pinned LeRobot original torch conversion function.
"""
from __future__ import annotations
import ast
from hashlib import sha256
from difflib import unified_diff
from json import dumps
from random import Random
import torch
from .lerobot_temporal_anchor_case import (
    load_actual_upstream_function_pair,MASK,UPSTREAM_GIT_BLOB_SHA1
)

ORIGINAL_CALLSITE="""def decode_chunk(chunk, current_state, initial_anchor, decoder, mask):
    return decoder(chunk, current_state, mask)
"""
SHA=sha256(ORIGINAL_CALLSITE.encode()).hexdigest()


def synthesize_authorized_minimal_temporal_patch(
    *,original_source:str,expected_sha256:str,write_authorized:bool,
    anchor_epoch:int,live_chunk_epoch:int,anchor_available:bool
):
    if sha256(original_source.encode()).hexdigest()!=expected_sha256:
        return {"decision":"REFUSE_UNTRUSTED_SOURCE_BYTES"},None
    if not write_authorized:
        return {"decision":"REFUSE_UNLICENSED_SOURCE_EDIT"},None
    if type(anchor_epoch) is not int or type(live_chunk_epoch) is not int or \
       anchor_epoch!=live_chunk_epoch or not anchor_available:
        return {"decision":"REFUSE_MISSING_OR_STALE_CHUNK_ANCHOR"},None
    tree=ast.parse(original_source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError("source not exactly one callsite")
    fn=tree.body[0]
    if fn.name!="decode_chunk" or [a.arg for a in fn.args.args]!=[
        "chunk","current_state","initial_anchor","decoder","mask"
    ] or len(fn.body)!=1 or not isinstance(fn.body[0],ast.Return):
        raise ValueError("callsite signature/source control flow changed")
    call=fn.body[0].value
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or \
       call.func.id!="decoder" or len(call.args)!=3 or call.keywords or \
       not isinstance(call.args[1],ast.Name) or call.args[1].id!="current_state":
        raise ValueError("expected author-injected callsite mismatch")
    call.args[1]=ast.copy_location(
        ast.Name(id="initial_anchor",ctx=ast.Load()),call.args[1])
    ast.fix_missing_locations(tree)
    fixed=ast.unparse(tree)+"\n"
    metadata={
        "decision":"SOURCE_AST_PATCH_SYNTHESIZED_NOT_INSTALLED",
        "author_bug_injected":True,
        "real_vendor_bug_reported":False,
        "vendor_kernel_rewritten":False,
        "real_source_file_byte_replacements":0,
        "author_callsite_file_changes":1,
        "changed_AST_argument_names":1,
        "source_write_permission_checked":True,
        "chunk_epoch_checked":True,
        "original_author_callsite_sha256":expected_sha256,
        "patched_author_callsite_sha256":sha256(fixed.encode()).hexdigest(),
        "pinned_vendor_kernel_git_blob":UPSTREAM_GIT_BLOB_SHA1,
        "patch_unified_diff":"".join(unified_diff(
            original_source.splitlines(True),fixed.splitlines(True),
            fromfile="injected-callsite.py",tofile="repaired-callsite.py")),
    }
    return metadata,fixed


def _compiled(source):
    ns={}
    exec(compile(source,"author-callsite.py","exec"),ns,ns)
    return ns["decode_chunk"]


def run_original_lerobot_ast_repair():
    _,decode,source_bytes_sha=load_actual_upstream_function_pair()
    result,patched=synthesize_authorized_minimal_temporal_patch(
        original_source=ORIGINAL_CALLSITE,expected_sha256=SHA,
        write_authorized=True,anchor_epoch=9,live_chunk_epoch=9,
        anchor_available=True)
    faulty=_compiled(ORIGINAL_CALLSITE)
    corrected=_compiled(patched)
    rng=Random(20261010)
    data=[]
    for case in range(128):
        chunk=torch.tensor([[[rng.uniform(-.2,.2) for _ in range(7)]
                             for _ in range(8)]],dtype=torch.float64)
        anchor=torch.tensor([[rng.uniform(-.5,.5) for _ in range(7)]],
                             dtype=torch.float64)
        current=anchor+torch.tensor(
            [[rng.choice((-1.,1.))*rng.uniform(.05,.3) for _ in range(7)]],
            dtype=torch.float64)
        reference=decode(chunk,anchor,MASK)
        broken=faulty(chunk,current,anchor,decode,MASK)
        repaired=corrected(chunk,current,anchor,decode,MASK)
        data.append({
            "case":case,
            "author_injected_bug_max_abs_action_error":float((reference-broken).abs().max()),
            "one_node_repair_max_abs_action_error":float((reference-repaired).abs().max()),
            "original_vendor_kernel_used_in_both_calls":True
        })
    return {
        "protocol":"author-ast-one-node-correct-source-argument-real-original-LeRobot-v1",
        "external_git_blob_sha1":UPSTREAM_GIT_BLOB_SHA1,
        "external_source_sha256":source_bytes_sha,
        "real_vendor_kernel_unchanged":True,
        "author_injected_fault":True,
        "actual_robot_episodes":0,
        "author_callsite_edit_metadata":result,
        "exact_repaired_callsite_source":patched,
        "source_repair_comparisons":len(data),
        "mean_faulty_max_abs_error":sum(r["author_injected_bug_max_abs_action_error"] for r in data)/len(data),
        "maximum_fixed_max_abs_error":max(r["one_node_repair_max_abs_action_error"] for r in data),
        "improved_case_count":sum(
            r["author_injected_bug_max_abs_action_error"]>
            r["one_node_repair_max_abs_action_error"]+1e-12 for r in data),
        "rows":data,
        "limitation":"author source callsite experiment, no LeRobot upstream reported bug, signed production patch or robot task",
    }


if __name__=="__main__":
    print(dumps(run_original_lerobot_ast_repair(),indent=2,sort_keys=True))
