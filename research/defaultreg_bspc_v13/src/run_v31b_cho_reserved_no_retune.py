#!/usr/bin/env python3
"""Sequential *reserved Cho2017 participants* with PREEXISTING frozen method.

Not independent acquisition, not preregistered before original first block,
not a new method. Fails closed if source-choice JSON differs by even 1 byte.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from moabb.datasets import Cho2017
import run_v31_cho_source_envelope as v31

ORIGINAL_LOCK="677f848de0e3bc3122892e5bec55b20dc7808ab04093bdaeb33f1bd0a923f796"
RESERVED=tuple(range(33,53))
FIRST_PROTO=Path(__file__).resolve().parents[1]/"V31_CHO2017_SOURCE_ENVELOPE_FIRST_RUN_PROTOCOL.md"
SECOND_PROTO=Path(__file__).resolve().parents[1]/"V31B_CHO2017_RESERVED_BLOCK_BEFORE_TARGET_PROTOCOL.md"
LOCKED_LAMBDAS={
    "envelope_r1":1.0,"source_macro_grid":0.5,
    "proper_source_mean_gaussian":0.1,
    "historical_class_centred":0.0,"envelope_r0":0.3,
}

def main(out:Path):
    if not (FIRST_PROTO.is_file() and SECOND_PROTO.is_file()):
        raise RuntimeError("missing pre-outcome protocols")
    paradigm=v31.old.MotorImagery(
        n_classes=2,events=list(v31.CLASSES),channels=list(v31.SENSORS),
        fmin=8,fmax=30,resample=160,baseline=None,
        reject_by_annotation=False)
    data=Cho2017()
    # This step reads ONLY the original twelve source persons.
    C,y,g,receipts=v31.load_source(paradigm,data)
    inv,state,manifest=v31.freeze_only_source(C,y,g,receipts,FIRST_PROTO)
    out.mkdir(parents=True,exist_ok=True)
    source_file=out/v31.FREEZE_NAME
    h=v31.save(manifest,source_file)
    if h!=ORIGINAL_LOCK or manifest["selected_lambdas"]!=LOCKED_LAMBDAS:
        raise AssertionError("original immutable source decisions not reproduced; forbid reserved evaluation")
    manifest["source_lock_sha256"]=h
    print("V31B_SAME_ORIGINAL_SOURCE_SHA_LOCKED_BEFORE_RESERVED_IO",h,flush=True)
    # Changing this list happens strictly AFTER byte-level source-choice
    # reproduction. NEVER change the original selection or methods.
    v31.HELD=RESERVED
    result=v31.evaluate(paradigm,data,inv,state,C,y,manifest)
    if ({r["subject"] for r in result["participant_rows"]}!=set(RESERVED)
        or result["n_held_subjects"]!=20):
        raise AssertionError("reserved group schema incomplete")
    result["status"]="V31B_CHO2017_SEQUENTIAL_RESERVED_BLOCK_COMPLETE_NOT_INDEPENDENT_ACQUISITION"
    result["original_preexisting_source_lock_sha256"]=h
    result["second_protocol_sha256"]=hashlib.sha256(SECOND_PROTO.read_bytes()).hexdigest()
    result["not_an_independent_acquisition_or_formal_preregistration"]=True
    result["first_block_outcomes_already_exposed_before_second_protocol"]=True
    v31.save(result,out/"CHO2017_V31B_RESERVED_TARGET_AUC.json")
    print("V31B_RESERVED_SUMMARY",json.dumps({
        k:result[k] for k in (
        "status","macro_auc_by_method",
        "primary_mean_delta_envelope_minus_source_macro",
        "harmed_gt_002","exact_two_sided_sign_flip_pvalue",
        "n_target_trials"
        )}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args().out)
