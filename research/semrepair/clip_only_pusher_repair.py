"""Demonstrate a *lower source intervention* than two-component atomic repair.

Same OFFICIAL external SAC-expert Pusher-v5 and already-inspected severe
source defect. Compare:
  A healthy original policy in native physical simulator;
  B real OLD on-disk P/C + software clip=.8 (fault);
  C SAME two erroneous P/C source matrices, but a two-signer approved
    *clip-only* file-byte repair changing software clip from .8 to 9.05;
  D source atomic rewrite of P and C to I, clip to native 2.

Because C=P^-1, no interstage saturation is the only obstruction.
Pusher action command each of the 7 motors is within [-2,2].
P on the FIRST two controls amplifies the 2D square by row norm 4.5,
so L must be >=9. We choose 9.05 solely as SOFTWARE intermediary clip,
NOT claimed real device's permissible internal controller safety cap.
If external middleware has real hard cap <9, clip-only is **unlicensed**
and must never be installed; the source selection algorithm can refuse it.

Both real file changes are signed using first-party-generated test keys,
and read from disk for EVERY robot action. No key from manufacturers,
no provenance of actual upstream bug, no trustworthy measured clip cap.
16 seeds were observed previously and do not create new heldout evidence.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import gymnasium as gym

from .signed_patch_reacher_pusher_task import (
    REPLAY_FAULT,REPLACEMENT,ACTOR_FAMILY,
    _rollout,TEST_SEEDS,STEPS
)
from .signed_source_activation import (
    _canonical,PROTOCOL,verified_atomic_source_activate
)
from .external_pusher_transfer import load_external_pusher_checkpoint
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

CLIP_ONLY={**REPLAY_FAULT,"clip":9.05}
EXTERNAL_ALLOWED_MAX_SOFTWARE_CLIP_HERE=9.1  # AUTHOR ASSUMPTION, not vendor permission


def _activate(path,old,target,patch_class,weight_sha,epoch):
    if target["clip"]>EXTERNAL_ALLOWED_MAX_SOFTWARE_CLIP_HERE:
        raise ValueError("proposed source clip exceeds EXOGENOUS ASSUMED license cap")
    roots={k:Ed25519PrivateKey.generate() for k in ("test-code-owner","test-independent-code-reviewer","test-third")}
    public={k:v.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw) for k,v in roots.items()}
    blob=_canonical(target)
    deployment=sha256(_canonical({
        "task":"official Pusher-v5","action_limit":2.0,
        "frozen_published_policy_sha256":weight_sha,
        "assumed_clip_cap":EXTERNAL_ALLOWED_MAX_SOFTWARE_CLIP_HERE})).hexdigest()
    nonce="signed-source-config-clip-only-20261010-"+str(epoch)
    manifest={"protocol":PROTOCOL,
              "current_source_sha256":sha256(old).hexdigest(),
              "replacement_source_sha256":sha256(blob).hexdigest(),
              "current_epoch":epoch,
              "target_epoch":epoch+1,
              "deployment_sha256":deployment,
              "expected_nonce":nonce,
              "authorized_repair":patch_class,
              "actor_family":ACTOR_FAMILY,
              "max_abs_input_action":2.}
    approvals=[
        {"issuer":k,"ed25519_hex":roots[k].sign(_canonical(manifest)).hex()}
        for k in ("test-code-owner","test-independent-code-reviewer")]
    certificate=verified_atomic_source_activate(
        active_config=path,replacement_bytes=blob,manifest=manifest,
        signatures=approvals,trusted_public_roots=public,
        expected_current_epoch=epoch,expected_nonce=nonce,
        expected_actor_family=ACTOR_FAMILY,
        expected_deployment_sha256=deployment,
        expected_action_radius=2.,threshold=2)
    return certificate,sha256(blob).hexdigest()


def run_clip_only_source_recovery():
    actor,weight_sha=load_external_pusher_checkpoint()
    old=_canonical(REPLAY_FAULT)
    expected_old_sha=sha256(old).hexdigest()
    rows=[]
    actual_records={}
    with TemporaryDirectory(prefix="semrepair-clip-only-") as temp:
        file=Path(temp)/"source.json"
        file.write_bytes(old)
        with gym.make("Pusher-v5") as env:
            for seed in TEST_SEEDS:
                rows.append({"arm":"official-health","source_module_edit_count":0,
                             **_rollout(actor,env,seed)})
                rows.append({"arm":"file-backed-fault","source_module_edit_count":0,
                             **_rollout(actor,env,seed,config_path=file,
                                        source_digest=expected_old_sha)})
            # Stage 1: minimal source repair, only ONE numeric source field
            certificate,new_sha=_activate(
                file,old,CLIP_ONLY,"clip-only",weight_sha,1)
            actual_records["clip-only"]=certificate
            if certificate["source_repair_class"]!="clip-only" or \
               file.read_bytes()!=_canonical(CLIP_ONLY):
                raise AssertionError("expected real clip-only file replacement did not occur")
            for seed in TEST_SEEDS:
                rows.append({"arm":"installed-signed-clip-only","source_module_edit_count":1,
                             **_rollout(actor,env,seed,config_path=file,
                                        source_digest=new_sha)})
            # Stage 2: compare against the more invasive traditional repair,
            # on a *separate* original faulty source file.
            file.write_bytes(old)
            replacement_certificate,ref_sha=_activate(
                file,old,REPLACEMENT,"atomic",weight_sha,3)
            actual_records["atomic"]=replacement_certificate
            for seed in TEST_SEEDS:
                rows.append({"arm":"installed-signed-atomic-matrices","source_module_edit_count":3,
                             **_rollout(actor,env,seed,config_path=file,
                                        source_digest=ref_sha)})
    arms=("official-health","file-backed-fault",
          "installed-signed-clip-only","installed-signed-atomic-matrices")
    report={}
    for arm in arms:
        data=[row for row in rows if row["arm"]==arm]
        report[arm]={
            "episodes":len(data),
            "declared_source_fields_edited":data[0]["source_module_edit_count"],
            "mean_original_reward":sum(x["official_return"] for x in data)/len(data),
            "mean_actual_object_goal_distance":sum(x["mean_object_goal_distance"] for x in data)/len(data),
        }
    if len(rows)!=4*len(TEST_SEEDS):
        raise AssertionError("missing required paired 4-arm official robot runs")
    per_seed=[]
    for seed in TEST_SEEDS:
        data={x["arm"]:x for x in rows if x["seed"]==seed}
        ref=data["official-health"]["official_return"]
        close=data["installed-signed-clip-only"]["official_return"]
        atomic=data["installed-signed-atomic-matrices"]["official_return"]
        per_seed.append({
            "seed":seed,
            "clip_only_minus_healthy_reward":close-ref,
            "atomic_minus_healthy_reward":atomic-ref,
            "clip_only_minus_atomic_reward":close-atomic,
            "clip_only_object_distance_minus_healthy":
                data["installed-signed-clip-only"]["mean_object_goal_distance"]-
                data["official-health"]["mean_object_goal_distance"],
        })
    return {
        "protocol":"semrepair-signed-minimal-clip-only-actual-pusher-recovery-v1",
        "original_external_actor_sha256":weight_sha,
        "old_seen_seed_count":len(TEST_SEEDS),
        "new_holdout_seeds":0,
        "actual_official_task_replays":len(rows),
        "software_clip_cap_is_author_assumed_NOT_independently_signed":True,
        "calibration_and_external_signer_roots_verified":False,
        "publisher_policy_retrained":False,
        "clip_only_original_source_matrices_unchanged":True,
        "signed_clip_only_installed_as_real_on_disk_file":True,
        "signed_atomic_installed_as_separate_real_on_disk_file":True,
        "actor_name":"Farama Minari pretrained Pusher-v5 SAC expert",
        "exact_full_action_box_clipping_threshold":9.0,
        "clip_only_candidate_magnitude":CLIP_ONLY["clip"],
        "signed_approval_records":actual_records,
        "summary":report,
        "seed_paired":per_seed,
        "raw_rows":rows,
        "limitations":"previously inspected seeds; first-party signed artificial software shear+clip, assumed 9.1 source clip cap; not vendor security or real robot",
    }


if __name__=="__main__":
    print(json.dumps(run_clip_only_source_recovery(),indent=2,sort_keys=True))
