"""Fully executed official Pusher SAC task through signed, FILE-LOADED adapter patch.

Mechanism bridge only, NOT a new held-out efficacy trial: use the first
16 previously evaluated Pusher severe-fault seeds and same pretrained SAC
actor; compare (i) healthy original official environment actions, (ii)
faulty ACTUAL on-disk JSON adapter loaded for EVERY physical step, and (iii)
same adapter after real Ed25519 threshold-signed byte-verified atomic config
replacement. This avoids the old "blind-source-patch-candidate" branch
that simply returned the ideal action without installing any artifact.

Three mock private roots are generated in the first-party test runner and
are NOT independent hardware vendor signing roots. No source sensor origin
is authenticated; no physical robot operating system patch is installed.
The "signed source" applies to an AUTHOR-AUTHORED file-based adapter before
the unchanged official Gymnasium Pusher-v5 MuJoCo actuator.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import gymnasium as gym

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .external_pusher_transfer import load_external_pusher_checkpoint
from .signed_source_activation import (
    PROTOCOL,_canonical,read_executable_control_config,
    verified_atomic_source_activate,execute_loaded_source_action,
)

ACTOR_FAMILY="Farama-Gymnasium-Pusher-v5-seven-torque"
TEST_SEEDS=tuple(range(23011,23027))
REPLAY_FAULT={"producer":[[1.,3.5],[0.,1.]],
              "controller":[[1.,-3.5],[0.,1.]],"clip":.8}
REPLACEMENT={"producer":[[1.,0.],[0.,1.]],
             "controller":[[1.,0.],[0.,1.]],"clip":2.}
STEPS=100


def _rollout(policy,env,seed,*,config_path=None,source_digest=None):
    obs,_=env.reset(seed=seed)
    reward=0.
    goal_dist=0.
    contact_dist=0.
    for i in range(STEPS):
        a,_=policy.predict(obs,deterministic=True)
        if config_path is not None:
            cfg=read_executable_control_config(config_path,source_digest)
            proposed=execute_loaded_source_action(a,cfg)
        else:
            proposed=np.asarray(a,dtype=np.float32)
        applied=np.clip(proposed,env.action_space.low,env.action_space.high)
        obs,r,done,truncated,info=env.step(applied)
        reward+=float(r)
        goal_dist+=-float(info["reward_dist"])
        contact_dist+=-float(info["reward_near"])/.5
        if done or truncated:
            break
    if i+1!=STEPS:
        raise AssertionError("frozen official Pusher task horizon changed")
    return {"seed":seed,"official_return":reward,
            "mean_object_goal_distance":goal_dist/STEPS,
            "mean_fingertip_object_distance":contact_dist/STEPS}


def _activate_demo_owner_signed_source(path,old,new,policy_checkpoint_sha):
    # Authenticates file bytes against a first-party TEST root set, not robot
    # manufacturer or independent maintenance-domain issuers.
    roots={name:Ed25519PrivateKey.generate() for name in
           ("test-maintainer-A","test-maintainer-B","test-reviewer-C")}
    pubs={name:key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw) for name,key in roots.items()}
    deployment=sha256(_canonical({
        "actor":ACTOR_FAMILY,
        "native_action_limit":2,
        "frozen_checkpoint_sha256":policy_checkpoint_sha,
        "first_two_shoulder_fault_family":"reciprocal-linear-shear-plus-internal-clip",
    })).hexdigest()
    nonce=sha256(b"fixed-demo-nonce-not-for-production-20261010").hexdigest()
    manifest={
        "protocol":PROTOCOL,
        "current_source_sha256":sha256(old).hexdigest(),
        "replacement_source_sha256":sha256(new).hexdigest(),
        "current_epoch":1,"target_epoch":2,
        "deployment_sha256":deployment,
        "expected_nonce":nonce,
        "authorized_repair":"atomic",
        "actor_family":ACTOR_FAMILY,
    }
    serialized=_canonical(manifest)
    signatures=[
        {"issuer":owner,"ed25519_hex":roots[owner].sign(serialized).hex()}
        for owner in ("test-maintainer-A","test-reviewer-C")]
    installed=verified_atomic_source_activate(
        active_config=path,replacement_bytes=new,
        manifest=manifest,signatures=signatures,
        trusted_public_roots=pubs,expected_current_epoch=1,
        expected_nonce=nonce,expected_actor_family=ACTOR_FAMILY,
        expected_deployment_sha256=deployment,threshold=2)
    return installed,manifest


def run_real_pusher_signed_patch_task():
    actor,checkpoint_sha=load_external_pusher_checkpoint()
    old=_canonical(REPLAY_FAULT)
    new=_canonical(REPLACEMENT)
    all_episodes=[]
    with TemporaryDirectory(prefix="semrepair-signed-source-") as tmp:
        config=Path(tmp)/"active-source.json"
        config.write_bytes(old)
        with gym.make("Pusher-v5") as env:
            for seed in TEST_SEEDS:
                all_episodes.append({"method":"healthy-unmodified",
                   **_rollout(actor,env,seed)})
                all_episodes.append({"method":"faulty-actual-file-loaded",
                   **_rollout(actor,env,seed,config_path=config,
                              source_digest=sha256(old).hexdigest())})
            record,manifest=_activate_demo_owner_signed_source(
                config,old,new,checkpoint_sha)
            if config.read_bytes()!=new:
                raise AssertionError("signed source installation did not actually change bytes")
            for seed in TEST_SEEDS:
                all_episodes.append({"method":"real-signed-atomic-file-loaded",
                   **_rollout(actor,env,seed,config_path=config,
                              source_digest=sha256(new).hexdigest())})
    summary={}
    for method in ("healthy-unmodified","faulty-actual-file-loaded",
                   "real-signed-atomic-file-loaded"):
        group=[r for r in all_episodes if r["method"]==method]
        summary[method]={
            "n_episodes":len(group),
            "mean_official_return":sum(x["official_return"] for x in group)/len(group),
            "mean_real_object_goal_distance":sum(
                x["mean_object_goal_distance"] for x in group)/len(group)}
    for seed in TEST_SEEDS:
        ref=next(x for x in all_episodes if x["seed"]==seed and
                 x["method"]=="healthy-unmodified")
        patched=next(x for x in all_episodes if x["seed"]==seed and
                     x["method"]=="real-signed-atomic-file-loaded")
        if abs(ref["official_return"]-patched["official_return"])>1e-7 or \
           abs(ref["mean_object_goal_distance"]-
               patched["mean_object_goal_distance"])>1e-9:
            raise AssertionError("signed file-backed runtime patch did not restore genuine physics")
    return {
        "protocol":"pusher-real-file-loaded-threshold-ed25519-atomic-repair-v1",
        "source_file_original_sha256":sha256(old).hexdigest(),
        "source_file_installed_sha256":sha256(new).hexdigest(),
        "verified_source_manifest":manifest,
        "activation_record":record,
        "authentic_external_robot_signer_key_count":0,
        "test_only_author_generated_signing_key_count":3,
        "actually_swapped_on_disk_runtime_adapter_file":True,
        "actual_manufacturer_operating_system_software_installation":False,
        "frozen_pretrained_pusher_weight_sha256":checkpoint_sha,
        "policy_training_by_authors":0,
        "new_holdout_seeds":0,
        "already_used_seed_count":len(TEST_SEEDS),
        "episode_count":len(all_episodes),
        "summary":summary,
        "rows":all_episodes,
        "limitation":"mechanism bridge on previously inspected seeds; test-only signing roots, not independent vendor certificate or trustworthy hardware tap",
    }

if __name__=="__main__":
    print(json.dumps(run_real_pusher_signed_patch_task(),indent=2,sort_keys=True))
