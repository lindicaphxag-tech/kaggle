"""SemRepair test-only trusted-origin challenge binding for actuator tap traces.

This module illustrates WHY a source-observation attestation is an *additional*
measurement rather than something one can infer from matching robot motion.

HMAC is a standard primitive, NOT a new contribution or hardware secure boot:
a protected issuer key and honest measurement agent are explicit assumptions.
No private production key is embedded or requested. This is a simulated
tamper/replay threat model with MuJoCo frames, not real secure hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from hmac import new as hmac_new, compare_digest
import json

from .mujoco_native_authority import WORLDS, rollout
from .native_measured_blind_replay import HELD_EXCITATIONS
from .open_catalog_source_gap import (
    known_source_digest,decide_from_sensor_traces,
)

CHANNELS=("external","producer_to_controller_ctrl","transmission_force")


@dataclass(frozen=True)
class SignedMeasurement:
    nonce: str
    source_sha256: str
    observation_sha256: str
    hmac_sha256: str


def _observations_digest(measured: dict) -> str:
    """Canonical raw floating sensor samples, without 8-decimal truncation."""
    raw={ch:measured[ch] for ch in CHANNELS}
    blob=json.dumps(raw,sort_keys=True,separators=(",",":"),allow_nan=False)
    return sha256(blob.encode()).hexdigest()


def _message(nonce: str, source_sha256: str, trace_sha256: str) -> bytes:
    if not nonce or len(source_sha256)!=64 or len(trace_sha256)!=64:
        raise ValueError("invalid claim binding")
    return json.dumps(
        [nonce,source_sha256,trace_sha256],
        separators=(",",":"),ensure_ascii=True).encode()


def issue_measurement(*, trusted_issuer_key:bytes, source_sha256:str,
                      observations:dict, nonce:str) -> SignedMeasurement:
    """Only the trusted measurement issuer, not the robot repair agent, signs."""
    if len(trusted_issuer_key)<16:
        raise ValueError("test issuer key too weak")
    digest=_observations_digest(observations)
    signature=hmac_new(
        trusted_issuer_key,_message(nonce,source_sha256,digest),"sha256"
    ).hexdigest()
    return SignedMeasurement(nonce,source_sha256,digest,signature)


def verify_measurement(*, trusted_verifier_key:bytes, attestation:SignedMeasurement,
                       observations:dict, requested_nonce:str,
                       catalog_source_hashes:set[str]) -> bool:
    if not requested_nonce or requested_nonce!=attestation.nonce:
        return False
    if attestation.source_sha256 not in catalog_source_hashes:
        return False
    try:
        digest=_observations_digest(observations)
        sig=hmac_new(
            trusted_verifier_key,
            _message(attestation.nonce,attestation.source_sha256,
                     attestation.observation_sha256),
            "sha256").hexdigest()
    except (TypeError,ValueError,KeyError):
        return False
    return (compare_digest(digest,attestation.observation_sha256)
            and compare_digest(sig,attestation.hmac_sha256))


def authorized_on_verified_trace(
    *, trusted_verifier_key:bytes, evidence:SignedMeasurement,
    observations:dict, requested_nonce:str,
    catalog_templates:dict[str,dict]
) -> dict:
    catalog={known_source_digest(w) for w in WORLDS}
    if not verify_measurement(
        trusted_verifier_key=trusted_verifier_key,attestation=evidence,
        observations=observations,requested_nonce=requested_nonce,
        catalog_source_hashes=catalog):
        return {"answer":"REFUSE","reason":"failed_origin_or_transcript",
                "attestation_verified":False}
    outcome=decide_from_sensor_traces(
        measured=observations,templates=catalog_templates,
        source_digest=evidence.source_sha256,
        enforce_source_provenance=True)
    return {**outcome,"attestation_verified":True}


def study() -> dict:
    # Fixture is ephemeral research-only shared HMAC key, not external
    # production credential or independently protected hardware device.
    fixture_key=sha256(b"toy-test-issuer-key-not-for-deployment-20261010").digest()
    valid=0
    substituted_rejected=0
    stale_rejected=0
    alias_cases=(("gain_fault","nuisance_alias"),
                 ("nuisance_alias","gain_fault"))
    across=[]
    for wave,commands in enumerate(HELD_EXCITATIONS):
        catalog={w.name:rollout(w,commands) for w in WORLDS}
        for w in WORLDS:
            nonce=f"wave-{wave}:source-{w.name}:challenge"
            signed=issue_measurement(
                trusted_issuer_key=fixture_key,
                source_sha256=known_source_digest(w),
                observations=catalog[w.name],nonce=nonce)
            response=authorized_on_verified_trace(
                trusted_verifier_key=fixture_key,evidence=signed,
                observations=catalog[w.name],requested_nonce=nonce,
                catalog_templates=catalog)
            if response["answer"]!=w.authority:
                raise AssertionError("signed legitimate observation was falsely rejected")
            valid+=1
            replay=authorized_on_verified_trace(
                trusted_verifier_key=fixture_key,evidence=signed,
                observations=catalog[w.name],requested_nonce=nonce+"-fresh",
                catalog_templates=catalog)
            if replay["answer"]!="REFUSE":
                raise AssertionError("stale challenge was accepted")
            stale_rejected+=1
        for source,foreign in alias_cases:
            w=next(w for w in WORLDS if w.name==source)
            nonce=f"wave-{wave}:tap-source-{w.name}:challenge"
            signed=issue_measurement(
                trusted_issuer_key=fixture_key,
                source_sha256=known_source_digest(w),
                observations=catalog[w.name],nonce=nonce)
            modified={**catalog[w.name],
                      "transmission_force":catalog[foreign]["transmission_force"]}
            response=authorized_on_verified_trace(
                trusted_verifier_key=fixture_key,evidence=signed,
                observations=modified,requested_nonce=nonce,
                catalog_templates=catalog)
            if response["answer"]!="REFUSE" or response["attestation_verified"]:
                raise AssertionError("forged actuator-force transcript was authorized")
            substituted_rejected+=1
            across.append({"source":source,"spoof":foreign,"wave":wave,
                           "post_signing_force_swap_refused":True})
    return {
        "experiment":"trusted-origin-tap-transcript-challenge-under-HMAC-assumption",
        "known_world_wave_cells":valid,
        "authentic_legitimate_authorized":valid,
        "post_signing_force_substitution_attempts":substituted_rejected,
        "post_signing_force_substitution_refused":substituted_rejected,
        "stale_nonce_replays_refused":stale_rejected,
        "attacked_cells":across,
        "critical_scope":"protected honest issuer assumed, in-memory test HMAC only; does not protect against compromised measurement agent, copied key, physical robot exploits, unavailable certified boot, or independent generalization",
    }


if __name__=="__main__":
    print(json.dumps(study(),indent=2,sort_keys=True))
