"""Dual-domain, conditional ROBUST source authorization with executable activation.

A nominally correct source patch can be signed by owners while still being
uncertain at real actuator scales. This adds a SECOND signer domain for the
producer/controller coefficient error bounds. The activation gate requires:

  1) exact expected calibration profile BYTES and SHA-256 pinned by caller;
  2) 2 distinct VALID Ed25519 signers from the calibration root set;
  3) NO calibration signing public key reused as a source patch signer root;
  4) calibration profile bound to actor, deployment digest, exact proposed
     replacement source SHA, deployment action radius, activation epoch;
  5) interval-robust certificate valid over ALL commands in [-R,R]^2:
      internal source never saturates under bounded P drift, and
      downstream infinity-norm action error <= signed budget;
  6) independent verified 2-of-3 signed patch source manifest including
     original/replacement bytes, epoch, nonce, action semantics, component
     allowlist; atomic actual on-disk config replacement.

This demonstrates dual-authority, fail-closed proof-carrying actuator adapter
CONFIG activation using first-party ephemeral signer roots. It is NOT real
manufacturer approval, secure calibration, trusted hardware telemetry,
vendor ROS deployment, or defense against compromised host root privileges.

STANDARD cryptography and interval arithmetic, not invented primitives.
The novel research claim is conditional composition of physical action
authority, calibration uncertainty, and runnable source semantics.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import math

from .signed_source_activation import (
    _canonical,_validate_source_config,_signature_quorum,
    verified_atomic_source_activate,SHA256_RE
)
from .robust_source_envelope_certificate import robust_repair_certificate

PROFILE_TYPE="semrepair-authority-calibration-profile-v1"
PROFILE_FIELDS=frozenset({
    "protocol","actor_family","deployment_sha256",
    "replacement_source_sha256","target_epoch","deployment_action_radius",
    "producer_abs_coeff_error","controller_abs_coeff_error",
    "max_allowed_abs_output_action_error"
})


def verify_calibration_and_activate(
    *,active_config:Path,replacement_bytes:bytes,
    patch_manifest:dict,patch_signatures:list[dict],
    trusted_patch_roots:dict[str,bytes],
    calibration_profile_bytes:bytes,calibration_signatures:list[dict],
    trusted_calibration_roots:dict[str,bytes],
    expected_calibration_profile_sha256:str,
    expected_current_epoch:int,expected_nonce:str,
    expected_actor_family:str,expected_deployment_sha256:str,
    expected_action_radius:float,threshold:int=2
)->dict:
    """Reject before writing unless independent signed bound attests robust repair."""
    if not isinstance(calibration_profile_bytes,bytes) or len(calibration_profile_bytes)>4096:
        raise ValueError("calibration bytes unavailable or too large")
    actual_cal_hash=sha256(calibration_profile_bytes).hexdigest()
    if not isinstance(expected_calibration_profile_sha256,str) or \
       not SHA256_RE.fullmatch(expected_calibration_profile_sha256) or \
       actual_cal_hash!=expected_calibration_profile_sha256:
        raise ValueError("calibration profile differs from independently pinned digest")
    if set(trusted_calibration_roots.values())&set(trusted_patch_roots.values()):
        raise ValueError("calibration and source signer root domains are not independent")
    try:
        from json import loads
        profile=loads(calibration_profile_bytes.decode("utf-8"))
    except (ValueError,UnicodeDecodeError,TypeError) as e:
        raise ValueError("malformed calibration bytes") from e
    if not isinstance(profile,dict) or set(profile)!=PROFILE_FIELDS or \
       _canonical(profile)!=calibration_profile_bytes:
        raise ValueError("uncanonicalized or incomplete calibration profile")
    if profile["protocol"]!=PROFILE_TYPE or \
       profile["actor_family"]!=expected_actor_family or \
       profile["actor_family"]!=patch_manifest.get("actor_family") or \
       profile["deployment_sha256"]!=expected_deployment_sha256 or \
       profile["deployment_sha256"]!=patch_manifest.get("deployment_sha256") or \
       profile["replacement_source_sha256"]!=sha256(replacement_bytes).hexdigest() or \
       profile["replacement_source_sha256"]!=patch_manifest.get("replacement_source_sha256") or \
       type(profile["target_epoch"]) is not int or \
       profile["target_epoch"]!=expected_current_epoch+1 or \
       type(profile["deployment_action_radius"]) not in (int,float) or \
       not math.isfinite(profile["deployment_action_radius"]) or \
       abs(profile["deployment_action_radius"]-expected_action_radius)>1e-12:
        raise ValueError("calibration profile does not bind exact source patch/deployment/action epoch")

    calibrator_issuers=_signature_quorum(
        calibration_profile_bytes,calibration_signatures,
        trusted_calibration_roots,threshold)
    new=_validate_source_config(replacement_bytes)
    cert=robust_repair_certificate(
        producer=new["producer"],controller=new["controller"],
        producer_abs_error_bounds=profile["producer_abs_coeff_error"],
        controller_abs_error_bounds=profile["controller_abs_coeff_error"],
        full_command_radius=expected_action_radius,
        internal_clip_limit=new["clip"],
        allowed_max_output_error=profile["max_allowed_abs_output_action_error"])
    if not cert["certified"]:
        raise ValueError("signed calibration cannot certify entire source deployment action interval")

    # Harden even under a trusted root that simply lies about profile
    # authentication: current code's caller is still responsible for
    # pinning actual source+calibration public root keys externally.
    installed=verified_atomic_source_activate(
        active_config=active_config,replacement_bytes=replacement_bytes,
        manifest=patch_manifest,signatures=patch_signatures,
        trusted_public_roots=trusted_patch_roots,
        expected_current_epoch=expected_current_epoch,
        expected_nonce=expected_nonce,
        expected_actor_family=expected_actor_family,
        expected_deployment_sha256=expected_deployment_sha256,
        expected_action_radius=expected_action_radius,
        threshold=threshold)
    return {
        "decision":"DEMONSTRATION_SIGNED_ROBUST_SOURCE_CONFIG_INSTALLED",
        "verified_source_signers":installed["verified_distinct_root_issuers"],
        "verified_independent_calibration_signers":list(calibrator_issuers),
        "verified_calibration_profile_sha256":actual_cal_hash,
        "source_config_installed_sha256":installed["installed_bytes_sha256"],
        "source_activation":installed,
        "worst_case_envelope_certificate":cert,
        "calibration_bounds_externally_measured":False,
        "vendor_signing_roots":0,
        "cryptographic_signature_domains":2,
        "claim_scope":"first-party temporary signing roots and EXOGENOUS assumed interval error magnitudes; not field-certified safety",
    }
