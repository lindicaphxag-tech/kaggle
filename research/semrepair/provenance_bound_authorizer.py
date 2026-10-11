"""SemRepair: decision-sufficient evidence + source trust + actual patch bytes.

An executable finite-world *integrity certificate* with explicit abstention.
The independently supplied probe trust map and source isolation are ASSUMED,
not established by these Python functions. No patch is applied to any robot.
"""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
from typing import Callable

from .bound_repair_authority import RepairAuthority, execute_bound_repair
from .blind_authority_runner import ProbeSpec
from .artifact_byte_authority import AuthorityFiles, attest_current_artifacts


def execute_provenance_bound_authorization(
    *, worlds: tuple[str,...],
    authorities: dict[str,RepairAuthority],
    probes: dict[str,ProbeSpec],
    policy: Callable[[frozenset[str]],str],
    observe: Callable[[str],Fraction],
    externally_trusted_probe_names: frozenset[str],
    candidate_files_by_binding: dict[str, AuthorityFiles],
    total_cost_budget: Fraction,
    max_single_probe_risk: Fraction,
) -> dict:
    """Refuse before requesting an untrusted probe or a missing patch file.

    The declared trusted-probe list must be supplied by an independent
    attestation mechanism. Caller-supplied booleans alone offer no security
    guarantee against an attacker who controls that metadata.
    """
    if not externally_trusted_probe_names <= set(probes):
        raise ValueError("unknown trusted probe ID")
    attempted=[]
    def trusted_policy(belief):
        chosen=policy(belief)
        if chosen not in (None,"stop","reject") and chosen not in externally_trusted_probe_names:
            attempted.append(chosen)
            return "reject"
        return chosen
    decision=execute_bound_repair(
        worlds=worlds,authorities=authorities,probes=probes,
        policy=trusted_policy,observe=observe,
        total_cost_budget=total_cost_budget,
        max_single_probe_risk=max_single_probe_risk)
    if attempted:
        return {
            "decision":"REFUSED_UNTRUSTED_EVIDENCE",
            "untrusted_probe":attempted[0],
            "authorization_digest":None,
            "spent_probe_cost":str(decision["probe_cost"]),
            "source_trust_externally_proven":False,
        }
    if decision["decision"]!="AUTHORIZED":
        return {
            "decision":decision["decision"],
            "authorization_digest":None,
            "spent_probe_cost":str(decision["probe_cost"]),
            "source_trust_externally_proven":False,
        }
    chosen=decision["bound_repair"]
    if not isinstance(chosen,RepairAuthority):
        raise AssertionError("bound authority agreed but no concrete object")
    binding=decision["authorization_digest"]
    if chosen.binding!=binding:
        raise AssertionError("authority mismatch")
    files=candidate_files_by_binding.get(binding)
    if files is None:
        return {
            "decision":"REFUSED_MISSING_PATCH_ARTIFACT",
            "authorization_digest":None,
            "spent_probe_cost":str(decision["probe_cost"]),
        }
    evidence=attest_current_artifacts(chosen,files)
    if not evidence["valid"]:
        return {
            "decision":"REFUSED_INVALID_PATCH_ARTIFACT",
            "authorization_digest":None,
            "failed_artifacts":evidence["failures"],
            "spent_probe_cost":str(decision["probe_cost"]),
        }
    return {
        "decision":"FILE_BYTES_AUTHORIZED_ONLY",
        "authorization_digest":binding,
        "surviving_worlds":decision["surviving_worlds"],
        "spent_probe_cost":str(decision["probe_cost"]),
        "file_evidence":evidence,
        "source_trust_externally_proven":False,
        "not_claimed":"trusted source is caller assumption; bytes can change after verification; no automated patch/actuation/robot safety",
    }
