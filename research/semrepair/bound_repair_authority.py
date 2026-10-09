"""Concrete repair authority binding, not mere string-label consensus.

An authorization binds the PATCH + IMPLEMENTATION + EVIDENCE + DEPENDENCY
digests. No actuation or code patch application is performed by this module.
Hidden-world source isolation must be provided by the external experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from fractions import Fraction
from typing import Callable

from .blind_authority_runner import ProbeSpec, execute_blind


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RepairAuthority:
    patch_sha256: str
    implementation_sha256: str
    evidence_sha256: str
    dependency_sha256: str

    def __post_init__(self):
        if any(not _SHA256.fullmatch(value) for value in (
            self.patch_sha256, self.implementation_sha256,
            self.evidence_sha256, self.dependency_sha256
        )):
            raise ValueError("authority requires four real SHA-256 hex digests")

    @property
    def binding(self) -> str:
        return sha256(json.dumps(self.__dict__, sort_keys=True,
                                 separators=(",", ":")).encode()).hexdigest()


def execute_bound_repair(
    *, worlds: tuple[str, ...],
    authorities: dict[str, RepairAuthority],
    probes: dict[str, ProbeSpec],
    policy: Callable[[frozenset[str]], str],
    observe: Callable[[str], Fraction],
    total_cost_budget: Fraction,
    max_single_probe_risk: Fraction,
) -> dict:
    """Return an authoritative digest only if all survivors agree exactly."""
    if set(authorities) != set(worlds):
        raise ValueError("missing bound repair authority")
    lookup = {a.binding: a for a in authorities.values()}
    out = execute_blind(
        worlds=worlds,
        repair_by_world={w: a.binding for w, a in authorities.items()},
        probes=probes, policy=policy, observe=observe,
        total_cost_budget=total_cost_budget,
        max_single_probe_risk=max_single_probe_risk)
    selected = lookup.get(out.authorized_repair)
    return {
        "decision": out.status,
        "bound_repair": selected,
        "authorization_digest": out.authorized_repair,
        "surviving_worlds": out.surviving_worlds,
        "probe_cost": out.probe_cost,
        "not_an_actuation_or_safety_certificate": True,
    }
