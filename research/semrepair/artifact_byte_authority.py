"""Verify actual content bytes before acting on a SemRepair authority digest.

Four syntactically valid 64-character strings are NOT evidence that the
corresponding bytes exist or match. This helper reopens all four artifacts and
binds authority to current bytes, refusing on tampering, missing files, or
symbolic links. It does not install a patch or certify robot physical safety.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
import os

from .bound_repair_authority import RepairAuthority


@dataclass(frozen=True)
class AuthorityFiles:
    patch: Path
    implementation: Path
    evidence: Path
    dependency: Path

    def items(self):
        return (
            ("patch_sha256", self.patch),
            ("implementation_sha256", self.implementation),
            ("evidence_sha256", self.evidence),
            ("dependency_sha256", self.dependency),
        )


def attest_current_artifacts(authority: RepairAuthority, files: AuthorityFiles) -> dict:
    """Fail closed if any artifact or digest is not an actual frozen byte object."""
    observed = {}
    failures = []
    for attr, path in files.items():
        path = Path(path)
        # Symlinks can change target after the caller selects paths. Prefer a
        # pinned immutable checkout/content-addressed snapshot for the study.
        if path.is_symlink() or not path.is_file():
            failures.append(attr + ":missing_or_symlink")
            continue
        observed[attr] = sha256(path.read_bytes()).hexdigest()
        if observed[attr] != getattr(authority, attr):
            failures.append(attr + ":sha256_mismatch")
    return {
        "valid": not failures,
        "failures": failures,
        "observed_sha256": observed,
        "authority_digest": authority.binding,
        "claim_scope": "files only, not execution or robot safety",
    }


def authorize_verified_files(decision: dict, files: AuthorityFiles) -> dict:
    """Upgrade an identified authority to file-byte authorization or REFUSE."""
    auth = decision.get("bound_repair")
    if decision.get("decision") != "AUTHORIZED" or not isinstance(auth, RepairAuthority):
        return {"decision": "REFUSED_NO_BOUND_AUTHORITY", "accepted": False}
    verification = attest_current_artifacts(auth, files)
    if not verification["valid"]:
        return {"decision": "REFUSED_ARTIFACT_MISMATCH", "accepted": False,
                "file_evidence": verification}
    return {"decision": "FILE_BYTES_AUTHORIZED_ONLY", "accepted": True,
            "file_evidence": verification,
            "warning": "revalidate immediately before actual patch execution; no physical safety result"}
