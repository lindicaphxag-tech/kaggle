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
import stat

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



def _digest_no_follow(path: Path) -> str:
    """Hash from ONE opened FD, refusing a replaced symlink or FIFO.

    This eliminates check-then-read symlink substitution; hashing itself
    cannot prevent modification after the FD closes and before execution.
    """
    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("secure no-follow open is unavailable on this platform")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os,"O_CLOEXEC",0) | getattr(os,"O_NONBLOCK",0)
    fd = os.open(os.fspath(path), flags)
    try:
        first = os.fstat(fd)
        if not stat.S_ISREG(first.st_mode):
            raise ValueError("non-regular patch/evidence artifact")
        digest=sha256()
        while True:
            part=os.read(fd,1024*1024)
            if not part:
                break
            digest.update(part)
        second=os.fstat(fd)
        first_state=(first.st_dev,first.st_ino,first.st_size,first.st_mtime_ns,first.st_ctime_ns)
        second_state=(second.st_dev,second.st_ino,second.st_size,second.st_mtime_ns,second.st_ctime_ns)
        if first_state!=second_state:
            raise ValueError("artifact changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(fd)

def attest_current_artifacts(authority: RepairAuthority, files: AuthorityFiles) -> dict:
    """Fail closed if any artifact or digest is not an actual frozen byte object."""
    observed = {}
    failures = []
    for attr, path in files.items():
        path = Path(path)
        try:
            observed[attr] = _digest_no_follow(path)
        except (OSError, ValueError) as exc:
            failures.append(attr + ":unreadable_symlink_or_nonregular")
            continue
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
