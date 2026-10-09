"""Pinned SemRepair artifact bytes: avoid verify-path/execute-different-path races.

The downstream consumer receives ONLY the exact byte buffers whose hashes
were checked, never fresh filenames. This does not implement sandboxed patch
execution, OS enforcement, robot safety, or adversarial hardware attestation.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable
import os
import stat

from .bound_repair_authority import RepairAuthority
from .artifact_byte_authority import AuthorityFiles

MAX_ARTIFACT_BYTES=128*1024*1024


@dataclass(frozen=True)
class VerifiedBytes:
    patch: bytes
    implementation: bytes
    evidence: bytes
    dependency: bytes
    authority_binding: str


def _pinned_bytes(path:Path,expected:str)->bytes:
    if not hasattr(os,"O_NOFOLLOW"):
        raise OSError("no-follow descriptor support unavailable")
    flags=os.O_RDONLY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NONBLOCK",0)
    fd=os.open(os.fspath(path),flags)
    try:
        pre=os.fstat(fd)
        if not stat.S_ISREG(pre.st_mode) or pre.st_size>MAX_ARTIFACT_BYTES:
            raise ValueError("artifact must be a small bounded regular file")
        blocks=[]
        used=0
        while True:
            chunk=os.read(fd,1024*1024)
            if not chunk:break
            used+=len(chunk)
            if used>MAX_ARTIFACT_BYTES:
                raise ValueError("artifact changed or exceeded byte limit")
            blocks.append(chunk)
        post=os.fstat(fd)
        attrs=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
        if attrs(pre)!=attrs(post):
            raise ValueError("artifact mutated during pinned descriptor read")
        content=b"".join(blocks)
        if sha256(content).hexdigest()!=expected:
            raise ValueError("pinned artifact bytes do not match authorized SHA-256")
        return content
    finally:
        os.close(fd)


def build_pinned_snapshot(authority:RepairAuthority, files:AuthorityFiles)->VerifiedBytes:
    expected=dict((name,getattr(authority,name)) for name,_ in files.items())
    measured={}
    for key,p in files.items():
        measured[key]=_pinned_bytes(Path(p),expected[key])
    return VerifiedBytes(
        patch=measured["patch_sha256"],
        implementation=measured["implementation_sha256"],
        evidence=measured["evidence_sha256"],
        dependency=measured["dependency_sha256"],
        authority_binding=authority.binding,
    )


def handoff_same_verified_bytes(
    *, decision:dict, files:AuthorityFiles,
    consumer:Callable[[VerifiedBytes],None]
)->dict:
    """Delivers a frozen byte snapshot; never promises code execution success."""
    authority=decision.get("bound_repair")
    if decision.get("decision")!="AUTHORIZED" or not isinstance(authority,RepairAuthority):
        return {"status":"REFUSE", "consumer_called":False}
    try:
        snap=build_pinned_snapshot(authority,files)
    except (OSError,ValueError):
        return {"status":"REFUSED_SOURCE_BYTES","consumer_called":False}
    # No path names are handed to the consumer. If a different component
    # reopens the paths, these guarantees explicitly cease to apply.
    consumer(snap)
    return {
        "status":"VERIFIED_BYTES_HANDED_OFF",
        "consumer_called":True,
        "authority_binding":authority.binding,
        "patch_sha256":sha256(snap.patch).hexdigest(),
        "warning":"NOT proof the consumer installed the patch or robot is safe",
    }
