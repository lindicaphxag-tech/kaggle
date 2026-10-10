"""PCPR Causal Evidence Closure (CEC), v0.4.

This is a SMALL, AUDITABLE REFERENCE CHECKER, not proof of patch correctness.
The graph reasons about independently sourced *obligation justifications*,
not the truth of the requirement or the completeness of the effect inventory.

Security model: hash identity here is local tamper detection, not independent
attestation of the remote Git tree, trusted runtime, test oracle or reviewer.
A patch may suggest what to measure but may never be its own requirement
oracle. Graph paths with unsupported edges do not gain authority by reachability.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Mapping, Sequence

ROOT_KINDS=frozenset({"UNCHANGED_CONSUMER","BASELINE_TEST","EXTERNAL_API_SPEC","REVIEWER_APPROVED_SPEC"})
TIERS=("operator","function","package","deployment")
PROPERTIES=frozenset({"dtype","device","shape","autograd","registration",
                      "checkpoint","aliasing","mask_boundary","optimizer_state",
                      "parameter_domain","numerical_finiteness","behavior"})

@dataclass(frozen=True)
class Vertex:
    key: str
    kind: str
    origin: str     # immutable BEFORE, UNCHANGED, EXTERNAL, AFTER or UNKNOWN
    source_sha256: str
    semantic_property: str

@dataclass(frozen=True)
class Dependency:
    source: str
    target: str
    relation: str
    verified: bool
    provenance: str    # BEFORE/UNCHANGED/EXTERNAL/AFTER/UNKNOWN

@dataclass(frozen=True)
class RunReceipt:
    requirement_vertex: str
    semantic_property: str
    tier: str
    context_sha256: str
    before_sha256: str
    after_sha256: str
    old_observation: str
    new_observation: str
    oracle_source_vertex: str
    negative_control: str   # "VIOLATES", "MISSING", "PASSES"
    preserving_control: str # "PASSES", "MISSING", "VIOLATES"
    native_replayed: bool
    independent_oracle_pass: bool
    evidence_scope: str

def sha(data: bytes|str):
    if isinstance(data,str): data=data.encode("utf8")
    return hashlib.sha256(data).hexdigest()

def _verified_hex(s):
    return isinstance(s,str) and len(s)==64 and all(c in "0123456789abcdef" for c in s)

class CausalEvidenceGraph:
    def __init__(self, vertices:Sequence[Vertex], edges:Sequence[Dependency], 
                 trusted_content_hashes:Mapping[str,str]):
        self.vertices={v.key:v for v in vertices}
        if len(self.vertices)!=len(vertices):raise ValueError("Duplicate graph vertex")
        self.edges=tuple(edges)
        self.trusted_hashes=dict(trusted_content_hashes)
        for k,v in self.trusted_hashes.items():
            if not k or not _verified_hex(v): raise ValueError("Invalid trusted digest")
        if any(e.source not in self.vertices or e.target not in self.vertices for e in edges):
            raise ValueError("Dangling dependency")
        for v in vertices:
            if v.semantic_property not in PROPERTIES:raise ValueError("Unknown semantic property")
            if not _verified_hex(v.source_sha256):raise ValueError("Missing source digest")
        self.incoming={v.key:[] for v in vertices}
        for e in edges:self.incoming[e.target].append(e)

    def root_authority(self,key):
        v=self.vertices[key]
        return (v.kind in ROOT_KINDS and v.origin in ("BEFORE","UNCHANGED","EXTERNAL")
                and self.trusted_hashes.get(key)==v.source_sha256)

    def closure(self,target:str,semantic_property:str)->dict:
        """All directed prerequisite paths, fail-closed on cycles or taint.
        
        A single verified independent root is necessary, but is deliberately
        INSUFFICIENT to establish correctness of the stated obligation.
        """
        if target not in self.vertices: 
            return {"status":"ABSTAIN_UNKNOWN_TARGET","roots":[],"unresolved":[target]}
        visited=set(); visiting=set(); roots=set(); problems=set(); edge_count=0
        def dfs(k):
            nonlocal edge_count
            if k in visiting:
                problems.add("CYCLE:"+k);return
            if k in visited:return
            visiting.add(k)
            v=self.vertices[k]
            if v.semantic_property!=semantic_property:
                problems.add("PROPERTY_MISMATCH:"+k)
            if v.origin in ("AFTER","UNKNOWN"):
                problems.add("TAINTED_ORIGIN:"+k)
            if not self.trusted_hashes.get(k)==v.source_sha256:
                problems.add("UNATTESTED_BYTES:"+k)
            if self.root_authority(k):roots.add(k)
            for e in self.incoming[k]:
                edge_count+=1
                if not e.verified:problems.add("UNVERIFIED_EDGE:"+e.source+"->"+k)
                if e.provenance in ("AFTER","UNKNOWN"):problems.add("TAINTED_EDGE:"+e.source+"->"+k)
                if not e.relation:problems.add("UNKNOWN_RELATION:"+e.source+"->"+k)
                dfs(e.source)
            visiting.remove(k);visited.add(k)
        dfs(target)
        if not roots:problems.add("NO_INDEPENDENT_SOURCE")
        return {"status":"INDEPENDENTLY_ANCHORED_HYPOTHESIS" if not problems else "ABSTAIN_INCOMPLETE_PROVENANCE",
                "roots":sorted(roots),"unresolved":sorted(problems),
                "visited":len(visited),"edges":edge_count,
                "scope":"Source-independent JUSTIFICATION CANDIDATE, not verified specification truth"}

    def evaluate(self, receipt:RunReceipt, code_hashes:Mapping[str,str])->dict:
        """Authorization NEVER exceeds one finite observation.

        Author supplied booleans are NOT a secure oracle; external verifier
        must independently replay receipts and check source identity.  This
        checker validates only the structure and allowed inference.
        """
        rejects=[]
        if receipt.semantic_property not in PROPERTIES:rejects.append("UNKNOWN_PROPERTY")
        if receipt.tier not in TIERS:rejects.append("UNKNOWN_TIER")
        graph=self.closure(receipt.requirement_vertex,receipt.semantic_property)
        oracle=self.closure(receipt.oracle_source_vertex,receipt.semantic_property)
        if graph["status"]!="INDEPENDENTLY_ANCHORED_HYPOTHESIS":
            rejects.append("MISSING_REQUIREMENT_PROVENANCE")
        if oracle["status"]!="INDEPENDENTLY_ANCHORED_HYPOTHESIS":
            rejects.append("MISSING_INDEPENDENT_ORACLE_PROVENANCE")
        if not self.root_authority(receipt.oracle_source_vertex):
            rejects.append("ORACLE_IS_NOT_INDEPENDENT_ROOT")
        for attr in ("before_sha256","after_sha256","context_sha256"):
            expected=code_hashes.get(attr)
            if not _verified_hex(expected) or getattr(receipt,attr)!=expected:
                rejects.append("SOURCE_OR_CONTEXT_IDENTITY_MISMATCH:"+attr)
        if not receipt.native_replayed:rejects.append("NOT_NATIVE_EXECUTED")
        if receipt.negative_control!="VIOLATES":rejects.append("NO_DISCRIMINATING_NEGATIVE_CONTROL")
        if receipt.preserving_control!="PASSES":rejects.append("NO_PRESERVING_CONTROL")
        if not receipt.independent_oracle_pass:rejects.append("ORACLE_NOT_SATISFIED")
        if receipt.old_observation==receipt.new_observation:
            rejects.append("NO_OBSERVED_DIFFERENCE")
        if receipt.evidence_scope!="MATCHED_SINGLE_CONTEXT":
            rejects.append("UNSUPPORTED_SCOPE_PROMOTION")
        verdict="LOCAL_EFFECT_OBSERVED" if not rejects else "ABSTAIN"
        # Independent requirements are hypotheses, not machine-proved truths.
        return {"verdict":verdict, "scope":receipt.evidence_scope,
                "semantic_property":receipt.semantic_property,
                "tier":receipt.tier,"reasons":sorted(set(rejects)),
                "requirement_roots":graph["roots"],"oracle_roots":oracle["roots"],
                "whole_patch_authorized":False, "environment_generalization":False,
                "independent_requirement_truth_proven":False}

def audit_bundle(graph:CausalEvidenceGraph,receipts:Sequence[RunReceipt],
                 expected_hashes:Mapping[str,str]):
    outcomes=[graph.evaluate(r,expected_hashes) for r in receipts]
    seen={}
    for r in receipts:
        key=(r.semantic_property,r.tier,r.context_sha256)
        seen.setdefault(key,set()).add((r.old_observation,r.new_observation))
    for i,r in enumerate(receipts):
        key=(r.semantic_property,r.tier,r.context_sha256)
        if len(seen[key])>1:
            outcomes[i]["verdict"]="ABSTAIN"
            outcomes[i]["reasons"]=sorted(set(outcomes[i]["reasons"]+["CONTRADICTORY_RECEIPTS"]))
    return {"schema":"pcpr-cec-0.4","outcomes":outcomes,
            "n_local_effects":sum(x["verdict"]=="LOCAL_EFFECT_OBSERVED" for x in outcomes),
            "n_abstain":sum(x["verdict"]=="ABSTAIN" for x in outcomes),
            "whole_patch_authorized":False,
            "provenance_limits":"Input hashes and flags alone are not independent remote attestation"}

def canonical_json_digest(record):
    return sha(json.dumps(record,sort_keys=True,separators=(",",":"),ensure_ascii=False))

def example():
    """DEVELOPMENT-ONLY constructed evidence; no external repository truth."""
    a,b,c=sha("before-source"),sha("after-source"),sha("fixed-tensor-context")
    roots=[Vertex("orig-consumer","UNCHANGED_CONSUMER","UNCHANGED",sha("consumer"),"checkpoint"),
           Vertex("api","EXTERNAL_API_SPEC","EXTERNAL",sha("nn.Module.state_dict"),"checkpoint"),
           Vertex("candidate","DERIVED_EFFECT","BEFORE",sha("before-state-use"),"checkpoint")]
    edges=[Dependency("orig-consumer","candidate","member-read",True,"BEFORE"),
           Dependency("api","candidate","PyTorch serialization contract",True,"EXTERNAL")]
    graph=CausalEvidenceGraph(roots,edges,{v.key:v.source_sha256 for v in roots})
    r=RunReceipt("candidate","checkpoint","operator",c,a,b,"missing-key","present-key",
                 "api","VIOLATES","PASSES",True,True,"MATCHED_SINGLE_CONTEXT")
    return audit_bundle(graph,[r],{"before_sha256":a,"after_sha256":b,"context_sha256":c})

if __name__=="__main__":
    print(json.dumps(example(),indent=2))
