import unittest
from dataclasses import replace
from causal_evidence_closure import (
    CausalEvidenceGraph,Vertex,Dependency,RunReceipt,audit_bundle,
    canonical_json_digest,sha,example,
)
P="checkpoint"
def setup():
    vertices=[
      Vertex("consumer","UNCHANGED_CONSUMER","UNCHANGED",sha("git consumer"),P),
      Vertex("oracle","EXTERNAL_API_SPEC","EXTERNAL",sha("PyTorch state API"),P),
      Vertex("use","DERIVED_EFFECT","BEFORE",sha("pre-change line"),P),
    ]
    edges=[Dependency("consumer","use","reads state",True,"BEFORE"),
           Dependency("oracle","use","API contract",True,"EXTERNAL")]
    g=CausalEvidenceGraph(vertices,edges,{v.key:v.source_sha256 for v in vertices})
    hashes={x:sha(x) for x in ("before_sha256","after_sha256","context_sha256")}
    receipt=RunReceipt("use",P,"operator",hashes["context_sha256"],
             hashes["before_sha256"],hashes["after_sha256"],
             "buffer-absent","buffer-present","oracle",
             "VIOLATES","PASSES",True,True,"MATCHED_SINGLE_CONTEXT")
    return vertices,edges,g,hashes,receipt

class CausalEvidenceTests(unittest.TestCase):
 def test_valid_bounded_observation(self):
    *_,g,hashes,r=setup()
    out=g.evaluate(r,hashes)
    self.assertEqual(out["verdict"],"LOCAL_EFFECT_OBSERVED")
    self.assertFalse(out["whole_patch_authorized"])
    self.assertFalse(out["independent_requirement_truth_proven"])
 def test_patch_cannot_prove_its_own_requirement(self):
    v,e,g,h,r=setup()
    v[0]=replace(v[0],origin="AFTER")
    self.assertEqual(CausalEvidenceGraph(v,e,{x.key:x.source_sha256 for x in v}).evaluate(r,h)["verdict"],"ABSTAIN")
 def test_after_edge_taints_valid_root(self):
    v,e,g,h,r=setup()
    e[0]=replace(e[0],provenance="AFTER")
    out=CausalEvidenceGraph(v,e,{x.key:x.source_sha256 for x in v}).evaluate(r,h)
    self.assertEqual(out["verdict"],"ABSTAIN")
 def test_unverified_edge_taints_proof(self):
    v,e,g,h,r=setup()
    e[0]=replace(e[0],verified=False)
    self.assertEqual(CausalEvidenceGraph(v,e,{x.key:x.source_sha256 for x in v}).evaluate(r,h)["verdict"],"ABSTAIN")
 def test_missing_anchor_hash(self):
    v,e,g,h,r=setup()
    hashes={x.key:x.source_sha256 for x in v};hashes.pop("consumer")
    out=CausalEvidenceGraph(v,e,hashes).evaluate(r,h)
    self.assertEqual(out["verdict"],"ABSTAIN")
 def test_forged_content_hash(self):
    v,e,g,h,r=setup()
    hashes={x.key:x.source_sha256 for x in v};hashes["consumer"]=sha("tampered")
    self.assertEqual(CausalEvidenceGraph(v,e,hashes).evaluate(r,h)["verdict"],"ABSTAIN")
 def test_after_oracle_denied(self):
    v,e,g,h,r=setup()
    v[1]=replace(v[1],origin="AFTER")
    self.assertEqual(CausalEvidenceGraph(v,e,{x.key:x.source_sha256 for x in v}).evaluate(r,h)["verdict"],"ABSTAIN")
 def test_derived_not_valid_oracle(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,oracle_source_vertex="use"),h)["verdict"],"ABSTAIN")
 def test_no_negative_control(self):
    v,e,g,h,r=setup()
    self.assertIn("NO_DISCRIMINATING_NEGATIVE_CONTROL",g.evaluate(replace(r,negative_control="MISSING"),h)["reasons"])
 def test_wrong_negative_control(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,negative_control="PASSES"),h)["verdict"],"ABSTAIN")
 def test_no_preserving_control(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,preserving_control="MISSING"),h)["verdict"],"ABSTAIN")
 def test_wrong_preserving_control(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,preserving_control="VIOLATES"),h)["verdict"],"ABSTAIN")
 def test_wrong_context_hash(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,context_sha256=sha("other-context")),h)["verdict"],"ABSTAIN")
 def test_wrong_base_identity(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,before_sha256=sha("other-base")),h)["verdict"],"ABSTAIN")
 def test_wrong_head_identity(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,after_sha256=sha("other-head")),h)["verdict"],"ABSTAIN")
 def test_unsupported_tier(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,tier="whole-patch"),h)["verdict"],"ABSTAIN")
 def test_cannot_promote_tier_by_relabeling(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,evidence_scope="WHOLE_PATCH"),h)["verdict"],"ABSTAIN")
 def test_untrusted_execution_denied(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,native_replayed=False),h)["verdict"],"ABSTAIN")
 def test_oracle_false_denied(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,independent_oracle_pass=False),h)["verdict"],"ABSTAIN")
 def test_identical_results_are_not_difference(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,new_observation=r.old_observation),h)["verdict"],"ABSTAIN")
 def test_conflicting_claims_all_abstain(self):
    v,e,g,h,r=setup()
    q=replace(r,new_observation="different")
    report=audit_bundle(g,[r,q],h)
    self.assertEqual(report["n_local_effects"],0)
    self.assertEqual(report["n_abstain"],2)
 def test_cycle_denied(self):
    v,e,g,h,r=setup()
    e.append(Dependency("use","consumer","cycle",True,"BEFORE"))
    self.assertEqual(CausalEvidenceGraph(v,e,{x.key:x.source_sha256 for x in v}).evaluate(r,h)["verdict"],"ABSTAIN")
 def test_property_mismatch_denied(self):
    v,e,g,h,r=setup()
    v[0]=replace(v[0],semantic_property="dtype")
    self.assertEqual(CausalEvidenceGraph(v,e,{x.key:x.source_sha256 for x in v}).evaluate(r,h)["verdict"],"ABSTAIN")
 def test_duplicate_vertex_denied(self):
    v,e,g,h,r=setup()
    with self.assertRaises(ValueError):CausalEvidenceGraph(v+[v[0]],e,{x.key:x.source_sha256 for x in v})
 def test_dangling_edge_denied(self):
    v,e,g,h,r=setup()
    e.append(Dependency("NOT-FOUND","use","imaginary",True,"EXTERNAL"))
    with self.assertRaises(ValueError):CausalEvidenceGraph(v,e,{x.key:x.source_sha256 for x in v})
 def test_unknown_requirement_denied(self):
    v,e,g,h,r=setup()
    self.assertEqual(g.evaluate(replace(r,requirement_vertex="unknown"),h)["verdict"],"ABSTAIN")
 def test_no_roots_denied(self):
    v,e,g,h,r=setup()
    v=[replace(x,kind="PATCH_DELTA") for x in v]
    self.assertEqual(CausalEvidenceGraph(v,e,{x.key:x.source_sha256 for x in v}).evaluate(r,h)["verdict"],"ABSTAIN")
 def test_tainted_unknown_origin_denied(self):
    v,e,g,h,r=setup()
    v[2]=replace(v[2],origin="UNKNOWN")
    self.assertEqual(CausalEvidenceGraph(v,e,{x.key:x.source_sha256 for x in v}).evaluate(r,h)["verdict"],"ABSTAIN")
 def test_property_unknown_denied(self):
    v,e,g,h,r=setup()
    with self.assertRaises(ValueError):CausalEvidenceGraph([replace(v[0],semantic_property="magic")]+v[1:],e,
                                                        {x.key:x.source_sha256 for x in v})
 def test_example_is_not_whole_patch(self):
    t=example()
    self.assertEqual(t["n_local_effects"],1)
    self.assertFalse(t["whole_patch_authorized"])
 def test_canonical_order_invariant(self):
    self.assertEqual(canonical_json_digest({"b":2,"a":1}),canonical_json_digest({"a":1,"b":2}))
if __name__=="__main__":unittest.main()
