"""Link verified Git-blob original source replays to CEC *local* receipts.

The pre-existing code/docstrings provide independent-of-PATCH hypothesis
anchors; generic PyTorch behavior is an API oracle candidate. Neither source
asserts a fully adjudicated project requirement. All cases were INSPECTED
before this implementation; this is development evidence ONLY.
"""
import ast,json
from dataclasses import asdict
from causal_evidence_closure import (CausalEvidenceGraph,Vertex,Dependency,
                                     RunReceipt,audit_bundle,sha)
from native_ast_lift_v03 import run_all,fetch_ref

def graph(name, property_name, before_source, anchor,oracle_label):
    # Source comes from Git-blob-verified BEFORE commit in run_all() and fetch.
    if anchor not in before_source:
        raise ValueError("INDEPENDENT_BEFORE_ANCHOR_ABSENT:"+name)
    b_hash=sha(before_source)
    spec_hash=sha(oracle_label)
    v=[Vertex(f"{name}:before","UNCHANGED_CONSUMER","BEFORE",b_hash,property_name),
       Vertex(f"{name}:api","EXTERNAL_API_SPEC","EXTERNAL",spec_hash,property_name),
       Vertex(f"{name}:hypothesis","DERIVED_EFFECT","BEFORE",sha(anchor),property_name)]
    e=[Dependency(f"{name}:before",f"{name}:hypothesis","source-use",True,"BEFORE"),
       Dependency(f"{name}:api",f"{name}:hypothesis","typed PyTorch API hypothesis",True,"EXTERNAL")]
    return CausalEvidenceGraph(v,e,{x.key:x.source_sha256 for x in v})

def receipt(name,property_name,source,before_source,after_source,context,
            old_observation,new_observation,negative,preserving):
    hashes={"before_sha256":sha(before_source),"after_sha256":sha(after_source),
            "context_sha256":sha(context)}
    return RunReceipt(f"{name}:hypothesis",property_name,"function",
       hashes["context_sha256"],hashes["before_sha256"],hashes["after_sha256"],
       old_observation,new_observation,f"{name}:api",
       "VIOLATES" if negative else "MISSING",
       "PASSES" if preserving else "MISSING",
       True,True,"MATCHED_SINGLE_CONTEXT"),hashes

def build():
    native=run_all()
    items={}
    anchors={
      "ctc":("mask_boundary","input_lengths","PyTorch padded sequence masking / CTC length semantics"),
      "line":("dtype","dtype=points.dtype","PyTorch dtype-preserving tensor factory / torch.where semantics"),
      "quaternion":("checkpoint","def to(","torch.nn.Module registered buffers appear in state_dict, parent.to propagates dtype")
    }
    for name in ("ctc","line","quaternion"):
      before=fetch_ref(name,"before");after=fetch_ref(name,"after")
      typ,anchor,api=anchors[name]
      g=graph(name,typ,before,anchor,api)
      if name=="ctc":
        n=native[name]
        negative=n["old_new_differ"] and n["before_alignment"][0][:5]!=n["after_short_alignment"][0]
        preserving=n["after_padding_prefix_stable"]
        before_obs=str(n["before_alignment"][0])
        after_obs=str(n["after_alignment"][0])
        ctx="CTC:5 valid frames+1 padded vs 5 valid; CPU FP32; target=[1,1]; length=5"
      elif name=="line":
        n=native[name]
        row=next(q for q in n["contexts"] if q["dtype"]=="torch.float16")
        fp32=next(q for q in n["contexts"] if q["dtype"]=="torch.float32")
        negative=(not row["before_preserves_input_dtype"]) and row["after_preserves_input_dtype"]
        preserving=fp32["before_preserves_input_dtype"] and fp32["after_preserves_input_dtype"]
        before_obs=row["old_direction_dtype"];after_obs=row["new_direction_dtype"]
        ctx="Vertical line 3 points CPU float16; preserving CPU float32"
      else:
        n=native[name]
        negative=(not n["counterfeit_plain"]["state_keys"]) and bool(n["after_plain"]["state_keys"])
        preserving=n["before_parameter"]["param_registered"] and n["after_parameter"]["param_registered"]
        before_obs=json.dumps(n["before_plain"]["state_keys"])
        after_obs=json.dumps(n["after_plain"]["state_keys"])
        ctx="Quaternion CPU float32 plain tensor; parent.to(float64); Parameter preserving control"
      rc,hashes=receipt(name,typ,n,before,after,ctx,before_obs,after_obs,negative,preserving)
      result=audit_bundle(g,[rc],hashes)
      if result["n_local_effects"]!=1:
        raise AssertionError("Failed to attach a bounded, evidenced effect: "+name)
      items[name]={"before_sha256":hashes["before_sha256"],
                   "after_sha256":hashes["after_sha256"],
                   "source_precondition":anchor,
                   "oracle_candidate":api,
                   "negative_control_observed":negative,
                   "preserving_control_observed":preserving,
                   "result":result}
    return {"status":"THREE_PREVIOUSLY_INSPECTED_REPO_CHANGES_WITH_LOCAL_CEC_RECEIPTS",
            "n_source_exact_cases":3,
            "n_local_observations":sum(x["result"]["n_local_effects"] for x in items.values()),
            "n_independently_adjudicated_specifications":0,
            "n_blind_unseen_prs":0,
            "n_whole_patch_decisions":0,
            "n_full_upstream_packages":0,
            "notes":"One CTC and two Kornia revisions; not three independent repositories. Anchors justify candidates, not objective specification correctness.",
            "cases":items}

if __name__=="__main__":
    r=build()
    with open("pcpr_v04_cec_real_source_receipts.json","w") as f:json.dump(r,f,indent=2,sort_keys=True)
    print(json.dumps({k:v for k,v in r.items() if k!="cases"},indent=2))
