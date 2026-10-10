"""PCPR v0.2: source-diff trigger -> bounded PyTorch semantics probe.

POST-HOC development after frozen v0.1 failed 0/6. Patterns are general
operation families, not real-pr gold: buffer registration, requires_grad
preservation, state-dict vs parameters, singleton axis broadcasting.
Native observations never authorize patch correctness.
"""
from __future__ import annotations
import ast,json,re,hashlib
from patch_scope_vnext import fetch,get_json,FROZEN

def _semantic_registration():
    import torch
    from torch import nn
    x=torch.tensor([1.,2.])
    class Before(nn.Module):
        def __init__(self):
            super().__init__()
            self._data=x.clone()
    class After(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_data",x.clone())
    a=Before();b=After()
    return {"old_state_keys":list(a.state_dict()),"new_state_keys":list(b.state_dict()),
            "new_named_buffers":list(dict(b.named_buffers())),
            "old_new_data_equal":bool(torch.equal(a._data,b._data)),
            "state_key_added":"_data" in b.state_dict() and "_data" not in a.state_dict()}

def _semantic_requires_grad():
    import torch
    from torch import nn
    t=torch.tensor([2.],requires_grad=False)
    old=nn.Parameter(t)
    new=nn.Parameter(t,requires_grad=t.requires_grad)
    return {"source_requires_grad":bool(t.requires_grad),
            "old_registered_requires_grad":bool(old.requires_grad),
            "new_registered_requires_grad":bool(new.requires_grad),
            "delta":bool(old.requires_grad)!=bool(new.requires_grad)}

def _semantic_state_dict():
    import torch
    from torch import nn
    m=nn.BatchNorm1d(3)
    with torch.no_grad():m.running_mean.copy_(torch.tensor([1.,2.,3.]))
    before=set(n for n,_ in m.named_parameters())
    after=set(m.state_dict())
    return {"old_parameter_names":sorted(before),"new_state_keys":sorted(after),
            "additional_buffer_names":sorted(after-before),
            "delta_includes_buffer":len(after-before)>0}

def _semantic_broadcast():
    import torch
    z=torch.tensor([[.3],[.6],[.9]])
    v=torch.tensor([1.,2.,3.])
    old=z*v
    new=z.squeeze(-1)*v
    return {"source_input":[3,1],"paired_vector":[3],
            "old_broadcast_shape":list(old.shape),
            "new_broadcast_shape":list(new.shape),
            "delta":tuple(old.shape)!=tuple(new.shape)}

PROBES={
  "registration_buffer":_semantic_registration,
  "parameter_grad":_semantic_requires_grad,
  "state_serialization":_semantic_state_dict,
  "axis_broadcast":_semantic_broadcast,
}

def classified_patch(patches):
    """Pattern candidates; no intent inferred from the patched output."""
    kinds=set();locations={}
    for file in patches:
        path=file.get("filename","")
        if not path.endswith(".py") or "/test" in path or path.startswith("tests/"):continue
        diff=file.get("patch") or ""
        plus="\n".join(s[1:] for s in diff.splitlines() if s.startswith("+") and not s.startswith("+++"))
        minus="\n".join(s[1:] for s in diff.splitlines() if s.startswith("-") and not s.startswith("---"))
        if "register_buffer(" in plus and ("self." in minus or "register_buffer(" not in minus):
            kinds.add("registration_buffer");locations["registration_buffer"]=path
        if ("nn.Parameter(" in plus and "nn.Parameter(" in minus and
            "requires_grad=" in plus and "requires_grad=" not in minus):
            kinds.add("parameter_grad");locations["parameter_grad"]=path
        if (".state_dict()" in plus and ".named_parameters()" in minus):
            kinds.add("state_serialization");locations["state_serialization"]=path
        if "squeeze(-1)" in plus and "return self." in minus:
            kinds.add("axis_broadcast");locations["axis_broadcast"]=path
    return {k:locations[k] for k in sorted(kinds)}

def audit(repo,pr):
    meta=get_json(f"https://api.github.com/repos/{repo}/pulls/{pr}")
    fs=get_json(f"https://api.github.com/repos/{repo}/pulls/{pr}/files?per_page=100")
    matched=classified_patch(fs)
    found=[]
    for mechanism,file in matched.items():
        try:
            observation=PROBES[mechanism]()
            found.append({"mechanism":mechanism,"file":file,
                          "observation":observation,
                          "source_identity":{"base":meta["base"]["sha"],
                                            "head":meta["head"]["sha"]},
                          "scope":"ISOLATED_PYTORCH_SEMANTIC_DIFFERENCE_ONLY",
                          "independent_requirement_validated":False,
                          "whole_patch_authorized":False})
        except Exception as exc:
            found.append({"mechanism":mechanism,"file":file,
                          "status":"ABSTAIN_NATIVE_PROBE_ERROR","exception":type(exc).__name__})
    return {"url":f"https://github.com/{repo}/pull/{pr}",
            "category":"POST_HOC_DEVELOPMENT","matched_mechanisms":list(matched),
            "native_operator_contexts":found,
            "n_scoped_native_semantics":sum("observation" in x for x in found),
            "whole_patch_authorized":False}

def main():
    results=[]
    for repo,pr in FROZEN:
        try:results.append(audit(repo,pr))
        except Exception as e:results.append({"url":f"https://github.com/{repo}/pull/{pr}",
                                               "status":"FETCH_OR_PARSE_FAILURE",
                                               "reason":type(e).__name__,
                                               "n_scoped_native_semantics":0})
    return {"status":"POST_HOC_DEVELOPMENT_NOT_INDEPENDENT_VALIDATION",
            "frozen_v0_1_result":"0/6 source-only candidates, 0/6 witnesses",
            "n_prs":len(FROZEN),
            "n_prs_with_operator_effects":sum(bool(x.get("n_scoped_native_semantics")) for x in results),
            "n_operator_effects":sum(x.get("n_scoped_native_semantics",0) for x in results),
            "n_real_repository_native_tests":0,
            "n_independently_verified_requirements":0,
            "n_whole_patch_authorizations":0,
            "results":results}

if __name__=="__main__":
    r=main()
    with open("pcpr_v02_posthoc_effects.json","w") as f:json.dump(r,f,indent=2,sort_keys=True)
    print(json.dumps({k:v for k,v in r.items() if k!="results"},indent=2))
    for q in r["results"]:
        print(q["url"],q.get("matched_mechanisms",[]),q.get("n_scoped_native_semantics",0))
