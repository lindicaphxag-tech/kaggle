"""PCPR vNext PRE-OUTCOME, source-only PR scanner.

Version lock: repository commit of this file precedes FIRST inspection of the
six frozen PR patches. Decision is conservative: candidate effect != justified
specification != native witness != whole-patch correctness.

Only stdlib; this is a PR-level first-pass coverage instrument, not APCA.
"""
from __future__ import annotations
import ast
import hashlib
import json
import sys
import urllib.request
from dataclasses import asdict,dataclass

@dataclass
class Edit:
    file: str
    class_name: str
    field: str
    old_rhs: str|None
    new_rhs: str|None
    old_kind: str
    new_kind: str
    old_consumer_line: int|None
    old_source_digest: str
    evidence_status: str
    reason: str

def qualified(x):
    if isinstance(x,ast.Name):return x.id
    if isinstance(x,ast.Attribute):
        p=qualified(x.value)
        return p+"."+x.attr if p else ""
    return ""

def self_name(x):
    if isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id=="self":
        return x.attr
    return None

def fetch(url):
    r=urllib.request.Request(url,headers={"Accept":"application/vnd.github+json",
                                         "User-Agent":"pcpr-prospective-source-auditor-0.1"})
    with urllib.request.urlopen(r,timeout=35) as f:return f.read()

def get_json(url):return json.loads(fetch(url))

def classes(source):
    return {n.name:n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)}

def field_writes(init):
    """Recognize guarded field writes only; ambiguous duplicates are excluded.

    A register_buffer/parameter call is a candidate but NOT an obligation.
    """
    out={}
    if init is None:return out
    for node in ast.walk(init):
        k=None;value=None;kind=None
        if isinstance(node,ast.AnnAssign):
            k=self_name(node.target);value=node.value;kind="assign"
        elif isinstance(node,ast.Assign) and len(node.targets)==1:
            k=self_name(node.targets[0]);value=node.value;kind="assign"
        elif isinstance(node,ast.Call) and qualified(node.func) in ("self.register_buffer","self.register_parameter"):
            if node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
                k=node.args[0].value
                value=node.args[1] if len(node.args)>1 else None
                kind=qualified(node.func).split(".")[-1]
        if k and value is not None:
            val=ast.unparse(value)
            if kind=="assign" and isinstance(value,ast.Call) and qualified(value.func) in ("nn.Parameter","torch.nn.Parameter"):
                kind="Parameter"
            out.setdefault(k,[]).append((kind,val))
    return {k:v[0] for k,v in out.items() if len(v)==1}

def methods(c):
    return {x.name:x for x in c.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}

def old_consumer_line(c,k):
    """A forward-read is independently before-patch-sourced evidence of use.
    It alone does NOT establish a trainability/serialization requirement."""
    f=methods(c).get("forward")
    if f is None:return None
    matches=[x.lineno for x in ast.walk(f) if isinstance(x,ast.Attribute)
             and isinstance(x.ctx,ast.Load) and self_name(x)==k]
    return min(matches) if matches else None

def score_one(file,old,new):
    try:b=classes(old);a=classes(new)
    except (SyntaxError,UnicodeError,ValueError):
        return [],"ABSTAIN_PARSE_OR_LANGUAGE"
    found=[];digest=hashlib.sha256(old.encode()).hexdigest()
    for name in sorted(b.keys()&a.keys()):
        ob=methods(b[name]).get("__init__")
        oa=methods(a[name]).get("__init__")
        if not ob or not oa:continue
        old_fields=field_writes(ob);new_fields=field_writes(oa)
        for k in sorted(old_fields.keys()|new_fields.keys()):
            if old_fields.get(k)==new_fields.get(k):continue
            before=old_fields.get(k);after=new_fields.get(k)
            line=old_consumer_line(b[name],k)
            # Never label changed behavior correct merely because after patch
            # uses an apparently desirable register_* or Parameter constructor.
            if line is None:
                evidence_status="ABSTAIN_NO_UNCHANGED_CONSUMER"
                reason="The BEFORE forward method does not refer to this field"
            elif before is None or after is None:
                evidence_status="ABSTAIN_UNPAIRED_FIELD"
                reason="An insertion/removal lacks a paired observable field"
            elif before[0]!=after[0]:
                evidence_status="SOURCE_ANCHORED_CANDIDATE_ONLY"
                reason="Field mechanism changed and BEFORE forward read exists; intent not proved"
            else:
                evidence_status="ABSTAIN_SAME_REGISTRATION_KIND"
                reason="RHS changed without registration-kind evidence"
            found.append(asdict(Edit(file,name,k,before[1] if before else None,
                                     after[1] if after else None,
                                     before[0] if before else "MISSING",
                                     after[0] if after else "MISSING",line,digest,
                                     evidence_status,reason)))
    return found,"PARSED"

def audit_pr(repo,number,max_files=15):
    api=f"https://api.github.com/repos/{repo}"
    meta=get_json(f"{api}/pulls/{number}")
    base=meta["base"]["sha"];head=meta["head"]["sha"]
    files=get_json(f"{api}/pulls/{number}/files?per_page=100")
    info={"url":f"https://github.com/{repo}/pull/{number}",
          "title":meta["title"],"base":base,"head":head,"source_level":"fixed upstream commits",
          "status":"ABSTAIN","edits":[],"file_statuses":[],"native_executed":False,
          "whole_patch_authorized":False}
    py=[f for f in files if f["filename"].endswith(".py") and f["status"]=="modified"]
    for f in py[:max_files]:
        path=f["filename"]
        raw=f"https://raw.githubusercontent.com/{repo}"
        try:
            old=fetch(f"{raw}/{base}/{path}").decode("utf8")
            new=fetch(f"{raw}/{head}/{path}").decode("utf8")
            edits,stat=score_one(path,old,new)
        except Exception as e:
            edits=[];stat="FETCH_OR_ANALYSIS_ERROR:"+type(e).__name__
        info["file_statuses"].append({"file":path,"status":stat})
        info["edits"].extend(edits)
    info["scope_candidates"]=sum(e["evidence_status"]=="SOURCE_ANCHORED_CANDIDATE_ONLY" for e in info["edits"])
    info["unsupported_python_files"]=max(0,len(py)-max_files)
    info["untouched_file_count"]=len(files)-len(py)
    info["status"]="SOURCE_ANCHORED_CANDIDATES_NOT_CERTIFIED" if info["scope_candidates"] else "ABSTAIN"
    # No post-patch intention and no AST lexical pattern is permitted to elevate
    # evidence to SCOPED_WITNESS without independently executed contexts.
    return info

FROZEN=[
    ("pytorch/torchtitan",4267),
    ("kornia/kornia",5446),
    ("huggingface/peft",3481),
    ("vllm-project/vllm",57163),
    ("jeshraghian/snntorch",453),
    ("kornia/kornia",5005),
]

def main():
    results=[]
    for repo,num in FROZEN:
        try:results.append(audit_pr(repo,num))
        except Exception as e:
            results.append({"url":f"https://github.com/{repo}/pull/{num}",
                            "status":"FETCH_FAILURE","error":type(e).__name__,
                            "whole_patch_authorized":False,"native_executed":False,
                            "scope_candidates":0})
    return {"protocol":"VNEXT_PRE_INSPECTION_COHORT_LOCK_20261010.md",
            "measurement":"source-independent FORWARD-USE candidates, NOT correctness",
            "n_fixed":len(FROZEN),"n_candidate_prs":sum(bool(r["scope_candidates"]) for r in results),
            "n_witness_prs":0,"n_patch_certified":0,
            "n_abstain":sum(r["status"] in ("ABSTAIN","FETCH_FAILURE") for r in results),
            "n_fetch_failure":sum(r["status"]=="FETCH_FAILURE" for r in results),
            "results":results}

if __name__=="__main__":
    result=main()
    with open("pcpr_vnext_frozen_first_score.json","w") as f:json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps({k:v for k,v in result.items() if k!="results"},indent=2))
    for p in result["results"]:
        print(p["url"],p["status"],p["scope_candidates"])
