"""Download exact authentic DeepChem PR #5190 source and run bounded AST+Torch audit.

The pinned SHA1 values are git BLOB identifiers independently read from GitHub.
This is a previously inspected, retrospective PR. It is NOT held-out discovery.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pcpr_source_closure import assess_pair

REPOSITORY="deepchem/deepchem"
BEFORE="455d07f3e17e880a4d980b5b0365e4a02b417e14"
AFTER="b2875b0d374304ebfa2005abb5a984171a1cdef9"
SOURCES={
    "before_layers":(BEFORE,"deepchem/models/torch_models/layers.py","907addd5443cfc9acd56961c1284eb6e45fc63ac"),
    "after_layers":(AFTER,"deepchem/models/torch_models/layers.py","e6318ee8a68ef1a3e26af6ea0b8e47c0d0b45767"),
    "unchanged_consumer":(BEFORE,"deepchem/models/torch_models/torch_model.py","8a73f0b8af1868576fac29ab9d202f4f7a3957cd"),
}
def git_blob(data: bytes):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()

def fetch_pinned():
    result={}
    for key,(revision,path,blob) in SOURCES.items():
        url=f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{path}"
        with urllib.request.urlopen(url,timeout=120) as resp: data=resp.read()
        if git_blob(data)!=blob:
            raise ValueError("SOURCE_ID_MISMATCH:"+key)
        result[key]=data.decode("utf-8")
    return result

def break_one_registration(source):
    tree=ast.parse(source)
    made=False
    for cl in tree.body:
        if not isinstance(cl,ast.ClassDef) or cl.name!="WeaveLayer":continue
        for stmt in ast.walk(cl):
            if not isinstance(stmt,ast.AnnAssign) or not isinstance(stmt.target,ast.Attribute):
                continue
            if stmt.target.attr!="W_AA":continue
            val=stmt.value
            if not isinstance(val,ast.Call) or not isinstance(val.func,ast.Attribute):
                raise ValueError("No valid registered WeaveLayer.W_AA in AFTER source")
            if val.func.attr!="Parameter" or len(val.args)!=1:
                raise ValueError("Unsupported Parameter AST")
            stmt.value=copy.deepcopy(val.args[0])
            made=True
    if not made:raise ValueError("Counterfeit mutation target missing")
    return ast.unparse(ast.fix_missing_locations(tree))

def run():
    src=fetch_pinned()
    data={"status":"RETROSPECTIVE_OPERATOR_ONLY_NOT_PATCH_CORRECTNESS",
          "upstream":"https://github.com/deepchem/deepchem/pull/5190",
          "source_git_blobs":{k:v[2] for k,v in SOURCES.items()},
          "classes":{}}
    for class_name in ("WeaveLayer","WeaveGather"):
        report=assess_pair(src["before_layers"],src["after_layers"],
                           src["unchanged_consumer"],class_name)
        data["classes"][class_name]=report
    statuses={x["field"]:x["status"] for x in data["classes"]["WeaveLayer"]["fields"]}
    if statuses.get("W_AA")!="BOUNDED_REGISTRATION_WITNESS" or statuses.get("b_AA")!="BOUNDED_REGISTRATION_WITNESS":
        raise RuntimeError("Authentic complete-file registration witness failed")
    counterfeit=break_one_registration(src["after_layers"])
    c=assess_pair(src["before_layers"],counterfeit,src["unchanged_consumer"],"WeaveLayer")
    mutations={x["field"]:x["status"] for x in c["fields"]}
    if mutations.get("W_AA")!="ABSTAIN_NO_REGISTRATION_GAIN":
        raise RuntimeError("Incorrect partial patch was improperly authorized")
    data["counterfeit_mutation"]={
      "mechanism":"after file with W_AA nn.Parameter wrapper removed, other fields unchanged",
      "W_AA_verdict":mutations["W_AA"],"synthetic_counterfeit_not_external_PR":True}
    data["summary"]={
      "bounded_witnesses":sum(x["status"]=="BOUNDED_REGISTRATION_WITNESS"
                               for cls in data["classes"].values() for x in cls["fields"]),
      "scope_refusals":sum(x["status"].startswith("ABSTAIN")
                            for cls in data["classes"].values() for x in cls["fields"]),
      "tested_classes":len(data["classes"]),
      "source_consumer_pinned":True,
      "external_PyTorch_package_integration":False}
    return data

if __name__=="__main__":
    report=run()
    out=Path("pcpr_deepchem_native_summary.json")
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],**report["summary"],
                      "counterfeit":report["counterfeit_mutation"]["W_AA_verdict"]}))
