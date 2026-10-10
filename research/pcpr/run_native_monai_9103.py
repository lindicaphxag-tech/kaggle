"""Original MONAI #9103 BarlowTwinsLoss class-level dtype counterfactual replay.

Retrospective exploration AFTER the v0.5 cohort's first source scan. Never
report as independent blind patch correctness. Pinned unmodified upstream Git
blobs and original two method bodies; one generic semantic intervention.
"""
import ast,copy,hashlib,json,urllib.request
import torch
from torch import nn
from native_ast_lift_v03 import SAFE_BUILTINS,check_replay_ast
from semantic_interventions_v06 import mutate

REPO="Project-MONAI/MONAI"
PATH="monai/losses/barlow_twins.py"
BEFORE="522697c1addaf46f033cd7bd0831078e4fd8de1f"
AFTER="d249307339477f7666523c733a06e99ec8d78032"
BEFORE_BLOB="699594493cb2f6c334c2784ff41f50bd129195e8"
AFTER_BLOB="dd38577eb3593b9afafaec520647c4cef151913f"

def git_blob(b):
 return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()

def fetch(rev,expected_blob):
 url=f"https://raw.githubusercontent.com/{REPO}/{rev}/{PATH}"
 request=urllib.request.Request(url,headers={"User-Agent":"pcpr-monai-source-exact-dev/0.6"})
 with urllib.request.urlopen(request,timeout=70) as r:data=r.read()
 if git_blob(data)!=expected_blob:raise ValueError("PINNED_UPSTREAM_BLOB_MISMATCH")
 return data.decode("utf8")

def lift_class(src):
 tree=ast.parse(src)
 found=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="BarlowTwinsLoss"]
 if len(found)!=1:raise ValueError("BARLOW_CLASS_MISSING_OR_DUPLICATE")
 cls=found[0]
 bodies=[]
 for method in ("__init__","forward"):
  matches=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==method]
  if len(matches)!=1:raise ValueError("MISSING_OR_MULTIPLE_METHOD:"+method)
  obj=copy.deepcopy(matches[0])
  if obj.decorator_list:raise ValueError("UNSUPPORTED_DECORATOR")
  check_replay_ast(obj)
  bodies.append(obj)
 module=ast.Module(body=[ast.ClassDef(name="BarlowTwinsLoss",
   bases=[ast.Name(id="_Loss",ctx=ast.Load())],
   keywords=[],body=bodies,decorator_list=[])],type_ignores=[])
 module=ast.fix_missing_locations(module)
 scope={"__builtins__":SAFE_BUILTINS,"torch":torch,
        "_Loss":nn.modules.loss._Loss}
 exec(compile(module,"<verified-MONAI-class-AST>","exec"),scope,scope)
 return scope["BarlowTwinsLoss"]

def run():
 old=fetch(BEFORE,BEFORE_BLOB)
 after=fetch(AFTER,AFTER_BLOB)
 altered,manifest=mutate(after,"BarlowTwinsLoss.forward","drop_tensor_factory_dtype")
 B,A,M=lift_class(old),lift_class(after),lift_class(altered)
 cases=[]
 for dt in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
  torch.manual_seed(392)
  x=torch.randn(4,8,dtype=dt)
  y=torch.randn(4,8,dtype=dt)
  z=[]
  for cls in (B,A,M):
   value=cls(lambd=0.005)(x.clone(),y.clone())
   z.append(value)
  out={"dtype":str(dt),
       "before_dtype":str(z[0].dtype),
       "after_dtype":str(z[1].dtype),
       "counterfactual_dtype":str(z[2].dtype),
       "after_preserves_input_dtype":z[1].dtype==dt,
       "mutation_changes_effect":z[1].dtype!=z[2].dtype,
       "original_matches_mutant":z[0].dtype==z[2].dtype,
       "native_context":"CPU; batch=4; features=8; fixed torch seed 392"}
  cases.append(out)
 half=cases[:2]
 if not all(z["after_preserves_input_dtype"] and z["mutation_changes_effect"]
            and z["original_matches_mutant"] for z in half):
  raise AssertionError("Original MONAI native half-precision counterfactual failed")
 if not all(z["after_preserves_input_dtype"] and not z["mutation_changes_effect"] 
            for z in cases[2:]):
  raise AssertionError("Full precision preserving control failed")
 return {"status":"MONAI_9103_RETROSPECTIVE_NATIVE_CLASS_METHOD_EXACT_REPLAY",
         "upstream":f"https://github.com/{REPO}/pull/9103",
         "before_blob":BEFORE_BLOB,"after_blob":AFTER_BLOB,
         "counterfactual":manifest,
         "n_tested_native_contexts":4,
         "n_half_precision_negative_contexts":2,
         "n_full_precision_preserving_contexts":2,
         "genuine_whole_package_import":False,
         "whole_PR_correctness_authorized":False,
         "independently_adjudicated_correctness":False,
         "cases":cases}

if __name__=="__main__":
 data=run()
 with open("pcpr_monai_9103_native_counterfactual.json","w") as f:json.dump(data,f,indent=2)
 print(json.dumps({k:v for k,v in data.items() if k!="cases"},indent=2))
 print(json.dumps(data["cases"],indent=2))
