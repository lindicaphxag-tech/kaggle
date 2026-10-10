"""PCPR v0.5: AST-based cross-project semantic EFFECT hypotheses.

This is a conservative *front-end*, not a specification miner or patch
correctness detector. The patch triggers hypotheses, but cannot prove their
truth. The BEFORE source provides the function boundary and context; no
upstream patch tests, CHANGELOG or commit title is used as an oracle.
 
Frozen transfer protocol: keep this exact commit fixed before opening the
new 8 pull requests in V05_PRE_INSPECTION_LOCK_20261010.md.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict,dataclass
import ast,hashlib,json,urllib.request

# Semantic operators/functions, deliberately NOT tuned to individual PRs.
# All descriptions are HYPOTHESES. A native verifier must demonstrate effects.
FACETS = {
 "state": ("register_buffer","register_parameter","state_dict","load_state_dict",
           "ModuleList","BufferList","named_buffers","named_parameters","_buffers","_parameters"),
 "gradient": ("requires_grad","detach","grad","backward","Parameter","no_grad",
              "autograd","retain_grad","grad_fn"),
 "precision": ("dtype","float16","bfloat16","float32","float64","to",
               "type_as","result_type","where","autocast","promote_types"),
 "layout": ("squeeze","unsqueeze","reshape","view","flatten","expand","repeat",
            "broadcast","permute","transpose","shape","size","dim"),
 "masking": ("masked_fill","masked_scatter","where","mask","lengths","padding",
             "nonzero","gather","scatter","index_select"),
 "optimizer": ("optimizer","optim","state","step","zero_grad","param_groups",
               "Adam","AdamW","SGD","momentum"),
 "numerical_domain": ("clamp","sigmoid","logit","isfinite","isnan","nan_to_num",
                      "log","exp","sqrt","epsilon","eps","inf"),
 "device": ("device","cuda","cpu","to","register_buffer","new_zeros",
            "zeros_like","ones_like"),
 "representation": ("contiguous","as_strided","view","clone","copy_","data",
                    "storage","alias","detach"),
}
SYMBOLS = {p:set(values) for p,values in FACETS.items()}

@dataclass(frozen=True)
class Hypothesis:
 file:str
 symbol:str
 facets:tuple[str,...]
 before_calls:tuple[str,...]
 after_calls:tuple[str,...]
 new_calls:tuple[str,...]
 dropped_calls:tuple[str,...]
 source_before_sha256:str
 source_after_sha256:str
 before_public_arguments:tuple[str,...]
 before_has_return:bool
 status:str="EFFECT_HYPOTHESIS_ONLY"

def name_of(node):
 if isinstance(node,ast.Name):return node.id
 if isinstance(node,ast.Attribute):return name_of(node.value)+"."+node.attr
 return ""

def tokenize(node):
 """Typed attributes + call arguments, *not* docs or comments."""
 seen=set()
 for n in ast.walk(node):
  if isinstance(n,ast.Call):seen.update(name_of(n.func).split("."))
  elif isinstance(n,ast.Attribute):seen.add(n.attr)
  elif isinstance(n,ast.Name):seen.add(n.id)
  elif isinstance(n,ast.keyword) and n.arg:seen.add(n.arg)
 return seen

def code_units(tree):
 """Class-method/function nested paths; no functions inside control-flow."""
 result={}
 def descend(body,prefix=""):
  for n in body:
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
    result[prefix+n.name]=n
   elif isinstance(n,ast.ClassDef):
    descend(n.body,prefix+n.name+".")
 descend(tree.body)
 return result

def call_sig(node):
 return Counter(name_of(n.func) for n in ast.walk(node) if isinstance(n,ast.Call) and name_of(n.func))

def uses_before_inputs(node):
 p=node.args
 return tuple(a.arg for a in p.posonlyargs+p.args+p.kwonlyargs if a.arg not in ("self","cls"))

def scan_pair(file:str,before:str,after:str):
 try:
  old=code_units(ast.parse(before));new=code_units(ast.parse(after))
 except (SyntaxError,UnicodeError,ValueError):
  return [],"ABSTAIN_UNPARSEABLE"
 result=[]
 for symbol in sorted(old.keys()&new.keys()):
  a,b=old[symbol],new[symbol]
  if ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False):
   continue
  all_tokens=tokenize(a)|tokenize(b)
  facets=tuple(sorted(k for k,v in SYMBOLS.items() if all_tokens & v))
  ca,cb=call_sig(a),call_sig(b)
  additions=tuple(sorted((cb-ca).elements()))
  removals=tuple(sorted((ca-cb).elements()))
  # Unchanged BEFORE arguments and return are a scope anchor, NOT proof
  # that their dtype/device/state must be preserved.
  result.append(asdict(Hypothesis(
    file=file,symbol=symbol,facets=facets,
    before_calls=tuple(sorted(ca.elements())),
    after_calls=tuple(sorted(cb.elements())),
    new_calls=additions,dropped_calls=removals,
    source_before_sha256=hashlib.sha256(before.encode()).hexdigest(),
    source_after_sha256=hashlib.sha256(after.encode()).hexdigest(),
    before_public_arguments=uses_before_inputs(a),
    before_has_return=any(isinstance(x,ast.Return) for x in ast.walk(a)),
    status="TYPED_EFFECT_HYPOTHESIS_NOT_VERIFIED" if facets else "UNCLASSIFIED_CHANGE_ABSTAIN"
  )))
 return result,"PARSED"

def request_json(url):
 req=urllib.request.Request(url,headers={"Accept":"application/vnd.github+json",
                                          "User-Agent":"pcpr-typed-effect-ir-0.5"})
 with urllib.request.urlopen(req,timeout=35) as r:return json.loads(r.read())
def request_text(url):
 req=urllib.request.Request(url,headers={"User-Agent":"pcpr-typed-effect-ir-0.5"})
 with urllib.request.urlopen(req,timeout=35) as r:return r.read().decode("utf-8")

def assess_real_pr(repo,pr,max_files=24):
 api="https://api.github.com/repos/"+repo
 metadata=request_json(f"{api}/pulls/{pr}")
 listing=request_json(f"{api}/pulls/{pr}/files?per_page=100")
 py=[f["filename"] for f in listing if f["filename"].endswith(".py") and f["status"]=="modified"
     and not f["filename"].startswith(("tests/","test/")) and "/tests/" not in f["filename"]]
 o={"url":f"https://github.com/{repo}/pull/{pr}","base":metadata["base"]["sha"],
    "head":metadata["head"]["sha"],"all_modified_py_count":len(py),"max_files":max_files,
    "raw_pr_file_count":len(listing),"hypotheses":[],"errors":[],
    "run_tier":"AST_SOURCE_SCOPE_ONLY","native_executions":0,
    "independent_specifications_proved":0,"patch_approved":False}
 for path in py[:max_files]:
  raw=f"https://raw.githubusercontent.com/{repo}"
  try:
   before=request_text(f"{raw}/{o['base']}/{path}")
   after=request_text(f"{raw}/{o['head']}/{path}")
   result,state=scan_pair(path,before,after)
   o["hypotheses"].extend(result)
   if state!="PARSED":o["errors"].append({"file":path,"reason":state})
  except Exception as exc:
   o["errors"].append({"file":path,"reason":"FETCH_OR_PARSE_ERROR_"+type(exc).__name__})
 o["skipped_py_files"]=max(0,len(py)-max_files)
 o["n_typed_changed_functions"]=sum(bool(h["facets"]) for h in o["hypotheses"])
 o["n_typed_facets"]=sum(len(h["facets"]) for h in o["hypotheses"])
 o["facet_union"]=sorted({f for h in o["hypotheses"] for f in h["facets"]})
 o["status"]="TYPED_EFFECT_HYPOTHESES_ONLY" if o["n_typed_changed_functions"] else "ABSTAIN"
 return o
