"""PCPR v0.3: source-exact AST-lifted native PyTorch observations.

Run ORIGINAL before/after function or constructor AST extracted from immutable
GitHub sources (verified against Git git-blob hashes). No reimplemented function
body, no imports of arbitrary upstream repo code. Semantics are native PyTorch
CPU at FUNCTION/CONSTRUCTOR tier, not full upstream package correctness.
All THREE cases are post-hoc previously inspected DEVELOPMENT references.
"""
from __future__ import annotations
import ast,copy,hashlib,json,urllib.request
from dataclasses import dataclass
from typing import Union

PINNED={
 "ctc":{
  "repo":"QwenAudio/SenseVoice","pr":358,"path":"utils/ctc_alignment.py",
  "base":"ea15219509625e5d4c5143c37c86970135886b5d",
  "head":"e3881391e47ba72129e9eea86ce78f819eef6d88",
  "before_blob":"f10fcf27e9ce421d858913006fd4d13089f3e993",
  "after_blob":"51e86d8bda5074a9b16fe09d619506118bf37b49"},
 "line":{
  "repo":"kornia/kornia","pr":5026,"path":"kornia/geometry/line.py",
  "base":"6888c039843596b28b06f5457310a7646604e850",
  "head":"687c0e461be0eac280db72a9638930ca087814be",
  "before_blob":"56ce53a1b38e533ffcfaaaf92d7c81de2efd5a74",
  "after_blob":"2856a88a6d4051a5d7f29b16d97d2759b040b150"},
 "quaternion":{
  "repo":"kornia/kornia","pr":5446,"path":"kornia/geometry/quaternion.py",
  "base":"457fa2ec3182483528251241fc312fc49768d44f",
  "head":"59a5de3c8ae87d3e9e020028510312451a138127",
  "before_blob":"586ca80d63e0100e2f4746d4ea0ad2133a11c9c4",
  "after_blob":"029dfc35f6bc93aa7e09cfdd79d4d22d51fe255e"}}

def blob_id(content):
 return hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()


# AST-lift defense in depth; NOT a secure process sandbox for arbitrary code.
# External source must still be Git-pinned and executed only on an isolated,
# least-privilege worker without credentials. Reject whole scripts/imports.
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__,dict)
                 else getattr(__builtins__,k)
                 for k in ("float","int","bool","str","len","range","enumerate",
                           "tuple","list","dict","set","sum","min","max","abs",
                           "isinstance","ValueError","TypeError","RuntimeError",
                           "super","type","slice")}
# PyTorch internals may lazily import dependencies during a permitted tensor
# operation. AST-level direct import/use is still refused below. This is
# defense-in-depth only: a container/process sandbox is required for hostile
# upstream Python and never supplied by this allowlist.
SAFE_BUILTINS["__import__"] = __import__
DANGEROUS_NAMES = {"eval","exec","compile","open","__import__","globals",
                   "locals","vars","getattr","setattr","delattr","breakpoint",
                   "input","help","dir"}
DANGEROUS_NODES=(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,
                 ast.With,ast.AsyncWith,ast.Try,ast.Lambda,ast.ClassDef)
def check_replay_ast(node):
 for n in ast.walk(node):
  if isinstance(n,DANGEROUS_NODES):
   raise ValueError("UNSAFE_AST_NODE:"+type(n).__name__)
  if isinstance(n,ast.Name) and (n.id in DANGEROUS_NAMES or n.id.startswith("__")):
   raise ValueError("UNSAFE_AST_NAME:"+n.id)
  if isinstance(n,ast.Attribute) and n.attr.startswith("__") and n.attr!="__init__":
   raise ValueError("UNSAFE_AST_ATTRIBUTE:"+n.attr)

def fetch_ref(family,version):
 spec=PINNED[family]
 ref=spec["base"] if version=="before" else spec["head"]
 url=f"https://raw.githubusercontent.com/{spec['repo']}/{ref}/{spec['path']}"
 request=urllib.request.Request(url,headers={"User-Agent":"pcpr-exact-AST-verified/0.3"})
 with urllib.request.urlopen(request,timeout=80) as f: payload=f.read()
 if blob_id(payload)!=spec[version+"_blob"]:
  raise ValueError("GIT_BLOB_IDENTITY_MISMATCH")
 return payload.decode("utf8")

def source_ast_function(source,name):
 tree=ast.parse(source)
 matched=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
 if len(matched)!=1:raise ValueError("Function must be unique")
 return copy.deepcopy(matched[0])

def source_ast_constructor(source,name):
 tree=ast.parse(source)
 cc=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name]
 if len(cc)!=1:raise ValueError("Class not unique")
 con=[n for n in cc[0].body if isinstance(n,ast.FunctionDef) and n.name=="__init__"]
 if len(con)!=1:raise ValueError("Constructor not unique")
 return copy.deepcopy(con[0])

def lift_function(source,name):
 import torch
 @dataclass
 class ParametrizedLine:
  origin:object
  direction:object
 node=source_ast_function(source,name)
 # Explicitly no decorators; original function body AST left UNMODIFIED.
 if node.decorator_list:raise ValueError("Decorated function unsupported")
 check_replay_ast(node)
 t=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
 space={"torch":torch,"ParametrizedLine":ParametrizedLine}
 exec(compile(t,"<pinned-original-function-AST>","exec"),{"__builtins__":SAFE_BUILTINS,**space},space)
 return space[name]

def lift_constructor(source,name="Quaternion",counterfeit=False):
 import torch
 from torch import nn
 init=source_ast_constructor(source,name)
 if counterfeit:
  replaced=False
  # Strictly handle only direct if in source constructor.
  for statement in init.body:
   if not isinstance(statement,ast.If):continue
   if not statement.orelse:continue
   for idx,child in enumerate(statement.orelse):
    if isinstance(child,ast.Expr) and isinstance(child.value,ast.Call):
     call=child.value
     if isinstance(call.func,ast.Attribute) and call.func.attr=="register_buffer" and len(call.args)==2:
      statement.orelse[idx]=ast.Assign(
       targets=[ast.Attribute(value=ast.Name(id="self",ctx=ast.Load()),
                              attr="_data",ctx=ast.Store())],
       value=copy.deepcopy(call.args[1]))
      replaced=True
  if not replaced:raise ValueError("Counterfeit AST anchor missing")
 check_replay_ast(init)
 cls=ast.ClassDef(name=name,bases=[ast.Attribute(
       value=ast.Name(id="nn",ctx=ast.Load()),attr="Module",ctx=ast.Load())],
       keywords=[],body=[init],decorator_list=[])
 tree=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
 space={"torch":torch,"nn":nn,"Union":Union}
 exec(compile(tree,"<pinned-original-constructor-AST>","exec"),
      {"__builtins__":__builtins__,**space},space)
 return space[name]

def run_ctc(pre,post):
 import torch
 old=lift_function(pre,"ctc_forced_align")
 new=lift_function(post,"ctc_forced_align")
 frames=torch.tensor([
  [-1.99,-1.09,-2.18,-1.83,-1.36],
  [-.29,-4.06,-4.18,-3.26,-1.72],
  [-.93,-1.8,-2.21,-3.37,-1.44],
  [-4.04,-4.09,-2.9,-2.12,-.23],
  [-3.59,-3.29,-.24,-2.61,-2.63]],dtype=torch.float32)
 # These are deliberately hand-constructed AFTER seeing the upstream PR.
 extras=torch.tensor([[0.,-8.,-8.,-8.,-8.]])
 short=frames.unsqueeze(0)
 padded=torch.cat([frames,extras],dim=0).unsqueeze(0)
 targets=torch.tensor([[1,1]])
 inp=torch.tensor([5]);tar=torch.tensor([2])
 def evaluate(f,x):
  return f(x,targets.clone(),inp.clone(),tar.clone())
 before=evaluate(old,padded);after=evaluate(new,padded)
 ref=evaluate(new,short)
 return {"before_alignment":before.tolist(),"after_alignment":after.tolist(),
         "after_short_alignment":ref.tolist(),
         "after_padding_prefix_stable":bool(torch.equal(after[:,:5],ref)),
         "old_new_differ":bool(not torch.equal(before,after)),
         "native_tier":"original complete ctc_forced_align AST"}

def run_line(pre,post):
 import torch
 old=lift_function(pre,"_fit_line_ols_2d")
 new=lift_function(post,"_fit_line_ols_2d")
 output=[]
 for dt in [torch.float16,torch.bfloat16,torch.float32,torch.float64]:
  pts=torch.tensor([[[1.,0.],[1.,1.],[1.,3.]]],dtype=dt)
  b=old(pts);a=new(pts)
  output.append({"dtype":str(dt),"old_direction_dtype":str(b.direction.dtype),
                 "new_direction_dtype":str(a.direction.dtype),
                 "input_dtype":str(pts.dtype),
                 "after_preserves_input_dtype":a.direction.dtype==pts.dtype,
                 "before_preserves_input_dtype":b.direction.dtype==pts.dtype,
                 "old_shape":list(b.direction.shape),"new_shape":list(a.direction.shape)})
 return {"contexts":output,"native_tier":"original complete _fit_line_ols_2d AST; result holder minimal"}

def run_quaternion(pre,post):
 import torch
 from torch import nn
 B=lift_constructor(pre);A=lift_constructor(post)
 counterfeit=lift_constructor(post,counterfeit=True)
 plain=torch.tensor([1.,0.,0.,0.])
 def stats(cl):
  obj=cl(plain.clone())
  parent=nn.Module();parent.rotation=obj
  before_dtype=obj._data.dtype
  parent.to(dtype=torch.float64)
  return {"state_keys":list(parent.state_dict()),
          "buffer_keys":list(dict(parent.named_buffers())),
          "parameter_keys":list(dict(parent.named_parameters())),
          "dtype_moved":obj._data.dtype==torch.float64,
          "initial_dtype":str(before_dtype)}
 results={"before_plain":stats(B),"after_plain":stats(A),
          "counterfeit_plain":stats(counterfeit)}
 for label,cl in [("before",B),("after",A)]:
  value=cl(nn.Parameter(plain.clone()))
  results[label+"_parameter"]={"param_registered":"_data" in dict(value.named_parameters()),
                                "buffer_registered":"_data" in dict(value.named_buffers())}
 return {**results,"native_tier":"original constructor AST; nn.Module persistence/parent move, not full Kornia"}

def run_all():
 src={key:{v:fetch_ref(key,v) for v in ("before","after")} for key in PINNED}
 results={
  "ctc":run_ctc(src["ctc"]["before"],src["ctc"]["after"]),
  "line":run_line(src["line"]["before"],src["line"]["after"]),
  "quaternion":run_quaternion(src["quaternion"]["before"],src["quaternion"]["after"])
 }
 if not results["quaternion"]["after_plain"]["dtype_moved"]:
  raise AssertionError("Kornia buffer not following parent move")
 if results["quaternion"]["counterfeit_plain"]["dtype_moved"]:
  raise AssertionError("Wrong partial fix was mistaken as persistence")
 if not all(row["after_preserves_input_dtype"] for row in results["line"]["contexts"]):
  raise AssertionError("Kornia line patched dtype must be stable")
 results["scope"]="THREE_POSTHOC_PINNED_SOURCE_FUNCTION_LEVEL_REPLAYS; NOT NEW BLINDED PRS"
 results["pinned"]={k:{x:y for x,y in v.items() if x!="repo"} for k,v in PINNED.items()}
 results["whole_patch_authorized"]=False
 return results

if __name__=="__main__":
 r=run_all()
 with open("pcpr_v03_original_source_native.json","w") as f:json.dump(r,f,indent=2,default=str)
 print(json.dumps({"status":r["scope"],"ctc":r["ctc"],"line":r["line"],
                   "quaternion":r["quaternion"],"full_patch":r["whole_patch_authorized"]},indent=2))
