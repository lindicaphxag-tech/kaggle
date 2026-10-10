"""PCPR v0.7: source-exact differential tensor-factory operator replay.

Goal: prospectively discover *runnable effect witnesses* across new PRs,
without importing a whole upstream package, executing arbitrary patch Python,
using developer-authored after-tests as truth, or approving whole patches.

This deliberately NARROW proof scope has no correctness oracle: "before
outputs float32, after outputs float16" is an *observed local effect*, not
proof that float16 was intended. Only matched, whitelisted torch calls are
compiled/executed and their Git blob identities are reported. Unsupported
calls must abstain. Independent obligation truth is NOT inferred from the
patch code's new dtype keyword. Source retrieval is performed separately.
"""
from __future__ import annotations
import ast,copy,hashlib,json
from dataclasses import dataclass,asdict

FACTORIES=frozenset(("torch.eye","torch.zeros","torch.ones","torch.full",
                     "torch.arange","torch.empty","torch.tensor"))
ALLOWED_METHODS=frozenset(("size","dim","numel"))
BUILTINS=frozenset(("tuple","list","int","float","len"))
NATIVE_FACETS=("dtype","shape","device")
SUPPORTED_KWS=frozenset(("dtype","device","requires_grad"))
SUPPORTED_NODES=(ast.Expression,ast.Call,ast.Attribute,ast.Name,ast.Load,
                 ast.Constant,ast.Tuple,ast.List,ast.keyword,ast.UnaryOp,
                 ast.USub,ast.UAdd,ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Div)
def name(x):
 if isinstance(x,ast.Name):return x.id
 if isinstance(x,ast.Attribute):
  a=name(x.value)
  return a+"."+x.attr if a else ""
 return ""

def names(node):
 return {x.id for x in ast.walk(node) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}

def ast_same(a,b):
 return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

def strip_dtype(call):
 out=copy.deepcopy(call)
 out.keywords=[k for k in out.keywords if k.arg!="dtype"]
 return out

def tree_functions(source):
 tree=ast.parse(source)
 found={}
 for n in tree.body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):found[n.name]=n
  elif isinstance(n,ast.ClassDef):
   for m in n.body:
    if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)):
     found[n.name+"."+m.name]=m
 return found

def _calls(node):
 return [x for x in ast.walk(node) if isinstance(x,ast.Call) and name(x.func) in FACTORIES]

def candidate_pairs(file,old,new):
 try: before=tree_functions(old);after=tree_functions(new)
 except (SyntaxError,UnicodeError,ValueError):
  return [],"ABSTAIN_PARSE"
 found=[]
 for symbol in sorted(before.keys()&after.keys()):
  a,b=before[symbol],after[symbol]
  if ast_same(a,b):continue
  old_calls=_calls(a);new_calls=_calls(b)
  for oc in old_calls:
   if any(k.arg=="dtype" for k in oc.keywords):continue
   for nc in new_calls:
    dtype=[k.value for k in nc.keywords if k.arg=="dtype"]
    if len(dtype)!=1:continue
    if not ast_same(strip_dtype(nc),oc):continue
    # All other source expressions match exactly; dtype addition isolated.
    refs=sorted(names(dtype[0])-{"torch"})
    if len(refs)!=1 or not isinstance(dtype[0],ast.Attribute) or dtype[0].attr!="dtype":
     continue
    root=dtype[0].value
    if not isinstance(root,ast.Name) or root.id not in refs or root.id=="self":continue
    d={"file":file,"symbol":symbol,"line_before":oc.lineno,
       "line_after":nc.lineno,"factory":name(oc.func),
       "input_root":root.id,
       "old_source_sha256":hashlib.sha256(old.encode()).hexdigest(),
       "new_source_sha256":hashlib.sha256(new.encode()).hexdigest(),
       "before":ast.unparse(oc),"after":ast.unparse(nc),
       "old_ast":oc,"new_ast":nc}
    found.append(d)
 # De-duplicate AST content within symbol, since pairwise class exploration
 # can otherwise make the same function fragment count several times.
 distinct={}
 for f in found:
  key=(f["file"],f["symbol"],f["before"],f["after"])
  distinct[key]=f
 return list(distinct.values()),"PARSED"

def check_expression(call,root):
 if name(call.func) not in FACTORIES:raise ValueError("NOT_WHITELISTED_FACTORY")
 for node in ast.walk(call):
  if not isinstance(node,SUPPORTED_NODES):
   raise ValueError("DISALLOWED_AST_NODE_"+type(node).__name__)
  if isinstance(node,ast.Attribute):
   if node.attr.startswith("__"):raise ValueError("DUNDER_ATTRIBUTE")
  if isinstance(node,ast.Call):
   q=name(node.func)
   allowed=(q in FACTORIES or q in BUILTINS or
            q==root+".size" or q==root+".dim" or q==root+".numel")
   if not allowed:raise ValueError("UNSAFE_CALL_"+q)
  if isinstance(node,ast.Name):
   if node.id not in {"torch",root,*BUILTINS}:raise ValueError("UNBOUND_NAME_"+node.id)
  if isinstance(node,ast.keyword):
   if node.arg not in SUPPORTED_KWS:raise ValueError("UNSUPPORTED_KEYWORD_"+str(node.arg))
 return True

def replay(candidate,dtypes=("float16","bfloat16","float32")):
 """Runs only standalone AST factory expressions. NOT full source function."""
 import torch
 old=candidate["old_ast"];new=candidate["new_ast"];root=candidate["input_root"]
 check_expression(old,root);check_expression(new,root)
 results=[]
 for d in dtypes:
  try:
   x=torch.ones((4,4),dtype=getattr(torch,d),device="cpu")
   env={"torch":torch,root:x,"tuple":tuple,"list":list,
        "int":int,"float":float,"len":len}
   a=eval(compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(old))),
                  "<isolated-before-operator>","eval"),
          {"__builtins__":{}},env)
   b=eval(compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(new))),
                  "<isolated-after-operator>","eval"),
          {"__builtins__":{}},env)
   if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor):
    raise ValueError("NON_TENSOR_OUTPUT")
   results.append({"dtype":d,"before_dtype":str(a.dtype),
                   "after_dtype":str(b.dtype),"input_dtype":str(x.dtype),
                   "before_shape":list(a.shape),"after_shape":list(b.shape),
                   "dtype_gain":bool(a.dtype!=x.dtype and b.dtype==x.dtype),
                   "preserved":bool(a.dtype==x.dtype and b.dtype==x.dtype),
                   "different":bool(a.dtype!=b.dtype or tuple(a.shape)!=tuple(b.shape)),
                   "status":"NATIVE_OPERATOR_REPLAY"})
  except Exception as e:
   results.append({"dtype":d,"status":"ABSTAIN_CONTEXT_FAILURE",
                   "reason":type(e).__name__})
 return {"operator":"MATCHED_FACTORY_DTYPE_INTERVENTION",
         "scope":"NATIVE_PYTORCH_OPERATOR_ONLY",
         "independent_requirement_validated":False,
         "whole_patch_authorized":False,
         "source_file":candidate["file"],"symbol":candidate["symbol"],
         "before_git_blob":None,"after_git_blob":None,
         "source_sha256_before":candidate["old_source_sha256"],
         "source_sha256_after":candidate["new_source_sha256"],
         "contexts":results,
         "n_native_contexts":sum(x["status"]=="NATIVE_OPERATOR_REPLAY" for x in results),
         "n_dtype_gains":sum(x.get("dtype_gain",False) for x in results),
         "n_preserving":sum(x.get("preserved",False) for x in results)}

def score_file(file,old,new,max_candidates=10):
 c,state=candidate_pairs(file,old,new)
 out=[]
 for candidate in c[:max_candidates]:
  try: r=replay(candidate)
  except Exception as e:
   r={"scope":"ABSTAIN","reason":type(e).__name__,
      "n_native_contexts":0,"n_dtype_gains":0,"n_preserving":0,
      "whole_patch_authorized":False}
  out.append(r)
 return {"file":file,"parse_status":state,"candidate_count":len(c),
         "evaluated_count":len(out),"skipped_candidates":max(0,len(c)-max_candidates),
         "n_native_contexts":sum(r["n_native_contexts"] for r in out),
         "n_positive_dtype_contexts":sum(r["n_dtype_gains"] for r in out),
         "native_results":out,
         "correctness_decisions":0}
