"""PCPR v0.8: bounded symbolic-context native operator replay.

Extends v0.7 to exact matched torch factory calls that add a dtype bound to
an existing tensor, either a local function tensor or a module-owned
self.weight. Only whitelisted original AST operator expressions are executed
under synthesized typed PyTorch CPU contexts, not the complete patch.

A matching output dtype is not a validated specification. Negative outcome,
unknown dimension, dynamic dispatch or unsupported syntax => ABSTAIN.
"""
from __future__ import annotations
import ast,copy,hashlib,types
from native_delta_ir_v07 import (FACTORIES,BUILTINS,SUPPORTED_NODES,
                                SUPPORTED_KWS,tree_functions,_calls,
                                name,ast_same,strip_dtype)

SCALAR_NAMES=frozenset({"T","N","B","C","L","dim","num_classes",
                        "num_outputs","size","n","length","batch_size","hidden_dim"})
ALLOWED_TENSOR_MEMBER=frozenset({"dtype","device","shape","size","dim","numel"})
BINDABLE_SELF_ATTRIBUTES=frozenset({"weight","bias","_weight","scale",
                                    "input","tensor","data","running_mean"})
def dtype_reference(node):
 if not isinstance(node,ast.Attribute) or node.attr!="dtype":return None
 obj=node.value
 if isinstance(obj,ast.Name) and obj.id not in ("torch","self"):
  return ("parameter",obj.id)
 if isinstance(obj,ast.Attribute) and isinstance(obj.value,ast.Name) and obj.value.id=="self":
  if obj.attr in BINDABLE_SELF_ATTRIBUTES:return ("module_field",obj.attr)
 return None

def candidate_pairs(file,old,new):
 try:before=tree_functions(old);after=tree_functions(new)
 except (SyntaxError,UnicodeError,ValueError):
  return [],"ABSTAIN_PARSE"
 proposals={}
 for symbol in sorted(before.keys()&after.keys()):
  if ast_same(before[symbol],after[symbol]):continue
  olds=_calls(before[symbol]);news=_calls(after[symbol])
  for a in olds:
   if any(k.arg=="dtype" for k in a.keywords):continue
   for b in news:
    ds=[kw.value for kw in b.keywords if kw.arg=="dtype"]
    if len(ds)!=1 or not ast_same(strip_dtype(b),a):continue
    ref=dtype_reference(ds[0])
    if not ref:continue
    key=(file,symbol,ast.unparse(a),ast.unparse(b))
    proposals[key]={
     "file":file,"symbol":symbol,"line_before":a.lineno,
     "line_after":b.lineno,"factory":name(a.func),
     "tensor_ref":ref,"before_expr":ast.unparse(a),"after_expr":ast.unparse(b),
     "base_sha256":hashlib.sha256(old.encode()).hexdigest(),
     "head_sha256":hashlib.sha256(new.encode()).hexdigest(),
     "before_ast":a,"after_ast":b}
 return list(proposals.values()),"PARSED"

def check(node,candidate):
 ref=candidate["tensor_ref"]
 allowed_names={"torch","tuple","list","int","float","len",*SCALAR_NAMES}
 if ref[0]=="parameter":allowed_names.add(ref[1])
 else:allowed_names.add("self")
 for x in ast.walk(node):
  if not isinstance(x,SUPPORTED_NODES):
   raise ValueError("UNSAFE_AST_NODE_"+type(x).__name__)
  if isinstance(x,ast.Name) and x.id not in allowed_names:
   raise ValueError("UNBOUND_NAME_"+x.id)
  if isinstance(x,ast.Attribute):
   if x.attr.startswith("__"):raise ValueError("DUNDER")
   q=name(x)
   if q.startswith("self."):
    components=q.split(".")
    if len(components)>3 or len(components)<2:raise ValueError("SELF_CHAIN_COMPLEXITY")
    if components[1]!=ref[1]:raise ValueError("UNRELATED_SELF_ATTRIBUTE")
    if len(components)==3 and components[2] not in ALLOWED_TENSOR_MEMBER:
     raise ValueError("UNSAFE_MEMBER")
  if isinstance(x,ast.Call):
   q=name(x.func)
   methods=((ref[1]+".size",ref[1]+".dim",ref[1]+".numel")
        if ref[0]=="parameter" else
        ("self."+ref[1]+".size","self."+ref[1]+".dim","self."+ref[1]+".numel"))
   if q not in FACTORIES and q not in BUILTINS and q not in methods:
    raise ValueError("UNSAFE_CALL_"+q)
  if isinstance(x,ast.keyword) and x.arg not in SUPPORTED_KWS:
   raise ValueError("UNSUPPORTED_KW_"+str(x.arg))
 return True

def replay(item,dtypes=("float16","bfloat16","float32","float64")):
 import torch
 a=item["before_ast"];b=item["after_ast"]
 check(a,item);check(b,item)
 outcome=[]
 for dt in dtypes:
  try:
   ref=item["tensor_ref"];x=torch.ones((4,4),dtype=getattr(torch,dt))
   env={"torch":torch,"tuple":tuple,"list":list,"int":int,
        "float":float,"len":len}
   env.update({n:4 for n in SCALAR_NAMES})
   if ref[0]=="parameter":env[ref[1]]=x
   else:env["self"]=types.SimpleNamespace(**{ref[1]:x})
   def evaluate(call):
    expr=ast.fix_missing_locations(ast.Expression(copy.deepcopy(call)))
    return eval(compile(expr,"<exact-native-factory-ast>","eval"),
                {"__builtins__":{}},env)
   before=evaluate(a);after=evaluate(b)
   if not isinstance(before,torch.Tensor) or not isinstance(after,torch.Tensor):
    raise ValueError("NON_TENSOR_RESULT")
   outcome.append({"input_dtype":dt,"status":"ORIGINAL_AST_OPERATOR_NATIVE",
                   "before_dtype":str(before.dtype),"after_dtype":str(after.dtype),
                   "before_shape":list(before.shape),"after_shape":list(after.shape),
                   "observed_dtype_gain":bool(before.dtype!=x.dtype and after.dtype==x.dtype),
                   "preserving_context":bool(before.dtype==x.dtype and after.dtype==x.dtype)})
  except Exception as exc:
   outcome.append({"input_dtype":dt,"status":"ABSTAIN_CONTEXT_ERROR",
                   "reason":type(exc).__name__})
 return {"tier":"ISOLATED_EXACT_FACTORY_OPERATOR",
         "path":item["file"],"symbol":item["symbol"],
         "tensor_reference":list(item["tensor_ref"]),
         "before_expression":item["before_expr"],
         "after_expression":item["after_expr"],
         "before_sha256":item["base_sha256"],
         "after_sha256":item["head_sha256"],
         "n_native_contexts":sum(x["status"]=="ORIGINAL_AST_OPERATOR_NATIVE" for x in outcome),
         "n_dtype_gain_contexts":sum(x.get("observed_dtype_gain",False) for x in outcome),
         "n_preserving_contexts":sum(x.get("preserving_context",False) for x in outcome),
         "n_native_patch_correctness_decisions":0,
         "independent_requirement_gold":False,
         "whole_patch_authorized":False,
         "context_type":"SYNTHESIZED_TENSOR_SCALAR_SHAPES",
         "outcomes":outcome}

def score_file(file,old,new,max_calls=8):
 matches,status=candidate_pairs(file,old,new)
 obs=[]
 for x in matches[:max_calls]:
  try:obs.append(replay(x))
  except Exception as e:
   obs.append({"tier":"ABSTAIN_UNSUPPORTED_AST","reason":type(e).__name__,
               "n_native_contexts":0,"n_dtype_gain_contexts":0,
               "n_preserving_contexts":0,"whole_patch_authorized":False})
 return {"path":file,"status":status,"n_candidates":len(matches),
         "n_skipped_budget":max(0,len(matches)-max_calls),
         "n_native_contexts":sum(x["n_native_contexts"] for x in obs),
         "n_dtype_gain_contexts":sum(x["n_dtype_gain_contexts"] for x in obs),
         "n_preserving_contexts":sum(x["n_preserving_contexts"] for x in obs),
         "observations":obs,"whole_patch_authorized":False}
