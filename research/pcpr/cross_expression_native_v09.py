"""PCPR v0.9: cross-expression dtype-flow counterfactual witness generator.

Addresses v0.7/v0.8 inability to replay dtype semantics when a fix wraps an
existing tensor operation or one operand with .to(dtype=other.dtype).
This is a CLOSED set of expression rewrites, not whole-source execution.
All input tensors are synthetic from a fixed 4-element schema. Executable
differences are observed effects, not faithful user requirements.

Two patterns:
  1. old: g.reshape(t.shape), new: g.reshape(t.shape).to(dtype=t.dtype)
  2. old: torch.tensordot(weights,jacobian,dims=1)
     new: torch.tensordot(weights.to(dtype=jacobian.dtype),jacobian,dims=1)

No native proof of whole-PR correctness, no arbitrary imports, no API call
beyond approved torch tensor methods/operators. Unrecognized source abstains.
"""
from __future__ import annotations
import ast,copy,hashlib,types
from native_delta_ir_v07 import name,ast_same,tree_functions

KERNELS=frozenset({"torch.tensordot","torch.mm","torch.matmul"})
TENSOR_METHODS=frozenset({"reshape","view","flatten","transpose","contiguous",
                          "sum","mean","clone","to","size","dim"})
BUILTIN_NAMES=frozenset(("torch","int","float","tuple","list","len"))
SAFE_AST=(ast.Expression,ast.Call,ast.Name,ast.Attribute,ast.Load,
          ast.Tuple,ast.List,ast.Constant,ast.keyword,ast.BinOp,ast.Add,
          ast.Sub,ast.Mult,ast.Div,ast.UnaryOp,ast.USub,ast.UAdd)
def dtype_tensor(call):
 if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute):
  return None
 if call.func.attr!="to" or call.args or len(call.keywords)!=1:
  return None
 kw=call.keywords[0]
 if kw.arg!="dtype" or not isinstance(kw.value,ast.Attribute) or kw.value.attr!="dtype":
  return None
 o=kw.value.value
 if not isinstance(o,ast.Name) or o.id in BUILTIN_NAMES:return None
 return o.id

def source_calls(node):
 return [n for n in ast.walk(node) if isinstance(n,ast.Call)]

def _wrap_cast_candidates(file,symbol,old_calls,new_calls):
 out=[]
 for old in old_calls:
  for new in new_calls:
   ref=dtype_tensor(new)
   if not ref or not ast_same(old,new.func.value):continue
   # Only wrap an actual tensor transform kernel from the original source.
   op=name(old.func)
   if op in KERNELS or (isinstance(old.func,ast.Attribute)
                         and old.func.attr in TENSOR_METHODS):
    out.append((old,new,ref,"OUTPUT_CAST"))
 return out

def _arg_cast_candidates(file,symbol,old_calls,new_calls):
 out=[]
 for oc in old_calls:
  opname=name(oc.func)
  if opname not in KERNELS:continue
  for nc in new_calls:
   if name(nc.func)!=opname or len(nc.args)!=len(oc.args) or len(nc.keywords)!=len(oc.keywords):
    continue
   changed=[]
   for j,(x,y) in enumerate(zip(oc.args,nc.args)):
    if ast_same(x,y):continue
    ref=dtype_tensor(y)
    if ref and ast_same(x,y.func.value):
     changed.append((j,ref))
    else:changed.append((j,None))
   if len(changed)!=1 or changed[0][1] is None:continue
   if not all(ast_same(x.value,y.value) and x.arg==y.arg
              for x,y in zip(oc.keywords,nc.keywords)):continue
   out.append((oc,nc,changed[0][1],"ARGUMENT_CAST"))
 return out

def candidate_pairs(file,old,new):
 try:a=tree_functions(old);b=tree_functions(new)
 except (SyntaxError,UnicodeError,ValueError):return [],"ABSTAIN_PARSE"
 out={}
 for symbol in sorted(a.keys()&b.keys()):
  if ast_same(a[symbol],b[symbol]):continue
  ac,bc=source_calls(a[symbol]),source_calls(b[symbol])
  matches=_wrap_cast_candidates(file,symbol,ac,bc)
  matches+=_arg_cast_candidates(file,symbol,ac,bc)
  for x,y,ref,kind in matches:
   key=(symbol,ast.unparse(x),ast.unparse(y),kind)
   out[key]={
     "path":file,"symbol":symbol,"pattern":kind,"tensor_reference":ref,
     "old_expression":ast.unparse(x),"new_expression":ast.unparse(y),
     "before_sha256":hashlib.sha256(old.encode()).hexdigest(),
     "after_sha256":hashlib.sha256(new.encode()).hexdigest(),
     "old_ast":x,"new_ast":y}
 return list(out.values()),"PARSED"

def check(call,allowed_roots):
 for x in ast.walk(call):
  if not isinstance(x,SAFE_AST):raise ValueError("UNSUPPORTED_AST_NODE")
  if isinstance(x,ast.Name):
   if x.id not in allowed_roots|BUILTIN_NAMES:raise ValueError("UNBOUND_NAME_"+x.id)
  if isinstance(x,ast.Attribute):
   if x.attr.startswith("__"):raise ValueError("DUNDER_ATTRIBUTE")
  if isinstance(x,ast.Call):
   q=name(x.func)
   if q not in KERNELS:
    if not isinstance(x.func,ast.Attribute) or x.func.attr not in TENSOR_METHODS:
     raise ValueError("UNSAFE_CALL_"+q)
    target=x.func.value
    root_direct=isinstance(target,ast.Name) and target.id in allowed_roots
    nested_cast=(x.func.attr=="to" and isinstance(target,ast.Call)
                 and ((name(target.func) in KERNELS) or
                      (isinstance(target.func,ast.Attribute)
                       and target.func.attr in TENSOR_METHODS)))
    if not (root_direct or nested_cast):
     raise ValueError("NON_ROOT_METHOD_"+q)
  if isinstance(x,ast.keyword) and x.arg not in {"dims","dtype","dim","keepdim"}:
   raise ValueError("UNSUPPORTED_KW_"+str(x.arg))
 return True

def replay(candidate,dtypes=("float16","bfloat16","float32")):
 import torch
 a,b=candidate["old_ast"],candidate["new_ast"]
 root=candidate["tensor_reference"]
 free={n.id for expr in (a,b) for n in ast.walk(expr) if isinstance(n,ast.Name)}
 roots=free-BUILTIN_NAMES
 if root not in roots or len(roots)>4:raise ValueError("UNSUPPORTED_ROOTS")
 check(a,roots);check(b,roots)
 outcomes=[]
 for dt in dtypes:
  try:
   env={"torch":torch,"tuple":tuple,"list":list,"float":float,"int":int,"len":len}
   for key in roots:
    dtype=getattr(torch,dt) if key==root else torch.float32
    size=(4,) if key in ("weights","w","weight") else (4,4)
    env[key]=torch.ones(size,dtype=dtype)
   def evaluate(expr):
    tree=ast.fix_missing_locations(ast.Expression(copy.deepcopy(expr)))
    return eval(compile(tree,"<pinned-source-cast-expression>","eval"),
                {"__builtins__":{}},env)
   err_old=None;err_new=None;old=None;new=None
   try:old=evaluate(a)
   except Exception as e:err_old=type(e).__name__
   try:new=evaluate(b)
   except Exception as e:err_new=type(e).__name__
   if new is None or not isinstance(new,torch.Tensor):
    raise ValueError("NEW_CONTEXT_NOT_RUNNABLE")
   recovered=bool(old is None and err_old and err_new is None)
   dtype_gain=bool(isinstance(old,torch.Tensor)
                   and old.dtype!=getattr(torch,dt) and new.dtype==getattr(torch,dt))
   outcomes.append({"dtype":dt,"status":"NATIVE_EXACT_CAST_EXPRESSION",
                    "old_error":err_old,"new_error":err_new,
                    "old_dtype":str(old.dtype) if isinstance(old,torch.Tensor) else None,
                    "new_dtype":str(new.dtype),
                    "recovered_execution":recovered,
                    "dtype_changed_to_reference":dtype_gain,
                    "preserving_control":bool(isinstance(old,torch.Tensor)
                                             and old.dtype==new.dtype)})
  except Exception as e:
   outcomes.append({"dtype":dt,"status":"ABSTAIN_UNRUNNABLE",
                    "error_class":type(e).__name__})
 return {"scope":"SOURCE_EXACT_CAST_EXPRESSION_ONLY",
         "pattern":candidate["pattern"],"symbol":candidate["symbol"],
         "original_before_sha256":candidate["before_sha256"],
         "original_after_sha256":candidate["after_sha256"],
         "n_native_contexts":sum(z["status"]=="NATIVE_EXACT_CAST_EXPRESSION" for z in outcomes),
         "n_recovered_execution":sum(z.get("recovered_execution",False) for z in outcomes),
         "n_dtype_reference_gains":sum(z.get("dtype_changed_to_reference",False) for z in outcomes),
         "n_preserving_controls":sum(z.get("preserving_control",False) for z in outcomes),
         "n_whole_patch_authorizations":0,"n_validated_external_specs":0,
         "outcomes":outcomes}

def score_file(file,old,new,budget=8):
 matches,status=candidate_pairs(file,old,new)
 observations=[]
 for q in matches[:budget]:
  try:observations.append(replay(q))
  except Exception as e:
   observations.append({"scope":"ABSTAIN_UNSUPPORTED_CAST",
                        "reason":type(e).__name__,"n_native_contexts":0,
                        "n_recovered_execution":0,"n_dtype_reference_gains":0,
                        "n_preserving_controls":0})
 return {"file":file,"status":status,"n_candidates":len(matches),
         "n_evaluated":len(observations),
         "n_native_contexts":sum(v["n_native_contexts"] for v in observations),
         "n_recovered_execution":sum(v["n_recovered_execution"] for v in observations),
         "n_dtype_reference_gains":sum(v["n_dtype_reference_gains"] for v in observations),
         "n_preserving_controls":sum(v["n_preserving_controls"] for v in observations),
         "observations":observations,"whole_patch_authorized":False}
