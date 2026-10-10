"""PCPR CEC v1.0: baseline-only independently originated test contracts.

This is an AST/read-only provenance miner, NOT a reliable oracle itself.
It excludes modified/new tests and never promotes a property without a
separately executed and verified test on the original environment.
A source assertion is a PREEXISTING obligation hypothesis only.
No imports, eval or execution of untrusted repository sources.
"""
from __future__ import annotations
import ast,hashlib,json
from dataclasses import asdict,dataclass

PROPS=frozenset(("shape","dtype","device","requires_grad"))
PROPERTY_STATUSES={"shape":"SHAPE_EQ","dtype":"DTYPE_EQ",
                   "device":"DEVICE_EQ","requires_grad":"GRAD_FLAG_EQ"}

def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()
def dump(x):return ast.dump(x,include_attributes=False)
def name(x):
 if isinstance(x,ast.Name):return x.id
 if isinstance(x,ast.Attribute):
  p=name(x.value)
  return p+"."+x.attr if p else ""
 return ""
def methods(source):
 tree=ast.parse(source)
 out={}
 for x in tree.body:
  if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name.startswith("test_"):
   out[x.name]=x
  elif isinstance(x,ast.ClassDef):
   for y in x.body:
    if isinstance(y,(ast.FunctionDef,ast.AsyncFunctionDef)) and y.name.startswith("test_"):
     out[x.name+"."+y.name]=y
 return out

def resolve_target(expr,aliases):
 if isinstance(expr,ast.Name):return aliases.get(expr.id,"")
 if isinstance(expr,ast.Attribute):
  q=name(expr)
  if q.startswith("self."):return ""
  return aliases.get(q,q)
 return ""

def track_stmts(node):
 aliases={}
 results={}
 assertions=[]
 for stmt in node.body:
  if isinstance(stmt,ast.Assign) and len(stmt.targets)==1 and isinstance(stmt.targets[0],ast.Name):
   var=stmt.targets[0].id
   expr=stmt.value
   if isinstance(expr,(ast.Name,ast.Attribute)):
    api=resolve_target(expr,aliases)
    if api and "." in api:aliases[var]=api
    else:aliases.pop(var,None)
    results.pop(var,None)
   elif isinstance(expr,ast.Call):
    api=resolve_target(expr.func,aliases)
    if api and "." in api:results[var]=api
    else:results.pop(var,None)
    aliases.pop(var,None)
   else:
    results.pop(var,None);aliases.pop(var,None)
  elif isinstance(stmt,ast.Assert):
   assertions.append(stmt.test)
  elif isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call):
   call=stmt.value
   if name(call.func) in ("self.assert_close","torch.testing.assert_close"):
    assertions.append(call)
 return results,assertions

def value_relation(expr,produced):
 if isinstance(expr,ast.Compare) and len(expr.ops)==1 and isinstance(expr.ops[0],(ast.Eq,ast.Is)):
  sides=(expr.left,expr.comparators[0])
  for side,other in (sides,tuple(reversed(sides))):
   if not (isinstance(side,ast.Attribute) and side.attr in PROPS):continue
   if not (isinstance(side.value,ast.Name) and side.value.id in produced):continue
   return produced[side.value.id],side.attr,dump(other)
 if isinstance(expr,ast.Call) and name(expr.func) in ("self.assert_close","torch.testing.assert_close"):
  if len(expr.args)>=2:
   for side,other in ((expr.args[0],expr.args[1]),(expr.args[1],expr.args[0])):
    if isinstance(side,ast.Name) and side.id in produced:
     return produced[side.id],"value_equality",dump(other)
 return None

@dataclass(frozen=True)
class Contract:
 source_test:str
 target_api:str
 property:str
 relation:str
 test_ast_sha256:str
 before_full_test_file_sha256:str
 after_full_test_file_sha256:str
 expected_ast_sha256:str
 origin:str="UNCHANGED_PREPATCH_TEST"
 test_executed:bool=False
 expected_value_validated:bool=False
 whole_patch_correctness:bool=False

def scan(before,after,source_test_file,api_prefix=None):
 try:b=methods(before);a=methods(after)
 except (SyntaxError,ValueError,UnicodeError):
  return {"status":"ABSTAIN_UNPARSEABLE","contracts":[],"n_removed_or_changed_tests":0}
 shared=set(b)&set(a)
 unchanged={key for key in shared if dump(b[key])==dump(a[key])}
 changed=len(set(b)-unchanged)
 found={}
 for key in sorted(unchanged):
  fn=b[key]
  produced,assertions=track_stmts(fn)
  for assertion in assertions:
   rel=value_relation(assertion,produced)
   if rel is None:continue
   api,prop,expected=rel
   if api_prefix and not (api==api_prefix or api.startswith(api_prefix+".")):continue
   tag=(key,api,prop,expected)
   found[tag]=asdict(Contract(
    source_test=source_test_file+"::"+key,target_api=api,
    property=prop,relation=PROPERTY_STATUSES.get(prop,"VALUE_EQ"),
    test_ast_sha256=sha(dump(fn)),
    before_full_test_file_sha256=sha(before),
    after_full_test_file_sha256=sha(after),
    expected_ast_sha256=sha(expected)))
 return {"status":"PREPATCH_TEST_CONTRACT_CANDIDATES",
         "source":source_test_file,
         "n_preexisting_test_functions":len(b),
         "n_unchanged_preexisting_tests":len(unchanged),
         "n_removed_or_changed_tests":changed,
         "n_contracts":len(found),
         "contracts":list(found.values()),
         "execution_claims":0,"whole_patch_correctness_claims":0,
         "not_supported":"NO_DYNAMIC_FIXTURE_ALIASES_NO_TEST_EXECUTION",
         "source_integrity_claim":"LOCAL_SHA256_NOT_REMOTE_ATTESTATION"}
