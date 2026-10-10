"""PCPR v0.6: syntax-guided counterfactual intervention operators.

Generates *single-edit* semantic ablations from actual pinned source AST:
  - omit dtype argument of a tensor factory,
  - remove local input cloning,
  - discard requires_grad forwarding to a Parameter,
  - demote register_buffer to a plain member assignment.

Generated variants are ARTIFICIAL, never counted as independently occurring
upstream incorrect fixes. The semantics must be checked by separate native
execution, not inferred from AST pattern alone. Mutation identity is logged.
"""
from __future__ import annotations
import ast,copy,hashlib,json
from dataclasses import asdict,dataclass

OPERATORS=frozenset({"drop_tensor_factory_dtype","drop_clone","drop_parameter_grad",
                     "demote_buffer_registration"})

@dataclass(frozen=True)
class Mutation:
 symbol:str
 operator:str
 input_sha256:str
 output_sha256:str
 changed_line:int
 n_operations:int
 kind:str="AUTHOR_SYNTHESIZED_SINGLE_AST_INTERVENTION"

def call_name(n):
 if isinstance(n,ast.Name):return n.id
 if isinstance(n,ast.Attribute):return call_name(n.value)+"."+n.attr
 return ""

class Intervention(ast.NodeTransformer):
 def __init__(self,operator):
  if operator not in OPERATORS:raise ValueError("Unknown mutation operator")
  self.operator=operator
  self.records=[]
 def visit_Call(self,node):
  node=self.generic_visit(node)
  name=call_name(node.func)
  if self.operator=="drop_tensor_factory_dtype" and name in ("torch.tensor","torch.as_tensor"):
   kw=[x for x in node.keywords if x.arg=="dtype"]
   if len(kw)==1:
    node.keywords=[x for x in node.keywords if x.arg!="dtype"]
    self.records.append(node.lineno)
  elif self.operator=="drop_clone" and isinstance(node.func,ast.Attribute) and node.func.attr=="clone" and not node.args and not node.keywords:
   self.records.append(node.lineno)
   return copy.deepcopy(node.func.value)
  elif self.operator=="drop_parameter_grad" and name in ("nn.Parameter","torch.nn.Parameter"):
   kw=[x for x in node.keywords if x.arg=="requires_grad"]
   if len(kw)==1:
    node.keywords=[x for x in node.keywords if x.arg!="requires_grad"]
    self.records.append(node.lineno)
  return node
 def visit_Expr(self,node):
  # Handle only plain buffer registration expression with a LITERAL key.
  # An arbitrary register_buffer overload remains unsupported.
  if self.operator=="demote_buffer_registration":
   v=node.value
   if isinstance(v,ast.Call) and call_name(v.func)=="self.register_buffer" and len(v.args)==2 and not v.keywords:
    name, rhs=v.args
    if isinstance(name,ast.Constant) and isinstance(name.value,str) and name.value.isidentifier():
     self.records.append(node.lineno)
     return ast.copy_location(ast.Assign(
        targets=[ast.Attribute(value=ast.Name(id="self",ctx=ast.Load()),
                               attr=name.value,ctx=ast.Store())],
        value=copy.deepcopy(rhs)),node)
  return self.generic_visit(node)

def _symbol(tree,symbol):
 parts=symbol.split(".")
 if len(parts)==1:
  found=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==parts[0]]
 elif len(parts)==2:
  cs=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==parts[0]]
  found=[n for c in cs for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==parts[1]]
 else:
  raise ValueError("Nested/multi-level symbols unsupported")
 if len(found)!=1:raise ValueError("AMBIGUOUS_OR_MISSING_TARGET")
 return found[0]

def mutate(source,symbol,operator):
 tree=ast.parse(source)
 original=_symbol(tree,symbol)
 worker=Intervention(operator)
 # Mutate only the selected function, never other code in the module.
 transformed=worker.visit(copy.deepcopy(original))
 if len(worker.records)!=1:raise ValueError("AMBIGUOUS_INTERVENTION_COUNT:"+str(len(worker.records)))
 if isinstance(tree.body[0],ast.ClassDef) and symbol.startswith(tree.body[0].name+"."):
  # Case handled in general traversal below, no special assumptions.
  pass
 parts=symbol.split(".")
 if len(parts)==1:
  for i,n in enumerate(tree.body):
   if n is original:tree.body[i]=transformed;break
 else:
  for c in tree.body:
   if isinstance(c,ast.ClassDef) and c.name==parts[0]:
    for i,n in enumerate(c.body):
     if n is original:c.body[i]=transformed;break
 tree=ast.fix_missing_locations(tree)
 changed=ast.unparse(tree)+"\n"
 m=Mutation(symbol,operator,hashlib.sha256(source.encode()).hexdigest(),
            hashlib.sha256(changed.encode()).hexdigest(),
            worker.records[0],len(worker.records))
 return changed,asdict(m)

def valid_single_edit(before,after):
 return (before!=after and hashlib.sha256(before.encode()).digest()!=
         hashlib.sha256(after.encode()).digest())
