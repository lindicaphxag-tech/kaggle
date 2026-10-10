"""PCPR v0.8 PRE-CHANGE unchanged-consumer dependency graph.

Identifies *syntactic* independent-from-PATCH consumers of changed functions
inside a source module. Works from paired, pinned complete files. The
outward caller closure uses only exact, narrowly resolved Python calls:

  - module function calls: f(...)
  - same-class self.method(...) calls
  - literal class-qualified calls: Class.method(...)

No dynamic getattr, imported aliases, dispatch across subclasses, monkey
patching, decorators or cross-file completeness is assumed. A verified source
edge is NOT a verified semantic correctness obligation or full runtime path.
Unknown edges/uses remain unseen, not certified. This is a candidate graph,
not a proof of patch correctness.
"""
from __future__ import annotations
import ast,hashlib,json,collections
from dataclasses import asdict,dataclass

PROPS=frozenset({"dtype","device","shape","state_dict","requires_grad",
                 "parameters","grad","size","masked_fill","where","clone"})

def funcs(src):
 tree=ast.parse(src)
 out={}
 for n in tree.body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):out[n.name]=n
  elif isinstance(n,ast.ClassDef):
   for m in n.body:
    if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)):
     out[n.name+"."+m.name]=m
 return out

def stable(node):
 return ast.dump(node,include_attributes=False)
def sha(data):
 return hashlib.sha256(data.encode("utf8")).hexdigest()

def resolve_call(node,owner,definitions,global_names,method_counts):
    if isinstance(node,ast.Name):
        return node.id if node.id in global_names else None
    if not isinstance(node,ast.Attribute):return None
    n=node.attr
    base=node.value
    cls=owner.split(".")[0] if "." in owner else None
    if isinstance(base,ast.Name) and base.id=="self" and cls and cls+"."+n in definitions:
        if method_counts.get(cls+"."+n)==1:
            return cls+"."+n
    if isinstance(base,ast.Name) and base.id+"."+n in definitions and base.id!="self":
        if method_counts.get(base.id+"."+n)==1:return base.id+"."+n
    return None

def dependency_graph(old,new,max_depth=3):
 try:a=funcs(old);b=funcs(new)
 except (SyntaxError,ValueError):
  return {"status":"ABSTAIN_PARSE","changed":[],"roots":[],"unknown":[]}
 shared=set(a)&set(b)
 changed={k for k in shared if stable(a[k])!=stable(b[k])}
 unchanged={k for k in shared if stable(a[k])==stable(b[k])}
 old_names={k for k in a if "." not in k}
 global_counts=collections.Counter(k for k in old_names)
 method_counts=collections.Counter(k for k in a if "." in k)
 reverse=collections.defaultdict(set)
 for caller,node in a.items():
  if caller not in unchanged:continue
  for n in ast.walk(node):
   if not isinstance(n,ast.Call):continue
   callee=resolve_call(n.func,caller,a,old_names,method_counts)
   if callee and callee!=caller:reverse[callee].add(caller)
 roots=[]
 for target in sorted(changed):
  queue=collections.deque([(target,0,(target,))]);seen={target}
  while queue:
   cur,depth,path=queue.popleft()
   if depth>=max_depth:continue
   for caller in sorted(reverse.get(cur,())):
    if caller in seen:continue
    seen.add(caller)
    body=a[caller]
    signals=sorted({n.attr for n in ast.walk(body)
                    if isinstance(n,ast.Attribute) and n.attr in PROPS})
    # Bind to BEFORE content, not derived from postpatch method text.
    roots.append({
      "changed_producer":target,"unchanged_consumer":caller,
      "depth":depth+1,"unchanged_consumer_ast_sha256":sha(stable(body)),
      "call_chain":list(path)+[caller],
      "property_signals":signals,
      "status":"BEFORE_ONLY_CALLGRAPH_CANDIDATE",
      "semantic_requirement_adjudicated":False,
      "runtime_dispatch_proven":False,
      "whole_patch_authorized":False
    })
    queue.append((caller,depth+1,path+(caller,)))
 return {"status":"SOURCE_CONSUMER_GRAPH",
         "source_before_sha256":sha(old),"source_after_sha256":sha(new),
         "n_original_functions":len(a),"n_changed_existing_functions":len(changed),
         "n_unchanged_functions":len(unchanged),
         "n_changed_producers_with_root":len({r["changed_producer"] for r in roots}),
         "n_syntactic_anchor_paths":len(roots),
         "changed":sorted(changed),
         "roots":roots,
         "unsupported_scope":"NO_CROSS_FILE_ALIAS_OR_DYNAMIC_DISPATCH",
         "runtime_correctness_assertions":0}

def summarise_file(file,old,new):
 d=dependency_graph(old,new)
 return {"file":file,
         "status":d["status"],
         "n_changed_producers_with_root":d.get("n_changed_producers_with_root",0),
         "n_syntactic_anchor_paths":d.get("n_syntactic_anchor_paths",0),
         "source_before_sha256":d.get("source_before_sha256"),
         "source_after_sha256":d.get("source_after_sha256"),
         "roots":d.get("roots",[]),
         "semantic_requirement_adjudicated":False,
         "whole_patch_authorized":False}
