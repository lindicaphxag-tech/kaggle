"""PCPR: authenticated-looking AST ownership path, not dynamic dispatch proof.

Source files are externally pinned before this inference. A path across class
constructors, ModuleList, and optimizer is evidence only of literal syntax.
"""
import ast
from hashlib import sha256


def qualified(n):
    if isinstance(n,ast.Name):return n.id
    if isinstance(n,ast.Attribute):
        p=qualified(n.value)
        return p+"."+n.attr if p else ""
    return ""

def classes(src):
    return {x.name:x for x in ast.parse(src).body if isinstance(x,ast.ClassDef)}

def method(cl,name):
    return next((f for f in cl.body if isinstance(f,ast.FunctionDef) and f.name==name),None)

def calls(tree,fn):
    return [x for x in ast.walk(tree) if isinstance(x,ast.Call) and fn(x)]

def ownership_route(weave_src,trainer_src):
    w=classes(weave_src);t=classes(trainer_src)
    if not all(k in w for k in ("Weave","WeaveModel")) or "TorchModel" not in t:
        return {"status":"ABSTAIN_MISSING_CLASS"}
    top=w["WeaveModel"];model=w["Weave"];trainer=t["TorchModel"]
    missing=[]
    if not any(isinstance(x,ast.Name) and x.id=="TorchModel" for x in top.bases):
        missing.append("NO_INHERITANCE")
    a=method(top,"__init__");b=method(model,"__init__");z=method(trainer,"_ensure_built")
    if a is None or b is None or z is None:
        return {"status":"ABSTAIN_MISSING_METHOD","missing":missing}
    if not any(isinstance(x,ast.Assign) and len(x.targets)==1
               and qualified(x.targets[0])=="self.model"
               and isinstance(x.value,ast.Call) and qualified(x.value.func)=="Weave"
               for x in ast.walk(a)):
        missing.append("NO_WEAVE_MODEL_OWNER")
    if not any(isinstance(x.func,ast.Attribute) and x.func.attr=="__init__"
               and isinstance(x.func.value,ast.Call) and qualified(x.func.value.func)=="super"
               and x.args and qualified(x.args[0])=="self.model"
               for x in calls(a,lambda x:True)):
        missing.append("NO_MODEL_TO_PARENT_INIT")
    if not any(isinstance(x,ast.AnnAssign) and qualified(x.target)=="self.layers"
               and isinstance(x.value,ast.Call) and qualified(x.value.func)=="nn.ModuleList"
               for x in ast.walk(b)):
        missing.append("NO_MODULELIST_OWNER")
    if not any(isinstance(x,ast.Assign)
               and any(isinstance(y,ast.Name) and y.id=="weave_layer" for y in x.targets)
               and isinstance(x.value,ast.Call)
               and qualified(x.value.func)=="torch_layers.WeaveLayer"
               for x in ast.walk(b)):
        missing.append("NO_WEAVELAYER_CTOR")
    if not calls(b,lambda x:qualified(x.func)=="self.layers.append"
                 and len(x.args)==1 and qualified(x.args[0])=="weave_layer"):
        missing.append("NO_WEAVELAYER_APPEND")
    if not calls(z,lambda x:qualified(x.func)=="self.optimizer._create_pytorch_optimizer"
                 and len(x.args)==1 and isinstance(x.args[0],ast.Call)
                 and qualified(x.args[0].func)=="self.model.parameters"):
        missing.append("NO_ACTUAL_OPTIMIZER_CALL")
    return {"status":"STATIC_OWNERSHIP_ROUTE" if not missing else "ABSTAIN_ROUTE_NOT_ESTABLISHED",
            "missing":missing,
            "source_sha256":{"owner":sha256(weave_src.encode()).hexdigest(),
                             "trainer":sha256(trainer_src.encode()).hexdigest()},
            "route":["WeaveModel(TorchModel)","self.model=Weave",
                     "super.__init__(self.model)","Weave.layers=nn.ModuleList",
                     "WeaveLayer appended","TorchModel._ensure_built:self.model.parameters()"],
            "scope":"Literal AST evidence only: no dynamic dispatch or full training execution"}
