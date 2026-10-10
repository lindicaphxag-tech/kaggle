"""PCPR scoped PyTorch parameter-registration audit, NOT whole-patch correctness.

The obligations are grounded in BEFORE-commit forward reads and an unchanged
optimizer consumer. AFTER-commit source cannot invent its own obligation.
Only source expressions accepted by a strict AST grammar are executed, in
an ISOLATED nn.Module with stub initialization (not a native package run).
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, asdict
from hashlib import sha256


@dataclass(frozen=True)
class FieldEvidence:
    class_name: str
    field: str
    line: int
    origin_hash: str
    rhs: str
    used_in_forward: bool
    rhs_parameter_wrapper: bool


def _q(n):
    if isinstance(n, ast.Name): return n.id
    if isinstance(n, ast.Attribute):
        r = _q(n.value)
        return r + "." + n.attr if r else ""
    return ""


def _self_attr(n):
    return n.attr if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self" else None


def _walk_class(source, class_name):
    tree = ast.parse(source)
    classes = [c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == class_name]
    if len(classes) != 1: raise ValueError("A unique real class is required")
    return classes[0]


def discover_fields(source: str, class_name: str) -> dict[str, FieldEvidence]:
    cl = _walk_class(source, class_name)
    init = next((f for f in cl.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == "__init__"), None)
    forward = next((f for f in cl.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == "forward"), None)
    if init is None or forward is None: raise ValueError("init and forward must be present")
    reads = {_self_attr(n) for n in ast.walk(forward)
             if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)}
    out = {}
    origin = sha256(source.encode("utf8")).hexdigest()
    for stmt in ast.walk(init):
        if isinstance(stmt, ast.AnnAssign):
            attr, val = _self_attr(stmt.target), stmt.value
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            attr, val = _self_attr(stmt.targets[0]), stmt.value
        else: continue
        if not attr or val is None or attr not in reads: continue
        if attr in out: raise ValueError("Ambiguous multiple assignments: " + attr)
        wrapped = isinstance(val, ast.Call) and _q(val.func) in ("nn.Parameter", "torch.nn.Parameter")
        out[attr] = FieldEvidence(class_name, attr, stmt.lineno, origin,
                                  ast.unparse(val), True, wrapped)
    return out


def has_optimizer_parameter_consumer(source: str) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "parameters" and not n.args and not n.keywords
               for n in ast.walk(ast.parse(source)))


def registration_runtime(rhs: str) -> dict:
    """Strict allowlist; expression from independently pinned source only."""
    import torch
    from torch import nn
    expr = ast.parse(rhs, mode="eval").body
    allowed = {"nn.Parameter", "torch.nn.Parameter", "torch.empty", "torch.zeros", "init"}
    nodes = (ast.List, ast.Tuple, ast.Constant, ast.Load, ast.BinOp, ast.Mult,
             ast.Add, ast.Sub, ast.USub, ast.UnaryOp, ast.Name, ast.Attribute, ast.Call)
    for n in ast.walk(expr):
        if not isinstance(n, nodes):
            raise ValueError("Forbidden syntax: " + type(n).__name__)
        if isinstance(n, ast.Call) and _q(n.func) not in allowed:
            raise ValueError("Unknown call: " + _q(n.func))
        if isinstance(n, ast.Attribute) and _q(n) not in allowed and not _q(n).startswith("self.n_"):
            raise ValueError("Unknown attribute: " + _q(n))
        if isinstance(n, ast.Name) and n.id not in ("torch", "nn", "self", "init"):
            raise ValueError("Unknown root: " + n.id)
    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("n_atom_input_feat", "n_pair_input_feat", "n_atom_output_feat",
                         "n_pair_output_feat", "n_hidden_AA", "n_hidden_AP",
                         "n_hidden_A", "n_hidden_P", "n_hidden_PA",
                         "n_hidden_PP", "n_input"):
                setattr(self, name, 4)
    module = Shell()
    initializer = lambda t: torch.nn.init.uniform_(t, -0.1, 0.1)
    v = eval(compile(ast.Expression(expr), "<audited-source-rhs>", "eval"),
             {"__builtins__": {}, "torch": torch, "nn": nn,
              "init": initializer, "self": module}, {})
    if not isinstance(v, torch.Tensor):
        raise ValueError("The audited expression did not produce a tensor")
    module.target = v
    params = dict(module.named_parameters())
    registered = "target" in params
    gradient = None
    if registered:
        params["target"].square().sum().backward()
        gradient = params["target"].grad is not None
    return {"registered": registered, "requires_grad": bool(v.requires_grad),
            "parameter_gradient": gradient, "shape": list(v.shape)}


def assess_pair(before: str, after: str, consumer: str, class_name: str,
                expected_before_sha: str | None = None,
                expected_after_sha: str | None = None) -> dict:
    hashes = {"before": sha256(before.encode()).hexdigest(),
              "after": sha256(after.encode()).hexdigest(),
              "consumer": sha256(consumer.encode()).hexdigest()}
    if expected_before_sha and hashes["before"] != expected_before_sha:
        raise ValueError("Before source hash mismatch")
    if expected_after_sha and hashes["after"] != expected_after_sha:
        raise ValueError("After source hash mismatch")
    if not has_optimizer_parameter_consumer(consumer):
        return {"status": "ABSTAIN_NO_SOURCE_INDEPENDENT_OPTIMIZER_CONSUMER",
                "hashes": hashes}
    old = discover_fields(before, class_name)
    new = discover_fields(after, class_name)
    records = []
    for field, prior in sorted(old.items()):
        if field not in new:
            records.append({"field": field, "status": "ABSTAIN_MISSING_FIELD_IN_PATCH"})
            continue
        latest = new[field]
        try:
            b = registration_runtime(prior.rhs)
            a = registration_runtime(latest.rhs)
        except (ValueError, TypeError, RuntimeError, AttributeError, NameError) as e:
            records.append({"field": field,
                            "status": "ABSTAIN_UNSUPPORTED_SOURCE_EXPRESSION",
                            "reason": type(e).__name__})
            continue
        decision = ("BOUNDED_REGISTRATION_WITNESS"
                    if not b["registered"] and a["registered"]
                    else "ABSTAIN_NO_REGISTRATION_GAIN"
                    if not a["registered"] else "ALREADY_REGISTERED")
        records.append({"field": field, "status": decision, "before": b,
                        "after": a, "before_line": prior.line,
                        "after_line": latest.line})
    return {"status": "SCOPED_OPERATOR_AUDIT_ONLY", "hashes": hashes,
            "class_name": class_name, "consumer_is_unchanged": True,
            "fields": records,
            "limits": [
                "Source consumer only establishes an optimizer path, not every field's intended trainability",
                "Isolated AST field RHS, not native DeepChem fit or GPU execution",
                "Retrospective development PR; no prospective discovery rate",
                "Never authorizes whole patch correctness"
            ]}
