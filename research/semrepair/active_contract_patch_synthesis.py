"""Contract-guided active-probe source patch synthesis on REAL upstream ManiSkill.

JSS/IST research-grade adversarial result, with *honest overfitting controls*.
Given a source SHA-pinned upstream controller method, create a finite
TYPED AST repair grammar for precisely one multiplication-source term.
The first (symmetric rotation-limit) training test accepts multiple WRONG
patches. Choose one extra asymmetrical limit counterexample from a fixed
public calibration probe bank to eliminate them, and verify all candidates
by an independent symbolic polynomial identity proof over the complete
declared domain lower<0 and upper>0.

Unlike simply hardcoding PR #1472's answer, the selector evaluates the
entire licensed candidate grammar, records all plausible but incorrect
patches, counts REAL unmodified upstream controller executions, and
returns a source AST patch + digest for the cheapest *contract-correct*
repair, or REFUSE when no authorized candidate has a proof.

The original helper+controller code come from the actual official
mani-skill/ManiSkill immutable pinned commit through the companion module;
the source patch remains a first-party candidate, NOT merged/upstream or
vendor-authorized. Contract "positive rotation gain = rot_upper" is
a declared controller-specification assumption, not separately signed by
the real robotics manufacturer. This is not a new generic APR theorem;
active distinguishing tests and polynomial identities are established
verification tools applied to authority-constrained robot software.
"""
from __future__ import annotations

import ast
import copy
from hashlib import sha256
from itertools import product
from random import Random
import math
import sympy as sp
import torch

from .maniskill_upstream_rotation_sign_case import (
    _frozen_upstream_source,_get_controller_method,_real_helper_function,
    _source_method_runtime,CONTROLLER,HELPER,CONTROLLER_BLOB_SHA1,
    HELPER_BLOB_SHA1,COMMIT,UPSTREAM_ISSUE_URL,UPSTREAM_OPEN_PR_URL
)

# Intentionally adversarial grammar. Five candidates are indistinguishable
# on the typical (lower,upper)=(-0.1,+0.1) symmetric training input.
# Only one has the correct semantics for arbitrary INDEPENDENT valid bounds.
GRAMMAR=(
 ("original_negative_lower","self.config.rot_lower",0),
 ("proposed_upper","self.config.rot_upper",1),
 ("negate_lower","-self.config.rot_lower",2),
 ("absolute_lower","abs(self.config.rot_lower)",2),
 ("hardcode_training_upper","0.1",2),
 ("half_span","(self.config.rot_upper-self.config.rot_lower)/2",3),
 ("average_positive_gains","(self.config.rot_upper+abs(self.config.rot_lower))/2",4),
)
TRAIN_PROBE=(-.1,.1)
# These available calibration probes are not statistical held-out data;
# selection is allowed to look at ALL their predicted candidate results.
COUNTEREXAMPLE_BANK=(
 (-.08,.16),(-.15,.03),(-.06,.18),(-.125,.21),
 (-.05,.09),(-.12,.07),(-.3,.04),
)
N_FINAL_HELDOUT=128
TOL=1e-9


def _terms(ast_expr):
    """Decode only safe source-term AST grammar to symbolic (lower,upper)."""
    lo,hi=sp.symbols("lower upper",real=True)
    def rec(n):
        if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Attribute) and \
           isinstance(n.value.value,ast.Name) and n.value.value.id=="self" and \
           n.value.attr=="config" and n.attr in ("rot_lower","rot_upper"):
            return lo if n.attr=="rot_lower" else hi
        if isinstance(n,ast.Constant) and type(n.value) in (int,float) and \
           math.isfinite(n.value):
            return sp.Rational(str(n.value))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):
            return -rec(n.operand)
        if isinstance(n,ast.BinOp):
            a,b=rec(n.left),rec(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n.op,ast.Div):return a/b
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and \
           n.func.id=="abs" and len(n.args)==1 and not n.keywords:
            return sp.Abs(rec(n.args[0]))
        raise ValueError("source term outside approved typed repair grammar")
    return rec(ast_expr)


def _full_domain_contract(expr):
    """Polynomial identity proof on independent lower<0 and upper>0.

    Since the grammar permits abs(lower) only, map |lower|=-lower.
    A patch is admitted only if the symbolic polynomial equals 'upper'
    for EVERY independently chosen lower<0, upper>0.
    """
    lo,hi=sp.symbols("lower upper",real=True)
    form=_terms(ast.parse(expr,mode="eval").body)
    form=form.subs(sp.Abs(lo),-lo)
    delta=sp.cancel(sp.together(form-hi))
    numerator,_=sp.fraction(delta)
    poly=sp.Poly(numerator,lo,hi)
    ok=poly.is_zero
    return {
        "declared_independent_source_domain":"rot_lower<0,rot_upper>0,not necessarily symmetric",
        "expression":expr,"normalized_residual":str(delta),
        "all_domain_symbolic_identity":bool(ok),
        "symbolic_polynomial_monomial_count":len(poly.terms()),
        "proof_assumptions":"original norm clipping stays unchanged, scalar rot limits, positive action orientation defined as rot_upper",
    }


def _mutate_actual_method(original_method,expression):
    method=copy.deepcopy(original_method)
    occurrences=0
    for node in ast.walk(method):
        if not isinstance(node,ast.BinOp) or not isinstance(node.op,ast.Mult):
            continue
        r=node.right
        if isinstance(r,ast.Attribute) and r.attr=="rot_lower" and \
           isinstance(r.value,ast.Attribute) and r.value.attr=="config" and \
           isinstance(r.value.value,ast.Name) and r.value.value.id=="self":
            node.right=ast.copy_location(
                ast.parse(expression,mode="eval").body,node.right)
            occurrences+=1
    if occurrences!=1:
        raise ValueError("actual original upstream rotation source AST site changed")
    ast.fix_missing_locations(method)
    return method


def _execute_vendor_controller(method,helper,lower,upper,rotation):
    from types import SimpleNamespace
    real=_source_method_runtime(method,helper)
    owner=SimpleNamespace(
        config=SimpleNamespace(rot_lower=float(lower),rot_upper=float(upper)),
        action_space_low=torch.tensor([-0.1]*6,dtype=torch.float64),
        action_space_high=torch.tensor([.1]*6,dtype=torch.float64),
    )
    inp=torch.tensor([[0.,0.,0.,*rotation]],dtype=torch.float64)
    output=real(owner,inp)
    if output.shape!=(1,6) or not bool(torch.isfinite(output).all()):
        raise ValueError("compiled actual vendor code returned invalid source controls")
    return tuple(float(v) for v in output[0,3:])


def _expected(upper,rotation):
    norm=math.sqrt(sum(v*v for v in rotation))
    gain=min(1.,1./norm) if norm>0 else 1.
    return tuple(v*gain*upper for v in rotation)


def _err(output,expected):
    return max(abs(a-b) for a,b in zip(output,expected))


def _choose_discriminating_probe(survivors,real_methods,helper):
    """One adaptive counterexample targeted to observed WRONG AST programs."""
    results=[]
    for lower,upper in COUNTEREXAMPLE_BANK:
        expected=_expected(upper,(1.,0.,0.))
        predicted={name:_execute_vendor_controller(
            real_methods[name],helper,lower,upper,(1.,0.,0.))
                   for name in survivors}
        kept=[name for name in survivors if _err(predicted[name],expected)<=TOL]
        results.append({
            "lower":lower,"upper":upper,
            "wrong_plausible_eliminated":len(survivors)-len(kept),
            "plausible_remaining":kept,
            "executions":len(survivors),
        })
    return max(results,key=lambda x:(x["wrong_plausible_eliminated"],
                                    -abs(x["upper"]),-abs(x["lower"])))


def run_constrained_active_source_patch_experiment():
    controller,controller_sha=_frozen_upstream_source(
        CONTROLLER,CONTROLLER_BLOB_SHA1)
    helper,helper_sha=_frozen_upstream_source(
        HELPER,HELPER_BLOB_SHA1)
    original=_get_controller_method(controller)
    actual_helper=_real_helper_function(helper)
    methods={}
    proofs={}
    candidates=[]
    for name,expr,edit_cost in GRAMMAR:
        method=_mutate_actual_method(original,expr)
        methods[name]=method
        proofs[name]=_full_domain_contract(expr)
        output=_execute_vendor_controller(
            method,actual_helper,*TRAIN_PROBE,(1.,0.,0.))
        error=_err(output,_expected(TRAIN_PROBE[1],(1.,0.,0.)))
        candidates.append({
            "patch_id":name,"source_expression":expr,
            "ast_edit_cost":edit_cost,"training_symmetric_error":error,
            "training_plausible":error<=TOL,
            "symbolic_domain_proof":proofs[name],
            "AST_method_sha256":sha256(ast.unparse(method).encode()).hexdigest(),
        })
    survivors=[x["patch_id"] for x in candidates if x["training_plausible"]]
    if len(survivors)<4:
        raise AssertionError("calibration cannot expose intended training overfitting")
    active=_choose_discriminating_probe(
        survivors,methods,actual_helper)
    remaining=active["plausible_remaining"]
    eligible=[x for x in candidates if x["patch_id"] in remaining and
              x["symbolic_domain_proof"]["all_domain_symbolic_identity"]]
    if not eligible:
        raise AssertionError("no truly contract-compliant source candidate in catalog")
    chosen=min(eligible,key=lambda p:(p["ast_edit_cost"],p["patch_id"]))
    # The 128 final (asymmetric) task-free policy test cases are
    # NOT available to choose the probe, patch or proof. Fresh PRNG seed
    # disclosed for replicability; one is NOT a new independent robot.
    rng=Random(20261011)
    outcomes={name:[] for name in methods}
    rotations=[]
    for case in range(N_FINAL_HELDOUT):
        lo=-rng.uniform(.02,.3)
        hi=rng.uniform(.025,.35)
        vec=tuple(rng.uniform(-1.4,1.4) for _ in range(3))
        rotations.append((lo,hi,vec))
        ideal=_expected(hi,vec)
        for name,method in methods.items():
            result=_execute_vendor_controller(method,actual_helper,lo,hi,vec)
            outcomes[name].append(_err(result,ideal))
    heldout=[{
        "patch_id":name,
        "mean_max_action_error":sum(errors)/len(errors),
        "max_action_error":max(errors),
        "cases_passed":sum(x<=TOL for x in errors),
        "cases_failed":sum(x>TOL for x in errors),
        "symbolic_contract_true":proofs[name]["all_domain_symbolic_identity"],
    } for name,errors in outcomes.items()]
    if next(x for x in heldout if x["patch_id"]==chosen["patch_id"])["cases_failed"]!=0:
        raise AssertionError("symbolically admitted patch fails truly frozen heldout tests")
    naive_bad=sum(x["training_plausible"] and not
        x["symbolic_domain_proof"]["all_domain_symbolic_identity"]
        for x in candidates)
    active_bad=sum(name in remaining and not proofs[name]["all_domain_symbolic_identity"]
                   for name in remaining)
    return {
        "protocol":"real-immutable-ManiSkill-typed-AST-active-counterexample-source-synthesis-v1",
        "vendor_source_commit":COMMIT,
        "vendor_original_controller_source_SHA256":controller_sha,
        "vendor_original_helper_source_SHA256":helper_sha,
        "vendor_issue":UPSTREAM_ISSUE_URL,
        "vendor_PR_open_unmerged":UPSTREAM_OPEN_PR_URL,
        "real_manufacturer_signer_permission_obtained":False,
        "new_native_robot_task_episodes":0,
        "candidate_count":len(candidates),
        "symmetric_training_probe":list(TRAIN_PROBE),
        "training_plausible_candidate_count":len(survivors),
        "training_overfit_wrong_candidate_count":naive_bad,
        "active_probes_selected_count":1,
        "active_probe":active,
        "active_remaining_candidate_count":len(remaining),
        "active_wrong_candidate_count":active_bad,
        "symbolically_certified_remaining_patch_count":len(eligible),
        "selected_minimal_verified_AST_patch":chosen["patch_id"],
        "selected_source_executable_AST":ast.unparse(methods[chosen["patch_id"]]),
        "heldout_distinct_asymmetric_config_action_examples":N_FINAL_HELDOUT,
        "heldout_result_per_candidate":heldout,
        "actual_upstream_torch_kernel_executions":len(candidates)+
            len(survivors)*len(COUNTEREXAMPLE_BANK)+
            len(methods)*N_FINAL_HELDOUT,
        "symbolic_contract_conditional_on_declared_rot_upper_semantics":True,
        "patch_author_generated_not_installed_or_merged":True,
        "limitations":"7 hand-enumerated typed AST source terms, first-party scalar rotation contract, no independent vendor signing, no SAPIEN task or general unrestricted APR success",
        "candidates":candidates,
    }


if __name__=="__main__":
    import json
    print(json.dumps(run_constrained_active_source_patch_experiment(),
                     sort_keys=True,indent=2))
