"""Minimal authorized interface repair over an EXACT box action contract.

The source actuator is t=C clip(P u,-L,L). Upstream action commands u lie
in the predeclared square [-R,R]^2; P,C invertible and FIXED. We test each
permitted concrete repair (noop/producer/controller/atomic), changing P
and/or C to identity in an actual chosen component. For linear matrices:

(1) C'P'=I and
(2) R*max_i sum_j |P'_ij| <= L

are sufficient for t(u)=u on the ENTIRE deployment action box, and
necessary under invertible C' and nonzero full-dimensional box: without (2),
some command saturates and cannot pass through invertible C' unchanged.

In contrast, reachability alone C'[-L,L]^2 covering the desired box does
NOT establish identity of actions or authorization to modify the component.

These are standard induced matrix norm/linear polytope facts, not novel
geometry. The unique robotics contribution candidate would require actual
trusted source authority and task-level execution before publication.
"""
from __future__ import annotations
from fractions import Fraction
from hashlib import sha256
from math import isfinite
import re
from typing import Mapping,Sequence
from .observable_interface_factorization import I, Mat, inverse, mmul, mv

PATCH_COSTS={"noop":Fraction(0),"producer-only":Fraction(1),
             "controller-only":Fraction(1),"atomic":Fraction(2)}
_SHA=re.compile(r"^[a-f0-9]{64}$")


def _max_matrix_abs_from(a:Mat,b:Mat)->float:
    return max(abs(a[i][j]-b[i][j]) for i in range(2) for j in range(2))


def proof_for_patch(*, producer:Mat,controller:Mat,patch:str,
                    action_radius:float,inner_clip:float,
                    matrix_tolerance:float=1e-9)->dict:
    if patch not in PATCH_COSTS:
        raise ValueError("undeclared concrete patch")
    if any(not isfinite(z) for row in (*producer,*controller) for z in row):
        raise ValueError("nonfinite control matrices")
    if not all(isfinite(z) for z in (action_radius,inner_clip,matrix_tolerance)) or (
         action_radius<=0 or inner_clip<=0 or matrix_tolerance<0):
        raise ValueError("invalid deployment contract")
    P=I if patch in ("producer-only","atomic") else producer
    C=I if patch in ("controller-only","atomic") else controller
    try:
        inverse(P);invC=inverse(C)
    except ValueError:
        return {"patch":patch,"certified":False,"failure":"singular interface",
                "patch_cost_exact":str(PATCH_COSTS[patch])}
    composed=mmul(C,P)
    mapping_error=_max_matrix_abs_from(composed,I)
    max_inner=action_radius*max(sum(abs(z) for z in row) for row in P)
    max_inverse_needed=action_radius*max(sum(abs(z) for z in row) for row in invC)
    reachable=max_inverse_needed<=inner_clip+matrix_tolerance
    no_internal_clip=max_inner<=inner_clip+matrix_tolerance
    mapping_match=mapping_error<=matrix_tolerance
    return {"patch":patch,"certified":bool(mapping_match and no_internal_clip),
            "constant_linear_composition_error":mapping_error,
            "maximum_producer_magnitude_on_action_box":max_inner,
            "maximum_required_actuator_inner_torque":max_inverse_needed,
            "unbounded_upstream_action_reachability":reachable,
            "full_action_box_no_internal_saturation":no_internal_clip,
            "repair_preserves_identity_mapping":mapping_match,
            "action_radius":action_radius,"inner_clip":inner_clip,
            "patch_cost_exact":str(PATCH_COSTS[patch]),
            "scope":"exact finite fixed 2x2 linear interface with componentwise clip and unbounded downstream motor torque; no source attestation"}


def choose_source_authorized_patch(
    *, producer:Mat,controller:Mat,action_radius:float,inner_clip:float,
    allowed_by_external_source:Sequence[str],external_source_verified:bool,
    source_digest:str|None,
)->dict:
    """Reject unless an OUTSIDE party has authenticated the source authority.

    This function never validates the boolean itself, a signer key or patch
    bytes; no claim of proof-carrying patch installation is made.
    """
    if not external_source_verified or not isinstance(source_digest,str) or \
       not _SHA.fullmatch(source_digest):
        return {"decision":"REFUSE_MISSING_EXTERNAL_SOURCE_PROVENANCE",
                "patch":None,"conditional_only":True}
    if not allowed_by_external_source:
        return {"decision":"REFUSE_NO_AUTHORIZED_PATCH", "patch":None}
    if any(x not in PATCH_COSTS for x in allowed_by_external_source):
        raise ValueError("source manifest contains unknown patch IDs")
    candidates=[proof_for_patch(
        producer=producer,controller=controller,patch=name,
        action_radius=action_radius,inner_clip=inner_clip)
        for name in sorted(set(allowed_by_external_source))]
    passed=[x for x in candidates if x["certified"]]
    if not passed:
        return {"decision":"REFUSE_NO_PATCH_VALID_OVER_DEPLOYMENT_ENVELOPE",
                "patch":None,"candidates":candidates,
                "source_digest":source_digest}
    best=min(passed,key=lambda x:(Fraction(x["patch_cost_exact"]),x["patch"]))
    return {"decision":"CONDITIONAL_CANDIDATE_ONLY_EXTERNAL_ATTESTATION_REQUIRED",
            "patch":best["patch"],"minimum_cost":best["patch_cost_exact"],
            "source_digest":source_digest,"candidates":candidates,
            "actual_patch_installed":False,
            "external_source_authentication_performed_by_this_module":False}


def freeze_example_report()->dict:
    P=((1.,.8),(0.,1.))
    C=((1.,-.8),(0.,1.))
    digest=sha256(b"demonstration-source-not-real-signed-issuer").hexdigest()
    return {
        "protocol":"two-by-two-compensated-interface-envelopes-v1",
        "small_development_envelope":choose_source_authorized_patch(
            producer=P,controller=C,action_radius=2.,inner_clip=9.,
            allowed_by_external_source=tuple(PATCH_COSTS),
            external_source_verified=True,source_digest=digest),
        "larger_deployment_envelope":choose_source_authorized_patch(
            producer=P,controller=C,action_radius=8.,inner_clip=9.,
            allowed_by_external_source=tuple(PATCH_COSTS),
            external_source_verified=True,source_digest=digest),
        "same_deployment_with_no_source_provenance":choose_source_authorized_patch(
            producer=P,controller=C,action_radius=8.,inner_clip=9.,
            allowed_by_external_source=tuple(PATCH_COSTS),
            external_source_verified=False,source_digest=None),
        "limitation":"demonstration authority flag is an ASSERTED assumption; no actual authenticated source or external physics model",
    }


if __name__=="__main__":
    import json
    print(json.dumps(freeze_example_report(),indent=2,sort_keys=True))
