"""Evidence-constrained temporal interface repair operator on REAL LeRobot code.

Research aim: determine when *source* mutation is ACTUALLY necessary, not
simply rewrite sources whenever a fault is detected. Unlike the linear
P/C/clip problem, this is a STATEFUL ABI problem: correct absolute robot
actions depend on an action-chunk reference state S0 that may no longer
be present in the current sensor observation S(t).

Given an observational equivalence class of plausible historical anchors
A={a_1,...,a_K}, exact frozen relative action R, live observation S(t),
and an allowed worst-case action tolerance epsilon, consider:

  1) current-only minimax output (center of coordinatewise anchor ranges):
     min_b max_{a in A} ||abs(R,a)-b||_infty
       = (1/2) max_j (max_a a_j - min_a a_j)
     on the converted action channels, provided all anchor worlds share
     the same gripper/unconverted source conventions.
  2) if this minimax ambiguity radius <= epsilon: a black-box current-only
     correction meets the contract; NO source patch is necessary.
  3) if a valid *historical observation cache* is allowed and binds to this
     exact chunk epoch, the blackbox can reconstruct perfectly; prefer
     that cheaper intervention.
  4) otherwise if a TRUSTED source-anchor read/repair is permitted,
     retain the correct source anchor for the entire chunk; no false
     current-state reconstruction is authorized.
  5) otherwise REFUSE rather than silently pick one plausible history.

This operator's minimax lower bound is exact under L_inf and the stated
explicit set of possible anchors (standard Chebyshev center theory, NOT a
novel minimax theorem). This evaluator does not verify Ed25519 signatures
or attest physical sensors; a source_anchor_verified Boolean is a CALLER
TRUST ASSERTION and cannot substitute for the separate signed artifact
gate. The actual corrective conversion ALWAYS runs original source-pinned
LeRobot function on a real PyTorch tensor. The injected callsite error
is AUTHOR-created, not a public vendor bug.

Unlike the previous source-only narrative, this explicitly allows the
history-enabled black-box to TIE the source method when information parity
holds. Future publication must retain those equal-information results.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any
import torch


@dataclass(frozen=True)
class InformationDecision:
    outcome:str
    operation:str
    worst_case_error_bound:float
    monetary_or_permission_cost:float
    minmax_radius:float
    authority_assumption:str
    source_file_modified:bool


def _validate_and_center(anchors,relative,current,mask):
    if not isinstance(relative,torch.Tensor) or relative.ndim!=3:
        raise ValueError("relative action chunk must be real original torch [B,T,A]")
    if relative.shape[0]!=1 or relative.shape[-1]!=len(mask):
        raise ValueError("requires one chunk and one-dimensional mask per action channel")
    if not isinstance(current,torch.Tensor) or current.shape!=(1,len(mask)):
        raise ValueError("current observation is [1,D] tensor")
    if not isinstance(anchors,(list,tuple)) or not anchors:
        raise ValueError("at least one plausible historical anchor required")
    if any(not isinstance(v,torch.Tensor) or v.shape!=current.shape
           for v in anchors):
        raise ValueError("every anchored world must use current tensor shape")
    if not all(bool(torch.isfinite(x).all()) for x in [relative,current,*anchors]):
        raise ValueError("nonfinite robot source action/state")
    if any(type(m) is not bool for m in mask):
        raise ValueError("invalid source channel mask")
    a=torch.cat([v.to(device=relative.device,dtype=relative.dtype)
                 for v in anchors],dim=0)
    mask_t=torch.tensor(mask,dtype=torch.bool,device=relative.device)
    low=a.min(dim=0).values
    high=a.max(dim=0).values
    midpoint=(low+high)/2.
    radius=float(((high-low)[mask_t]/2.).amax()) if bool(mask_t.any()) else 0.
    return midpoint.unsqueeze(0),radius


def select_information_authority(
    *,relative,current,mask,possible_historical_anchors,
    allowable_max_action_error:float,
    historical_cache=None,source_anchor=None,
    cache_verified=False,source_anchor_verified=False,
    cache_epoch=None,source_epoch=None,chunk_epoch=None,
    cost_current_only=0.,cost_cache=.1,cost_source=1.,
):
    """Choose lowest declared cost of robustly correct permitted interventions.

    The masks mark action channels converted relative to reference state.
    The source function consumer must use the chosen *explicit* reference;
    hence no intervention ever silently substitutes hidden true world
    information for a black-box current-only actor.
    """
    midpoint,radius=_validate_and_center(
        possible_historical_anchors,relative,current,mask)
    for name,v in (("allowable_max_action_error",allowable_max_action_error),
                   ("cost_current_only",cost_current_only),
                   ("cost_cache",cost_cache),("cost_source",cost_source)):
        if type(v) not in (float,int) or not isfinite(v) or v<0:
            raise ValueError(f"invalid declared cost or tolerance {name}")
    # Reject contradictory purported trustworthy history BEFORE choosing
    # the cheapest intervention. Otherwise a forged 'verified' source
    # witness could be silently ignored whenever CURRENT_ONLY_MINIMAX has
    # zero cost, defeating fail-closed provenance semantics and giving a
    # false impression that a calibration-source identity was checked.
    for flag,claim,label in (
        (cache_verified,historical_cache,"cache"),
        (source_anchor_verified,source_anchor,"source"),
    ):
        if not flag:
            continue
        if not isinstance(claim,torch.Tensor) or claim.shape!=current.shape or \
           not bool(torch.isfinite(claim).all()):
            raise ValueError("purported verified historical reference missing or invalid "+label)
        if not any(bool(torch.allclose(claim.to(v),v,atol=1e-12,rtol=0))
                   for v in possible_historical_anchors):
            raise ValueError("purported verified historical reference contradicts source-world evidence")
    options=[]
    if radius<=allowable_max_action_error:
        options.append((float(cost_current_only),"CURRENT_ONLY_MINIMAX",
                        midpoint,radius,False))
    if cache_verified and cache_epoch is not None and chunk_epoch==cache_epoch:
        if not isinstance(historical_cache,torch.Tensor) or \
           historical_cache.shape!=current.shape:
            raise ValueError("verified history cache missing its tensor")
        options.append((float(cost_cache),"USE_VERIFIED_HISTORY",
                        historical_cache,0.,False))
    if source_anchor_verified and source_epoch is not None and \
       chunk_epoch==source_epoch:
        if not isinstance(source_anchor,torch.Tensor) or \
           source_anchor.shape!=current.shape:
            raise ValueError("verified source anchor missing its tensor")
        options.append((float(cost_source),"AUTHORIZED_SOURCE_ANCHOR",
                        source_anchor,0.,True))
    if not options:
        return InformationDecision(
            "REFUSE_UNCERTIFIED_CURRENT_STATE",
            "ABSTAIN",radius,float("inf"),radius,
            "none of the offered information authorities proves the requested command bound",
            False),None
    cost,kind,reference,guarantee,source_mutation=min(
        options,key=lambda x:(x[0],x[1]))
    # A verified cache or source witness MUST agree with at least one
    # plausible historical world, and each protected mask channel must
    # be exact. Otherwise a caller's "verified" Boolean can lead to false
    # authorization even under the model's declared knowledge set.
    if kind in ("USE_VERIFIED_HISTORY","AUTHORIZED_SOURCE_ANCHOR"):
        compatible=any(bool(torch.allclose(reference.to(v),v,atol=1e-12,rtol=0))
                       for v in possible_historical_anchors)
        if not compatible:
            raise ValueError("purported verified historical reference contradicts source-world evidence")
    return InformationDecision(
        "AUTHORIZE_WITHIN_EXPLICIT_INFORMATION_MODEL",
        kind,guarantee,cost,radius,
        "caller-supplied verification and epoch identity are assumed, not independently authenticated here; this operator selects authority but does not write bytes",
        False),reference


def execute_original_lerobot_conversion(
    original_to_absolute,relative,decision,explicit_reference,mask
):
    """Call ORIGINAL separately hash-verified publisher kernel, not a mock."""
    if decision.outcome!="AUTHORIZE_WITHIN_EXPLICIT_INFORMATION_MODEL" or \
       explicit_reference is None:
        raise ValueError("source action is not authorized")
    out=original_to_absolute(relative,explicit_reference,mask)
    if out.shape!=relative.shape or not bool(torch.isfinite(out).all()):
        raise ValueError("external source conversion did not preserve contract")
    return out
