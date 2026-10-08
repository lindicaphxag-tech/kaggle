"""Sample-conditioned C2 transport for MONAI DiNTS default ReLU/InstanceNorm.

This proof is sound in exact arithmetic *conditional on* trusted uniform
input world-space intervals and Hessian bounds. ReLU is linear on a region
known to stay positive or negative; InstanceNorm3d on one fixed sample has
a constant per-channel affine scale determined by actual pre-norm samples.
It does NOT prove the user-provided input premises or quantify roundoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .weight_curvature import _check_conv_branch


@dataclass(frozen=True)
class ConditionalMONAICurvature:
    branch_output_bounds: tuple[float, ...]
    normalized_output_bounds: tuple[float, ...]
    fiber_bounds: tuple[float, float]
    fiber_starts: tuple[int, int]
    normalization_mean: tuple[float, ...]
    normalization_scale: tuple[float, ...]
    normalization_offset: tuple[float, ...]
    relu_active: tuple[bool, ...]
    premise_intervals: tuple[tuple[float, float], ...]
    input_hessian_bounds: tuple[float, ...]
    weights_and_sample_sha256: str

    def by_fiber_start(self) -> dict[int, float]:
        return dict(zip(self.fiber_starts,self.fiber_bounds))


def _data(value: Sequence[float] | float, count: int, label: str) -> np.ndarray:
    v=np.asarray(value,dtype=np.float64)
    if v.ndim==0:
        v=np.full(count,float(v),dtype=np.float64)
    if v.shape!=(count,) or not np.all(np.isfinite(v)) or np.any(v<0):
        raise ValueError(f"{label} must be finite, nonnegative, one per input channel")
    return v


def derive_default_monai_sample_conditional_curvature(
    block: nn.Module,
    input_tensor: torch.Tensor,
    *,
    input_value_intervals: Sequence[tuple[float,float]],
    uniform_input_world_hessian_bounds: float | Sequence[float],
) -> ConditionalMONAICurvature:
    """Carry a declared input C2 premise through ReLU/Conv/InstanceNorm.

    Only one fixed N=1 input sample is admitted: instance normalization
    has sample-dependent statistics. This creates a POST-EXECUTION proof
    of the function obtained by freezing those measured statistics.
    """
    from monai.networks.blocks.dints_block import FactorizedReduceBlock
    if type(block) is not FactorizedReduceBlock:
        raise ValueError("unsupported model topology")
    if type(block.act) is not nn.ReLU:
        raise ValueError("requires exact ReLU activation, not an unsupported operator")
    if type(block.norm) is not nn.InstanceNorm3d:
        raise ValueError("requires exact InstanceNorm3d normalization")
    norm=block.norm
    if norm.track_running_stats or not norm.affine or not np.isfinite(norm.eps) or norm.eps<=0:
        raise ValueError("unsupported InstanceNorm running statistics, affine or epsilon")
    if input_tensor.ndim!=5 or input_tensor.shape[0]!=1 or not input_tensor.is_floating_point():
        raise ValueError("requires one floating NCDHW input sample")
    if not torch.isfinite(input_tensor).all():
        raise ValueError("input has nonfinite values")
    first,second=block.conv_1,block.conv_2
    count=int(input_tensor.shape[1])
    _check_conv_branch(first,count)
    _check_conv_branch(second,count)
    if any(n%2 for n in input_tensor.shape[-3:]):
        raise ValueError("factorized reduction expects even input dimensions")
    B=_data(uniform_input_world_hessian_bounds,count,"input Hessian bounds")
    ranges=np.asarray(input_value_intervals,dtype=np.float64)
    if ranges.shape!=(count,2) or not np.all(np.isfinite(ranges)):
        raise ValueError("input interval premise must be finite and one interval per channel")
    if np.any(ranges[:,0]>ranges[:,1]):
        raise ValueError("input interval lower bound exceeds upper bound")
    # Including an endpoint at zero is safe: ReLU(f)=f for f>=0
    # and ReLU(f)=0 for f<=0, even when f touches zero.
    if np.any((ranges[:,0]<0)&(ranges[:,1]>0)):
        raise ValueError("ReLU interval crosses zero; unproved kink: fail closed")
    active = ranges[:,0]>=0
    B_after_relu=np.where(active,B,0.0)
    if not torch.isfinite(norm.weight).all() or not torch.isfinite(norm.bias).all():
        raise ValueError("nonfinite InstanceNorm affine parameters")
    with torch.no_grad():
        activated=block.act(input_tensor.detach().clone())
        raw=torch.cat((
            first(activated),
            second(activated[:,:,1:,1:,1:]),
        ),dim=1)
        # Use the frozen sample statistics of the actual operator input.
        means=raw.mean(dim=(2,3,4))[0].double().cpu().numpy()
        variance=raw.var(dim=(2,3,4),unbiased=False)[0].double().cpu().numpy()
        if not np.all(np.isfinite(variance)) or np.any(variance<0):
            raise ValueError("invalid measured sample variance")
        gamma=norm.weight.detach().double().cpu().numpy()
        beta=norm.bias.detach().double().cpu().numpy()
        scale=gamma/np.sqrt(variance+float(norm.eps))
        offset=beta-scale*means
    w1=first.weight.detach().double().cpu().numpy().reshape(first.out_channels,count)
    w2=second.weight.detach().double().cpu().numpy().reshape(second.out_channels,count)
    W=np.concatenate((w1,w2),axis=0)
    conv_B=np.abs(W)@B_after_relu
    effective_B=np.abs(scale)*conv_B
    if not np.all(np.isfinite(effective_B)):
        raise ValueError("nonfinite derived normalized curvature bound")
    starts=(0,first.out_channels)
    fb=(float(effective_B[:starts[1]].max()),
        float(effective_B[starts[1]:].max()))
    provenance=sha256()
    provenance.update(b"monai-relu-instancenorm-conditional-C2-v1")
    for value in (B,ranges,W,means,variance,gamma,beta):
        arr=np.ascontiguousarray(value,dtype="<f8")
        provenance.update(str(arr.shape).encode("ascii"))
        provenance.update(arr.tobytes())
    provenance.update(str(float(norm.eps)).encode("ascii"))
    return ConditionalMONAICurvature(
        branch_output_bounds=tuple(float(v) for v in conv_B),
        normalized_output_bounds=tuple(float(v) for v in effective_B),
        fiber_bounds=fb, fiber_starts=starts,
        normalization_mean=tuple(float(v) for v in means),
        normalization_scale=tuple(float(v) for v in scale),
        normalization_offset=tuple(float(v) for v in offset),
        relu_active=tuple(bool(v) for v in active),
        premise_intervals=tuple((float(a),float(b)) for a,b in ranges),
        input_hessian_bounds=tuple(float(v) for v in B),
        weights_and_sample_sha256=provenance.hexdigest(),
    )


def certify_and_repair_default_monai(
    block: nn.Module,
    x: torch.Tensor,
    target: "FeatureLattice",
    *,
    input_voxel_to_patient_mm: np.ndarray,
    maximum_value_error: float,
    input_value_intervals: Sequence[tuple[float,float]],
    uniform_input_world_hessian_bounds: float | Sequence[float],
) -> tuple["RepairedChannelFibers",ConditionalMONAICurvature]:
    from .channel_fiber_runtime import repair_channel_fibers
    from .torch_fx_bundle import compile_fx_bundle_geometry
    proof=derive_default_monai_sample_conditional_curvature(
        block,x,input_value_intervals=input_value_intervals,
        uniform_input_world_hessian_bounds=uniform_input_world_hessian_bounds,
    )
    bundle=compile_fx_bundle_geometry(
        block,input_shape=tuple(x.shape[-3:]),input_channels=int(x.shape[1])
    ).output_bundle
    if tuple(f.channel_start for f in bundle.fibers)!=proof.fiber_starts:
        raise ValueError("runtime fiber layout differs from curvature derivation")
    # Compute the actual original module output. The frozen sample norm
    # statistics in the certificate are generated from its pre-norm tensor.
    out=block(x.detach().clone())
    repaired=repair_channel_fibers(
        out,bundle,target,role="feature",
        input_voxel_to_patient_mm=input_voxel_to_patient_mm,
        maximum_value_error=maximum_value_error,
        uniform_world_hessian_bound_by_channel_start=proof.by_fiber_start(),
    )
    return repaired,proof
