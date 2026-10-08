"""Conditional patient-space Hessian bounds derived from actual model weights.

For a channelwise C² input field with externally justified world-Hessian
operator-norm upper bounds B_c, a 1x1 Conv3d output feature
  y_o(p) = sum_c W[o,c] * f_c(p) + bias_o
has a valid bound B_o <= sum_c abs(W[o,c]) * B_c.

Spatial stride/slice alter sampling phase, NOT this continuous-field bound.
This narrow compiler only accepts the audited MONAI DiNTS factorized reduce
topology with identity activation/normalization and 1x1 Conv3d branches.
Any unproved nonlinear stage or operator fails closed; it does not infer
learned-feature input curvature from voxel values.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class WeightDerivedCurvatureCertificate:
    per_output_channel_bounds: tuple[float, ...]
    fiber_channel_starts: tuple[int, int]
    per_fiber_uniform_bounds: tuple[float, float]
    source_input_uniform_bounds: tuple[float, ...]
    model_parameter_digest: str
    derivation: str

    def by_fiber_start(self) -> dict[int, float]:
        return dict(zip(self.fiber_channel_starts, self.per_fiber_uniform_bounds))


def _check_conv_branch(conv: nn.Module, input_channels: int) -> nn.Conv3d:
    if type(conv) is not nn.Conv3d:
        raise ValueError("curvature propagation requires exact Conv3d branch")
    if conv.in_channels != input_channels or conv.groups != 1:
        raise ValueError("unsupported grouped or mismatched convolution channels")
    if (tuple(conv.kernel_size) != (1, 1, 1)
            or tuple(conv.stride) != (2, 2, 2)
            or tuple(conv.padding) != (0, 0, 0)
            or tuple(conv.dilation) != (1, 1, 1)):
        raise ValueError("curvature derivation only supports unpadded 1x1 stride-2 branches")
    if not torch.isfinite(conv.weight).all():
        raise ValueError("non-finite branch weights cannot support a curvature bound")
    if conv.bias is not None and not torch.isfinite(conv.bias).all():
        raise ValueError("non-finite branch bias invalidates feature provenance")
    return conv


def derive_monai_factorized_reduce_curvature(
    block: nn.Module,
    *,
    uniform_input_world_hessian_bounds: float | Sequence[float],
) -> WeightDerivedCurvatureCertificate:
    """Derive, rather than guess, per-fiber curvature bounds from weights.

    The caller must establish uniform C² bounds for the CONTINUOUS input
    fields over every affected source cell. This routine proves the bound
    transfer through the admitted frozen linear branches; it is not a proof
    of that input premise or of clinical prediction accuracy.
    """
    # Import inside the function so NumPy-only ClinGeom tools are not forced
    # to depend on MONAI. Require the exact audited class, not a subclass
    # with an arbitrary overridden forward method.
    from monai.networks.blocks.dints_block import FactorizedReduceBlock
    if type(block) is not FactorizedReduceBlock:
        raise ValueError("unsupported model topology: not exact MONAI FactorizedReduceBlock")
    if type(block.act) is not nn.Identity or type(block.norm) is not nn.Identity:
        raise ValueError("cannot propagate C2 curvature through unverified nonlinear activation or normalization")
    first = getattr(block, "conv_1", None)
    second = getattr(block, "conv_2", None)
    if type(first) is not nn.Conv3d or type(second) is not nn.Conv3d:
        raise ValueError("MONAI curvature proof requires two explicit Conv3d branches")
    n_input = first.in_channels
    _check_conv_branch(first, n_input)
    _check_conv_branch(second, n_input)
    if isinstance(uniform_input_world_hessian_bounds, (int,float,np.number)):
        bounds = np.full(n_input, float(uniform_input_world_hessian_bounds), dtype=np.float64)
    else:
        bounds = np.asarray(uniform_input_world_hessian_bounds, dtype=np.float64)
    if bounds.shape != (n_input,) or not np.all(np.isfinite(bounds)) or np.any(bounds < 0):
        raise ValueError("input C2 curvature bounds must be finite, nonnegative, one per input channel")

    weights1 = first.weight.detach().cpu().numpy().astype(np.float64,copy=True).reshape(first.out_channels,n_input)
    weights2 = second.weight.detach().cpu().numpy().astype(np.float64,copy=True).reshape(second.out_channels,n_input)
    values1 = np.abs(weights1) @ bounds
    values2 = np.abs(weights2) @ bounds
    per_output = tuple(float(v) for v in np.concatenate([values1,values2]))
    starts = (0, first.out_channels)
    fiber_bounds = (float(np.max(values1)),float(np.max(values2)))

    digest = sha256()
    digest.update(b"monai-factorized-reduce-C2-bound-v1")
    digest.update(np.asarray(bounds,dtype="<f8").tobytes())
    for conv in (first,second):
        for name,tensor in (("weight",conv.weight),("bias",conv.bias)):
            digest.update(name.encode("ascii"))
            if tensor is None:
                digest.update(b"NONE")
            else:
                array = tensor.detach().cpu().contiguous().numpy()
                digest.update(str(tuple(array.shape)).encode("ascii"))
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(array.tobytes())

    return WeightDerivedCurvatureCertificate(
        per_output_channel_bounds=per_output,
        fiber_channel_starts=starts,
        per_fiber_uniform_bounds=fiber_bounds,
        source_input_uniform_bounds=tuple(float(v) for v in bounds),
        model_parameter_digest=digest.hexdigest(),
        derivation=("C2 input bounds transported through two audited, unpadded "
                    "1x1 stride-2 linear Conv3d branches; "
                    "B_out[o] <= sum_c abs(W[o,c]) B_in[c]; identity act/norm"),
    )


def certify_and_repair_monai_factorized_reduce(
    block: nn.Module,
    input_tensor: torch.Tensor,
    target: "FeatureLattice",
    *,
    input_voxel_to_patient_mm: np.ndarray,
    maximum_value_error: float,
    uniform_input_world_hessian_bounds: float | Sequence[float],
) -> tuple["RepairedChannelFibers", WeightDerivedCurvatureCertificate]:
    """Run frozen model and transport channel fibers using weight-bound evidence.

    The block is executed only after the model/topology/weights are checked;
    the bound is refreshed for the exact current weights on every call.
    """
    from .channel_fiber_runtime import RepairedChannelFibers, repair_channel_fibers
    from .clingeom import FeatureLattice
    from .torch_fx_bundle import compile_fx_bundle_geometry
    if input_tensor.ndim != 5 or not torch.is_floating_point(input_tensor):
        raise ValueError("expected floating NCDHW model input")
    evidence = derive_monai_factorized_reduce_curvature(
        block, uniform_input_world_hessian_bounds=uniform_input_world_hessian_bounds,
    )
    geometry = compile_fx_bundle_geometry(
        block, input_shape=tuple(input_tensor.shape[-3:]),
        input_channels=int(input_tensor.shape[1]),
    ).output_bundle
    if len(geometry.fibers) != 2 or tuple(f.channel_start for f in geometry.fibers) != evidence.fiber_channel_starts:
        raise ValueError("traced channel-fiber geometry differs from derivation assumptions")
    output = block(input_tensor)
    repaired = repair_channel_fibers(
        output, geometry, target,
        input_voxel_to_patient_mm=input_voxel_to_patient_mm,
        maximum_value_error=maximum_value_error,
        uniform_world_hessian_bound_by_channel_start=evidence.by_fiber_start(),
    )
    return repaired,evidence

