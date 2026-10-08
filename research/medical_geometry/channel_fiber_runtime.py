"""Execute channel-fibered patient-space repairs before mixed-channel readout.

This operation only supports one point-valued lattice per channel fiber.
It deliberately refuses dense multi-support channels: interpolating them
as if they were samples of one field would make an unjustified claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .clingeom import FeatureLattice
from .lattice_bundle import LatticeBundle
from .interpolation_certificate import certify_interpolation_error
from .repair_synthesis import (
    RepairRole, synthesize_lattice_repair, verify_repair_plan,
)
from .torch_repair import LatticeRepair3d


@dataclass(frozen=True)
class RepairedChannelFibers:
    tensor: torch.Tensor
    target: FeatureLattice
    source_fiber_count: int
    plan_kinds: tuple[str, ...]
    plan_certificate_digests: tuple[str, ...]
    interpolation_fiber_count: int
    conditional_value_error_bounds: tuple[float, ...] | None = None


def repair_channel_fibers(
    x: torch.Tensor,
    bundle: LatticeBundle,
    target: FeatureLattice,
    *,
    role: RepairRole = "feature",
    input_voxel_to_patient_mm: np.ndarray | None = None,
    maximum_value_error: float | None = None,
    uniform_world_hessian_bound_by_channel_start: Mapping[int, float] | None = None,
) -> RepairedChannelFibers:
    """Rebind every *point-valued* channel fiber to one physical target grid.

    A failure cannot be repaired by inventing a missing source sample or
    by treating a dense/multi-support channel as one point-valued lattice.
    This is a runtime operation; model prediction equivalence is not implied.
    """
    if maximum_value_error is not None:
        if input_voxel_to_patient_mm is None:
            raise ValueError("value-error budget requires input patient affine")
        if uniform_world_hessian_bound_by_channel_start is None:
            raise ValueError("value-error budget requires per-fiber curvature evidence")
    if x.ndim != 5:
        raise ValueError("expected NCDHW tensor")
    if tuple(x.shape[2:]) != tuple(bundle.shape):
        raise ValueError("tensor spatial shape differs from channel-bundle contract")
    if int(x.shape[1]) != int(bundle.channels):
        raise ValueError("tensor channel count differs from channel-bundle contract")
    repaired = []
    kinds = []
    digests = []
    interpolation_count = 0
    conditional_bounds: list[float] = []
    for fiber in bundle.fibers:
        if len(fiber.supports) != 1:
            raise ValueError(
                "multi-support channel fiber is not a point-valued feature; "
                "cannot losslessly collapse it to one target affine"
            )
        support = fiber.supports[0]
        plan = synthesize_lattice_repair(support, target, role=role)
        if plan.kind == "refuse" or not verify_repair_plan(
            support, target, plan, role=role
        ):
            raise ValueError(
                f"channel fiber [{fiber.channel_start}:{fiber.channel_stop}] "
                "cannot be physically transported to the requested target"
            )
        if maximum_value_error is not None:
            curvature = uniform_world_hessian_bound_by_channel_start.get(
                fiber.channel_start
            )
            evidence = certify_interpolation_error(
                support, target, plan,
                input_voxel_to_patient_mm=input_voxel_to_patient_mm,
                maximum_allowed_error=maximum_value_error,
                world_hessian_operator_norm_bound=curvature,
                role=role,
            )
            if not evidence.authorized:
                raise ValueError(
                    f"channel fiber [{fiber.channel_start}:{fiber.channel_stop}] "
                    f"exceeds or lacks conditional value-error certificate: "
                    f"{evidence.reason}"
                )
            conditional_bounds.append(float(evidence.max_absolute_error_bound))
        part = x[:, fiber.channel_start:fiber.channel_stop]
        repaired.append(LatticeRepair3d(plan)(part))
        kinds.append(plan.kind)
        digests.append(plan.certificate_digest)
        interpolation_count += int(plan.requires_interpolation)
    if not repaired:
        raise ValueError("channel-bundle contract cannot be empty")
    output = torch.cat(repaired, dim=1)
    if tuple(output.shape[2:]) != tuple(target.shape):
        raise RuntimeError("runtime channel-fiber repair violated target shape")
    return RepairedChannelFibers(
        tensor=output, target=target,
        source_fiber_count=len(bundle.fibers),
        plan_kinds=tuple(kinds),
        plan_certificate_digests=tuple(digests),
        interpolation_fiber_count=interpolation_count,
        conditional_value_error_bounds=(
            tuple(conditional_bounds) if maximum_value_error is not None else None
        ),
    )
