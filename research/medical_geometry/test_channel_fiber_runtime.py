"""Patient-space channel-fiber repair must change actual sampled values."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.channel_fiber_runtime import repair_channel_fibers
from research.medical_geometry.lattice_bundle import (
    bundle_from_lattice, factorized_reduce_bundle, dense_conv_mix_bundle,
)


def _reduce_bundle():
    return factorized_reduce_bundle(
        bundle_from_lattice(1, FeatureLattice.input_grid((8, 8, 8))),
        out_channels=2,
    )


def _target():
    return FeatureLattice(
        (3,3,3), np.full(3,2.0), np.full(3,1.5), np.ones(3)
    )


def test_runtime_fiber_repair_has_no_free_extrapolation():
    bundle=_reduce_bundle()
    x=torch.zeros(1,2,4,4,4)
    target=FeatureLattice(
        (4,4,4), np.full(3,2.0), np.full(3,0.5), np.ones(3)
    )
    with pytest.raises(ValueError, match="cannot be physically transported"):
        repair_channel_fibers(x,bundle,target)


def test_multi_support_channel_is_not_silently_projected_to_single_affine():
    source = _reduce_bundle()
    mixed = dense_conv_mix_bundle(source,out_channels=3,kernel_size=1)
    assert len(mixed.fibers[0].supports) == 2
    with pytest.raises(ValueError,match="multi-support"):
        repair_channel_fibers(torch.zeros(1,3,4,4,4),mixed,_target())


def test_runtime_checks_channel_shape_not_only_spatial_shape():
    bundle=_reduce_bundle()
    with pytest.raises(ValueError,match="channel count"):
        repair_channel_fibers(torch.zeros(1,3,4,4,4),bundle,_target())


def test_runtime_repairs_two_point_value_phases_of_linear_mm_field():
    # Construct a deterministic input patient-space coordinate field. This
    # validates physical sample transport, not a pretrained model accuracy.
    spacing=np.array([0.7,0.7,5.0])
    d,h,w=torch.meshgrid(
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),indexing="ij",
    )
    source_field=spacing[0]*d+spacing[1]*h+spacing[2]*w
    x=source_field[None,None]
    phase0=x[:,:,0::2,0::2,0::2]
    phase1=x[:,:,1::2,1::2,1::2]
    y=torch.cat((phase0,phase1),dim=1)
    repaired=repair_channel_fibers(y,_reduce_bundle(),_target())
    assert repaired.plan_kinds==("resample","resample")
    assert repaired.interpolation_fiber_count==2
    t0,t1,t2=torch.meshgrid(
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,indexing="ij",
    )
    expected=(spacing[0]*t0+spacing[1]*t1+spacing[2]*t2)[None,None]
    naive=y[:,:,:3,:3,:3]
    repaired_error=(repaired.tensor-expected).abs().max().item()
    naive_error=(naive-expected).abs().mean().item()
    assert naive_error > 3.0
    assert repaired_error < 1e-4


def test_curvature_budget_fails_closed_without_external_bound():
    bundle=_reduce_bundle()
    x=torch.zeros(1,2,4,4,4)
    with pytest.raises(ValueError, match='per-fiber curvature evidence'):
        repair_channel_fibers(
            x, bundle, _target(), maximum_value_error=1.0,
            input_voxel_to_patient_mm=np.eye(4),
        )
    with pytest.raises(ValueError, match='lacks conditional value-error certificate'):
        repair_channel_fibers(
            x, bundle, _target(), maximum_value_error=1.0,
            input_voxel_to_patient_mm=np.eye(4),
            uniform_world_hessian_bound_by_channel_start={0:0.1},
        )


def test_curvature_budget_limits_actual_two_fiber_transport():
    bundle=_reduce_bundle()
    x=torch.zeros(1,2,4,4,4)
    patient_affine=np.diag([0.7,0.7,5.0,1.0])
    # Each MONAI factorized branch has a 2-index-step source cell.
    # Target at 1.5,3.5,5.5 is at 1/4 or 3/4 of each 2-index cell.
    # For B=0.02, sharp maximum is 0.19485 value units.
    with pytest.raises(ValueError, match='exceeds or lacks'):
        repair_channel_fibers(
            x,bundle,_target(),
            input_voxel_to_patient_mm=patient_affine,
            maximum_value_error=0.19,
            uniform_world_hessian_bound_by_channel_start={0:0.02,1:0.02},
        )
    result=repair_channel_fibers(
        x,bundle,_target(),
        input_voxel_to_patient_mm=patient_affine,
        maximum_value_error=0.20,
        uniform_world_hessian_bound_by_channel_start={0:0.02,1:0.02},
    )
    assert result.conditional_value_error_bounds == pytest.approx((0.19485,0.19485))
    assert result.interpolation_fiber_count == 2


def test_legacy_geometry_only_path_never_claims_value_error_certificate():
    bundle=_reduce_bundle()
    result=repair_channel_fibers(torch.zeros(1,2,4,4,4),bundle,_target())
    assert result.conditional_value_error_bounds is None
