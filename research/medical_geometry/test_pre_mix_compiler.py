"""The only admissible 1x1 fusion order is typed repair, then mix."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.lattice_bundle import (
    bundle_from_lattice, factorized_reduce_bundle, dense_conv_mix_bundle,
)
from research.medical_geometry.pre_mix_compiler import physically_align_then_mix


def setup():
    source = FeatureLattice.input_grid((8,8,8))
    bundle = factorized_reduce_bundle(bundle_from_lattice(1,source),out_channels=2)
    target = FeatureLattice((3,3,3),np.full(3,2.0),np.full(3,1.5),np.ones(3))
    mix = nn.Conv3d(2,1,1,bias=False)
    with torch.no_grad(): mix.weight.fill_(1.0)
    x=torch.zeros(1,2,4,4,4)
    return x,bundle,target,mix


def test_irreversible_mixed_channel_cannot_get_a_fabricated_affine():
    x,bundle,target,mix=setup()
    mixed_bundle=dense_conv_mix_bundle(bundle,out_channels=1,kernel_size=1)
    assert len(mixed_bundle.fibers[0].supports)==2
    from research.medical_geometry.channel_fiber_runtime import repair_channel_fibers
    with pytest.raises(ValueError,match="multi-support"):
        repair_channel_fibers(mix(x),mixed_bundle,target)
    safe=physically_align_then_mix(x,bundle,target,mix)
    assert safe.tensor.shape==(1,1,3,3,3)
    assert len(safe.output_bundle.fibers[0].supports)==1


def test_refuses_non_pointwise_mixer_and_shape_mismatch():
    x,bundle,target,mix=setup()
    with pytest.raises(ValueError,match="pointwise dense"):
        physically_align_then_mix(x,bundle,target,nn.Conv3d(2,1,3,padding=1))
    with pytest.raises(ValueError,match="pointwise dense"):
        physically_align_then_mix(x,bundle,target,nn.Conv3d(2,2,1,groups=2))
    with pytest.raises(ValueError,match="channel count"):
        physically_align_then_mix(x[:,0:1],bundle,target,mix)


def test_output_value_error_budget_propagates_weighted_fiber_interpolation():
    x,bundle,target,mix=setup()
    with torch.no_grad(): mix.weight[0,0,0,0,0]=2.0
    with torch.no_grad(): mix.weight[0,1,0,0,0]=-3.0
    affine=np.diag([0.7,0.7,5.0,1.0])
    bounds={0:0.02,1:0.02}
    with pytest.raises(ValueError,match="exceeds output budget"):
        physically_align_then_mix(
            x,bundle,target,mix,input_voxel_to_patient_mm=affine,
            maximum_fiber_value_error=0.2,
            fiber_world_hessian_bounds=bounds,
            maximum_output_value_error=0.9
        )
    good=physically_align_then_mix(
        x,bundle,target,mix,input_voxel_to_patient_mm=affine,
        maximum_fiber_value_error=0.2,
        fiber_world_hessian_bounds=bounds,
        maximum_output_value_error=1.0
    )
    assert good.per_output_value_error_bounds==pytest.approx((0.97425,))
    assert len(good.conv_weight_provenance_sha256)==64
    with torch.no_grad():mix.weight[0,1,0,0,0]=-4.0
    changed=physically_align_then_mix(
        x,bundle,target,mix,input_voxel_to_patient_mm=affine,
        maximum_fiber_value_error=0.2,
        fiber_world_hessian_bounds=bounds,
        maximum_output_value_error=2.0
    )
    assert changed.conv_weight_provenance_sha256!=good.conv_weight_provenance_sha256
    assert changed.per_output_value_error_bounds==pytest.approx((1.1691,))


def test_requested_output_proof_refuses_missing_curvature():
    x,bundle,target,mix=setup()
    with pytest.raises(ValueError,match="per-fiber bound"):
        physically_align_then_mix(x,bundle,target,mix,maximum_output_value_error=1.0)
    base=physically_align_then_mix(x,bundle,target,mix)
    assert base.per_output_value_error_bounds is None
