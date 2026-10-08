"""Real default MONAI ReLU + InstanceNorm 3-D runtime; conditional C2 proof."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from monai.networks.blocks.dints_block import FactorizedReduceBlock

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.default_monai_curvature import (
    derive_default_monai_sample_conditional_curvature,
    certify_and_repair_default_monai,
)

B=0.02
SPACING=(0.7,0.7,5.0)

def model():
    block=FactorizedReduceBlock(in_channel=1,out_channel=2,spatial_dims=3)
    assert type(block.act) is nn.ReLU
    assert type(block.norm) is nn.InstanceNorm3d
    assert not block.norm.track_running_stats
    with torch.no_grad():
        block.conv_1.weight.fill_(2.0)
        block.conv_2.weight.fill_(-3.0)
        block.norm.weight.fill_(1.0)
        block.norm.bias.zero_()
    return block.eval()

def input_field(sign=1.0):
    d,h,w=torch.meshgrid(
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),indexing="ij",
    )
    f=1.0+0.5*B*((SPACING[0]*d)**2+(SPACING[1]*h)**2+(SPACING[2]*w)**2)
    return sign*f[None,None]

def target():
    return FeatureLattice(
        (3,3,3),np.full(3,2.0),np.full(3,1.5),np.ones(3)
    )

def validate_actual():
    x=input_field()
    mod=model()
    repaired, proof=certify_and_repair_default_monai(
        mod,x,target(),input_voxel_to_patient_mm=np.diag([*SPACING,1.0]),
        maximum_value_error=2.0,input_value_intervals=[(1.0,14.0)],
        uniform_input_world_hessian_bounds=B,
    )
    t0,t1,t2=torch.meshgrid(
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,indexing="ij",
    )
    f=1.0+0.5*B*((SPACING[0]*t0)**2+(SPACING[1]*t1)**2+(SPACING[2]*t2)**2)
    ideal=torch.stack([
        (proof.normalization_scale[0]*2.0*f+proof.normalization_offset[0]),
        (proof.normalization_scale[1]*(-3.0)*f+proof.normalization_offset[1]),
    ],dim=0)[None]
    error=(repaired.tensor-ideal).abs()
    return mod,x,proof,repaired,error

def test_default_monai_actual_norm_sample_is_certified_under_positive_interval():
    mod,x,proof,repaired,error=validate_actual()
    assert proof.relu_active == (True,)
    assert proof.branch_output_bounds == pytest.approx((0.04,0.06))
    assert len(proof.weights_and_sample_sha256)==64
    assert repaired.conditional_value_error_bounds is not None
    assert error.max().item() <= max(repaired.conditional_value_error_bounds)+2e-4
    # Quadratic field with same-sign linear scaling attains the bound.
    for i,bound in enumerate(repaired.conditional_value_error_bounds):
        assert error[:,i:i+1].max().item()==pytest.approx(bound,abs=2e-4)

def test_kinked_relu_interval_fails_closed_even_when_samples_are_all_positive():
    x=input_field()
    with pytest.raises(ValueError,match="crosses zero"):
        derive_default_monai_sample_conditional_curvature(
            model(),x,input_value_intervals=[(-1.0,14.0)],
            uniform_input_world_hessian_bounds=B,
        )

def test_proven_negative_interval_yields_exact_zero_curvature():
    x=input_field(-1.0)
    mod=model()
    proof=derive_default_monai_sample_conditional_curvature(
        mod,x,input_value_intervals=[(-14.0,-1.0)],
        uniform_input_world_hessian_bounds=B
    )
    assert proof.relu_active==(False,)
    assert proof.normalized_output_bounds==pytest.approx((0.0,0.0))
    repaired,_=certify_and_repair_default_monai(
        mod,x,target(),input_voxel_to_patient_mm=np.diag([*SPACING,1]),
        maximum_value_error=0.0,input_value_intervals=[(-14.0,-1.0)],
        uniform_input_world_hessian_bounds=B,
    )
    assert repaired.conditional_value_error_bounds==pytest.approx((0.0,0.0))

def test_changed_sample_stats_and_norm_gamma_change_certificate_digest():
    mod=model()
    p=derive_default_monai_sample_conditional_curvature(
        mod,input_field(),input_value_intervals=[(1.0,14.0)],
        uniform_input_world_hessian_bounds=B,
    )
    with torch.no_grad():
        mod.norm.weight[0]=2.0
    q=derive_default_monai_sample_conditional_curvature(
        mod,input_field(),input_value_intervals=[(1.0,14.0)],
        uniform_input_world_hessian_bounds=B,
    )
    assert q.normalized_output_bounds[0]==pytest.approx(2*p.normalized_output_bounds[0])
    assert q.weights_and_sample_sha256!=p.weights_and_sample_sha256

def test_running_stats_and_batch_conditioning_are_refused():
    mod=model()
    mod.norm=nn.InstanceNorm3d(2,affine=True,track_running_stats=True)
    with pytest.raises(ValueError,match="running statistics"):
        derive_default_monai_sample_conditional_curvature(
            mod,input_field(),input_value_intervals=[(1,14)],
            uniform_input_world_hessian_bounds=B,
        )
    mod=model()
    with pytest.raises(ValueError,match="one floating NCDHW"):
        derive_default_monai_sample_conditional_curvature(
            mod,input_field().expand(2,-1,-1,-1,-1),
            input_value_intervals=[(1,14)],uniform_input_world_hessian_bounds=B,
        )

def test_unsupported_activation_is_refused_not_silently_approximated():
    mod=model()
    mod.act=nn.SiLU()
    with pytest.raises(ValueError,match="requires exact ReLU"):
        derive_default_monai_sample_conditional_curvature(
            mod,input_field(),input_value_intervals=[(1,14)],
            uniform_input_world_hessian_bounds=B,
        )

def test_default_instance_norm_refused_as_structurally_local_but_conditional_transport_admitted():
    from research.medical_geometry.torch_fx_bundle import compile_fx_bundle_geometry
    from research.medical_geometry.torch_geometry import UnsupportedGeometryOp

    block=model()
    x=input_field()
    # The live default MONAI network depends on volume-wide InstanceNorm
    # statistics, including far-away voxels; no local-support proof is valid.
    with pytest.raises(UnsupportedGeometryOp, match="spatial statistics"):
        compile_fx_bundle_geometry(
            block,input_shape=tuple(x.shape[-3:]),input_channels=int(x.shape[1])
        )

    # But once this exact input has been executed, its normalization parameters
    # are frozen constants for the conditional continuous output field. The
    # specialized sample-conditioned C2 transport remains admissible, and
    # must never be described as a structural locality certificate.
    _,_,proof,repaired,error=validate_actual()
    assert proof.weights_and_sample_sha256
    assert max(float(v) for v in error.flatten()) <= (
        max(repaired.conditional_value_error_bounds) + 2e-4
    )
