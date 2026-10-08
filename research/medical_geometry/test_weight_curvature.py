"""Weight-derived curvature certificates on a real MONAI forward topology."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from monai.networks.blocks.dints_block import FactorizedReduceBlock

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.weight_curvature import (
    certify_and_repair_monai_factorized_reduce,
    derive_monai_factorized_reduce_curvature,
)


def _block():
    block = FactorizedReduceBlock(in_channel=1,out_channel=2,spatial_dims=3)
    block.act=nn.Identity()
    block.norm=nn.Identity()
    with torch.no_grad():
        block.conv_1.weight.fill_(2.0)
        block.conv_2.weight.fill_(-3.0)
    return block.eval()


def _target():
    return FeatureLattice(
        (3,3,3), np.full(3,2.0), np.full(3,1.5), np.ones(3)
    )


def test_provenance_derives_distinct_signed_weight_curvature_bounds():
    block = _block()
    cert = derive_monai_factorized_reduce_curvature(
        block, uniform_input_world_hessian_bounds=0.02
    )
    assert cert.per_output_channel_bounds == pytest.approx((0.04,0.06))
    assert cert.per_fiber_uniform_bounds == pytest.approx((0.04,0.06))
    assert cert.by_fiber_start() == pytest.approx({0:0.04, 1:0.06})
    assert len(cert.model_parameter_digest)==64
    with torch.no_grad():
        block.conv_2.weight.fill_(-4.0)
    revised=derive_monai_factorized_reduce_curvature(
        block, uniform_input_world_hessian_bounds=0.02
    )
    assert revised.per_fiber_uniform_bounds == pytest.approx((0.04,0.08))
    assert revised.model_parameter_digest != cert.model_parameter_digest


def test_unproven_nonlinearity_and_normalization_fail_closed():
    block = _block()
    block.act=nn.ReLU()
    with pytest.raises(ValueError,match="nonlinear activation"):
        derive_monai_factorized_reduce_curvature(
            block,uniform_input_world_hessian_bounds=0.02
        )
    block.act=nn.Identity()
    block.norm=nn.BatchNorm3d(2)
    with pytest.raises(ValueError,match="normalization"):
        derive_monai_factorized_reduce_curvature(
            block,uniform_input_world_hessian_bounds=0.02
        )


def test_unsupported_model_structure_rejects_unproven_C2_law():
    block = _block()
    with pytest.raises(ValueError,match="nonnegative"):
        derive_monai_factorized_reduce_curvature(
            block,uniform_input_world_hessian_bounds=-0.2
        )
    with pytest.raises(ValueError,match="finite"):
        derive_monai_factorized_reduce_curvature(
            block,uniform_input_world_hessian_bounds=float("nan")
        )
    with pytest.raises(ValueError,match="model topology"):
        derive_monai_factorized_reduce_curvature(
            nn.Identity(),uniform_input_world_hessian_bounds=0.02
        )


def test_weight_derived_error_budget_controls_actual_real_monai_forward():
    block = _block()
    spacing=(0.7,0.7,5.0)
    d,h,w = torch.meshgrid(
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32), indexing="ij"
    )
    B=0.02
    f=0.5*B*((spacing[0]*d)**2 + (spacing[1]*h)**2 + (spacing[2]*w)**2)
    x=f[None,None]
    affine=np.diag([*spacing,1.0])

    with pytest.raises(ValueError,match="exceeds or lacks"):
        certify_and_repair_monai_factorized_reduce(
            block,x,_target(), input_voxel_to_patient_mm=affine,
            maximum_value_error=0.58,uniform_input_world_hessian_bounds=B
        )
    repaired,proof = certify_and_repair_monai_factorized_reduce(
        block,x,_target(),input_voxel_to_patient_mm=affine,
        maximum_value_error=0.60,uniform_input_world_hessian_bounds=B
    )
    assert proof.by_fiber_start()==pytest.approx({0:0.04,1:0.06})
    assert repaired.conditional_value_error_bounds==pytest.approx((0.3897,0.58455))
    t0,t1,t2=torch.meshgrid(
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,indexing="ij"
    )
    target_truth=0.5*B*(
        (spacing[0]*t0)**2 + (spacing[1]*t1)**2 + (spacing[2]*t2)**2
    )
    expected=torch.cat([2*target_truth[None,None],-3*target_truth[None,None]],dim=1)
    errors=(repaired.tensor-expected).abs()
    assert float(errors[:,0].detach().max())==pytest.approx(0.3897,abs=2e-5)
    assert float(errors[:,1].detach().max())==pytest.approx(0.58455,abs=2e-5)
    assert float(errors.detach().max())<=max(repaired.conditional_value_error_bounds)+2e-5


def test_no_curvature_proof_from_nonlinear_active_monai_block():
    block=FactorizedReduceBlock(in_channel=1,out_channel=2,spatial_dims=3)
    assert not (type(block.act) is nn.Identity and type(block.norm) is nn.Identity)
    with pytest.raises(ValueError,match="nonlinear activation or normalization"):
        derive_monai_factorized_reduce_curvature(
            block,uniform_input_world_hessian_bounds=0.02
        )
