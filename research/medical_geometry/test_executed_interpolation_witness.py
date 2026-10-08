"""Falsification-first tests: ideal analytical error versus executed FP32."""
import numpy as np
import pytest
import torch

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.repair_synthesis import synthesize_lattice_repair
from research.medical_geometry.interpolation_certificate import certify_interpolation_error
from research.medical_geometry.executed_interpolation_witness import certify_executed_interpolation
from research.medical_geometry.torch_repair import LatticeRepair3d

def case(s=0.3):
    source=FeatureLattice((8,1,1),np.ones(3),np.zeros(3),np.ones(3))
    target=FeatureLattice((1,1,1),np.ones(3),np.array([s,0,0]),np.ones(3))
    plan=synthesize_lattice_repair(source,target)
    x=torch.arange(8,dtype=torch.float32).reshape(1,1,8,1,1)*1000
    return x,source,target,plan

def test_linear_field_zero_hessian_does_not_bound_float32_rounding_error():
    x,source,target,plan=case()
    analytic=certify_interpolation_error(
        source,target,plan,input_voxel_to_patient_mm=np.eye(4),
        maximum_allowed_error=0.0,world_hessian_operator_norm_bound=0.0)
    assert analytic.authorized and analytic.max_absolute_error_bound==0
    native=LatticeRepair3d(plan)(x)
    assert abs(native.item()-300.0)>1e-6
    witness=certify_executed_interpolation(
        x,source,target,plan,output=native,analytic_interpolation_bound=0.0)
    assert witness.max_exact_barycentric_residual>0
    assert witness.conditional_combined_bound>0
    assert witness.audited_elements==1

def test_exact_half_cell_can_have_zero_executed_roundoff():
    x,source,target,plan=case(0.5)
    witness=certify_executed_interpolation(
        x,source,target,plan,analytic_interpolation_bound=0.)
    assert witness.max_exact_barycentric_residual==0
    assert witness.conditional_combined_bound==0

def test_tampered_output_fails_closed_under_exact_execution_replay():
    x,source,target,plan=case()
    invalid=LatticeRepair3d(plan)(x)+1.0
    with pytest.raises(ValueError,match="execution replay"):
        certify_executed_interpolation(x,source,target,plan,output=invalid)

def test_source_fingerprint_changes_when_unsampled_region_changes():
    x,source,target,plan=case()
    first=certify_executed_interpolation(x,source,target,plan)
    altered=x.clone();altered[0,0,7,0,0]+=1
    second=certify_executed_interpolation(altered,source,target,plan)
    assert first.source_sha256!=second.source_sha256
    assert first.output_sha256==second.output_sha256

def test_actual_three_axis_fp32_interpolation_has_finite_exact_residual():
    source=FeatureLattice((5,5,5),np.ones(3),np.zeros(3),np.ones(3))
    target=FeatureLattice((3,3,3),np.ones(3),np.full(3,.3),np.ones(3))
    plan=synthesize_lattice_repair(source,target)
    torch.manual_seed(8)
    x=torch.randn(1,2,5,5,5,dtype=torch.float32)
    result=certify_executed_interpolation(x,source,target,plan,max_elements=54)
    assert result.audited_elements==54
    assert 0<=result.max_exact_barycentric_residual<1e-4
    assert len(result.output_sha256)==64
    with pytest.raises(ValueError,match="proof budget"):
        certify_executed_interpolation(x,source,target,plan,max_elements=1)

def test_invalid_bound_or_nonfinite_source_fails_closed():
    x,source,target,plan=case()
    with pytest.raises(ValueError,match="invalid analytic"):
        certify_executed_interpolation(
            x,source,target,plan,analytic_interpolation_bound=-1.)
    with pytest.raises(ValueError,match="finite CPU"):
        certify_executed_interpolation(x*float("nan"),source,target,plan)
