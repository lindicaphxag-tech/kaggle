"""Publish a falsifiable exact-arithmetic vs actual-FP32 execution witness."""
import json
from dataclasses import asdict
import numpy as np
import torch
from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.repair_synthesis import synthesize_lattice_repair
from research.medical_geometry.interpolation_certificate import certify_interpolation_error
from research.medical_geometry.executed_interpolation_witness import certify_executed_interpolation
from research.medical_geometry.torch_repair import LatticeRepair3d

def run():
    torch.set_num_threads(1)
    source=FeatureLattice((8,1,1),np.ones(3),np.zeros(3),np.ones(3))
    target=FeatureLattice((1,1,1),np.ones(3),np.array([.3,0.,0.]),np.ones(3))
    plan=synthesize_lattice_repair(source,target)
    x=torch.arange(8,dtype=torch.float32).reshape(1,1,8,1,1)*1000.
    analytic=certify_interpolation_error(
        source,target,plan,input_voxel_to_patient_mm=np.eye(4),
        maximum_allowed_error=0.,
        world_hessian_operator_norm_bound=0.)
    executed=LatticeRepair3d(plan)(x)
    witness=certify_executed_interpolation(
        x,source,target,plan,output=executed,
        analytic_interpolation_bound=analytic.max_absolute_error_bound)
    ideal=300.
    value=float(executed.item())
    assert analytic.authorized and analytic.max_absolute_error_bound==0.
    assert abs(value-ideal)>0.
    assert witness.max_exact_barycentric_residual>0.
    assert witness.conditional_combined_bound>0.
    result={
        "title":"real arithmetic is not the executed FP32 operator",
        "source_field":"f(x,y,z)=1000*x; exactly linear; Hessian=0",
        "target_source_index":[.3,0,0],
        "ideal_continuous_value_at_declared_target":ideal,
        "actual_torch_grid_sample_fp32_value":value,
        "observed_error_against_ideal_target":abs(value-ideal),
        "analytic_interpolation_error_bound_real_arithmetic":0.,
        "conditional_executed_numeric_error_bound_at_computed_coords":
            witness.max_exact_barycentric_residual,
        "conditional_analytic_plus_numeric_bound_at_computed_coords":
            witness.conditional_combined_bound,
        "witness":asdict(witness),
        "torch_version":torch.__version__,
        "is_clinical_data":False,
        "is_new_math_theorem":False,
        "is_external_validation":False,
        "warning":"Certificate excludes computing ideal patient-space index from affine; "
                  "source smoothness supplied as premise; witness covers the stored "
                  "float inputs and executed float64 target-to-source coordinates only",
    }
    return result

if __name__=="__main__":
    print(json.dumps(run(),indent=2,sort_keys=True))
