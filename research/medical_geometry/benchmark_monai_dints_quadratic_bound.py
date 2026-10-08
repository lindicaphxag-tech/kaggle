"""Pinned MONAI DiNTS: sharp non-linear interpolation error + fail-closed execution.

Unlike linear-phase demo, this quadratic patient-space field has non-zero
interpolation residual. Every channel-fiber resampling must independently
meet a **declared** physical Hessian envelope and output-value error budget.
The Hessian here is analytically known from the synthetic input function;
no theorem about learned neural activations is inferred.
"""

from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks.dints_block import FactorizedReduceBlock

from research.medical_geometry.channel_fiber_runtime import repair_channel_fibers
from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.torch_fx_bundle import compile_fx_bundle_geometry


def run() -> dict:
    spacing=(0.7,0.7,5.0)
    hessian=0.02
    budget=0.10
    patient_affine=np.diag([*spacing,1.0])
    block=FactorizedReduceBlock(in_channel=1,out_channel=2,spatial_dims=3)
    block.act=nn.Identity()
    block.norm=nn.Identity()
    with torch.no_grad():
        block.conv_1.weight.fill_(1.0)
        block.conv_2.weight.fill_(1.0)
    block.eval()
    d,h,w=torch.meshgrid(
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),indexing="ij"
    )
    # f(p)=(B/2)||p||^2 with uniform Hessian B*I in physical mm.
    f=0.5*hessian*(
        (spacing[0]*d)**2+(spacing[1]*h)**2+(spacing[2]*w)**2
    )
    with torch.no_grad():
        runtime=block(f[None,None])
    typed=compile_fx_bundle_geometry(
        block,input_shape=(8,8,8),input_channels=1
    ).output_bundle
    target=FeatureLattice(
        (3,3,3),np.full(3,2.0),np.full(3,1.5),np.ones(3)
    )
    bound_map={fiber.channel_start:hessian for fiber in typed.fibers}
    guarded=repair_channel_fibers(
        runtime,typed,target,
        input_voxel_to_patient_mm=patient_affine,
        maximum_value_error=budget,
        uniform_world_hessian_bound_by_channel_start=bound_map,
    )
    t0,t1,t2=torch.meshgrid(
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,indexing="ij"
    )
    expected=0.5*hessian*(
        (spacing[0]*t0)**2+(spacing[1]*t1)**2+(spacing[2]*t2)**2
    )
    target_truth=expected[None,None]
    actual_error=(guarded.tensor-target_truth).abs()
    naive_error=(runtime[:,:,:3,:3,:3]-target_truth).abs()
    certified=guarded.conditional_value_error_bounds
    assert certified is not None and len(certified)==2
    assert all(abs(b-0.097425)<2e-9 for b in certified)
    assert float(actual_error.max()) <= max(certified)+2e-5
    assert float(actual_error.max()) >= max(certified)-2e-5
    assert float(naive_error.mean()) > float(actual_error.mean())
    result={
        "runtime":"pinned MONAI DiNTS FactorizedReduceBlock",
        "field":"analytic quadratic patient-space scalar f(p)=B/2 ||p||^2",
        "known_uniform_world_hessian_norm":hessian,
        "patient_mm_spacing":list(spacing),
        "number_of_fibers":len(typed.fibers),
        "declared_value_error_budget":budget,
        "computed_value_error_bounds":list(certified),
        "naive_common_index_mean_absolute_error":float(naive_error.mean()),
        "physically_repaired_mean_absolute_error":float(actual_error.mean()),
        "physically_repaired_max_absolute_error":float(actual_error.max()),
        "curvature_evidence_source":"analytic synthetic function, NOT measured learned-feature curvature",
        "pretrained_checkpoint":False,
        "clinical_performance_claim":False
    }
    return result


if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True,indent=2))
