"""Public controlled MONAI proof: actual weights derive nonlinear value bounds."""

import json
import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks.dints_block import FactorizedReduceBlock

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.weight_curvature import (
    certify_and_repair_monai_factorized_reduce,
)

def run():
    B=0.02
    spacing=[0.7,0.7,5.0]
    patient_affine=np.diag([*spacing,1.0])
    block=FactorizedReduceBlock(in_channel=1,out_channel=2,spatial_dims=3)
    block.act=nn.Identity()
    block.norm=nn.Identity()
    with torch.no_grad():
        block.conv_1.weight.fill_(2.0)
        block.conv_2.weight.fill_(-3.0)
    block.eval()
    d,h,w=torch.meshgrid(
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),indexing="ij",
    )
    src=0.5*B*((spacing[0]*d)**2+(spacing[1]*h)**2+(spacing[2]*w)**2)
    target=FeatureLattice((3,3,3),np.full(3,2.0),np.full(3,1.5),np.ones(3))
    with torch.no_grad():
        res, proof = certify_and_repair_monai_factorized_reduce(
            block, src[None,None], target,
            input_voxel_to_patient_mm=patient_affine,
            maximum_value_error=0.60,
            uniform_input_world_hessian_bounds=B,
        )
    a,b,c=torch.meshgrid(
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,indexing="ij",
    )
    ideal=0.5*B*((spacing[0]*a)**2+(spacing[1]*b)**2+(spacing[2]*c)**2)
    ideal=torch.cat([2.0*ideal[None,None],-3.0*ideal[None,None]],dim=1)
    errors=(res.tensor-ideal).abs()
    naive=(block(src[None,None])[:,:,:3,:3,:3]-ideal).abs()
    report={
        "external_runtime":"pinned MONAI DiNTS FactorizedReduceBlock",
        "field":"known C2 quadratic world field, not learned medical features",
        "uniform_input_hessian_bound":B,
        "frozen_conv_branch_weights":[2.0,-3.0],
        "derived_output_world_hessian_bounds":list(proof.per_output_channel_bounds),
        "per_fiber_max_value_error_bounds":list(res.conditional_value_error_bounds),
        "observed_per_fiber_max_absolute_errors":[
            float(errors[:,i:i+1].detach().max()) for i in range(2)
        ],
        "naive_common_index_mean_absolute_error":float(naive.detach().mean()),
        "physically_repaired_mean_absolute_error":float(errors.detach().mean()),
        "physical_spacing_mm":spacing,
        "weight_provenance_sha256":proof.model_parameter_digest,
        "model_has_identity_activation_and_norm":True,
        "pretrained_checkpoint":False,
        "clinical_efficacy":False,
    }
    bounds=res.conditional_value_error_bounds
    if bounds is None:
        raise AssertionError("missing weight-derived runtime authorization")
    if any(float(errors[:,i:i+1].max()) > bounds[i]+2e-5 for i in range(2)):
        raise AssertionError("executed MONAI interpolation exceeded certified bound")
    if not float(naive.mean()) > float(errors.mean()):
        raise AssertionError("typed coordinate repair did not improve analytical error")
    return report

if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True,indent=2))
