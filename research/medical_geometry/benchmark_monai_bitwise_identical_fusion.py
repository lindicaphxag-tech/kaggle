"""Bitwise-identical post-fusion MONAI tensors with different physical truths.

The earlier 0.7 mm witness was exactly equal in real arithmetic but
differed by ~5e-7 when evaluated in float32. Here a dyadic 0.5 mm interval
makes the collision itself BITWISE EXACT in the executed PyTorch tensors.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks.dints_block import FactorizedReduceBlock

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.torch_fx_bundle import compile_fx_bundle_geometry
from research.medical_geometry.pre_mix_compiler import physically_align_then_mix


def run():
    s=0.5  # exactly representable in binary32
    d,h,w=torch.meshgrid(
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),indexing="ij"
    )
    zero=torch.zeros(1,2,8,8,8,dtype=torch.float32)
    counter=torch.stack((s*d,s-s*d),dim=0).unsqueeze(0)
    block=FactorizedReduceBlock(in_channel=2,out_channel=2,spatial_dims=3)
    block.act=nn.Identity()
    block.norm=nn.Identity()
    with torch.no_grad():
        block.conv_1.weight.zero_()
        block.conv_2.weight.zero_()
        block.conv_1.weight[0,0,0,0,0]=1.
        block.conv_2.weight[0,1,0,0,0]=1.
    block.eval()
    mix=nn.Conv3d(2,1,1,bias=False)
    with torch.no_grad():
        mix.weight.fill_(1.)
        beforeA=block(zero)
        beforeB=block(counter)
        afterA=mix(beforeA)
        afterB=mix(beforeB)
    # Bit-level reinterpretation also rejects +0 versus -0, which plain
    # float equality considers equal. Prove byte-identical IEEE-754 output.
    bitwise_equal = torch.equal(afterA.view(torch.int32), afterB.view(torch.int32))
    assert bitwise_equal, "binary32 adversarial collision not bitwise exact"
    assert torch.count_nonzero(afterB)==0
    bundle=compile_fx_bundle_geometry(
        block,input_shape=(8,8,8),input_channels=2
    ).output_bundle
    target=FeatureLattice((3,3,3),np.full(3,2.0),np.full(3,1.5),np.ones(3))
    with torch.no_grad():
        correctedA=physically_align_then_mix(beforeA,bundle,target,mix)
        correctedB=physically_align_then_mix(beforeB,bundle,target,mix)
    expected0=torch.zeros_like(correctedA.tensor)
    expected1=torch.full_like(correctedB.tensor,s)
    err0=float((correctedA.tensor-expected0).abs().max())
    err1=float((correctedB.tensor-expected1).abs().max())
    assert err0 < 1e-6 and err1 < 1e-6
    assert len(correctedB.output_bundle.fibers)==1
    return {
        "external_runtime":"pinned MONAI FactorizedReduceBlock + PyTorch dense 1x1 Conv3d",
        "tested_dtype":"torch.float32",
        "patient_spacing_mm":s,
        "post_fusion_tensors_bitwise_equal":bool(bitwise_equal),
        "post_fusion_max_abs_difference":float((afterA-afterB).abs().max()),
        "correct_patient_target_difference":s,
        "minimax_lower_bound_for_any_post_fusion_only_estimator":s/2,
        "premix_world_A_max_abs_error":err0,
        "premix_world_B_max_abs_error":err1,
        "source_repair_count":len(correctedB.source_repair_digests),
        "pretrained_checkpoint":False,
        "clinical_task_benefit":False
    }

if __name__=="__main__":
    print(json.dumps(run(),indent=2,sort_keys=True))