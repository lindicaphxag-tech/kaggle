"""Quantitative patient-space re-reading of a REAL pinned MONAI DiNTS block.

This controls the module weights to isolate sampling geometry; it neither
trains nor evaluates a medical segmentation checkpoint. The claim is a
mathematically testable reduction in coordinate-field readout error.
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
from research.medical_geometry.lattice_bundle import certify_single_lattice

def run() -> dict:
    block=FactorizedReduceBlock(in_channel=1,out_channel=2,spatial_dims=3)
    block.act=nn.Identity()
    block.norm=nn.Identity()
    with torch.no_grad():
        block.conv_1.weight.fill_(1.0)
        block.conv_2.weight.fill_(1.0)
    block.eval()
    spacing=(0.7,0.7,5.0)
    d,h,w=torch.meshgrid(
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),indexing="ij")
    x=(spacing[0]*d+spacing[1]*h+spacing[2]*w)[None,None]
    with torch.no_grad():
        y=block(x)
    typed=compile_fx_bundle_geometry(
        block,input_shape=(8,8,8),input_channels=1
    ).output_bundle
    aff=np.diag([*spacing,1.0])
    single=certify_single_lattice(typed,aff)
    target=FeatureLattice(
        (3,3,3),np.full(3,2.0),np.full(3,1.5),np.ones(3)
    )
    repaired=repair_channel_fibers(y,typed,target)
    td,th,tw=torch.meshgrid(
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,
        torch.arange(3,dtype=torch.float32)*2+1.5,indexing="ij"
    )
    expected=(spacing[0]*td+spacing[1]*th+spacing[2]*tw)[None,None]
    # Negative control: crop both output channels with identical array indices
    # while pretending that channel fibers share a target physical chart.
    naive=y[:,:,:3,:3,:3]
    before=naive-expected
    after=repaired.tensor-expected
    result={
        "external_runtime":"MONAI FactorizedReduceBlock",
        "task":"analytic patient-mm coordinate field under controlled 1x1 weights",
        "pretrained_checkpoint":False,
        "input_spacing_mm":list(spacing),
        "channel_fiber_count":len(typed.fibers),
        "single_feature_affine_valid":bool(single.valid),
        "fiber_phase_distance_mm":float(single.max_intra_tensor_displacement_mm),
        "target_support_shape":list(target.shape),
        "naive_shared_index_readout_mae":float(before.abs().mean().item()),
        "typed_fiber_repair_readout_mae":float(after.abs().mean().item()),
        "typed_fiber_repair_max_abs_error":float(after.abs().max().item()),
        "repair_plan_kinds":list(repaired.plan_kinds),
        "repair_plan_digests":list(repaired.plan_certificate_digests),
        "interpretation":"controlled linear-field readout, NOT downstream Dice or clinical benefit"
    }
    if single.valid:
        raise AssertionError("MONAI factorized reduction unexpectedly admits one feature affine")
    if result["naive_shared_index_readout_mae"] <= 3.0:
        raise AssertionError("controlled naive negative control did not show drift")
    if result["typed_fiber_repair_max_abs_error"] >= 2e-4:
        raise AssertionError("patient-space fiber repair failed linear-field parity")
    return result

if __name__ == "__main__":
    print(json.dumps(run(),indent=2,sort_keys=True))