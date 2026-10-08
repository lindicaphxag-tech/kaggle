"""Identical post-mix tensors cannot determine the correct patient-space fusion."""

from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks.dints_block import FactorizedReduceBlock

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.lattice_bundle import dense_conv_mix_bundle, certify_single_lattice
from research.medical_geometry.torch_fx_bundle import compile_fx_bundle_geometry
from research.medical_geometry.pre_mix_compiler import physically_align_then_mix
from research.medical_geometry.channel_fiber_runtime import repair_channel_fibers


def run():
    # Two different physical input fields (A and B). MONAI factorized reduction
    # samples channel 0 at even positions and channel 1 at odd positions.
    spacing=0.7
    d,h,w=torch.meshgrid(
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),
        torch.arange(8,dtype=torch.float32),indexing="ij"
    )
    zeros=torch.zeros(1,2,8,8,8)
    nonzero=torch.stack((spacing*d,-spacing*d+spacing),dim=0)[None]
    block=FactorizedReduceBlock(in_channel=2,out_channel=2,spatial_dims=3)
    block.act=nn.Identity()
    block.norm=nn.Identity()
    with torch.no_grad():
        block.conv_1.weight.zero_()
        block.conv_2.weight.zero_()
        block.conv_1.weight[0,0,0,0,0]=1.0
        block.conv_2.weight[0,1,0,0,0]=1.0
    block.eval()
    mixer=nn.Conv3d(2,1,kernel_size=1,bias=False)
    with torch.no_grad():
        mixer.weight.fill_(1.0)
        reduced_a=block(zeros)
        reduced_b=block(nonzero)
        postmixed_a=mixer(reduced_a)
        postmixed_b=mixer(reduced_b)
    # Any deterministic postmix-only correction receives indistinguishable
    # observations, but the target physical outputs differ by 0.7 everywhere.
    input_equal=float((postmixed_a-postmixed_b).abs().max())
    if input_equal>2e-6:
        raise AssertionError("adversarial postmix observations are distinguishable")
    bundle=compile_fx_bundle_geometry(
        block,input_shape=(8,8,8),input_channels=2
    ).output_bundle
    patient_affine=np.diag([spacing,spacing,5.0,1.0])
    mixed_type=dense_conv_mix_bundle(bundle,out_channels=1,kernel_size=1)
    assert not certify_single_lattice(mixed_type,patient_affine).valid
    target=FeatureLattice((3,3,3),np.full(3,2.0),np.full(3,1.5),np.ones(3))
    with torch.no_grad():
        corrected_a=physically_align_then_mix(reduced_a,bundle,target,mixer)
        corrected_b=physically_align_then_mix(reduced_b,bundle,target,mixer)
    truth_a=torch.zeros_like(corrected_a.tensor)
    truth_b=torch.full_like(corrected_b.tensor,spacing)
    err_a=float((corrected_a.tensor-truth_a).abs().max())
    err_b=float((corrected_b.tensor-truth_b).abs().max())
    if max(err_a,err_b)>2e-6:
        raise AssertionError("pre-mix transport failed physically correct target fusion")
    effective_rf=corrected_b.output_bundle.fibers[0].supports[0].receptive_field
    # The fused tensor has one sampling CENTER lattice, but trilinear
    # re-reading uses neighboring source samples. Retaining the user-declared
    # nominal receptive field would underreport actual physical support.
    np.testing.assert_allclose(effective_rf,[4.0,4.0,4.0],atol=1e-9)
    # Minimax impossibility: for any single postmix estimate z (same input for
    # both hypotheses), at least one world has absolute error >= 0.7/2.
    return {
        "external_runtime":"pinned MONAI FactorizedReduceBlock followed by nn.Conv3d(2,1,1)",
        "two_different_input_fields_have_same_postmix_output":input_equal<=2e-6,
        "postmix_output_max_difference":input_equal,
        "true_target_fusion_difference":spacing,
        "worst_case_lower_bound_any_postmix_only_estimator":spacing/2.0,
        "premix_corrected_case_A_max_error":err_a,
        "premix_corrected_case_B_max_error":err_b,
        "output_bundle_after_premix_has_single_lattice":len(corrected_a.output_bundle.fibers)==1,
        "source_repairs":len(corrected_b.source_repair_digests),
        "nominal_target_receptive_field_input_voxels":list(target.receptive_field),
        "certified_output_receptive_support_envelope_input_voxels":list(effective_rf),
        "support_envelope_is_conservative_not_exact_recovered_signal":True,
        "known_input_fields":"controlled linear signed functions; not clinical images",
        "pretrained_network":False,
        "clinical_effect_claim":False
    }

if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True,indent=2))
