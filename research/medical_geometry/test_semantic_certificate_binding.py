"""Semantic provenance must distinguish identical tensors in different patient frames.

A SHA-256 fingerprint is not evidence if it commits only to convolution weights
and voxel-index maps while ignoring the physical millimetre chart and premises
used to authorize interpolation.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.lattice_bundle import ChannelFiber, LatticeBundle
from research.medical_geometry.pre_mix_compiler import physically_align_then_mix


def case():
    source=FeatureLattice(
        (8,8,8), np.ones(3), np.zeros(3), np.ones(3)
    )
    fibers=(
        ChannelFiber(0,1,(source,),"left"),
        ChannelFiber(1,2,(source,),"right"),
    )
    bundle=LatticeBundle((8,8,8),2,fibers)
    target=FeatureLattice(
        (7,7,7), np.ones(3), np.full(3,0.5), np.ones(3)
    )
    mixer=nn.Conv3d(2,1,kernel_size=1,bias=False)
    with torch.no_grad():
        mixer.weight.fill_(1.0)
    return torch.ones(1,2,8,8,8),bundle,target,mixer


def run(*,spacing=(1,1,1),hessian=0.02,frame_uid="1.2.3"):
    x,bundle,target,mix=case()
    return physically_align_then_mix(
        x,bundle,target,mix,
        input_voxel_to_patient_mm=np.diag([*spacing,1.0]),
        maximum_fiber_value_error=100.0,
        fiber_world_hessian_bounds={0:hessian,1:hessian},
        maximum_output_value_error=100.0,
        patient_frame_uid=frame_uid,
    )


def test_same_voxel_values_different_patient_mm_scale_have_distinct_digest():
    isotropic=run(spacing=(1,1,1))
    thick_slice=run(spacing=(1,1,5))
    assert torch.equal(isotropic.tensor,thick_slice.tensor)
    assert isotropic.source_repair_digests==thick_slice.source_repair_digests
    assert isotropic.per_output_value_error_bounds != thick_slice.per_output_value_error_bounds
    assert isotropic.conv_weight_provenance_sha256!=thick_slice.conv_weight_provenance_sha256


def test_same_voxel_data_new_curvature_proof_changes_semantics_digest():
    a=run(hessian=0.02)
    b=run(hessian=0.04)
    assert torch.equal(a.tensor,b.tensor)
    assert a.per_output_value_error_bounds != b.per_output_value_error_bounds
    assert a.conv_weight_provenance_sha256!=b.conv_weight_provenance_sha256


def test_patient_frame_identity_must_be_part_of_certificate():
    a=run(frame_uid="1.2.3")
    b=run(frame_uid="1.2.4")
    assert torch.equal(a.tensor,b.tensor)
    assert a.per_output_value_error_bounds==b.per_output_value_error_bounds
    assert a.conv_weight_provenance_sha256!=b.conv_weight_provenance_sha256
    with pytest.raises(ValueError,match="nonblank"):
        run(frame_uid="  ")


def test_nonfinite_or_noninvertible_patient_chart_fails_closed():
    x,bundle,target,mix=case()
    for bad in (np.diag([1.0,0.0,1.0,1.0]), np.diag([1.0,np.nan,1.0,1.0])):
        with pytest.raises(ValueError,match="patient affine"):
            physically_align_then_mix(
                x,bundle,target,mix,input_voxel_to_patient_mm=bad
            )


def test_same_source_and_premises_remain_deterministic():
    a=run()
    b=run()
    assert a.conv_weight_provenance_sha256==b.conv_weight_provenance_sha256
    assert len(a.conv_weight_provenance_sha256)==64


def test_unknown_patient_frame_is_not_silently_substituted_for_declared_frame():
    unknown=run(frame_uid=None)
    declared=run(frame_uid="1.2.3")
    assert unknown.conv_weight_provenance_sha256!=declared.conv_weight_provenance_sha256
