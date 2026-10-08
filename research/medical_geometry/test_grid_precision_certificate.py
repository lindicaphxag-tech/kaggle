"""Falsify unsafe mixed-precision patient-coordinate certificates.

The real-valued C2 interpolation theorem does not include rounding normalized
grid_sample coordinates to FP16/BF16. This suite probes actual CPU PyTorch
trilinear sampling, oblique patient affines, exact-crop negative controls,
and the compiler's fail-closed patient-mm budget.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.grid_precision_certificate import (
    certify_grid_coordinate_precision,
)
from research.medical_geometry.lattice_bundle import bundle_from_lattice
from research.medical_geometry.pre_mix_compiler import physically_align_then_mix


def lattice(shape, start=(0, 0, 0), jump=(1, 1, 1), rf=(1, 1, 1)):
    return FeatureLattice(
        tuple(shape),
        np.asarray(jump, dtype=np.float64),
        np.asarray(start, dtype=np.float64),
        np.asarray(rf, dtype=np.float64),
    )


def affine():
    a = np.eye(4, dtype=np.float64)
    a[:3, :3] = np.array([
        [0.7, -0.35, 0.2], [0.2, 1.15, -0.25], [0.1, 0.2, 5.0],
    ])
    a[:3, 3] = [20, -15, 6]
    return a


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_eight_corner_certificate_equals_brute_force_real_voxel_grid(dtype):
    src = lattice((9, 8, 7))
    dst = lattice((8, 7, 6), start=(0.5, 0.25, 0.75))
    a = affine()
    cert = certify_grid_coordinate_precision(
        src, dst, input_voxel_to_patient_mm=a, input_dtype=dtype
    )
    per_axis = []
    for size, count, phase in zip(src.shape, dst.shape, dst.start):
        s = torch.arange(count, dtype=torch.float64) + float(phase)
        stored = (2 * s / (size - 1) - 1).to(dtype).to(torch.float64)
        drift = (stored + 1) * (size - 1) / 2 - s
        per_axis.append(drift.numpy())
    components = np.meshgrid(*per_axis, indexing="ij")
    every_drift = np.stack(components, -1).reshape(-1, 3)
    displacements = every_drift @ a[:3, :3].T
    brute = np.linalg.norm(displacements, axis=1).max()
    assert cert.max_world_coordinate_displacement_mm == pytest.approx(
        float(brute), rel=1e-11, abs=1e-11
    )
    assert cert.max_world_coordinate_displacement_mm >= 0


def test_real_linear_3d_pytorch_fp16_and_bf16_violate_zero_curvature_idealization():
    # f(i,j,k)=i has zero Hessian. Ideal trilinear interpolation should be
    # exact at every i+1/2 target sample, yet numerical coordinate storage
    # produces sizeable value discrepancies, even on CPU.
    n = 64
    true = torch.arange(n - 1, dtype=torch.float64) + 0.5
    for dtype, lower in [(torch.float16, 0.01), (torch.bfloat16, 0.10)]:
        x = torch.arange(n, dtype=dtype).reshape(1, 1, n, 1, 1)
        pos = torch.arange(n - 1, dtype=torch.float64) + 0.5
        norm = (2 * pos / (n - 1) - 1).to(dtype).view(1, n - 1, 1, 1)
        g = torch.stack((torch.zeros_like(norm), torch.zeros_like(norm), norm), -1)
        out = F.grid_sample(x, g, mode="bilinear", align_corners=True).flatten()
        measured = float((out.float() - true.float()).abs().max())
        assert measured > lower

    src = lattice((n, 1, 1))
    dst = lattice((n - 1, 1, 1), start=(0.5, 0, 0))
    physical = np.diag([0.7, 1.0, 5.0, 1.0])
    half = certify_grid_coordinate_precision(
        src, dst, input_voxel_to_patient_mm=physical, input_dtype=torch.float16
    )
    bf = certify_grid_coordinate_precision(
        src, dst, input_voxel_to_patient_mm=physical, input_dtype=torch.bfloat16
    )
    assert 0 < half.max_world_coordinate_displacement_mm
    assert bf.max_world_coordinate_displacement_mm > half.max_world_coordinate_displacement_mm


def test_sheared_affine_and_anisotropic_mm_modify_authorization_result():
    src = lattice((64, 2, 2))
    dst = lattice((63, 2, 2), start=(0.5, 0, 0))
    base = np.eye(4)
    anisotropic = affine()
    a = certify_grid_coordinate_precision(
        src, dst, input_voxel_to_patient_mm=base, input_dtype=torch.float16
    )
    b = certify_grid_coordinate_precision(
        src, dst, input_voxel_to_patient_mm=anisotropic, input_dtype=torch.float16
    )
    assert b.max_world_coordinate_displacement_mm != a.max_world_coordinate_displacement_mm
    assert b.max_absolute_source_index_displacement == pytest.approx(
        a.max_absolute_source_index_displacement
    )


def test_exact_crop_is_zero_grid_storage_drift_even_at_low_precision():
    src = lattice((64, 2, 2))
    dst = lattice((60, 2, 2), start=(2, 0, 0))
    p = certify_grid_coordinate_precision(
        src, dst, input_voxel_to_patient_mm=affine(),
        input_dtype=torch.bfloat16, resampling_executed=False
    )
    assert p.max_world_coordinate_displacement_mm == 0.0
    assert p.source_index_error_ranges == ((0, 0), (0, 0), (0, 0))


def test_compiler_refuses_FP16_patient_mm_budget_but_accepts_FP32():
    src = lattice((64, 2, 2))
    dst = lattice((63, 2, 2), start=(0.5, 0, 0))
    bundle = bundle_from_lattice(1, src)
    mm = np.diag([5.0, 1.0, 1.0, 1.0])
    budget = 0.001
    half_input = torch.ones((1, 1, 64, 2, 2), dtype=torch.float16)
    half_mixer = nn.Conv3d(1, 1, 1).half()
    with pytest.raises(ValueError, match="exceeds patient-mm budget"):
        physically_align_then_mix(
            half_input, bundle, dst, half_mixer,
            input_voxel_to_patient_mm=mm,
            maximum_grid_coordinate_drift_mm=budget,
        )
    # Equivalent FP32 coordinates satisfy the same budget. This is a
    # coordinate-only gate: floating-point value arithmetic is NOT certified.
    mixer = nn.Conv3d(1, 1, 1)
    good = physically_align_then_mix(
        half_input.float(), bundle, dst, mixer,
        input_voxel_to_patient_mm=mm,
        maximum_grid_coordinate_drift_mm=budget,
    )
    assert good.per_fiber_grid_coordinate_drift_mm is not None
    assert good.per_fiber_grid_coordinate_drift_mm[0] <= budget
    assert good.tensor.shape == (1, 1, 63, 2, 2)


def test_precision_budget_changes_semantic_digest_even_when_tensor_matches():
    src = lattice((64, 2, 2))
    dst = lattice((63, 2, 2), start=(0.5, 0, 0))
    bundle = bundle_from_lattice(1, src)
    x = torch.ones((1, 1, 64, 2, 2))
    mixer = nn.Conv3d(1, 1, 1, bias=False)
    mm = np.eye(4)
    a = physically_align_then_mix(
        x, bundle, dst, mixer, input_voxel_to_patient_mm=mm,
        maximum_grid_coordinate_drift_mm=0.01
    )
    b = physically_align_then_mix(
        x, bundle, dst, mixer, input_voxel_to_patient_mm=mm,
        maximum_grid_coordinate_drift_mm=0.02
    )
    torch.testing.assert_close(a.tensor, b.tensor, rtol=0, atol=0)
    assert a.conv_weight_provenance_sha256 != b.conv_weight_provenance_sha256


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_precision_budget_fails_closed(bad):
    src = lattice((8, 2, 2))
    dst = lattice((7, 2, 2), start=(0.5, 0, 0))
    bundle = bundle_from_lattice(1, src)
    with pytest.raises(ValueError, match="maximum grid-coordinate drift"):
        physically_align_then_mix(
            torch.ones((1, 1, 8, 2, 2)), bundle, dst, nn.Conv3d(1, 1, 1),
            input_voxel_to_patient_mm=np.eye(4),
            maximum_grid_coordinate_drift_mm=bad,
        )


def test_precision_budget_without_patient_frame_refused():
    src = lattice((8, 2, 2))
    dst = lattice((7, 2, 2), start=(0.5, 0, 0))
    with pytest.raises(ValueError, match="requires patient affine"):
        physically_align_then_mix(
            torch.ones((1, 1, 8, 2, 2)), bundle_from_lattice(1, src),
            dst, nn.Conv3d(1, 1, 1), maximum_grid_coordinate_drift_mm=1.0
        )
