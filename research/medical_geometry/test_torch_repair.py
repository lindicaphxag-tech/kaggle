import numpy as np
import torch

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.repair_synthesis import synthesize_lattice_repair
from research.medical_geometry.torch_repair import LatticeRepair3d


def lattice(shape, jump, start):
    return FeatureLattice(
        tuple(shape),
        np.asarray(jump, dtype=np.float64),
        np.asarray(start, dtype=np.float64),
        np.ones(3, dtype=np.float64),
    )


def test_exact_crop_executes_certificate_without_interpolation():
    src = lattice((20, 20, 20), (2, 2, 2), (0, 0, 0))
    dst = lattice((16, 16, 16), (2, 2, 2), (4, 2, 6))
    plan = synthesize_lattice_repair(src, dst)
    assert plan.kind == "exact_crop"

    x = torch.arange(20 * 20 * 20).reshape(1, 1, 20, 20, 20)
    y = LatticeRepair3d(plan)(x)
    assert tuple(y.shape[-3:]) == dst.shape
    assert torch.equal(y, x[:, :, 2:18, 1:17, 3:19])


def test_trilinear_repair_recovers_linear_patient_coordinate_field():
    # Source centers: 0.5, 2.5, ..., 62.5
    src = lattice((32, 32, 32), (2, 2, 2), (0.5, 0.5, 0.5))
    # Common target centers: 2, 4, ..., 62
    dst = lattice((31, 31, 31), (2, 2, 2), (2.0, 2.0, 2.0))
    plan = synthesize_lattice_repair(src, dst, role="feature")
    assert plan.kind == "resample"
    assert plan.interpolation == "trilinear"

    coords = 0.5 + 2.0 * torch.arange(32, dtype=torch.float32)
    d, h, w = torch.meshgrid(coords, coords, coords, indexing="ij")
    # A linear physical field is reproduced exactly by trilinear interpolation
    # up to ordinary floating-point arithmetic.
    x = (d + 2.0 * h + 3.0 * w)[None, None]

    target = 2.0 + 2.0 * torch.arange(31, dtype=torch.float32)
    td, th, tw = torch.meshgrid(target, target, target, indexing="ij")
    expected = (td + 2.0 * th + 3.0 * tw)[None, None]

    y = LatticeRepair3d(plan)(x)
    assert tuple(y.shape[-3:]) == dst.shape
    assert torch.max(torch.abs(y - expected)).item() < 2e-4


def test_nearest_label_repair_preserves_integer_label_set():
    src = lattice((8, 8, 8), (2, 2, 2), (0.5, 0.5, 0.5))
    dst = lattice((7, 7, 7), (2, 2, 2), (2.0, 2.0, 2.0))
    plan = synthesize_lattice_repair(src, dst, role="label")
    assert plan.interpolation == "nearest"

    x = torch.zeros(1, 1, 8, 8, 8, dtype=torch.int64)
    x[:, :, 3:, :, :] = 7
    y = LatticeRepair3d(plan)(x)
    assert y.dtype == torch.int64
    assert set(torch.unique(y).tolist()).issubset({0, 7})


def test_extrapolating_resample_is_refused_before_runtime():
    src = lattice((16, 16, 16), (2, 2, 2), (0, 0, 0))
    # Same size but shifted one full source sample to the left, so target index
    # zero maps outside the source lattice.
    dst = lattice((16, 16, 16), (2, 2, 2), (-2, 0, 0))
    plan = synthesize_lattice_repair(src, dst, role="feature")
    assert plan.kind == "refuse"
    assert "extrapolation" in plan.reason
