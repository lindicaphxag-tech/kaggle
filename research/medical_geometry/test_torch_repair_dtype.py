"""Data-type semantics: integer labels must never travel through floating values."""

import numpy as np
import torch

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.repair_synthesis import synthesize_lattice_repair
from research.medical_geometry.torch_repair import LatticeRepair3d


def frame(shape, start):
    return FeatureLattice(
        tuple(shape), np.array([2.0, 2.0, 2.0]),
        np.array(start, dtype=np.float64), np.ones(3),
    )


def test_nearest_int64_region_ids_above_2pow53_are_bitwise_preserved():
    source = frame((8, 8, 8), (0.5, 0.5, 0.5))
    target = frame((7, 7, 7), (2.0, 2.0, 2.0))
    plan = synthesize_lattice_repair(source, target, role="label")
    assert plan.kind == "resample" and plan.interpolation == "nearest"
    d, h, w = torch.meshgrid(
        torch.arange(8, dtype=torch.int64),
        torch.arange(8, dtype=torch.int64),
        torch.arange(8, dtype=torch.int64), indexing="ij",
    )
    x = (2**53 + 1000000 * d + 1000 * h + w)[None, None]
    y = LatticeRepair3d(plan)(x)
    # Source coords are i+0.75. Nearest is source index i+1 on each axis.
    expected = x[:, :, 1:, 1:, 1:]
    assert y.dtype is torch.int64
    assert torch.equal(y, expected)


def test_nearest_bool_mask_preserves_boolean_dtype_and_support():
    source = frame((8, 8, 8), (0.5, 0.5, 0.5))
    target = frame((7, 7, 7), (2.0, 2.0, 2.0))
    plan = synthesize_lattice_repair(source, target, role="label")
    x = torch.zeros(1, 1, 8, 8, 8, dtype=torch.bool)
    x[:, :, 3:6, 2:5, 1:4] = True
    y = LatticeRepair3d(plan)(x)
    assert y.dtype is torch.bool
    assert torch.equal(y, x[:, :, 1:, 1:, 1:])


def test_trilinear_float64_feature_grid_matches_dtype_and_backward():
    source = frame((8, 8, 8), (0.5, 0.5, 0.5))
    target = frame((7, 7, 7), (2.0, 2.0, 2.0))
    plan = synthesize_lattice_repair(source, target, role="feature")
    d, h, w = torch.meshgrid(
        torch.arange(8, dtype=torch.float64),
        torch.arange(8, dtype=torch.float64),
        torch.arange(8, dtype=torch.float64), indexing="ij",
    )
    x = (0.7*d + 1.1*h + 1.9*w)[None, None].detach().requires_grad_(True)
    executor = LatticeRepair3d(plan)
    y = executor(x)
    assert y.dtype is torch.float64
    # A linear field is reproduced at fractional coordinates without loss.
    i,j,k = torch.meshgrid(
        torch.arange(7,dtype=torch.float64)+0.75,
        torch.arange(7,dtype=torch.float64)+0.75,
        torch.arange(7,dtype=torch.float64)+0.75, indexing="ij",
    )
    exact = (0.7*i + 1.1*j + 1.9*k)[None,None]
    torch.testing.assert_close(y, exact, rtol=0, atol=1e-12)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0


def test_nearest_int64_round_half_uses_even_index_for_ties():
    source = frame((8,8,8),(0,0,0))
    target = frame((7,7,7),(1,1,1))
    plan = synthesize_lattice_repair(source,target,role="label")
    x = torch.arange(8,dtype=torch.int64).reshape(1,1,8,1,1).expand(1,1,8,8,8)
    y = LatticeRepair3d(plan)(x)
    # Float nearest rounding is ties-to-even; verify against grid_sample.
    as_float = LatticeRepair3d(plan)(x.float())
    assert torch.equal(y, as_float.long())