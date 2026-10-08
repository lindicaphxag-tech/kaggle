"""Adversarial exact-chart (not merely rounded-source-map) CPU witnesses."""
from fractions import Fraction
import math

import numpy as np
import pytest
import torch

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.repair_synthesis import synthesize_lattice_repair
from research.medical_geometry.torch_repair import LatticeRepair3d
from research.medical_geometry.executed_interpolation_witness import (
    _exact_barycentric,
    _exact_declared_chart_indices,
    certify_executed_interpolation,
)


def _case(dtype=torch.float32):
    # All chart fields are exact binary floats in memory, but 0.1/0.3 is NOT
    # generally an exact binary-rational division, so FP64 source indexing
    # differs from the quotient of the STORED chart declarations.
    src = FeatureLattice(
        (8, 1, 1), np.array([0.3, 1., 1.]),
        np.array([0., 0., 0.]), np.ones(3),
    )
    dst = FeatureLattice(
        (1, 1, 1), np.ones(3),
        np.array([0.1, 0., 0.]), np.ones(3),
    )
    plan = synthesize_lattice_repair(src, dst)
    x = torch.arange(8, dtype=dtype).reshape(1, 1, 8, 1, 1) * 1e8
    return x, src, dst, plan


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_declared_chart_rounding_is_visible(dtype):
    x, src, dst, plan = _case(dtype)
    assert plan.kind == "resample"
    witness = certify_executed_interpolation(
        x, src, dst, plan, analytic_interpolation_bound=0.0
    )
    index = _exact_declared_chart_indices(src, dst, (0, 0, 0))[0]
    computed = Fraction.from_float(float(plan.target_to_source_index[0, 3]))
    assert index != computed
    assert witness.max_declared_vs_computed_source_index_error_voxels > 0
    assert witness.max_exact_declared_chart_residual >= 0
    assert math.isfinite(witness.max_exact_declared_chart_residual)
    assert "chart-index FP64 quantization" in witness.proof_scope

    # Independent exact-rational oracle, not a second invocation of the new
    # residual implementation. This checks an ACTUAL stored CPU tensor value.
    outcome = Fraction.from_float(float(LatticeRepair3d(plan)(x).item()))
    exact_value = _exact_barycentric(
        x.detach().cpu().numpy(), (index, Fraction(0), Fraction(0)), 0, 0
    )
    exact_discrepancy = abs(outcome - exact_value)
    assert witness.max_exact_declared_chart_residual >= float(exact_discrepancy)


def test_exactly_representable_chart_needs_no_index_rounding_budget():
    src = FeatureLattice(
        (3, 1, 1), np.ones(3), np.zeros(3), np.ones(3)
    )
    dst = FeatureLattice(
        (1, 1, 1), np.ones(3), np.array([0.5, 0., 0.]), np.ones(3)
    )
    p = synthesize_lattice_repair(src, dst)
    x = torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1, 1)
    a = certify_executed_interpolation(x, src, dst, p)
    assert a.max_declared_vs_computed_source_index_error_voxels == 0
    assert a.max_exact_declared_chart_residual == 0


def test_modified_source_lattice_chart_changes_exact_index_obligation():
    x, src, dst, plan = _case()
    a = certify_executed_interpolation(x, src, dst, plan)
    other_source = FeatureLattice(
        src.shape, np.array([0.31, 1., 1.]),
        src.start.copy(), src.receptive_field.copy(),
    )
    other_plan = synthesize_lattice_repair(other_source, dst)
    b = certify_executed_interpolation(x, other_source, dst, other_plan)
    assert a.plan_digest != b.plan_digest
    assert a.mapped_coordinate_sha256 != b.mapped_coordinate_sha256
    assert a.max_declared_vs_computed_source_index_error_voxels >= 0
    assert b.max_declared_vs_computed_source_index_error_voxels >= 0


def test_auditor_cannot_reinterpret_source_boundary_rounded_inside():
    # A target center just beyond the exact stored source chart boundary can
    # be rounded into range by finite floating-point arithmetic. The exact
    # chart must fail closed rather than quietly use the boundary sample.
    src = FeatureLattice(
        (2, 1, 1), np.array([0.3, 1., 1.]),
        np.zeros(3), np.ones(3)
    )
    dst = FeatureLattice(
        (1, 1, 1), np.ones(3),
        np.array([np.nextafter(0.3, math.inf), 0., 0.]), np.ones(3)
    )
    plan = synthesize_lattice_repair(src, dst)
    if plan.kind == "resample":
        with pytest.raises(ValueError, match="outside source grid"):
            certify_executed_interpolation(
                torch.ones((1, 1, 2, 1, 1)), src, dst, plan
            )
    else:
        assert plan.kind == "refuse"
