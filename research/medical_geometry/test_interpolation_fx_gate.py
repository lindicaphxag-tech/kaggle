"""The FX repair compiler must not call interpolation 'safe' without a bound."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.fx_rewrite import rewrite_patient_space_merges
from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.repair_synthesis import synthesize_lattice_repair
from research.medical_geometry.interpolation_certificate import certify_interpolation_error
from research.medical_geometry.torch_repair import LatticeRepair3d
from research.medical_geometry.torch_geometry import UnsupportedGeometryOp


class PhaseShiftAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x) + self.pool(x)


def rewrite(**kwargs):
    return rewrite_patient_space_merges(
        PhaseShiftAdd(), input_shape=(64, 64, 64),
        input_affine=np.eye(4), **kwargs
    )


def test_opt_in_budget_requires_per_branch_curvature_provenance():
    with pytest.raises(UnsupportedGeometryOp, match='explicit per-node'):
        rewrite(maximum_value_error=1.0)
    with pytest.raises(UnsupportedGeometryOp, match='cannot certify'):
        rewrite(maximum_value_error=1.0, feature_hessian_bounds_by_node={})


def test_frozen_curvature_envelope_authorizes_bounded_rewrite():
    result = rewrite(
        maximum_value_error=0.2,
        feature_hessian_bounds_by_node={'pool': 0.1},
    )
    assert len(result.rewrites) == 1
    assert result.rewrites[0].value_error_bounds is not None
    assert result.rewrites[0].value_error_bounds[0] == 0
    assert result.rewrites[0].value_error_bounds[1] == pytest.approx(0.1125)
    assert len(result.postcondition_report.repair_obligations) == 0
    y = result.graph_module(torch.zeros(1,1,64,64,64))
    assert y.shape == (1,1,31,31,31)


def test_curvature_above_error_budget_rejects_rewrite():
    with pytest.raises(UnsupportedGeometryOp, match='value-error budget'):
        rewrite(
            maximum_value_error=0.1,
            feature_hessian_bounds_by_node={'pool': 0.1},
        )


def test_opt_out_keeps_geometry_only_claim_not_feature_error_proof():
    result = rewrite()
    assert result.rewrites[0].value_error_bounds is None


def test_nearest_neighbor_label_cannot_claim_c2_value_recovery():
    with pytest.raises(UnsupportedGeometryOp, match='cannot certify'):
        rewrite(
            role='label', maximum_value_error=10.0,
            feature_hessian_bounds_by_node={'pool': 0.001},
        )


def test_real_torch_interpolation_obeys_sharp_quadratic_certificate():
    source = FeatureLattice(
        (5, 5, 5), np.ones(3), np.zeros(3), np.ones(3)
    )
    target = FeatureLattice(
        (4, 4, 4), np.ones(3), np.full(3, 0.5), np.ones(3)
    )
    plan = synthesize_lattice_repair(source, target, role='feature')
    patient_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    proof = certify_interpolation_error(
        source, target, plan, input_voxel_to_patient_mm=patient_affine,
        world_hessian_operator_norm_bound=0.2,
        maximum_allowed_error=0.300001,
        role='feature',
    )
    assert proof.authorized
    assert proof.max_absolute_error_bound == pytest.approx(0.3)

    i, j, k = torch.meshgrid(
        torch.arange(5, dtype=torch.float32),
        torch.arange(5, dtype=torch.float32),
        torch.arange(5, dtype=torch.float32), indexing='ij',
    )
    samples = (0.5 * 0.2 * (4*i*i + 4*j*j + 4*k*k))[None,None]
    repaired = LatticeRepair3d(plan)(samples)
    u, v, w = torch.meshgrid(
        torch.arange(4, dtype=torch.float32) + 0.5,
        torch.arange(4, dtype=torch.float32) + 0.5,
        torch.arange(4, dtype=torch.float32) + 0.5, indexing='ij',
    )
    ideal = (0.5 * 0.2 * (4*u*u + 4*v*v + 4*w*w))[None,None]
    actual_max_error = float((repaired - ideal).abs().max())
    assert actual_max_error == pytest.approx(
        proof.max_absolute_error_bound, abs=2e-5
    )
    assert actual_max_error <= proof.max_absolute_error_bound + 2e-5
