"""Adversarial patient-space locality and receptive-support provenance tests.

These are source-code correctness regressions, NOT new math, clinical experiments
or MONAI bugs. Each test contrasts executable PyTorch dependency with the
ClinGeom compiler's claimed physical support.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.medical_geometry.torch_geometry import (
    UnsupportedGeometryOp,
    compile_sequential_geometry,
)
from research.medical_geometry.torch_fx_bundle import compile_fx_bundle_geometry


def input_pair():
    left = torch.zeros(1, 1, 9, 9, 9, dtype=torch.float32)
    left[0, 0, 4, 4, 4] = 1.0
    right = left.clone()
    right[0, 0, 0, 0, 0] = 8.0  # 4 voxels away on every spatial axis
    return left, right


@pytest.mark.parametrize("name,layer", [
    ("InstanceNorm3d training", nn.InstanceNorm3d(1, track_running_stats=False)),
    ("InstanceNorm3d eval no running stats", nn.InstanceNorm3d(1, track_running_stats=False).eval()),
    ("GroupNorm training", nn.GroupNorm(1, 1)),
    ("GroupNorm eval", nn.GroupNorm(1, 1).eval()),
    ("BatchNorm3d training", nn.BatchNorm3d(1).train()),
    ("BatchNorm3d eval no running stats", nn.BatchNorm3d(1, track_running_stats=False).eval()),
])
def test_far_away_voxel_changes_center_output_for_global_statistics(name, layer):
    x, y = input_pair()
    with torch.no_grad():
        a = layer(x)[0, 0, 4, 4, 4]
        b = layer(y)[0, 0, 4, 4, 4]
    assert abs(float(a-b)) > 1.0, name

    model = nn.Sequential(layer)
    with pytest.raises(UnsupportedGeometryOp, match="spatial|local|global"):
        compile_sequential_geometry(
            model,
            input_shape=(9, 9, 9),
            input_affine=np.diag([0.7, 0.7, 5.0, 1.0]),
        )
    with pytest.raises(UnsupportedGeometryOp, match="spatial|local|global"):
        compile_fx_bundle_geometry(
            model, input_shape=(9, 9, 9), input_channels=1
        )


@pytest.mark.parametrize("layer", [
    nn.BatchNorm3d(1, track_running_stats=True).eval(),
    nn.InstanceNorm3d(1, track_running_stats=True).eval(),
])
def test_frozen_eval_running_statistics_are_local(layer):
    x, y = input_pair()
    with torch.no_grad():
        a = layer(x)[0, 0, 4, 4, 4]
        b = layer(y)[0, 0, 4, 4, 4]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    report = compile_sequential_geometry(
        nn.Sequential(layer),
        input_shape=(9, 9, 9),
        input_affine=np.diag([0.7, 0.7, 5.0, 1.0]),
    )
    np.testing.assert_array_equal(report[-1].lattice.receptive_field, [1, 1, 1])
    fx = compile_fx_bundle_geometry(
        nn.Sequential(layer), input_shape=(9, 9, 9), input_channels=1
    )
    np.testing.assert_array_equal(
        fx.output_bundle.fibers[0].supports[0].receptive_field, [1, 1, 1]
    )


class ResidualWideContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(
            kernel_size=(3, 5, 1),
            stride=1,
            padding=(1, 2, 0),
            count_include_pad=True
        )

    def forward(self, x):
        return x + self.pool(x)


def test_pointwise_residual_addition_preserves_other_branches_wide_support():
    model = ResidualWideContext()
    x = torch.zeros(1, 1, 9, 9, 9)
    x[0, 0, 4, 6, 4] = 15.0  # 2 positions away from center, INSIDE pool support
    with torch.no_grad():
        out = model(x)
    assert float(x[0, 0, 4, 4, 4]) == 0.0
    assert float(out[0, 0, 4, 4, 4]) > 0.0
    report = compile_fx_bundle_geometry(
        model,
        input_shape=(9, 9, 9),
        input_channels=1
    )
    support = report.output_bundle.fibers[0].supports[0]
    np.testing.assert_array_equal(support.start, [0, 0, 0])
    np.testing.assert_array_equal(support.jump, [1, 1, 1])
    np.testing.assert_array_equal(support.receptive_field, [3, 5, 1])


def test_addition_output_support_is_order_invariant():
    class ReverseWideContext(ResidualWideContext):
        def forward(self,x):
            return self.pool(x) + x
    forward = compile_fx_bundle_geometry(
        ResidualWideContext(), input_shape=(9, 9, 9), input_channels=1
    )
    backward = compile_fx_bundle_geometry(
        ReverseWideContext(), input_shape=(9, 9, 9), input_channels=1
    )
    a=forward.output_bundle.fibers[0].supports[0]
    b=backward.output_bundle.fibers[0].supports[0]
    np.testing.assert_array_equal(a.receptive_field, [3, 5, 1])
    np.testing.assert_array_equal(b.receptive_field, [3, 5, 1])
