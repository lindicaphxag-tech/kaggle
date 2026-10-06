import numpy as np
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.torch_geometry import (
    UnsupportedGeometryOp,
    compile_sequential_geometry,
)


def test_compiled_shape_matches_real_sequential_conv_pool_forward():
    model = nn.Sequential(
        nn.Conv3d(1, 4, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
    )
    trace = compile_sequential_geometry(
        model, input_shape=(64, 80, 48), input_affine=np.eye(4)
    )
    x = torch.zeros(1, 1, 64, 80, 48)
    y = model(x)
    assert trace[-1].lattice.shape == tuple(y.shape[-3:])
    assert trace[-1].lattice.shape == (16, 20, 12)
    assert np.allclose(trace[-1].lattice.jump, [4.0, 4.0, 4.0])
    assert np.allclose(trace[-1].lattice.start, [1.0, 1.0, 1.0])
    assert np.allclose(trace[-1].lattice.receptive_field, [13.0, 13.0, 13.0])


def test_patient_affine_respects_anisotropic_input_spacing():
    model = nn.Sequential(nn.Conv3d(1, 2, kernel_size=3, stride=4, padding=1))
    affine = np.eye(4)
    affine[:3, :3] = np.diag([0.7, 0.7, 5.0])
    trace = compile_sequential_geometry(
        model, input_shape=(64, 64, 64), input_affine=affine
    )
    A = trace[-1].patient_affine
    assert np.allclose(np.diag(A)[:3], [2.8, 2.8, 20.0])
    assert np.allclose(A[:3, 3], [0.0, 0.0, 0.0])


def test_nested_sequential_is_serially_compiled():
    model = nn.Sequential(
        nn.Sequential(
            nn.Conv3d(1, 2, 3, stride=2, padding=1),
            nn.BatchNorm3d(2),
            nn.ReLU(),
        ),
        nn.AvgPool3d(kernel_size=2, stride=2),
    )
    trace = compile_sequential_geometry(
        model, input_shape=(32, 32, 32), input_affine=np.eye(4)
    )
    assert trace[-1].lattice.shape == (8, 8, 8)
    assert np.allclose(trace[-1].lattice.jump, [4.0, 4.0, 4.0])


def test_arbitrary_nonsequential_container_fails_closed():
    class ResidualLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 1, 3, padding=1)

        def forward(self, x):
            return self.conv(x) + x

    with pytest.raises(UnsupportedGeometryOp, match="branch/merge"):
        compile_sequential_geometry(
            ResidualLike(), input_shape=(16, 16, 16), input_affine=np.eye(4)
        )


def test_adaptive_pool_fails_closed():
    model = nn.Sequential(nn.AdaptiveAvgPool3d((8, 8, 8)))
    with pytest.raises(UnsupportedGeometryOp, match="no certified"):
        compile_sequential_geometry(
            model, input_shape=(31, 35, 39), input_affine=np.eye(4)
        )


def test_maxpool_ceil_mode_fails_closed():
    model = nn.Sequential(nn.MaxPool3d(3, stride=2, ceil_mode=True))
    with pytest.raises(UnsupportedGeometryOp, match="ceil_mode"):
        compile_sequential_geometry(
            model, input_shape=(20, 20, 20), input_affine=np.eye(4)
        )
