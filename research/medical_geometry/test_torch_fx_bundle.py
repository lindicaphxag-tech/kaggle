import numpy as np
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.lattice_bundle import certify_single_lattice
from research.medical_geometry.torch_fx_bundle import (
    BundleGeometryError,
    compile_fx_bundle_geometry,
)


class FactorizedReduce3D(nn.Module):
    """Minimal PyTorch reproduction of MONAI/DARTS factorized reduction."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv3d(
            c_in, c_out // 2, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.conv_2 = nn.Conv3d(
            c_in, c_out - c_out // 2, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.bn = nn.BatchNorm3d(c_out)

    def forward(self, x):
        x = self.relu(x)
        return self.bn(
            torch.cat(
                [
                    self.conv_1(x),
                    self.conv_2(x[:, :, 1:, 1:, 1:]),
                ],
                dim=1,
            )
        )


class ReduceThenMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = FactorizedReduce3D(8, 16)
        self.mix = nn.Conv3d(16, 24, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.mix(self.reduce(x))


class TwoAlignedBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv3d(4, 4, 3, padding=1, bias=False)
        self.b = nn.Conv3d(4, 4, 3, padding=1, bias=False)

    def forward(self, x):
        return self.a(x) + self.b(x)


class TwoMisalignedBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv3d(
            4, 4, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False
        )
        self.pad = nn.ConstantPad3d((1, 0, 0, 0, 0, 0), 0.0)
        self.b = nn.Conv3d(4, 4, kernel_size=(1, 1, 2), bias=False)

    def forward(self, x):
        return self.a(x) + self.b(self.pad(x))


def patient_affine():
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag([0.7, 0.7, 5.0])
    return affine


def test_fx_compiler_recovers_factorized_reduce_channel_fibers():
    model = FactorizedReduce3D(8, 16)
    model.eval()  # frozen BatchNorm is spatially local in inference
    x = torch.zeros(1, 8, 32, 32, 32)
    y = model(x)
    assert tuple(y.shape) == (1, 16, 16, 16, 16)

    result = compile_fx_bundle_geometry(
        model, input_shape=(32, 32, 32), input_channels=8
    )
    bundle = result.output_bundle
    assert bundle.shape == (16, 16, 16)
    assert bundle.channels == 16
    assert len(bundle.fibers) == 2
    assert (bundle.fibers[0].channel_start, bundle.fibers[0].channel_stop) == (0, 8)
    assert (bundle.fibers[1].channel_start, bundle.fibers[1].channel_stop) == (8, 16)
    assert np.allclose(bundle.fibers[0].supports[0].start, [0.0, 0.0, 0.0])
    assert np.allclose(bundle.fibers[1].supports[0].start, [1.0, 1.0, 1.0])

    cert = certify_single_lattice(bundle, patient_affine())
    assert cert.valid is False
    assert cert.unique_center_lattices == 2
    assert cert.max_intra_tensor_displacement_mm == pytest.approx(
        np.linalg.norm([0.7, 0.7, 5.0])
    )


def test_dense_conv_after_factorized_reduce_produces_multi_support_output_channels():
    model = ReduceThenMix()
    result = compile_fx_bundle_geometry(
        model, input_shape=(32, 32, 32), input_channels=8
    )
    bundle = result.output_bundle
    assert bundle.shape == (16, 16, 16)
    assert bundle.channels == 24
    assert len(bundle.fibers) == 1
    assert len(bundle.fibers[0].supports) == 2

    cert = certify_single_lattice(bundle, patient_affine())
    assert cert.valid is False
    assert cert.unique_center_lattices == 2


def test_bundle_compiler_allows_aligned_pointwise_add():
    model = TwoAlignedBranches()
    result = compile_fx_bundle_geometry(
        model, input_shape=(12, 12, 12), input_channels=4
    )
    cert = certify_single_lattice(result.output_bundle, np.eye(4))
    assert cert.valid


def test_bundle_compiler_rejects_misaligned_pointwise_add():
    model = TwoMisalignedBranches()
    x = torch.zeros(1, 4, 12, 12, 12)
    assert model.a(x).shape == model.b(model.pad(x)).shape
    with pytest.raises(BundleGeometryError, match="pointwise addition requires matching"):
        compile_fx_bundle_geometry(
            model, input_shape=(12, 12, 12), input_channels=4
        )
