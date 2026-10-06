import numpy as np
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.fx_geometry import (
    PatientSpaceMergeError,
    compile_fx_geometry,
)
from research.medical_geometry.torch_geometry import UnsupportedGeometryOp


class AlignedCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv3d(1, 2, kernel_size=3, stride=2, padding=1)
        self.b = nn.Conv3d(1, 2, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        return torch.cat([self.a(x), self.b(x)], dim=1)


class MisalignedCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv3d(1, 2, kernel_size=3, stride=2, padding=1)
        self.b = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return torch.cat([self.a(x), self.b(x)], dim=1)


class MisalignedAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv3d(1, 1, kernel_size=3, stride=2, padding=1)
        self.b = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.a(x) + self.b(x)


def test_aligned_same_shape_cat_is_legal_even_with_different_receptive_fields():
    model = AlignedCat()
    x = torch.zeros(1, 1, 64, 64, 64)
    assert tuple(model(x).shape[-3:]) == (32, 32, 32)

    report = compile_fx_geometry(
        model, input_shape=(64, 64, 64), input_affine=np.eye(4)
    )
    assert report.repair_obligations == ()
    assert report.output_lattice.shape == (32, 32, 32)
    assert np.allclose(report.output_lattice.start, [0, 0, 0])
    # Context/RF differs across branches but feature centers agree.
    assert np.allclose(report.output_lattice.receptive_field, [3, 3, 3])


def test_generic_misaligned_channel_cat_preserves_fibers_instead_of_auto_repair():
    model = MisalignedCat()
    x = torch.zeros(1, 1, 64, 64, 64)
    assert tuple(model(x).shape[-3:]) == (32, 32, 32)
    with pytest.raises(UnsupportedGeometryOp, match="LatticeBundle"):
        compile_fx_geometry(
            model, input_shape=(64, 64, 64), input_affine=np.eye(4), strict=False
        )


def test_explicit_aligned_cat_contract_emits_atomic_repair():
    model = MisalignedCat()
    report = compile_fx_geometry(
        model,
        input_shape=(64, 64, 64),
        input_affine=np.eye(4),
        strict=False,
        require_channel_aligned_cat=True,
    )
    assert len(report.repair_obligations) == 1
    obligation = report.repair_obligations[0]
    assert obligation.merge_kind == "cat"
    assert obligation.bundle.target.shape == (31, 31, 31)
    assert obligation.bundle.left.kind == "exact_crop"
    assert obligation.bundle.right.kind == "resample"
    assert obligation.bundle.interpolation_count == 1
    assert report.output_lattice.shape == (31, 31, 31)


def test_strict_mode_rejects_tensor_valid_patient_space_invalid_cat():
    with pytest.raises(PatientSpaceMergeError, match="patient-space"):
        compile_fx_geometry(
            MisalignedCat(),
            input_shape=(64, 64, 64),
            input_affine=np.eye(4),
            strict=True,
            require_channel_aligned_cat=True,
        )


def test_tensor_valid_add_is_also_checked():
    x = torch.zeros(1, 1, 64, 64, 64)
    assert tuple(MisalignedAdd()(x).shape[-3:]) == (32, 32, 32)

    report = compile_fx_geometry(
        MisalignedAdd(),
        input_shape=(64, 64, 64),
        input_affine=np.eye(4),
        strict=False,
    )
    assert len(report.repair_obligations) == 1
    assert report.repair_obligations[0].merge_kind == "add"


def test_fx_compiler_refuses_unknown_spatial_operator():
    model = nn.Sequential(nn.AdaptiveAvgPool3d((8, 8, 8)))
    with pytest.raises(UnsupportedGeometryOp):
        compile_fx_geometry(
            model, input_shape=(31, 35, 39), input_affine=np.eye(4)
        )
