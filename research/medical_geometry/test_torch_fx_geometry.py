import numpy as np
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.torch_fx_geometry import (
    GeometryMergeError,
    certify_merge,
    compile_fx_geometry,
    plan_pairwise_merge_repair,
)


class AlignedResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.branch(x) + x


class SameShapeMisalignedAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.good = nn.Conv3d(
            1,
            1,
            kernel_size=(1, 1, 3),
            stride=1,
            padding=(0, 0, 1),
            bias=False,
        )
        self.bad = nn.Sequential(
            # Pad one voxel on the left of W, then k=2 restores the original shape.
            # The resulting feature centers are shifted by -0.5 voxel along W.
            nn.ConstantPad3d((1, 0, 0, 0, 0, 0), 0.0),
            nn.Conv3d(
                1,
                1,
                kernel_size=(1, 1, 2),
                stride=1,
                padding=0,
                bias=False,
            ),
        )

    def forward(self, x):
        a = self.good(x)
        b = self.bad(x)
        return a + b


class SameShapeMisalignedCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv3d(
            1,
            1,
            kernel_size=(1, 1, 3),
            padding=(0, 0, 1),
            bias=False,
        )
        self.b = nn.Sequential(
            nn.ConstantPad3d((0, 1, 0, 0, 0, 0), 0.0),
            nn.Conv3d(
                1,
                1,
                kernel_size=(1, 1, 2),
                padding=0,
                bias=False,
            ),
        )

    def forward(self, x):
        return torch.cat([self.a(x), self.b(x)], dim=1)


def test_aligned_residual_merge_is_certified():
    model = AlignedResidual()
    result = compile_fx_geometry(
        model, input_shape=(16, 16, 16), input_affine=np.eye(4)
    )
    x = torch.zeros(1, 1, 16, 16, 16)
    assert tuple(model(x).shape[-3:]) == result.output_lattice.shape
    assert len(result.merges) == 1
    cert = result.merges[0]
    assert cert.valid
    assert cert.shape_equal
    assert cert.max_center_displacement_mm == 0.0


def test_same_shape_add_can_be_patient_space_misaligned():
    model = SameShapeMisalignedAdd()
    x = torch.zeros(1, 1, 16, 16, 16)
    a = model.good(x)
    b = model.bad(x)
    assert a.shape == b.shape

    with pytest.raises(
        GeometryMergeError,
        match="sample different patient-space locations",
    ):
        compile_fx_geometry(
            model, input_shape=(16, 16, 16), input_affine=np.eye(4)
        )


def test_same_shape_add_misalignment_scales_with_patient_spacing():
    model = SameShapeMisalignedAdd()
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag([0.7, 0.7, 5.0])

    with pytest.raises(GeometryMergeError, match="2.5 mm"):
        compile_fx_geometry(
            model, input_shape=(16, 16, 16), input_affine=affine
        )


def test_same_shape_concat_is_checked_too():
    model = SameShapeMisalignedCat()
    x = torch.zeros(1, 1, 12, 12, 12)
    assert model.a(x).shape[-3:] == model.b(x).shape[-3:]
    assert model(x).shape[1] == 2

    with pytest.raises(
        GeometryMergeError,
        match="sample different patient-space locations",
    ):
        compile_fx_geometry(
            model, input_shape=(12, 12, 12), input_affine=np.eye(4)
        )


def test_merge_tolerance_can_explicitly_accept_submillimeter_drift():
    model = SameShapeMisalignedAdd()
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag([0.5, 0.5, 0.5])
    result = compile_fx_geometry(
        model,
        input_shape=(16, 16, 16),
        input_affine=affine,
        merge_tolerance_mm=0.251,
    )
    cert = result.merges[0]
    assert cert.valid
    assert cert.max_center_displacement_mm == pytest.approx(0.25)


def test_merge_tolerance_is_patient_space_not_voxel_space():
    model = SameShapeMisalignedAdd()
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag([0.5, 0.5, 4.0])
    # The branch shift is along W/K, so 0.5 voxel means 2 mm here.
    with pytest.raises(GeometryMergeError, match="2 mm"):
        compile_fx_geometry(
            model,
            input_shape=(16, 16, 16),
            input_affine=affine,
            merge_tolerance_mm=0.3,
        )


def test_fractional_cell_merge_requires_interpolation_repair():
    model = SameShapeMisalignedAdd()
    # Compile branch lattices independently up to the merge by reproducing the
    # exact operator geometry through the FX checker internals in a permissive run.
    from research.medical_geometry.clingeom import FeatureLattice, propagate_conv_lattice

    reference = propagate_conv_lattice(
        FeatureLattice.input_grid((16, 16, 16)),
        kernel_size=(1, 1, 3),
        padding=(0, 0, 1),
    )
    padded = FeatureLattice(
        (16, 16, 17),
        np.ones(3),
        np.array([0.0, 0.0, -1.0]),
        np.ones(3),
    )
    candidate = propagate_conv_lattice(
        padded,
        kernel_size=(1, 1, 2),
        padding=0,
    )
    affine = np.eye(4)
    affine[:3, :3] = np.diag([0.7, 0.7, 5.0])
    plan = plan_pairwise_merge_repair(reference, candidate, affine)
    assert plan.kind == "resample"
    assert plan.exact_on_retained_domain is False
    assert plan.shift_cells == pytest.approx((0.0, 0.0, -0.5))
    assert plan.max_center_displacement_mm == pytest.approx(2.5)


def test_integer_cell_shift_gets_exact_common_overlap_crop_plan():
    from research.medical_geometry.clingeom import FeatureLattice

    reference = FeatureLattice(
        (8, 9, 10),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 3.0, 3.0]),
    )
    candidate = FeatureLattice(
        (8, 9, 10),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([3.0, 3.0, 2.0]),
    )
    plan = plan_pairwise_merge_repair(reference, candidate, np.eye(4))
    assert plan.kind == "crop_to_common_lattice"
    assert plan.exact_on_retained_domain
    assert plan.shift_cells == (0.0, 0.0, -1.0)
    assert plan.reference_slices == ((0, 8), (0, 9), (0, 9))
    assert plan.candidate_slices == ((0, 8), (0, 9), (1, 10))
    assert plan.overlap_shape == (8, 9, 9)


def test_aligned_pair_gets_noop_repair_plan():
    from research.medical_geometry.clingeom import FeatureLattice

    lattice = FeatureLattice.input_grid((5, 6, 7))
    plan = plan_pairwise_merge_repair(lattice, lattice, np.eye(4))
    assert plan.kind == "aligned"
    assert plan.exact_on_retained_domain
    assert plan.overlap_shape == (5, 6, 7)


def test_corner_certificate_equals_bruteforce_max_over_full_3d_grid():
    import itertools

    from research.medical_geometry.clingeom import FeatureLattice, feature_affine

    rng = np.random.default_rng(20261006)
    for case in range(32):
        shape = tuple(int(x) for x in rng.integers(2, 7, size=3))
        jump_a = rng.uniform(0.4, 3.0, size=3)
        jump_b = rng.uniform(0.4, 3.0, size=3)
        start_a = rng.uniform(-2.0, 2.0, size=3)
        start_b = rng.uniform(-2.0, 2.0, size=3)
        a = FeatureLattice(shape, jump_a, start_a, rng.uniform(1.0, 9.0, size=3))
        b = FeatureLattice(shape, jump_b, start_b, rng.uniform(1.0, 9.0, size=3))

        linear = rng.normal(size=(3, 3))
        while abs(np.linalg.det(linear)) < 0.2:
            linear = rng.normal(size=(3, 3))
        input_affine = np.eye(4)
        input_affine[:3, :3] = linear
        input_affine[:3, 3] = rng.uniform(-50.0, 50.0, size=3)

        cert = certify_merge(
            f"case_{case}", "add", [a, b], input_affine, tolerance_mm=1e12
        )
        A = feature_affine(input_affine, a)
        B = feature_affine(input_affine, b)
        brute = 0.0
        for idx in itertools.product(*(range(n) for n in shape)):
            x = np.ones(4)
            x[:3] = idx
            brute = max(brute, float(np.linalg.norm((A @ x)[:3] - (B @ x)[:3])))

        assert cert.max_center_displacement_mm == pytest.approx(
            brute, rel=1e-12, abs=1e-12
        )


class UNetLikeAlignedMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=False
        )

    def forward(self, x):
        low = self.pool(x)
        up = self.up(low)
        return torch.cat([x, up], dim=1)


class UNetLikeMisalignedMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True
        )

    def forward(self, x):
        low = self.pool(x)
        up = self.up(low)
        return torch.cat([x, up], dim=1)


def test_unet_like_align_corners_false_restores_skip_lattice():
    model = UNetLikeAlignedMerge()
    x = torch.zeros(1, 1, 16, 16, 16)
    y = model(x)
    assert y.shape[-3:] == (16, 16, 16)

    result = compile_fx_geometry(
        model, input_shape=(16, 16, 16), input_affine=np.eye(4)
    )
    assert len(result.merges) == 1
    assert result.merges[0].valid
    assert result.merges[0].max_center_displacement_mm == pytest.approx(0.0)


def test_unet_like_align_corners_true_is_same_shape_but_not_same_patient_grid():
    model = UNetLikeMisalignedMerge()
    x = torch.zeros(1, 1, 16, 16, 16)
    low = model.pool(x)
    up = model.up(low)
    assert up.shape[-3:] == x.shape[-3:]

    with pytest.raises(GeometryMergeError, match="sample different patient-space locations"):
        compile_fx_geometry(
            model, input_shape=(16, 16, 16), input_affine=np.eye(4)
        )


def test_unet_like_align_corners_true_error_is_mm_scaled():
    model = UNetLikeMisalignedMerge()
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag([0.7, 0.7, 5.0])

    # Each axis differs by 0.5 input voxel at an extreme corner.
    expected = float(np.linalg.norm([0.35, 0.35, 2.5]))
    with pytest.raises(GeometryMergeError) as exc:
        compile_fx_geometry(
            model, input_shape=(16, 16, 16), input_affine=affine
        )
    message = str(exc.value)
    assert f"{expected:.6g} mm" in message
