"""Execute the actual two-phase FX graph before/after typed patient-space repair."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.repair_placement_execute import (
    rewrite_two_phase_fusion,
)
from research.medical_geometry.torch_fx_bundle import (
    compile_fx_bundle_geometry,
)
from research.medical_geometry.lattice_bundle import certify_single_lattice


class PhaseMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Conv3d(2, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.mix.weight[0, 0, 0, 0, 0] = 1.0
            self.mix.weight[0, 1, 0, 0, 0] = -1.0

    def forward(self, x):
        even = x[:, :, 0::2, ::2, ::2]
        odd = x[:, :, 1::2, 1::2, 1::2]
        return self.mix(torch.cat((even, odd), dim=1))


def fixture():
    model = PhaseMix().eval().requires_grad_(False)
    # Real patient mm chart; field is affine in x patient-mm, so H=0 is an
    # externally proven mathematical premise for THIS constructed example.
    affine = np.diag([0.7, 0.7, 5.0, 1.0])
    target = FeatureLattice(
        (3, 3, 3), np.full(3, 2.0), np.full(3, 1.0), np.ones(3)
    )
    args = dict(
        input_shape=(8, 8, 8),
        target=target,
        input_patient_affine_mm=affine,
        patient_frame_uid="synthetic-test-only-not-a-dicom-uid",
        world_hessian_bounds_by_channel_start={0: 0.0, 1: 0.0},
        maximum_output_value_error=1e-4,
    )
    return model, args


def test_real_fx_phase_centers_are_not_fabricated_as_one_affine():
    model, args = fixture()
    r = compile_fx_bundle_geometry(
        model, input_shape=(8, 8, 8), input_channels=1
    )
    a, b = r.bundles["getitem"], r.bundles["getitem_1"]
    assert a.shape == b.shape == (4, 4, 4)
    assert np.array_equal(a.fibers[0].supports[0].jump, [2, 2, 2])
    assert np.array_equal(b.fibers[0].supports[0].jump, [2, 2, 2])
    assert np.array_equal(a.fibers[0].supports[0].start, [0, 0, 0])
    assert np.array_equal(b.fibers[0].supports[0].start, [1, 1, 1])
    assert not certify_single_lattice(r.bundles["cat"], args["input_patient_affine_mm"]).valid


def test_pre_mix_rewrite_executes_and_matches_analytic_colocated_target():
    model, args = fixture()
    pos = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1)
    x = pos.expand(1, 1, 8, 8, 8).clone() * 0.7
    original = model(x)
    assert original.shape == (1, 1, 4, 4, 4)
    assert torch.allclose(original, torch.full_like(original, -0.7), atol=1e-6)
    rewritten = rewrite_two_phase_fusion(model, **args)
    y = rewritten.graph_module(x)
    assert y.shape == (1, 1, 3, 3, 3)
    assert torch.allclose(y, torch.zeros_like(y), atol=2e-6)
    # Input model is left alone: repair is inserted in a separate traced GM.
    assert torch.equal(original, model(x))
    assert rewritten.selected_sites == ("cat",)
    assert rewritten.conditional_per_output_value_error == (0.0,)
    assert len(rewritten.certificate_digest) == 64
    assert rewritten.output_support.shape == args["target"].shape
    assert np.all(rewritten.output_support.receptive_field >= [1, 1, 1])


def test_refuse_missing_curvature_and_unsupported_extrapolation():
    model, args = fixture()
    with pytest.raises(ValueError, match="missing or over-budget"):
        rewrite_two_phase_fusion(
            model, **{**args, "world_hessian_bounds_by_channel_start": {}}
        )
    with pytest.raises(ValueError, match="exceeds|over-budget"):
        rewrite_two_phase_fusion(
            model, **{**args,
                      "world_hessian_bounds_by_channel_start": {0: 2.0, 1: 2.0},
                      "maximum_output_value_error": 0.01}
        )
    bad_target = FeatureLattice(
        (4, 4, 4), np.full(3, 2.0), np.full(3, 1.0), np.ones(3)
    )
    with pytest.raises(ValueError, match="transport refused"):
        rewrite_two_phase_fusion(model, **{**args, "target": bad_target})


def test_refuse_trainable_model_and_altered_mixer_topology():
    model, args = fixture()
    with pytest.raises(ValueError, match="frozen eval"):
        rewrite_two_phase_fusion(PhaseMix(), **args)
    model.mix = nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False)
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="pointwise"):
        rewrite_two_phase_fusion(model, **args)


class ExtraSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Conv3d(2, 1, 1, bias=False)

    def forward(self, x):
        a = x[:, :, ::2, ::2, ::2]
        b = x[:, :, 1::2, 1::2, 1::2]
        z = torch.cat((a, b), dim=1)
        return self.mix(z) + a


def test_refuse_unmodeled_skip_consumer_and_bad_frame():
    model, args = fixture()
    skip = ExtraSkip().eval().requires_grad_(False)
    with pytest.raises(ValueError, match="unsupported topology"):
        rewrite_two_phase_fusion(skip, **args)
    with pytest.raises(ValueError, match="frame"):
        rewrite_two_phase_fusion(model, **{**args, "patient_frame_uid": ""})


def test_spatial_slicer_refuses_negative_or_zero_stride():
    class BadStride(nn.Module):
        def forward(self, x):
            return x[:, :, ::-1, :, :]
    with pytest.raises(Exception, match="positive"):
        compile_fx_bundle_geometry(
            BadStride(), input_shape=(8, 8, 8), input_channels=1
        )


def test_minimum_cost_switch_executes_distinct_real_fx_repair_positions():
    model, args = fixture()
    x = (torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1)
         .expand(1, 1, 8, 8, 8).clone() * 0.7)
    common = rewrite_two_phase_fusion(
        model, **args, authorized_site_costs={"cat": 5, "left": 3, "right": 3}
    )
    branches = rewrite_two_phase_fusion(
        model, **args, authorized_site_costs={"cat": 7, "left": 3, "right": 3}
    )
    assert common.selected_sites == ("cat",)
    assert branches.selected_sites == ("getitem", "getitem_1")
    assert common.certificate_digest != branches.certificate_digest
    common_result = common.graph_module(x)
    branch_result = branches.graph_module(x)
    assert tuple(common_result.shape) == (1, 1, 3, 3, 3)
    assert torch.allclose(common_result, branch_result, atol=1e-6)
    assert torch.allclose(branch_result, torch.zeros_like(branch_result), atol=2e-6)
    assert "branch_0" not in dict(common.graph_module.named_modules())
    assert "_clingeom_certified_pre_mix_branch_0" in dict(
        branches.graph_module.named_modules()
    )
    assert "_clingeom_certified_pre_mix_branch_1" in dict(
        branches.graph_module.named_modules()
    )
    assert branches.conditional_per_output_value_error == (
        common.conditional_per_output_value_error
    )
    # Original pretrained/learned weights must not be mutated by this rewrite.
    assert torch.allclose(model(x), torch.full((1, 1, 4, 4, 4), -0.7))


def test_cost_choice_refuses_unprotected_branch_or_malformed_cost():
    model, args = fixture()
    from research.medical_geometry.repair_placement import UnrepairablePath
    with pytest.raises(UnrepairablePath):
        rewrite_two_phase_fusion(
            model, **args, authorized_site_costs={"left": 2}
        )
    with pytest.raises(UnrepairablePath):
        rewrite_two_phase_fusion(
            model, **args, authorized_site_costs={"right": 2}
        )
    for costs in ({"cat": True}, {"cat": 0}, {"left": -1}, {"anything": 5}, {}):
        with pytest.raises(ValueError, match="positive integer costs"):
            rewrite_two_phase_fusion(
                model, **args, authorized_site_costs=costs
            )


def test_mismatched_fusion_weights_preserve_scheme_equivalence():
    model, args = fixture()
    with torch.no_grad():
        model.mix.weight[0, 0, 0, 0, 0] = 2.5
        model.mix.weight[0, 1, 0, 0, 0] = -4.5
    torch.manual_seed(17)
    x = torch.randn(2, 1, 8, 8, 8)
    a = rewrite_two_phase_fusion(
        model, **args, authorized_site_costs={"cat": 2, "left": 3, "right": 3}
    )
    b = rewrite_two_phase_fusion(
        model, **args, authorized_site_costs={"cat": 10, "left": 3, "right": 3}
    )
    assert torch.allclose(a.graph_module(x), b.graph_module(x), atol=2e-6, rtol=2e-6)
