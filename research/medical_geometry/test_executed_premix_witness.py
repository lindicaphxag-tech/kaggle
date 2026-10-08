"""Actual CPU fusion adversarial attestation, not just interpolation checks."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.lattice_bundle import ChannelFiber, LatticeBundle
from research.medical_geometry.pre_mix_compiler import physically_align_then_mix
from research.medical_geometry.executed_premix_witness import (
    certify_executed_premix,
)


def _case(dtype=torch.float32):
    first = FeatureLattice(
        (8, 1, 1), np.ones(3), np.zeros(3), np.ones(3)
    )
    second = FeatureLattice(
        (8, 1, 1), np.ones(3), np.array([0.2, 0., 0.]), np.ones(3)
    )
    bundle = LatticeBundle((8, 1, 1), 2, (
        ChannelFiber(0, 1, (first,), "first medical feature phase"),
        ChannelFiber(1, 2, (second,), "second medical feature phase"),
    ))
    target = FeatureLattice(
        (3, 1, 1), np.ones(3),
        np.array([.37, 0., 0.]), np.ones(3),
    )
    x = torch.zeros((1, 2, 8, 1, 1), dtype=dtype)
    x[0, 0, :, 0, 0] = torch.arange(8, dtype=dtype) * 1e7
    x[0, 1, :, 0, 0] = torch.arange(8, dtype=dtype).square() * 1e6
    mix = nn.Conv3d(2, 1, 1, bias=True).to(dtype)
    with torch.no_grad():
        mix.weight[0, 0, 0, 0, 0] = .37
        mix.weight[0, 1, 0, 0, 0] = -.91
        mix.bias[0] = .13
    return x, bundle, target, mix


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_reference_covers_actual_grid_sample_and_dense_conv(dtype):
    x, bundle, target, mix = _case(dtype)
    real = physically_align_then_mix(x, bundle, target, mix)
    result = certify_executed_premix(
        x, bundle, target, mix, output=real.tensor,
        maximum_audited_execution_error=0,
    )
    assert result.exact_declared_float_fusion_residual > 0
    assert result.within_audited_execution_budget is False
    assert result.audited_output_elements == 3
    assert result.output_tensor_sha256
    assert result.compiler_provenance_sha256 == real.conv_weight_provenance_sha256
    assert "future inputs" in result.proof_scope
    # Exact replay was required before the rational proof is authorized.


def test_explicit_numerical_budget_is_snapshot_scoped():
    x, bundle, target, mix = _case()
    result = certify_executed_premix(
        x, bundle, target, mix, maximum_audited_execution_error=1e6
    )
    assert result.within_audited_execution_budget is True
    assert 0 < result.exact_declared_float_fusion_residual < 1e6


def test_exact_integer_aligned_two_channel_mix_can_have_zero_residual():
    first = FeatureLattice.input_grid((2, 1, 1))
    bundle = LatticeBundle((2, 1, 1), 2, (
        ChannelFiber(0, 1, (first,), "a"),
        ChannelFiber(1, 2, (first,), "b"),
    ))
    x = torch.tensor([1., 3., 2., 5.]).reshape(1, 2, 2, 1, 1)
    mix = nn.Conv3d(2, 1, 1, bias=False)
    with torch.no_grad():
        mix.weight.fill_(1.0)
    result = certify_executed_premix(
        x, bundle, first, mix, maximum_audited_execution_error=0,
    )
    assert result.exact_declared_float_fusion_residual == 0
    assert result.within_audited_execution_budget is True


def test_forged_output_fails_original_cpu_execution_replay():
    x, bundle, target, mix = _case()
    y = physically_align_then_mix(x, bundle, target, mix).tensor
    with pytest.raises(ValueError, match="execution replay"):
        certify_executed_premix(x, bundle, target, mix, output=y + 1.)


def test_modified_stored_weights_require_new_model_and_output_fingerprints():
    x, bundle, target, mix = _case()
    first = certify_executed_premix(x, bundle, target, mix)
    with torch.no_grad():
        mix.weight[0, 0, 0, 0, 0] += .25
    second = certify_executed_premix(x, bundle, target, mix)
    assert first.source_tensor_sha256 == second.source_tensor_sha256
    assert first.model_parameters_sha256 != second.model_parameters_sha256
    assert first.output_tensor_sha256 != second.output_tensor_sha256
    assert first.compiler_provenance_sha256 != second.compiler_provenance_sha256


def test_exact_work_quota_and_audited_budget_invalid_values_fail_closed():
    x, bundle, target, mix = _case()
    with pytest.raises(ValueError, match="work budget"):
        certify_executed_premix(
            x, bundle, target, mix, max_exact_operations=1
        )
    with pytest.raises(ValueError, match="budget"):
        certify_executed_premix(
            x, bundle, target, mix, maximum_audited_execution_error=-1
        )
    with pytest.raises(ValueError, match="positive exact"):
        certify_executed_premix(
            x, bundle, target, mix, max_exact_operations=False
        )


def test_gpu_or_dtype_change_must_not_inherit_cpu_fp32_witness():
    x, bundle, target, mix = _case()
    with pytest.raises(ValueError, match="scalar dtype"):
        certify_executed_premix(x.double(), bundle, target, mix)
    with pytest.raises(ValueError, match="finite CPU"):
        certify_executed_premix(x * float("nan"), bundle, target, mix)
