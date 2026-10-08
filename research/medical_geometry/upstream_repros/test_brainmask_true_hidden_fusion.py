"""Structural controls for authentic-hidden-feature experimental cut."""
import numpy as np
import pytest

from research.medical_geometry.upstream_repros.brainmask_true_hidden_fusion import (
    orient_hidden_to_acquisition,
    orient_acquisition_to_hidden,
)


@pytest.mark.parametrize("plane,axes", [
    ("ax", (0, 1, 2)), ("cor", (0, 2, 1)), ("sag", (2, 0, 1))
])
def test_exact_spatial_axes_only_preserve_channel_identity(plane, axes):
    x = np.arange(4 * 5 * 6 * 3, dtype=np.float32).reshape(4, 5, 6, 3)
    result = orient_hidden_to_acquisition(x, plane)
    np.testing.assert_array_equal(result, np.transpose(x, axes + (3,)))
    assert result.shape[-1] == 3
    assert np.shares_memory(result, x)


@pytest.mark.parametrize("plane", ["ax", "cor", "sag"])
def test_pointwise_frozen_classifier_commutes_with_orientation(plane):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(4, 5, 6, 3)).astype(np.float32)
    weights = np.array([0.23, -0.51, 0.76], np.float32)
    bias = np.float32(-0.27)
    # 1x1x1 trained-head topology, controlled numbers in the pure unit test.
    raw_head = np.einsum("ijkc,c->ijk", features, weights, optimize=False) + bias
    mapped_features = orient_hidden_to_acquisition(features, plane)
    mapped_head = np.einsum(
        "ijkc,c->ijk", mapped_features, weights, optimize=False
    ) + bias
    np.testing.assert_array_equal(
        mapped_head, orient_hidden_to_acquisition(raw_head[..., None], plane)[..., 0]
    )


def test_nonfinite_learned_hidden_fails_closed():
    x = np.ones((3, 4, 5, 7), np.float32)
    x[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        orient_hidden_to_acquisition(x, "cor")


def test_unknown_plane_fails_closed():
    with pytest.raises(ValueError, match="undocumented"):
        orient_hidden_to_acquisition(np.ones((3, 4, 5, 2)), "oblique")


def test_channel_semantics_not_implied_by_spatial_shape_agreement():
    x = np.ones((4, 5, 6, 2), np.float32)
    y = x.copy()
    y[..., 0] = -1
    # Exact charts / shapes do not make independently computed learned
    # feature values equivalent across orientations: an explicit nonclaim.
    assert x.shape == y.shape
    assert not np.array_equal(x, y)


def test_conv1x1_prehead_average_equals_same_frozen_head_logit_average():
    rng = np.random.default_rng(998)
    feats = rng.normal(size=(3, 4, 5, 6, 32)).astype(np.float64)
    weights = rng.normal(size=(32,))
    bias = -0.37
    aligned_mixed = np.mean(feats, axis=0)
    before_head = np.einsum("ijkc,c->ijk", aligned_mixed, weights) + bias
    each_head = np.einsum("nijkc,c->nijk", feats, weights) + bias
    after_head = np.mean(each_head, axis=0)
    np.testing.assert_allclose(before_head, after_head, rtol=1e-13, atol=1e-13)


def test_logit_mean_and_probability_mean_can_reverse_final_labels():
    logits = np.array([8.0, -3.0, -3.0], dtype=np.float64)
    logit_ensemble = 1 / (1 + np.exp(-np.mean(logits)))
    prob_ensemble = np.mean(1 / (1 + np.exp(-logits)))
    assert logit_ensemble > 0.5
    assert prob_ensemble < 0.5
    # This difference does not require learned-channel semantics at all.


@pytest.mark.parametrize("plane", ["ax", "cor", "sag"])
def test_native_canonical_nonlinear_cut_charts_are_two_sided_exact_inverses(plane):
    rng = np.random.default_rng(143)
    source = rng.normal(size=(4, 5, 6, 8)).astype(np.float32)
    mapped = orient_hidden_to_acquisition(source, plane)
    restored = orient_acquisition_to_hidden(mapped, plane)
    np.testing.assert_array_equal(restored, source)
    assert restored.shape[-1] == source.shape[-1]


def test_orientation_sensitive_3d_kernel_requires_operator_transport():
    source = np.zeros((4, 5, 6, 1), np.float32)
    source[1, 2, 3, 0] = 1.0

    def directional_spatial_conv(x):
        # Controlled anisotropic 3-D kernel: unit step on local axis 0.
        z = np.zeros(x.shape[:3], np.float32)
        z[1:] = x[:-1, :, :, 0]
        return z

    native_source = orient_acquisition_to_hidden(source, "sag")
    native_output = directional_spatial_conv(native_source)
    result_in_original_grid = orient_hidden_to_acquisition(
        native_output[..., None], "sag"
    )[..., 0]
    naive_untransported = directional_spatial_conv(source)
    assert not np.array_equal(result_in_original_grid, naive_untransported)
    assert np.sum(result_in_original_grid) == np.sum(naive_untransported) == 1
    # Controlled analytic unit test; real learned-kernel commutator is
    # independently computed from authentic pretrained weights by CI.


def test_acquisition_inverse_rejects_wrong_plane_or_nonfinite_values():
    with pytest.raises(ValueError, match="undocumented"):
        orient_acquisition_to_hidden(np.ones((4, 5, 6, 2)), "synthetic")
    arr = np.zeros((4, 5, 6, 2), dtype=np.float32)
    arr[2, 1, 4, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        orient_acquisition_to_hidden(arr, "sag")
