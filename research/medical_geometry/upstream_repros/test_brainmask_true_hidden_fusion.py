"""Structural controls for authentic-hidden-feature experimental cut."""
import numpy as np
import pytest

from research.medical_geometry.upstream_repros.brainmask_true_hidden_fusion import (
    orient_hidden_to_acquisition,
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
