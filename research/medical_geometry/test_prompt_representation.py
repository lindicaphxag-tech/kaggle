import numpy as np

from research.medical_geometry.prompt_representation import (
    Exactness,
    PromptPrimitive,
    certify_aabb_approximation_debt,
    certify_aabb_representation,
    has_orthogonal_columns,
    is_similarity,
    minimal_exact_lift_for_aabb,
    sample_tight_aabb_iou,
)


def _rotation(degrees):
    t = np.deg2rad(degrees)
    return np.array(
        [[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]]
    )


def test_45_degree_rotation_has_half_tight_aabb_prompt_iou():
    cert = certify_aabb_representation(_rotation(45.0), np.ones(3))
    assert not cert.closed_as_aabb
    np.testing.assert_allclose(cert.exact_region_volume, 1.0, atol=1e-12)
    np.testing.assert_allclose(cert.tight_aabb_volume, 2.0, atol=1e-12)
    np.testing.assert_allclose(cert.overcoverage_ratio, 1.0, atol=1e-12)
    np.testing.assert_allclose(cert.tight_aabb_iou, 0.5, atol=1e-12)


def test_shear_has_analytic_two_thirds_prompt_iou():
    shear = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cert = certify_aabb_representation(shear, np.ones(3))
    np.testing.assert_allclose(cert.overcoverage_ratio, 0.5, atol=1e-12)
    np.testing.assert_allclose(cert.tight_aabb_iou, 2.0 / 3.0, atol=1e-12)


def test_instance_obb_representability_differs_from_global_obb_closure():
    anisotropic = np.diag([1.0, 2.0, 3.0])
    assert has_orthogonal_columns(anisotropic)
    assert not is_similarity(anisotropic)


def test_minimal_exact_lift_hierarchy():
    supported = {
        PromptPrimitive.AABB,
        PromptPrimitive.OBB,
        PromptPrimitive.PARALLELEPIPED,
        PromptPrimitive.MASK,
    }
    axis = minimal_exact_lift_for_aabb(np.diag([-1.0, 2.0, 0.5]), supported=supported)
    assert axis.target is PromptPrimitive.AABB
    assert axis.complexity_level == 0

    obb = minimal_exact_lift_for_aabb(
        _rotation(31.0) @ np.diag([1.0, 2.0, 0.5]), supported=supported
    )
    assert obb.target is PromptPrimitive.OBB
    assert obb.exactness is Exactness.CONTINUUM
    assert obb.complexity_level == 1

    shear = np.array([[1.0, 0.35, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]])
    para = minimal_exact_lift_for_aabb(shear, supported=supported)
    assert para.target is PromptPrimitive.PARALLELEPIPED
    assert para.complexity_level == 2

    mask = minimal_exact_lift_for_aabb(
        shear, supported={PromptPrimitive.AABB, PromptPrimitive.MASK}
    )
    assert mask.target is PromptPrimitive.MASK
    assert mask.exactness is Exactness.TARGET_GRID
    assert mask.complexity_level == 3


def test_sampled_fidelity_tracks_analytic_rotation_curve():
    for degrees in range(0, 46, 5):
        result = sample_tight_aabb_iou(_rotation(degrees), np.ones(3), samples_per_axis=65)
        assert result.absolute_error < 0.002


def test_inverse_rotation_round_trip_has_quarter_eager_prompt_iou():
    r = _rotation(45.0)
    cert = certify_aabb_approximation_debt([r, r.T], np.ones(3))
    np.testing.assert_allclose(cert.total_linear_map, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(cert.direct_target_widths, np.ones(3), atol=1e-12)
    np.testing.assert_allclose(cert.eager_target_widths, [2.0, 2.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(cert.direct_prompt_iou, 1.0, atol=1e-12)
    np.testing.assert_allclose(cert.eager_prompt_iou, 0.25, atol=1e-12)
    np.testing.assert_allclose(cert.path_overcoverage_ratio, 3.0, atol=1e-12)
    np.testing.assert_allclose(cert.avoidable_prompt_iou_loss, 0.75, atol=1e-12)
    assert cert.should_defer_projection


def test_one_stage_has_zero_placement_debt():
    cert = certify_aabb_approximation_debt([_rotation(23.0)], np.ones(3))
    np.testing.assert_allclose(
        cert.eager_stagewise_aabb_volume, cert.direct_tight_aabb_volume, atol=1e-12
    )
    assert not cert.should_defer_projection


def test_eager_chain_never_beats_deferred_projection():
    maps = [
        np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.1], [0.0, 0.0, 1.0]]),
        _rotation(17.0) @ np.diag([1.2, 0.8, 1.1]),
        np.array([[1.0, 0.0, -0.15], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    cert = certify_aabb_approximation_debt(maps, np.array([2.0, 3.0, 4.0]))
    assert np.all(cert.eager_target_widths + 1e-12 >= cert.direct_target_widths)
    assert cert.eager_stagewise_aabb_volume + 1e-12 >= cert.direct_tight_aabb_volume
    assert cert.eager_prompt_iou <= cert.direct_prompt_iou + 1e-12
