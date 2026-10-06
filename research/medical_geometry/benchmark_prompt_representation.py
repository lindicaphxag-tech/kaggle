from __future__ import annotations

import numpy as np

from research.medical_geometry.prompt_representation import (
    certify_aabb_approximation_debt,
    sample_tight_aabb_iou,
)


def rotation(degrees: float) -> np.ndarray:
    t = np.deg2rad(degrees)
    return np.array(
        [[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]]
    )


print('rotation_deg,analytic_iou,sampled_iou_n65,absolute_error')
max_error = 0.0
for degrees in range(0, 46, 5):
    result = sample_tight_aabb_iou(rotation(degrees), np.ones(3), samples_per_axis=65)
    max_error = max(max_error, result.absolute_error)
    print(
        f'{degrees},{result.analytic_iou:.9f},{result.sampled_iou:.9f},'
        f'{result.absolute_error:.9f}'
    )
assert max_error < 0.002
print(f'max_absolute_error={max_error:.9f}')


r = rotation(45.0)
debt = certify_aabb_approximation_debt([r, r.T], np.ones(3))
np.testing.assert_allclose(debt.total_linear_map, np.eye(3), atol=1e-12)
np.testing.assert_allclose(debt.direct_prompt_iou, 1.0, atol=1e-12)
np.testing.assert_allclose(debt.eager_prompt_iou, 0.25, atol=1e-12)
print(f'roundtrip_45_direct_iou={debt.direct_prompt_iou:.9f}')
print(f'roundtrip_45_eager_iou={debt.eager_prompt_iou:.9f}')
print(f'roundtrip_45_avoidable_iou_loss={debt.avoidable_prompt_iou_loss:.9f}')

print(f'roundtrip_45_log_volume_debt={debt.log_volume_debt:.9f}')
print(f'roundtrip_45_fidelity_retention={debt.fidelity_retention:.9f}')
np.testing.assert_allclose(debt.log_volume_debt, np.log(4.0), atol=1e-12)
np.testing.assert_allclose(debt.fidelity_retention, 0.25, atol=1e-12)
