from __future__ import annotations

import numpy as np

from research.medical_geometry.prompt_representation import sample_tight_aabb_iou


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
