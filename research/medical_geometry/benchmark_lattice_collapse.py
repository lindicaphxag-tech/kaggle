from __future__ import annotations

import json

import numpy as np

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.lattice_bundle import (
    ChannelFiber,
    LatticeBundle,
    bundle_from_lattice,
    certify_parallel_lattice_collapse,
    factorized_reduce_bundle,
    shift_lattice_origin,
)


def anisotropic_affine() -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag([0.7, 0.7, 5.0])
    return affine


source = bundle_from_lattice(16, FeatureLattice.input_grid((24, 24, 24)))
two_phase = factorized_reduce_bundle(source, out_channels=32)
two = certify_parallel_lattice_collapse(two_phase, anisotropic_affine())

base = FeatureLattice.input_grid((12, 12, 12))
supports = tuple(
    shift_lattice_origin(base, (x, y, z))
    for x in (0, 1)
    for y in (0, 1)
    for z in (0, 1)
)
eight_bundle = LatticeBundle(
    base.shape,
    8,
    tuple(
        ChannelFiber(i, i + 1, (support,), f'phase-{i}')
        for i, support in enumerate(supports)
    ),
)
eight = certify_parallel_lattice_collapse(eight_bundle, np.eye(4))

report = {
    'schema_version': 1,
    'claim': (
        'For patient-space support lattices with one shared linear part, the '
        'best single-affine worst-case collapse error equals the minimum-'
        'enclosing-ball radius of their translation vectors.'
    ),
    'two_phase_anisotropic': {
        'spacing_mm': [0.7, 0.7, 5.0],
        'num_lattices': two.num_unique_lattices,
        'optimal_translation_mm': two.optimal_translation_mm.tolist(),
        'exact_minimax_radius_mm': two.exact_minimax_radius_mm,
        'diameter_lower_bound_mm': two.diameter_lower_bound_mm,
    },
    'eight_phase_unit_cube': {
        'num_lattices': eight.num_unique_lattices,
        'optimal_translation_mm': eight.optimal_translation_mm.tolist(),
        'exact_minimax_radius_mm': eight.exact_minimax_radius_mm,
        'diameter_lower_bound_mm': eight.diameter_lower_bound_mm,
        'analytic_expected_radius': float(np.sqrt(3.0) / 2.0),
    },
    'claim_boundary': (
        'This theorem applies only when all support affines share one linear '
        'part. Minimum enclosing balls are classical computational geometry; '
        'the contribution is their use as an exact patient-lattice semantic '
        'collapse certificate.'
    ),
}

assert abs(
    report['two_phase_anisotropic']['exact_minimax_radius_mm']
    - report['two_phase_anisotropic']['diameter_lower_bound_mm']
) < 1e-12
assert abs(
    report['eight_phase_unit_cube']['exact_minimax_radius_mm']
    - report['eight_phase_unit_cube']['analytic_expected_radius']
) < 1e-12

print(json.dumps(report, indent=2, sort_keys=True))
