"""The execution boundary must not trust an unchecked lattice repair plan."""

import numpy as np
import pytest
import torch

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.repair_synthesis import (
    _make_plan, synthesize_lattice_repair, target_to_source_index_map,
)
from research.medical_geometry.torch_repair import LatticeRepair3d


def lattice(shape, jump, start):
    return FeatureLattice(
        tuple(shape), np.asarray(jump, dtype=float),
        np.asarray(start, dtype=float), np.ones(3, dtype=float),
    )


def test_runtime_rejects_input_shape_different_from_signed_source_grid():
    source = lattice((8, 8, 8), (2, 2, 2), (0, 0, 0))
    target = lattice((7, 7, 7), (2, 2, 2), (1.0, 1.0, 1.0))
    plan = synthesize_lattice_repair(source, target)
    assert plan.kind == 'resample'
    executor = LatticeRepair3d(plan)
    with pytest.raises(ValueError, match='source shape'):
        executor(torch.zeros(1, 1, 9, 8, 8))
    assert executor(torch.zeros(1, 1, 8, 8, 8)).shape == (1, 1, 7, 7, 7)


def test_runtime_rejects_signed_out_of_domain_plan_before_sampling():
    source = lattice((8, 8, 8), (2, 2, 2), (0, 0, 0))
    target = lattice((8, 8, 8), (2, 2, 2), (1.0, 0, 0))
    signed = _make_plan(
        kind='resample', source=source, target=target,
        mapping=target_to_source_index_map(source, target),
        exact=False, requires_interpolation=True,
        interpolation='trilinear', crop_start=None, crop_stop=None,
        reason='forged full-extent sample beyond valid source points',
    )
    with pytest.raises(ValueError, match='outside source'):
        LatticeRepair3d(signed)


def test_runtime_shape_validation_also_applies_to_exact_identity_and_crop():
    source = lattice((8, 8, 8), (2, 2, 2), (0, 0, 0))
    same = synthesize_lattice_repair(source, source)
    crop_to = lattice((6, 6, 6), (2, 2, 2), (2, 2, 2))
    crop = synthesize_lattice_repair(source, crop_to)
    for plan in (same, crop):
        with pytest.raises(ValueError, match='source shape'):
            LatticeRepair3d(plan)(torch.ones(1, 1, 9, 8, 8))
