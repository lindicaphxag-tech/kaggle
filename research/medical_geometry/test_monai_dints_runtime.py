import math

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.dints_block import FactorizedReduceBlock

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.lattice_bundle import (
    bundle_from_lattice,
    certify_single_lattice,
    factorized_reduce_bundle,
)


def test_monai_dints_factorized_reduce_has_two_channel_sampling_phases():
    block = FactorizedReduceBlock(
        in_channel=1,
        out_channel=2,
        spatial_dims=3,
    )
    # Keep the production forward topology but expose its sampling relation
    # without activation/normalization changing the coordinate witness.
    block.act = nn.Identity()
    block.norm = nn.Identity()
    with torch.no_grad():
        block.conv_1.weight.fill_(1.0)
        block.conv_2.weight.fill_(1.0)

    d, h, w = torch.meshgrid(
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, dtype=torch.float32),
        indexing="ij",
    )
    x = (d + 10.0 * h + 100.0 * w)[None, None]
    y = block(x)

    assert y.shape == (1, 2, 4, 4, 4)
    phase0 = x[:, :, 0::2, 0::2, 0::2]
    phase1 = x[:, :, 1::2, 1::2, 1::2]
    assert torch.equal(y[:, 0:1], phase0)
    assert torch.equal(y[:, 1:2], phase1)
    assert torch.equal(y[:, 1:2] - y[:, 0:1], torch.full_like(phase0, 111.0))


def test_clingeom_bundle_matches_monai_dints_phase_structure():
    source = bundle_from_lattice(
        1,
        FeatureLattice.input_grid((8, 8, 8)),
        provenance="MONAI DiNTS input",
    )
    reduced = factorized_reduce_bundle(source, out_channels=2)

    assert reduced.shape == (4, 4, 4)
    assert len(reduced.fibers) == 2
    a = reduced.fibers[0].supports[0]
    b = reduced.fibers[1].supports[0]
    assert np.allclose(a.jump, [2, 2, 2])
    assert np.allclose(b.jump, [2, 2, 2])
    assert np.allclose(a.start, [0, 0, 0])
    assert np.allclose(b.start, [1, 1, 1])

    patient_affine = np.eye(4, dtype=np.float64)
    patient_affine[:3, :3] = np.diag([0.7, 0.7, 5.0])
    cert = certify_single_lattice(reduced, patient_affine)

    expected = math.sqrt(0.7**2 + 0.7**2 + 5.0**2)
    assert cert.valid is False
    assert math.isclose(cert.max_intra_tensor_displacement_mm, expected, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        cert.single_affine_minimax_lower_bound_mm,
        expected / 2.0,
        rel_tol=0,
        abs_tol=1e-12,
    )
