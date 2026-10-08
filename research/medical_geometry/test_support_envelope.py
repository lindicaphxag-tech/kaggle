"""Adversarial post-mix receptive-support provenance tests.

Center-aligning two sampled features cannot erase their different original
footprints, nor the extra source-grid neighbors used by interpolation.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.lattice_bundle import ChannelFiber, LatticeBundle
from research.medical_geometry.pre_mix_compiler import physically_align_then_mix
from research.medical_geometry.support_envelope import (
    interpolation_receptive_envelope, premix_output_support_envelope,
)


def _lattice(shape=(8,8,8),rf=(1,1,1),start=(0,0,0)):
    return FeatureLattice(tuple(shape), np.ones(3,dtype=np.float64),
                          np.asarray(start,dtype=np.float64),
                          np.asarray(rf,dtype=np.float64))


def _mixed_bundle(second_rf=(3,5,1)):
    left=_lattice(rf=(1,1,1))
    right=_lattice(rf=second_rf)
    return LatticeBundle((8,8,8),2,(
        ChannelFiber(0,1,(left,),"input-point"),
        ChannelFiber(1,2,(right,),"wide-anisotropic-context"),
    ))


def _mix():
    layer=nn.Conv3d(2,1,1,bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    return layer


def test_same_patient_centers_do_not_erase_different_receptive_fields():
    source=_mixed_bundle()
    target=_lattice(rf=(1,1,1))
    aligned=physically_align_then_mix(torch.ones(1,2,8,8,8),source,target,_mix())
    out=aligned.output_bundle.fibers[0].supports[0]
    np.testing.assert_allclose(out.start,target.start)
    np.testing.assert_allclose(out.jump,target.jump)
    np.testing.assert_allclose(out.receptive_field,[3,5,1])
    assert aligned.output_receptive_field_envelope == pytest.approx((3,5,1))
    assert out.receptive_field[1] > target.receptive_field[1]


def test_fractional_resampling_tracks_all_contributing_source_footprints():
    source=_mixed_bundle()
    target=_lattice(shape=(7,7,7),start=(0.5,0.5,0.5),rf=(1,1,1))
    result=physically_align_then_mix(torch.ones(1,2,8,8,8),source,target,_mix())
    rf=result.output_bundle.fibers[0].supports[0].receptive_field
    # Two half-cell neighbors are both used on each axis. Their union
    # enlarges the source receptive footprint by one source index unit.
    np.testing.assert_allclose(rf,[4,6,2],atol=1e-12)
    assert result.output_receptive_field_envelope == pytest.approx((4,6,2))
    assert result.tensor.shape == (1,1,7,7,7)


def test_analytic_neighbor_union_is_within_reported_envelope():
    src=_lattice(rf=(3,5,1))
    target=_lattice(shape=(7,7,7),start=(0.5,0.5,0.5))
    envelope=interpolation_receptive_envelope(src,target)
    for axis in range(3):
        rf=src.receptive_field[axis]
        for center in np.arange(7,dtype=float)+0.5:
            neighbor_centers=(np.floor(center),np.ceil(center))
            low=min(neighbor_centers)-rf/2
            high=max(neighbor_centers)+rf/2
            assert low >= center-envelope[axis]/2-1e-12
            assert high <= center+envelope[axis]/2+1e-12


def test_receptive_provenance_changes_certificate_digest_even_when_affines_match():
    target=_lattice(shape=(7,7,7),start=(0.5,0.5,0.5))
    tensor=torch.ones(1,2,8,8,8)
    mix=_mix()
    a=physically_align_then_mix(tensor,_mixed_bundle((3,5,1)),target,mix)
    b=physically_align_then_mix(tensor,_mixed_bundle((7,5,1)),target,mix)
    assert torch.equal(a.tensor,b.tensor)
    assert a.output_receptive_field_envelope != b.output_receptive_field_envelope
    assert a.conv_weight_provenance_sha256 != b.conv_weight_provenance_sha256


@pytest.mark.parametrize("rf",[(-1,1,1),(float("nan"),1,1),(float("inf"),1,1)])
def test_invalid_source_receptive_support_fails_before_mixing(rf):
    source=_mixed_bundle(rf)
    target=_lattice(shape=(7,7,7),start=(0.5,0.5,0.5))
    with pytest.raises(ValueError,match="receptive field"):
        physically_align_then_mix(torch.ones(1,2,8,8,8),source,target,_mix())


def test_nearest_node_source_support_does_not_inflate_for_exact_crop():
    source=_mixed_bundle()
    target=_lattice(shape=(6,6,6),start=(1,1,1))
    out=premix_output_support_envelope(source,target)
    np.testing.assert_allclose(out.receptive_field,[3,5,1])


def test_real_pytorch_anisotropic_pool_and_pointwise_mix_keep_wide_footprint():
    # Neither centered sampling coordinates nor equal NCDHW shapes can reveal
    # that one branch actually aggregates a wider anatomical neighborhood.
    x=torch.arange(8*8*8,dtype=torch.float32).reshape(1,1,8,8,8)
    point=x
    wide=F.avg_pool3d(x,kernel_size=(3,5,1),stride=1,padding=(1,2,0))
    assert point.shape==wide.shape
    bundle=_mixed_bundle(second_rf=(3,5,1))
    target=_lattice()
    result=physically_align_then_mix(
        torch.cat((point,wide),dim=1),bundle,target,_mix()
    )
    torch.testing.assert_close(result.tensor,point+wide,rtol=0,atol=0)
    np.testing.assert_allclose(
        result.output_bundle.fibers[0].supports[0].receptive_field,
        [3.0,5.0,1.0],
    )
    assert not np.allclose(
        result.output_bundle.fibers[0].supports[0].receptive_field,
        target.receptive_field
    )
