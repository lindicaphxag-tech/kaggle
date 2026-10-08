"""Conservative physical-support envelope for typed channel-fiber interpolation.

Matching sample-center affines does not imply matching receptive support.
Under separable trilinear regridding each target value depends on at most
two source centers on each axis. Its support is the union of the receptive
footprints of EVERY contributing source feature, not the caller-declared
target receptive field. This file returns a conservative uniform symmetric
box containing that union at every target center.

This is a structural support envelope, NOT recovery of an identical
continuous feature function, a learned semantic correspondence, or a
clinical guarantee.
"""

from __future__ import annotations

import numpy as np

from .clingeom import FeatureLattice
from .lattice_bundle import LatticeBundle
from .repair_synthesis import target_to_source_index_map


def _check_rf(lattice: FeatureLattice, *, label: str) -> np.ndarray:
    rf=np.asarray(lattice.receptive_field,dtype=np.float64)
    if rf.shape!=(3,) or not np.all(np.isfinite(rf)) or np.any(rf<=0):
        raise ValueError(f"{label} receptive field must be finite and positive")
    for name in ("jump","start"):
        v=np.asarray(getattr(lattice,name),dtype=np.float64)
        if v.shape!=(3,) or not np.all(np.isfinite(v)):
            raise ValueError(f"{label} {name} must contain finite 3-D coordinates")
        if name=="jump" and np.any(v<=0):
            raise ValueError(f"{label} jump must be positive")
    return rf


def interpolation_receptive_envelope(
    source: FeatureLattice,
    target: FeatureLattice,
) -> np.ndarray:
    """Upper-bound the full source footprint contributing at target centers.

    Both receptive fields and source jumps are in INPUT-VOXEL index units.
    Exact grid-node sampling adds zero neighbor span. At an interior
    fractional index alpha, both floor and ceil values contribute, giving
    maximum center separation max(alpha,1-alpha) * source.jump.
    The returned target-centered box has width
    source.receptive_field + 2*max_center_displacement.
    The target RF field is deliberately ignored: it cannot overwrite
    actual source support provenance.
    """
    rf=_check_rf(source,label="source")
    _check_rf(target,label="target")
    m=target_to_source_index_map(source,target)
    if m.shape!=(4,4) or not np.all(np.isfinite(m)):
        raise ValueError("nonfinite source-index transform")
    if not np.allclose(m[:3,:3],np.diag(np.diag(m[:3,:3])),atol=1e-12,rtol=0):
        raise ValueError("nonseparable grid map has no v1 support envelope")
    result=rf.copy()
    for axis,count in enumerate(target.shape):
        coords=m[axis,axis]*np.arange(count,dtype=np.float64)+m[axis,3]
        if (np.any(coords < -1e-9)
                or np.any(coords > float(source.shape[axis]-1)+1e-9)):
            raise ValueError("support envelope requires in-domain source sampling")
        alpha=np.mod(coords,1.0)
        # A full-cell neighbor is not used at an exactly integer sample.
        # A tiny but nonzero interpolation weight still expands the support.
        radius_cells=np.where(alpha==0,0.0,np.maximum(alpha,1.0-alpha))
        result[axis]+=2.0*float(source.jump[axis])*float(radius_cells.max(initial=0.0))
    return result


def premix_output_support_envelope(
    source_bundle: LatticeBundle,
    target: FeatureLattice,
) -> FeatureLattice:
    """Construct the uniform output support envelope across all mixed fibers.

    Dense mixing is admitted ONLY after each point-valued fiber has been
    mapped to target centers. A 1×1 dense mixer may depend on any input
    fiber, so its output receives the componentwise union envelope.
    """
    envelopes=[]
    for fiber in source_bundle.fibers:
        if len(fiber.supports)!=1:
            raise ValueError("multi-support fibers cannot obtain one center support envelope")
        envelopes.append(interpolation_receptive_envelope(fiber.supports[0],target))
    if not envelopes:
        raise ValueError("empty source bundle")
    upper=np.max(np.stack(envelopes),axis=0)
    return FeatureLattice(
        target.shape, np.asarray(target.jump,dtype=np.float64).copy(),
        np.asarray(target.start,dtype=np.float64).copy(),upper,
    )

