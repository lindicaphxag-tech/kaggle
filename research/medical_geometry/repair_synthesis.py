"""ClinGeom constructive repair synthesis for patient-space feature lattices.

The compiler distinguishes exact representation repairs from interpolating
repairs.  Exact repairs are preferred and are only emitted when target feature
centers are a subset of source feature centers.  Otherwise the compiler emits
an explicit resampling map (or refuses when the semantic role has no safe
interpolation rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal

import numpy as np

from .clingeom import FeatureLattice


RepairKind = Literal["identity", "exact_crop", "resample", "refuse"]
RepairRole = Literal["feature", "intensity", "probability", "label", "landmark"]


@dataclass(frozen=True)
class LatticeRepairPlan:
    kind: RepairKind
    source_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]
    target_to_source_index: np.ndarray
    exact: bool
    requires_interpolation: bool
    interpolation: str | None
    crop_start: tuple[int, int, int] | None
    crop_stop: tuple[int, int, int] | None
    reason: str
    certificate_digest: str


def target_to_source_index_map(
    source: FeatureLattice,
    target: FeatureLattice,
) -> np.ndarray:
    """Map target feature indices to source feature indices.

    Feature lattices are represented in the same input-voxel coordinate frame:

        x = start + jump * i.

    Therefore:

        i_source = diag(jump_target / jump_source) i_target
                   + (start_target - start_source) / jump_source.
    """

    sj = np.asarray(source.jump, dtype=np.float64)
    tj = np.asarray(target.jump, dtype=np.float64)
    ss = np.asarray(source.start, dtype=np.float64)
    ts = np.asarray(target.start, dtype=np.float64)
    if np.any(~np.isfinite(sj)) or np.any(~np.isfinite(tj)):
        raise ValueError("lattice jump must be finite")
    if np.any(sj <= 0) or np.any(tj <= 0):
        raise ValueError("lattice jump must be strictly positive")

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.diag(tj / sj)
    M[:3, 3] = (ts - ss) / sj
    return M


def _canonical_digest(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(text.encode("utf-8")).hexdigest()


def _interpolation_for_role(role: RepairRole) -> str | None:
    if role == "label":
        return "nearest"
    if role in {"feature", "intensity", "probability"}:
        return "trilinear"
    if role == "landmark":
        return None
    raise ValueError(f"unsupported semantic role: {role}")


def _make_plan(
    *,
    kind: RepairKind,
    source: FeatureLattice,
    target: FeatureLattice,
    mapping: np.ndarray,
    exact: bool,
    requires_interpolation: bool,
    interpolation: str | None,
    crop_start: tuple[int, int, int] | None,
    crop_stop: tuple[int, int, int] | None,
    reason: str,
) -> LatticeRepairPlan:
    payload = {
        "kind": kind,
        "source_shape": list(source.shape),
        "target_shape": list(target.shape),
        "source_jump": np.asarray(source.jump, dtype=float).round(12).tolist(),
        "source_start": np.asarray(source.start, dtype=float).round(12).tolist(),
        "target_jump": np.asarray(target.jump, dtype=float).round(12).tolist(),
        "target_start": np.asarray(target.start, dtype=float).round(12).tolist(),
        "target_to_source_index": np.asarray(mapping, dtype=float).round(12).tolist(),
        "exact": exact,
        "requires_interpolation": requires_interpolation,
        "interpolation": interpolation,
        "crop_start": None if crop_start is None else list(crop_start),
        "crop_stop": None if crop_stop is None else list(crop_stop),
        "reason": reason,
    }
    return LatticeRepairPlan(
        kind=kind,
        source_shape=source.shape,
        target_shape=target.shape,
        target_to_source_index=np.asarray(mapping, dtype=np.float64),
        exact=exact,
        requires_interpolation=requires_interpolation,
        interpolation=interpolation,
        crop_start=crop_start,
        crop_stop=crop_stop,
        reason=reason,
        certificate_digest=_canonical_digest(payload),
    )


def synthesize_lattice_repair(
    source: FeatureLattice,
    target: FeatureLattice,
    *,
    role: RepairRole = "feature",
    atol: float = 1e-9,
) -> LatticeRepairPlan:
    """Construct the least-destructive certified repair from source to target.

    Preference order:

    1. identity;
    2. exact integer crop (zero interpolation);
    3. explicit resampling with role-aware interpolation;
    4. refusal if interpolation would be semantically undefined.

    Padding/extrapolation is intentionally not called an exact repair: if target
    feature centers fall outside the source lattice, values do not already
    exist and must be synthesized.
    """

    M = target_to_source_index_map(source, target)

    same_jump = np.allclose(source.jump, target.jump, atol=atol, rtol=0)
    same_start = np.allclose(source.start, target.start, atol=atol, rtol=0)
    same_shape = tuple(source.shape) == tuple(target.shape)

    if same_jump and same_start and same_shape:
        return _make_plan(
            kind="identity",
            source=source,
            target=target,
            mapping=M,
            exact=True,
            requires_interpolation=False,
            interpolation=None,
            crop_start=(0, 0, 0),
            crop_stop=tuple(int(x) for x in source.shape),
            reason="source and target feature lattices are identical",
        )

    if same_jump:
        offset = M[:3, 3]
        rounded = np.rint(offset).astype(int)
        if np.allclose(offset, rounded, atol=atol, rtol=0):
            target_shape = np.asarray(target.shape, dtype=int)
            source_shape = np.asarray(source.shape, dtype=int)
            start = rounded
            stop = start + target_shape
            if np.all(start >= 0) and np.all(stop <= source_shape):
                return _make_plan(
                    kind="exact_crop",
                    source=source,
                    target=target,
                    mapping=M,
                    exact=True,
                    requires_interpolation=False,
                    interpolation=None,
                    crop_start=tuple(int(x) for x in start),
                    crop_stop=tuple(int(x) for x in stop),
                    reason="target lattice is an exact integer-index subset of source lattice",
                )

    interpolation = _interpolation_for_role(role)
    if interpolation is None:
        return _make_plan(
            kind="refuse",
            source=source,
            target=target,
            mapping=M,
            exact=False,
            requires_interpolation=True,
            interpolation=None,
            crop_start=None,
            crop_stop=None,
            reason=(
                f"target lattice requires interpolation, but semantic role {role!r} "
                "has no certified interpolation rule"
            ),
        )

    # Refuse extrapolation by default. Interpolation is only authorized when
    # every target feature center maps inside the source sampling lattice.
    target_shape = np.asarray(target.shape, dtype=np.float64)
    source_shape = np.asarray(source.shape, dtype=np.float64)
    corners = np.array(
        [
            [i, j, k, 1.0]
            for i in (0.0, target_shape[0] - 1.0)
            for j in (0.0, target_shape[1] - 1.0)
            for k in (0.0, target_shape[2] - 1.0)
        ],
        dtype=np.float64,
    )
    mapped = (M @ corners.T).T[:, :3]
    in_bounds = bool(
        np.all(mapped >= -atol)
        and np.all(mapped <= (source_shape - 1.0) + atol)
    )
    if not in_bounds:
        return _make_plan(
            kind="refuse",
            source=source,
            target=target,
            mapping=M,
            exact=False,
            requires_interpolation=True,
            interpolation=None,
            crop_start=None,
            crop_stop=None,
            reason="target lattice requires extrapolation outside source feature centers",
        )

    return _make_plan(
        kind="resample",
        source=source,
        target=target,
        mapping=M,
        exact=False,
        requires_interpolation=True,
        interpolation=interpolation,
        crop_start=None,
        crop_stop=None,
        reason=(
            "feature centers are not an exact integer subset; explicit "
            f"{interpolation} resampling is required without extrapolation"
        ),
    )


def verify_repair_plan(
    source: FeatureLattice,
    target: FeatureLattice,
    plan: LatticeRepairPlan,
    *,
    role: RepairRole | None = None,
    atol: float = 1e-9,
) -> bool:
    """Verify a repair independently, including its signed semantic claims.

    A matching index matrix is necessary but not sufficient: a mathematically
    valid mapping can still sample beyond the available physical lattice.
    The certificate must also be bound to its claimed geometry/parameters and
    the caller's value role (when explicitly provided).
    """
    if tuple(plan.source_shape) != tuple(source.shape):
        return False
    if tuple(plan.target_shape) != tuple(target.shape):
        return False

    expected = target_to_source_index_map(source, target)
    mapping = np.asarray(plan.target_to_source_index, dtype=np.float64)
    if mapping.shape != (4, 4) or not np.all(np.isfinite(mapping)):
        return False
    if not np.allclose(expected, mapping, atol=atol, rtol=0):
        return False

    # Recalculate rather than merely trusting the hash carried by the plan.
    payload = {
        'kind': plan.kind,
        'source_shape': list(source.shape),
        'target_shape': list(target.shape),
        'source_jump': np.asarray(source.jump, dtype=float).round(12).tolist(),
        'source_start': np.asarray(source.start, dtype=float).round(12).tolist(),
        'target_jump': np.asarray(target.jump, dtype=float).round(12).tolist(),
        'target_start': np.asarray(target.start, dtype=float).round(12).tolist(),
        'target_to_source_index': mapping.round(12).tolist(),
        'exact': plan.exact,
        'requires_interpolation': plan.requires_interpolation,
        'interpolation': plan.interpolation,
        'crop_start': None if plan.crop_start is None else list(plan.crop_start),
        'crop_stop': None if plan.crop_stop is None else list(plan.crop_stop),
        'reason': plan.reason,
    }
    if _canonical_digest(payload) != plan.certificate_digest:
        return False

    if plan.kind == 'identity':
        return (
            plan.exact
            and not plan.requires_interpolation
            and plan.interpolation is None
            and tuple(source.shape) == tuple(target.shape)
            and np.allclose(source.jump, target.jump, atol=atol, rtol=0)
            and np.allclose(source.start, target.start, atol=atol, rtol=0)
            and plan.crop_start == (0, 0, 0)
            and plan.crop_stop == tuple(source.shape)
        )

    if plan.kind == 'exact_crop':
        if plan.crop_start is None or plan.crop_stop is None:
            return False
        start = np.asarray(plan.crop_start, dtype=int)
        stop = np.asarray(plan.crop_stop, dtype=int)
        target_shape = np.asarray(target.shape, dtype=int)
        source_shape = np.asarray(source.shape, dtype=int)
        return (
            plan.exact
            and not plan.requires_interpolation
            and plan.interpolation is None
            and np.all(stop - start == target_shape)
            and np.all(start >= 0)
            and np.all(stop <= source_shape)
            and np.allclose(source.jump, target.jump, atol=atol, rtol=0)
            and np.allclose(expected[:3, 3], start, atol=atol, rtol=0)
        )

    if plan.kind == 'resample':
        if (plan.exact or not plan.requires_interpolation
                or plan.interpolation not in {'nearest', 'trilinear'}
                or plan.crop_start is not None or plan.crop_stop is not None):
            return False
        if role is not None and plan.interpolation != _interpolation_for_role(role):
            return False
        # An affine map's extrema over a rectangular domain lie at corners.
        # Refuse extrapolation even for a correctly signed index transform.
        target_max = np.asarray(target.shape, dtype=np.float64) - 1.0
        source_max = np.asarray(source.shape, dtype=np.float64) - 1.0
        corners = np.array([
            [i, j, k, 1.0]
            for i in (0.0, target_max[0])
            for j in (0.0, target_max[1])
            for k in (0.0, target_max[2])
        ], dtype=np.float64)
        mapped = (expected @ corners.T).T[:, :3]
        return bool(np.all(mapped >= -atol) and np.all(mapped <= source_max + atol))

    if plan.kind == 'refuse':
        return (
            not plan.exact
            and plan.requires_interpolation
            and plan.interpolation is None
            and plan.crop_start is None
            and plan.crop_stop is None
        )

    return False

@dataclass(frozen=True)
class MergeRepairBundle:
    """Atomic repair bundle for aligning two feature lattices before fusion."""

    target: FeatureLattice
    left: LatticeRepairPlan
    right: LatticeRepairPlan
    interpolation_count: int
    retained_fraction_left: float
    retained_fraction_right: float
    bundle_digest: str
    reason: str


def _lattice_extent(lattice: FeatureLattice) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(lattice.start, dtype=np.float64)
    jump = np.asarray(lattice.jump, dtype=np.float64)
    shape = np.asarray(lattice.shape, dtype=np.int64)
    end = start + jump * (shape - 1)
    return np.minimum(start, end), np.maximum(start, end)


def _target_on_anchor_phase(
    anchor: FeatureLattice,
    other: FeatureLattice,
    *,
    atol: float,
) -> FeatureLattice | None:
    """Largest anchor-phase lattice whose centers lie in both physical extents."""

    if not np.allclose(anchor.jump, other.jump, atol=atol, rtol=0):
        return None

    jump = np.asarray(anchor.jump, dtype=np.float64)
    a_lo, a_hi = _lattice_extent(anchor)
    b_lo, b_hi = _lattice_extent(other)
    lo = np.maximum(a_lo, b_lo)
    hi = np.minimum(a_hi, b_hi)
    if np.any(hi < lo - atol):
        return None

    anchor_start = np.asarray(anchor.start, dtype=np.float64)
    first = np.ceil((lo - anchor_start) / jump - atol).astype(int)
    last = np.floor((hi - anchor_start) / jump + atol).astype(int)
    if np.any(last < first):
        return None

    target_start = anchor_start + first * jump
    target_shape = tuple(int(x) for x in (last - first + 1))
    return FeatureLattice(
        target_shape,
        np.asarray(anchor.jump, dtype=np.float64).copy(),
        np.asarray(target_start, dtype=np.float64),
        np.maximum(
            np.asarray(anchor.receptive_field, dtype=np.float64),
            np.asarray(other.receptive_field, dtype=np.float64),
        ),
    )


def _retained_fraction(source: FeatureLattice, target: FeatureLattice) -> float:
    return float(np.prod(target.shape) / np.prod(source.shape))


def _bundle_digest(
    target: FeatureLattice,
    left: LatticeRepairPlan,
    right: LatticeRepairPlan,
) -> str:
    payload = {
        "target_shape": list(target.shape),
        "target_jump": np.asarray(target.jump, dtype=float).round(12).tolist(),
        "target_start": np.asarray(target.start, dtype=float).round(12).tolist(),
        "left": left.certificate_digest,
        "right": right.certificate_digest,
    }
    return _canonical_digest(payload)


def synthesize_merge_repair(
    left: FeatureLattice,
    right: FeatureLattice,
    *,
    role: RepairRole = "feature",
    atol: float = 1e-9,
) -> MergeRepairBundle:
    """Find a deterministic minimum-damage common lattice for feature fusion.

    v1 considers the two existing sampling phases when jumps are equal.  It
    chooses the candidate with:
      1. fewer interpolating repairs;
      2. greater minimum retained spatial fraction;
      3. greater total retained fraction;
      4. left-anchored phase as deterministic tie-break.

    This is intentionally conservative: incompatible sampling rates are refused
    until a separately certified scale-change rule is implemented.
    """

    if not np.allclose(left.jump, right.jump, atol=atol, rtol=0):
        refuse_target = FeatureLattice(
            left.shape,
            np.asarray(left.jump, dtype=np.float64).copy(),
            np.asarray(left.start, dtype=np.float64).copy(),
            np.asarray(left.receptive_field, dtype=np.float64).copy(),
        )
        lplan = synthesize_lattice_repair(left, refuse_target, role=role, atol=atol)
        rmap = target_to_source_index_map(right, refuse_target)
        rplan = _make_plan(
            kind="refuse",
            source=right,
            target=refuse_target,
            mapping=rmap,
            exact=False,
            requires_interpolation=True,
            interpolation=None,
            crop_start=None,
            crop_stop=None,
            reason="merge synthesis v1 refuses unequal feature sampling rates",
        )
        return MergeRepairBundle(
            refuse_target,
            lplan,
            rplan,
            int(lplan.requires_interpolation) + int(rplan.requires_interpolation),
            _retained_fraction(left, refuse_target),
            min(1.0, _retained_fraction(right, refuse_target)),
            _bundle_digest(refuse_target, lplan, rplan),
            "no certified common-lattice rule for unequal jumps",
        )

    candidates: list[tuple[tuple[float, float, float, int], MergeRepairBundle]] = []
    for anchor_index, target in enumerate(
        (
            _target_on_anchor_phase(left, right, atol=atol),
            _target_on_anchor_phase(right, left, atol=atol),
        )
    ):
        if target is None:
            continue
        lp = synthesize_lattice_repair(left, target, role=role, atol=atol)
        rp = synthesize_lattice_repair(right, target, role=role, atol=atol)
        if "refuse" in {lp.kind, rp.kind}:
            continue
        il = int(lp.requires_interpolation)
        ir = int(rp.requires_interpolation)
        fl = _retained_fraction(left, target)
        fr = _retained_fraction(right, target)
        bundle = MergeRepairBundle(
            target=target,
            left=lp,
            right=rp,
            interpolation_count=il + ir,
            retained_fraction_left=fl,
            retained_fraction_right=fr,
            bundle_digest=_bundle_digest(target, lp, rp),
            reason="minimum-damage common patient-space lattice for feature fusion",
        )
        # min() on this tuple prefers fewer interpolations and larger retention.
        score = (float(il + ir), -min(fl, fr), -(fl + fr), anchor_index)
        candidates.append((score, bundle))

    if not candidates:
        target = FeatureLattice(
            left.shape,
            np.asarray(left.jump, dtype=np.float64).copy(),
            np.asarray(left.start, dtype=np.float64).copy(),
            np.asarray(left.receptive_field, dtype=np.float64).copy(),
        )
        lp = synthesize_lattice_repair(left, target, role=role, atol=atol)
        rmap = target_to_source_index_map(right, target)
        rp = _make_plan(
            kind="refuse",
            source=right,
            target=target,
            mapping=rmap,
            exact=False,
            requires_interpolation=True,
            interpolation=None,
            crop_start=None,
            crop_stop=None,
            reason="feature lattices have no overlapping certifiable common grid",
        )
        return MergeRepairBundle(
            target, lp, rp, 1, 1.0, 0.0,
            _bundle_digest(target, lp, rp),
            "no overlapping certifiable common patient-space lattice",
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def verify_merge_bundle(
    left: FeatureLattice,
    right: FeatureLattice,
    bundle: MergeRepairBundle,
    *,
    atol: float = 1e-9,
) -> bool:
    """Independently verify both component repairs and atomic bundle digest."""

    return (
        verify_repair_plan(left, bundle.target, bundle.left, atol=atol)
        and verify_repair_plan(right, bundle.target, bundle.right, atol=atol)
        and bundle.bundle_digest
        == _bundle_digest(bundle.target, bundle.left, bundle.right)
    )
