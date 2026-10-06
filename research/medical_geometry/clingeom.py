"""ClinGeom: proof-carrying patient-space semantics for medical AI.

This prototype focuses on discrete 3-D reindexing (axis permutation / flip)
and RAS/LPS coordinate-frame conversion. A representation-only reindex is
valid only when the voxel-to-patient/world mapping is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product, permutations
from typing import Iterable, Literal, Sequence

import numpy as np

Frame = Literal["RAS", "LPS"]
Role = Literal["intensity", "label", "probability"]


@dataclass(frozen=True)
class PatientSpaceSpec:
    """Minimal physical-space contract for a 3-D medical image tensor."""

    shape: tuple[int, int, int]
    affine: np.ndarray
    frame: Frame = "RAS"
    role: Role = "intensity"
    source_id: str = ""

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(int(n) <= 0 for n in self.shape):
            raise ValueError("shape must contain three positive spatial dimensions")
        a = np.asarray(self.affine, dtype=np.float64)
        if a.shape != (4, 4):
            raise ValueError("affine must be 4x4")
        if not np.all(np.isfinite(a)):
            raise ValueError("affine must contain only finite values")
        if abs(np.linalg.det(a[:3, :3])) < 1e-12:
            raise ValueError("affine must be nonsingular")
        object.__setattr__(self, "affine", a)


@dataclass(frozen=True)
class ReindexCertificate:
    """Certificate for a representation-only discrete reindex."""

    valid: bool
    source_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    permutation: tuple[int, int, int]
    flips: tuple[bool, bool, bool]
    max_matrix_error_mm: float
    index_map_det: float
    requires_interpolation: bool
    reason: str


def _validate_permutation(permutation: Sequence[int]) -> tuple[int, int, int]:
    p = tuple(int(x) for x in permutation)
    if sorted(p) != [0, 1, 2]:
        raise ValueError("permutation must contain 0, 1, 2 exactly once")
    return p


def _validate_flips(flips: Sequence[bool]) -> tuple[bool, bool, bool]:
    f = tuple(bool(x) for x in flips)
    if len(f) != 3:
        raise ValueError("flips must have length 3")
    return f


def signed_reindex_map(
    source_shape: Sequence[int],
    permutation: Sequence[int],
    flips: Sequence[bool],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return homogeneous H mapping output voxel indices to source indices."""

    shape = tuple(int(x) for x in source_shape)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError("source_shape must contain three positive dimensions")
    p = _validate_permutation(permutation)
    f = _validate_flips(flips)
    output_shape = tuple(shape[p[k]] for k in range(3))

    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = 0.0
    H[:3, 3] = 0.0
    for out_axis, src_axis in enumerate(p):
        if f[out_axis]:
            H[src_axis, out_axis] = -1.0
            H[src_axis, 3] = output_shape[out_axis] - 1
        else:
            H[src_axis, out_axis] = 1.0
    return H, output_shape


def apply_reindex(
    array: np.ndarray,
    permutation: Sequence[int],
    flips: Sequence[bool],
) -> np.ndarray:
    """Apply the discrete reindex represented by signed_reindex_map."""

    p = _validate_permutation(permutation)
    f = _validate_flips(flips)
    if np.asarray(array).ndim != 3:
        raise ValueError("prototype currently accepts exactly 3 spatial dimensions")
    out = np.transpose(np.asarray(array), axes=p)
    for axis, do_flip in enumerate(f):
        if do_flip:
            out = np.flip(out, axis=axis)
    return np.ascontiguousarray(out)


def expected_reindexed_affine(
    source_affine: np.ndarray,
    source_shape: Sequence[int],
    permutation: Sequence[int],
    flips: Sequence[bool],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    H, output_shape = signed_reindex_map(source_shape, permutation, flips)
    return np.asarray(source_affine, dtype=np.float64) @ H, output_shape


def certify_reindex(
    source: PatientSpaceSpec,
    output_affine: np.ndarray | None,
    permutation: Sequence[int],
    flips: Sequence[bool],
    *,
    atol_mm: float = 1e-6,
) -> ReindexCertificate:
    """Fail-closed verification of patient-space preservation."""

    p = _validate_permutation(permutation)
    f = _validate_flips(flips)
    H, output_shape = signed_reindex_map(source.shape, p, f)
    det = float(np.linalg.det(H[:3, :3]))

    if output_affine is None:
        return ReindexCertificate(
            False, source.shape, output_shape, p, f, float("inf"), det, False,
            "missing output affine: patient-space preservation cannot be established",
        )

    out = np.asarray(output_affine, dtype=np.float64)
    if out.shape != (4, 4) or not np.all(np.isfinite(out)):
        return ReindexCertificate(
            False, source.shape, output_shape, p, f, float("inf"), det, False,
            "malformed output affine",
        )

    expected = source.affine @ H
    error = float(np.max(np.abs(expected - out)))
    valid = bool(error <= atol_mm)
    return ReindexCertificate(
        valid, source.shape, output_shape, p, f, error, det, False,
        "patient-space map preserved" if valid else "array reindex and affine disagree",
    )


def voxel_to_world(affine: np.ndarray, ijk: Sequence[float]) -> np.ndarray:
    x = np.ones(4, dtype=np.float64)
    x[:3] = np.asarray(ijk, dtype=np.float64)
    return (np.asarray(affine, dtype=np.float64) @ x)[:3]


def frame_conversion_matrix(source: Frame, target: Frame) -> np.ndarray:
    """Coordinate change between DICOM-style LPS and NIfTI-style RAS."""

    if source == target:
        return np.eye(4, dtype=np.float64)
    if {source, target} != {"LPS", "RAS"}:
        raise ValueError(f"unsupported frame conversion: {source} -> {target}")
    return np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def convert_affine_frame(affine: np.ndarray, source: Frame, target: Frame) -> np.ndarray:
    return frame_conversion_matrix(source, target) @ np.asarray(affine, dtype=np.float64)


def voxel_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Voxel-index Dice; intentionally blind to affine/patient-space semantics."""

    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if aa.shape != bb.shape:
        raise ValueError("voxel Dice requires equal array shapes")
    denom = int(aa.sum()) + int(bb.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(aa, bb).sum() / denom)


def all_signed_axis_reindexings() -> Iterable[tuple[tuple[int, int, int], tuple[bool, bool, bool]]]:
    """Enumerate all 48 discrete 3-D axis permutation/flip reindexings."""

    for p in permutations((0, 1, 2)):
        for f in product((False, True), repeat=3):
            yield tuple(p), tuple(f)

@dataclass(frozen=True)
class MetadataSubstitutionCertificate:
    """Certificate for replacing geometry metadata while leaving voxels unchanged."""

    valid: bool
    max_corner_displacement_mm: float
    handedness_changed: bool
    tolerance_mm: float
    reason: str


def physical_corners(shape: Sequence[int], affine: np.ndarray) -> np.ndarray:
    """Return the eight voxel-center corners in patient/world coordinates."""

    s = tuple(int(x) for x in shape)
    if len(s) != 3 or any(n <= 0 for n in s):
        raise ValueError("shape must contain three positive dimensions")
    pts = []
    for i in (0, s[0] - 1):
        for j in (0, s[1] - 1):
            for k in (0, s[2] - 1):
                pts.append(voxel_to_world(affine, (i, j, k)))
    return np.asarray(pts, dtype=np.float64)


def certify_metadata_substitution(
    shape: Sequence[int],
    old_affine: np.ndarray,
    new_affine: np.ndarray,
    *,
    tolerance_mm: float = 1e-3,
) -> MetadataSubstitutionCertificate:
    """Verify whether an affine overwrite is only a negligible metadata correction.

    Data are assumed unchanged. Large patient-space changes therefore cannot be
    justified as harmless metadata replacement and fail closed.
    """

    old = np.asarray(old_affine, dtype=np.float64)
    new = np.asarray(new_affine, dtype=np.float64)
    if old.shape != (4, 4) or new.shape != (4, 4):
        raise ValueError("old_affine and new_affine must both be 4x4")
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ValueError("affines must contain only finite values")
    if tolerance_mm < 0:
        raise ValueError("tolerance_mm must be nonnegative")

    old_pts = physical_corners(shape, old)
    new_pts = physical_corners(shape, new)
    disp = np.linalg.norm(old_pts - new_pts, axis=1)
    max_disp = float(np.max(disp))

    old_det = float(np.linalg.det(old[:3, :3]))
    new_det = float(np.linalg.det(new[:3, :3]))
    handedness_changed = bool(np.sign(old_det) != np.sign(new_det))

    valid = bool(max_disp <= tolerance_mm and not handedness_changed)
    if handedness_changed:
        reason = "metadata overwrite changes coordinate handedness"
    elif max_disp > tolerance_mm:
        reason = "metadata overwrite moves voxel centers in patient space"
    else:
        reason = "metadata difference is within the declared patient-space tolerance"

    return MetadataSubstitutionCertificate(
        valid, max_disp, handedness_changed, float(tolerance_mm), reason
    )

@dataclass(frozen=True)
class FeatureLattice:
    """Spatial semantics of a feature grid in input-voxel coordinates."""

    shape: tuple[int, int, int]
    jump: np.ndarray
    start: np.ndarray
    receptive_field: np.ndarray

    @classmethod
    def input_grid(cls, shape: Sequence[int]) -> "FeatureLattice":
        s = tuple(int(x) for x in shape)
        if len(s) != 3 or any(n <= 0 for n in s):
            raise ValueError("shape must contain three positive dimensions")
        return cls(
            s,
            np.ones(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.ones(3, dtype=np.float64),
        )


@dataclass(frozen=True)
class FeatureGeometryCertificate:
    """Certificate binding a feature tensor to its model-derived patient-space lattice."""

    valid: bool
    expected_affine: np.ndarray
    max_matrix_error_mm: float
    reason: str


def _triple(value: int | Sequence[int], name: str) -> np.ndarray:
    if isinstance(value, (int, np.integer)):
        out = np.array([int(value)] * 3, dtype=np.int64)
    else:
        out = np.asarray(tuple(int(x) for x in value), dtype=np.int64)
    if out.shape != (3,):
        raise ValueError(f"{name} must be an int or length-3 sequence")
    return out


def propagate_conv_lattice(
    lattice: FeatureLattice,
    *,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> FeatureLattice:
    """Propagate exact receptive-field center geometry through Conv/Pool-like ops."""

    k = _triple(kernel_size, "kernel_size")
    s = _triple(stride, "stride")
    p = _triple(padding, "padding")
    d = _triple(dilation, "dilation")
    if np.any(k <= 0) or np.any(s <= 0) or np.any(d <= 0) or np.any(p < 0):
        raise ValueError("invalid convolution geometry")

    in_shape = np.asarray(lattice.shape, dtype=np.int64)
    effective_kernel = d * (k - 1) + 1
    out_shape = np.floor((in_shape + 2 * p - effective_kernel) / s + 1).astype(int)
    if np.any(out_shape <= 0):
        raise ValueError("operation produces an empty spatial lattice")

    new_start = lattice.start + (((k - 1) * d) / 2.0 - p) * lattice.jump
    new_rf = lattice.receptive_field + (k - 1) * d * lattice.jump
    new_jump = lattice.jump * s
    return FeatureLattice(
        tuple(int(x) for x in out_shape),
        np.asarray(new_jump, dtype=np.float64),
        np.asarray(new_start, dtype=np.float64),
        np.asarray(new_rf, dtype=np.float64),
    )


def feature_affine(input_affine: np.ndarray, lattice: FeatureLattice) -> np.ndarray:
    """Map feature indices to patient coordinates of receptive-field centers."""

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.diag(lattice.jump)
    M[:3, 3] = lattice.start
    return np.asarray(input_affine, dtype=np.float64) @ M


def center_matched_affine(
    input_affine: np.ndarray,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> np.ndarray:
    """Shape-only FOV-center affine, useful as a naive comparison baseline."""

    A = np.asarray(input_affine, dtype=np.float64)
    in_shape = np.asarray(tuple(int(x) for x in input_shape), dtype=np.float64)
    out_shape = np.asarray(tuple(int(x) for x in output_shape), dtype=np.float64)
    if in_shape.shape != (3,) or out_shape.shape != (3,) or np.any(out_shape <= 0):
        raise ValueError("input_shape and output_shape must be positive length-3 sequences")

    linear = A[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    direction = linear / spacing
    new_spacing = spacing * (in_shape / out_shape)
    input_center = A[:3, 3] + direction @ (((in_shape - 1) / 2.0) * spacing)
    new_origin = input_center - direction @ (((out_shape - 1) / 2.0) * new_spacing)

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = direction * new_spacing
    out[:3, 3] = new_origin
    return out


def certify_feature_geometry(
    input_affine: np.ndarray,
    lattice: FeatureLattice,
    claimed_affine: np.ndarray | None,
    *,
    atol_mm: float = 1e-6,
) -> FeatureGeometryCertificate:
    """Verify a feature map affine against model-derived receptive-field geometry."""

    expected = feature_affine(input_affine, lattice)
    if claimed_affine is None:
        return FeatureGeometryCertificate(
            False, expected, float("inf"), "feature affine is missing"
        )
    claimed = np.asarray(claimed_affine, dtype=np.float64)
    if claimed.shape != (4, 4) or not np.all(np.isfinite(claimed)):
        return FeatureGeometryCertificate(
            False, expected, float("inf"), "feature affine is malformed"
        )
    error = float(np.max(np.abs(expected - claimed)))
    valid = bool(error <= atol_mm)
    return FeatureGeometryCertificate(
        valid,
        expected,
        error,
        "model-derived feature geometry matches" if valid else "shape-only or supplied geometry disagrees with model lattice",
    )

@dataclass(frozen=True)
class LateralityCertificate:
    """Certificate describing whether semantic left/right labels require remapping."""

    valid: bool
    physical_reflection: bool
    requires_label_swap: bool
    axis: int | None
    reason: str


def physical_transform_handedness(transform_world: np.ndarray) -> float:
    """Return determinant of a 3-D world-space transform linear part."""

    T = np.asarray(transform_world, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("transform_world must be a finite 4x4 matrix")
    return float(np.linalg.det(T[:3, :3]))


def certify_laterality_transform(
    *,
    representation_only: bool,
    transform_world: np.ndarray | None = None,
    left_right_axis: int = 0,
) -> LateralityCertificate:
    """Distinguish representation flips from physical left/right reflection.

    A representation-only reindex with a compensating affine preserves patient
    anatomy and therefore does not require semantic left/right label exchange.
    A physical world-space reflection reverses handedness and requires a
    declared label swap when laterality-bearing labels are present.
    """

    if left_right_axis not in (0, 1, 2):
        raise ValueError("left_right_axis must be 0, 1, or 2")

    if representation_only:
        if transform_world is not None:
            return LateralityCertificate(
                False, False, False, None,
                "representation-only reindex must not also declare a physical transform",
            )
        return LateralityCertificate(
            True, False, False, None,
            "patient-space anatomy is unchanged; laterality labels are preserved",
        )

    if transform_world is None:
        return LateralityCertificate(
            False, False, False, None,
            "physical augmentation requires an explicit world-space transform",
        )

    T = np.asarray(transform_world, dtype=np.float64)
    det = physical_transform_handedness(T)
    reflected = bool(det < 0)
    if not reflected:
        return LateralityCertificate(
            True, False, False, None,
            "physical transform preserves coordinate handedness",
        )

    # Determine whether the declared left/right world axis changes sign without
    # mixing; general reflections are rejected because laterality is ambiguous.
    row = T[left_right_axis, :3]
    dominant = int(np.argmax(np.abs(row)))
    pure_lr = (
        dominant == left_right_axis
        and row[left_right_axis] < 0
        and np.count_nonzero(np.abs(row) > 1e-9) == 1
    )
    if not pure_lr:
        return LateralityCertificate(
            False, True, False, None,
            "reflection is not an unambiguous left/right-axis reflection",
        )

    return LateralityCertificate(
        True, True, True, left_right_axis,
        "physical left/right reflection requires laterality-label exchange",
    )


def swap_label_pairs(
    labels: np.ndarray,
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Atomically exchange declared laterality label IDs."""

    src = np.asarray(labels)
    out = src.copy()
    seen: set[int] = set()
    for left, right in pairs:
        left = int(left)
        right = int(right)
        if left == right or left in seen or right in seen:
            raise ValueError("laterality label pairs must be disjoint and non-identical")
        seen.update((left, right))
        left_mask = src == left
        right_mask = src == right
        out[left_mask] = right
        out[right_mask] = left
    return out
