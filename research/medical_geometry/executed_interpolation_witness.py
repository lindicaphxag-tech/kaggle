"""Exact-rational execution witness for a CPU PyTorch trilinear repair.

Separates ideal-arithmetic C2 interpolation error from actual finite-precision
execution error. Independently computes (a) the barycentric reference at
executed FP64 source indices and (b) the barycentric reference at EXACT
RATIONAL source indices derived from STORED float64 chart declarations.
Thus chart-index arithmetic rounding is now directly measurable too.
This is a bounded-size a-posteriori CPU audit, NOT a universal numerical
stability bound or a guarantee of true acquisition-world coordinates.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import product
import math
import numpy as np
import torch
from .clingeom import FeatureLattice
from .repair_synthesis import LatticeRepairPlan, verify_repair_plan
from .torch_repair import LatticeRepair3d

@dataclass(frozen=True)
class ExecutedInterpolationWitness:
    plan_digest: str
    source_sha256: str
    output_sha256: str
    mapped_coordinate_sha256: str
    output_dtype: str
    audited_elements: int
    max_exact_barycentric_residual: float
    # Stronger independent view: exact fractional chart coordinates derived
    # from stored IEEE-754 lattice declarations *before* the FP64 affine map.
    max_exact_declared_chart_residual: float
    max_declared_vs_computed_source_index_error_voxels: float
    declared_analytic_interpolation_bound: float | None
    conditional_combined_bound: float | None
    proof_scope: str

def _fingerprint(tensor: torch.Tensor) -> str:
    a = tensor.detach().cpu().contiguous().numpy()
    h = sha256(repr((a.shape, a.dtype.str)).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()

def _round_up_exact(number: Fraction) -> float:
    if number == 0:
        return 0.0
    candidate = float(number)
    if not math.isfinite(candidate):
        raise ValueError("nonfinite exact-rational discrepancy")
    return math.nextafter(candidate, math.inf)

def _exact_barycentric(x: np.ndarray, indices: np.ndarray,
                       batch: int, channel: int) -> Fraction:
    axes = []
    for axis, position in enumerate(indices):
        coord = (position if isinstance(position, Fraction)
                 else Fraction.from_float(float(position)))
        first = math.floor(coord)
        if not 0 <= coord <= x.shape[axis + 2] - 1:
            raise ValueError("source point outside stored voxel-center domain")
        fractional = coord - first
        if fractional == 0:
            axes.append(((first, Fraction(1)),))
        else:
            axes.append(((first, 1 - fractional), (first + 1, fractional)))
    result = Fraction(0)
    for vertices in product(*axes):
        source_index = tuple(p for p, _ in vertices)
        weight = math.prod((weight for _, weight in vertices),
                           start=Fraction(1))
        stored_value = Fraction.from_float(
            float(x[(batch, channel) + source_index])
        )
        result += weight * stored_value
    return result


def _exact_declared_chart_indices(
    source: FeatureLattice, target: FeatureLattice, index: tuple[int, int, int],
) -> tuple[Fraction, Fraction, Fraction]:
    """Exact rational chart map from the STORED source/target float64 fields.

    This deliberately avoids float64 subtraction, division and affine
    multiplication. A valid coordinate must be inside the source grid as an
    *exact fraction*, even when rounded arithmetic would suggest otherwise.
    """
    positions = []
    for axis, i in enumerate(index):
        source_step = Fraction.from_float(float(source.jump[axis]))
        target_step = Fraction.from_float(float(target.jump[axis]))
        source_origin = Fraction.from_float(float(source.start[axis]))
        target_origin = Fraction.from_float(float(target.start[axis]))
        if source_step <= 0 or target_step <= 0:
            raise ValueError("invalid exact declared chart step")
        pos = (target_origin + i * target_step - source_origin) / source_step
        if not 0 <= pos <= int(source.shape[axis]) - 1:
            raise ValueError("declared rational chart samples outside source grid")
        positions.append(pos)
    return tuple(positions)


def certify_executed_interpolation(
    x: torch.Tensor,
    source: FeatureLattice,
    target: FeatureLattice,
    plan: LatticeRepairPlan,
    *,
    output: torch.Tensor | None = None,
    analytic_interpolation_bound: float | None = None,
    max_elements: int = 4096,
) -> ExecutedInterpolationWitness:
    """Audit ONE actual CPU execution versus an exact binary-rational reference.

    The legacy conditional combined bound uses a caller-supplied ideal
    interpolation bound PLUS the actual execution residual at FP64 indices.
    The independent exact declared-chart residual now exposes coordinate-map
    roundoff as well, but is kept separate: the caller's analytic Hessian
    bound is not automatically a rigorous outward-rounded chart bound.
    Never present either witness as a universal patient-space guarantee.
    """
    if not verify_repair_plan(source, target, plan, role="feature"):
        raise ValueError("unverified repair plan")
    if plan.kind != "resample" or plan.interpolation != "trilinear":
        raise ValueError("witness requires a true trilinear resampling plan")
    if (x.device.type != "cpu" or x.ndim != 5
            or x.dtype not in (torch.float32, torch.float64)
            or tuple(x.shape[-3:]) != tuple(source.shape)
            or not bool(torch.isfinite(x).all())):
        raise ValueError("expected finite CPU float32/float64 NCDHW source")
    if (isinstance(max_elements, bool) or not isinstance(max_elements, int)
            or max_elements < 1):
        raise ValueError("max_elements must be positive integer")
    elements = int(x.shape[0]) * int(x.shape[1]) * math.prod(target.shape)
    if elements > max_elements:
        raise ValueError("proof budget exceeded: audited elements")
    if analytic_interpolation_bound is not None and (
        not math.isfinite(analytic_interpolation_bound)
        or analytic_interpolation_bound < 0
    ):
        raise ValueError("invalid analytic interpolation bound")
    repair = LatticeRepair3d(plan)
    with torch.no_grad():
        expected_output = repair(x)
    if output is None:
        output = expected_output
    elif (tuple(output.shape) != tuple(expected_output.shape)
          or output.dtype != expected_output.dtype
          or output.device.type != "cpu"
          or not torch.equal(output, expected_output)):
        raise ValueError("provided output fails exact CPU execution replay")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("nonfinite executed output")
    coordinates = repair._source_coordinates(x).detach().cpu().numpy()
    stored = x.detach().cpu().numpy()
    produced = output.detach().cpu().numpy()
    maximum = Fraction(0)
    declared_maximum = Fraction(0)
    coordinate_maximum = Fraction(0)
    for d, h, w in np.ndindex(*target.shape):
        coordinate = coordinates[d, h, w]
        rational_chart = _exact_declared_chart_indices(source, target, (d, h, w))
        for computed, declared in zip(coordinate, rational_chart):
            coordinate_maximum = max(
                coordinate_maximum,
                abs(Fraction.from_float(float(computed)) - declared),
            )
        for batch in range(int(x.shape[0])):
            for channel in range(int(x.shape[1])):
                ideal = _exact_barycentric(stored, coordinate, batch, channel)
                declared_ideal = _exact_barycentric(
                    stored, rational_chart, batch, channel
                )
                actual = Fraction.from_float(
                    float(produced[batch, channel, d, h, w])
                )
                maximum = max(maximum, abs(ideal - actual))
                declared_maximum = max(
                    declared_maximum, abs(declared_ideal - actual)
                )
    numeric = _round_up_exact(maximum)
    declared_numeric = _round_up_exact(declared_maximum)
    coordinate_roundoff = _round_up_exact(coordinate_maximum)
    total = None
    if analytic_interpolation_bound is not None:
        summed = analytic_interpolation_bound + numeric
        if not math.isfinite(summed):
            raise ValueError("combined bound overflow")
        total = math.nextafter(summed, math.inf) if summed else 0.0
    return ExecutedInterpolationWitness(
        plan_digest=plan.certificate_digest,
        source_sha256=_fingerprint(x),
        output_sha256=_fingerprint(output),
        mapped_coordinate_sha256=sha256(coordinates.tobytes()).hexdigest(),
        output_dtype=str(x.dtype),
        audited_elements=elements,
        max_exact_barycentric_residual=numeric,
        max_exact_declared_chart_residual=declared_numeric,
        max_declared_vs_computed_source_index_error_voxels=coordinate_roundoff,
        declared_analytic_interpolation_bound=analytic_interpolation_bound,
        conditional_combined_bound=total,
        proof_scope=(
            "exact rational barycentric sum of stored source float values "
            "at ACTUAL computed float64 index coordinates; plus externally "
            "justified analytic Hessian error if supplied; independent "
            "also audits exact rational indices computed from STORED float64 "
            "lattice declarations, capturing chart-index FP64 quantization; "
            "original continuous patient-space acquisition truth and a "
            "universal worst-case runtime bound remain UNPROVED"
        ),
    )
