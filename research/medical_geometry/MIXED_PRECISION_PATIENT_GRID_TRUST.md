# ClinGeom: mixed-precision patient-coordinate authorization

## A concrete failure in the previous claim boundary

A continuous linear scalar field has identically zero Hessian. Therefore its
**real-arithmetic** trilinear interpolation error is exactly zero. That fact
does **not** imply that a PyTorch implementation executing normalized
coordinates in FP16 or BF16 has zero error.

The current runtime's `LatticeRepair3d._grid` calculates the source
coordinates in float64, normalizes them to [-1,1], **then casts the grid
to the input feature dtype** before `grid_sample`. At source-axis size 64,
the actual CPU `grid_sample` output of a linear ramp at half-cell locations
has observed maximum absolute discrepancies of approximately **0.03125
(FP16)** and **0.25 (BF16)** in input-voxel signal units. These are
deterministic controlled numeric tests, not patient clinical errors.

If an external Hessian assumption were zero, a curvature-only certificate
could claim mathematical interpolation error zero while the actual execution
uses displaced sampling coordinates.

## New mechanism: separable-grid, oblique-patient-affine precision certificate

For a source grid with side length `n` and target index `i`, let `s(i)`
be the **planned source index** from ClinGeom's exact index map.

Define the normalized grid coordinate `g(i)=2*s(i)/(n-1)-1`, the
stored mixed-precision grid coordinate `g_q(i)=cast_dtype(g(i))`, and the
corresponding stored-coordinate source displacement

`delta(i) = (g_q(i)+1)*(n-1)/2 - s(i)`.

For a real 3-D patient affine `A` and source lattice jumps `J`, define
`B = A[:3,:3] diag(J)`. The **stored-coordinate physical displacement**
at a target triple is

`d(i,j,k) = || B @ [delta_x(i), delta_y(j), delta_z(k)] ||_2`.

Enumerating all `D×H×W` triples wastes memory. Each source-index map is
separable. The norm of a linear map is convex, so its maximum on a box
occurs at one of the eight vertices. Because the extrema of each
axis's finite coordinate-error set occur on actual target positions, the
maximum on the full Cartesian product equals the maximum over just
`2³ = 8` combinations of per-axis minima and maxima.

**Complexity:** `O(D+H+W+8)` time and `O(D+H+W)` temporary memory,
versus `O(DHW)` exhaustive voxel expansion. This remains correct for
off-diagonal/sheared, anisotropic patient-space affine matrices.

This is a mathematically **exact bound for the normalized grid-coordinate
STORAGE component**, up to float64 host arithmetic. It is **not a bound for
the full grid_sample kernel's value roundoff, source-value quantization,
convolution or trained-model error**. Small rounding or value discrepancies
remain possible even if this physical-coordinate budget is zero.

## Actual intervention, not just a benchmark

`physically_align_then_mix(..., maximum_grid_coordinate_drift_mm=budget)`
now checks the physical-coordinate drift for every source fiber **before**
resampling and irreversible channel mixing. It refuses invalid budgets,
missing patient affine, or exceedance, and records all approved per-fiber
drifts in the provenance digest and result object.

A mixed precision tensor need not be banned outright. For a particular
patient affine, target sampling phase and physical budget, FP32 may be
admissible while FP16/BF16 must be rejected. Exact index-preserving crop
and identity paths remain zero-grid-storage-drift even at FP16/BF16.

## Reproduction and falsification

```bash
python -m pytest -q research/medical_geometry/test_grid_precision_certificate.py
```

Independent tests include:
- actual CPU PyTorch FP16/BF16 linear-ramp interpolation with zero analytic
  Hessian, exposing finite-execution error;
- brute-force Cartesian-grid comparison against the analytical **eight
  corners** under a genuinely sheared 3-D affine;
- anisotropic patient-mm gating; mixed-precision refusal, FP32 authorization;
- no-normalized-grid negative control (exact crop);
- corrupt / negative / nonfinite budget rejection and proof-digest changes.

### Reviewer caveat

The new numerical calculation is a **correctness mechanism** and its
eight-corner reduction is a derived elementary convexity result, not an
independently established novel mathematical theorem. Research novelty, if
any, lies in integrating execution precision with patient-space geometry
contracts and applying it in actual multi-fiber medical inference pipelines.
Clinical benefit, external validation and model accuracy are not inferred.
