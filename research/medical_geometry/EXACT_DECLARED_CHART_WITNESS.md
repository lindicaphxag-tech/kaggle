# Exact declared-chart runtime witness — ClinGeom numerical closure

## Why the existing executed witness was incomplete

A prior `executed_interpolation_witness.py` already used exact binary-rational
arithmetic to compare an actual CPU PyTorch `grid_sample` result against
trilinear interpolation at the **computed float64 source indices**. It
correctly disclaimed another unmeasured step: the map from source/target
**declared lattice fields** to those float64 indices.

For example, the stored binary64 representations of source jump `0.3` and
target start `0.1` define the exact rational source coordinate

```text
q_declared = Fraction.from_float(0.1) / Fraction.from_float(0.3)
```

That is generally different from the rounded binary64 quotient returned by
`(0.1 / 0.3)`. The difference is usually tiny in *source-index units*, but
a linear field can have arbitrarily large slope while its Hessian is exactly
zero. Consequently **a zero C² interpolation error estimate cannot bound
either coordinate conversion error or actual floating-point kernel error**.

## New evidence, not a new mathematical law

The exact-chart witness now computes, for every audited output voxel:

1. **Stored-chart rational source index:** for each axis `j`,
   `q = (Fraction(target.start[j]) + i*Fraction(target.jump[j])
         - Fraction(source.start[j])) / Fraction(source.jump[j])`;
2. **Computed index:** the source index emitted by the actual float64
   PyTorch grid construction, as already audited;
3. **Exact barycentric ideal on stored input samples:** independently evaluate
   both indices against exact-rational representations of each stored
   float32/float64 tensor value;
4. **Actual kernel output:** replay CPU PyTorch `grid_sample` and require
   any supplied output to match replay bit-for-bit;
5. **Outward-rounded a-posteriori residuals:** independently return the
   worst absolute error against the computed-index ideal and the exact
   declared-chart ideal, plus the largest index-coordinate discrepancy.

The existing witness fields and their scope are preserved; new fields are

```text
max_exact_declared_chart_residual
max_declared_vs_computed_source_index_error_voxels
```

Because exact rational arithmetic is used, the new chart-index computation
does not share the original float64 quotient implementation. This makes
the chart stage independently falsifiable.

## Fail-closed boundaries

- If an exact rational declared-grid index lies outside the declared source
  voxel-center grid, refuse even if floating mapping tolerances would have
  silently rounded it inside.
- Audit only a **frozen, bounded-size CPU tensor execution** (default 4,096
  elements), not all hardware or future executions.
- No claim is made about the precision of an original hospital DICOM affine
  **before** its declared float64 representation was formed.
- The external uniform Hessian bound may still be unverified. The existing
  `conditional_combined_bound` remains a conditional computed-index result;
  it is not silently promoted to an unconditional chart/world guarantee.
- The new a-posteriori residual is *not* a general forward error theorem for
  arbitrary model weights, devices, dtypes or larger volumes.

## Reproduce and falsify

```bash
python -m pytest -q \
  research/medical_geometry/test_executed_interpolation_witness.py \
  research/medical_geometry/test_exact_declared_chart_witness.py
```

The new tests use independent `Fraction` calculations, float32 and float64
executions, exact half-cell negative controls, changed source charts, and
the near-boundary case where computed floating indices and the exact declared
chart disagree on admissibility.

**Research interpretation:** this closes an explicitly documented part of the
medical-feature proof pipeline (declared source chart → executed numerical
index → actual tensor). It does not imply better segmentation Dice, naturally
occurring clinical harm, independent validation, or an accepted MONAI/TorchIO
contribution. Its value is correctness of the *proof-carrying* interface.
