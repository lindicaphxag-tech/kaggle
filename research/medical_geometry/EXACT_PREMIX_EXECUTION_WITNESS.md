# ClinGeom: exact-rational proof of one executed 3-D channel fusion

## The concrete proof gap

A medical feature tensor can have a correctly repaired **patient-space
sampling lattice**, and its interpolation may pass a smoothness-conditional
bound, while the eventual `1×1 Conv3d` changes numerical values through
ordinary floating-point multiplication and accumulation.

A prior CPU interpolation witness covered the *sampling operation*, and
[PR #64](https://github.com/lindicaphxag-tech/kaggle/pull/64) extended it
to the exact **stored input feature-chart** rather than only already-rounded
float64 source indices. Neither alone checked the **actual subsequent dense
channel-fusion execution**.

This artifact checks the remaining numerical edge, on bounded-size tensors.

## Whole-operation exact reference

For each output index `i`, input channel `c`, and output channel `o`:

```text
q_c(i) = (target.start + target.jump*i - source_c.start) / source_c.jump

v_c(i) = exact trilinear barycentric sum of STORED source-channel floats at q_c(i)

y_o(i) = EXACT_FLOAT_BIAS[o] + sum_c EXACT_FLOAT_WEIGHT[o,c] * v_c(i)
```

Every division, interpolation coefficient, source value, weight, bias,
multiply and sum in the **reference** uses `fractions.Fraction` on the
*actually stored IEEE-754 values*. No FP32/FP64 coordinate-map computation
or PyTorch convolution kernel is used for the reference.

The tested path separately runs the **actual CPU PyTorch runtime** through
`physically_align_then_mix()` and requires any externally supplied output to
match a fresh replay. The exact reference is compared directly against this
executed tensor with an **upward-rounded max-absolute residual**.

The result binds the source tensor SHA-256, fused output SHA-256, model
parameter SHA-256, original compiler geometry/provenance fingerprint,
audited element count and the measured discrepancy.

Optional `maximum_audited_execution_error` succeeds **only if this actual
frozen computation satisfies it**. E.g. a mathematical zero-Hessian input
may still fail an exact-zero executed numerical error budget. This is a
*one-execution audit*, never a universal theorem.

## Strictly limited evidence scope

The bound is exact (outward rounded into `float`) on the residual between
this finite **observed CPU output** and the ideal rational composition of
stored feature samples, declared source/target charts and stored mixer
parameters. It does **not** prove:

- source samples match any ideal continuous patient field;
- declared source/target charts are accurate measurements of the patient's
  original acquisition coordinate frame;
- the external Hessian bound of a medical feature is correct;
- numerical safety holds for a different tensor, different weights, GPU,
  kernel, dtype, or larger array;
- improved clinical segmentation accuracy;
- an external upstream project has adopted this work.

For reviewability the default exact-reference work quota is 16,384
per-output-channel input terms. Large biomedical 3D volumes should use
small independently specified ROIs or a separate sound scalable numerical
bound, not pretend this proof automatically scales.

## Targeted reproducibility

```bash
python -m pytest -q \
  research/medical_geometry/test_exact_declared_chart_witness.py \
  research/medical_geometry/test_executed_premix_witness.py
```

The accompanying `clingeom-premix-execution.yml` runs this and the prior
interpolation witness suite under CPU PyTorch on Python 3.10 and 3.12.
The public core CI independently runs it with the existing MONAI DiNTS
fused-grid tests.

This is a **numerical proof-carrying execution** research component,
not just a visualization or a Dice/HD95 benchmark. Its significance is
preventing mismatched ideal-world and executed-world correctness claims.
