# ClinGeom value-type preserving transport contract

Status: executable regression scope, **not a clinical correctness certificate**.

## Core invariant

An object declared as a discrete segmentation / instance-label tensor is
not a sampled scalar image. Transporting an `int64` label through a `float32`
surrogate violates discrete identity, even if the spatial affine and sample
coordinates are entirely correct.

Counterexample: two distinct integers above 2^53 can map to the same
float32 value. Thus no `int64 -> float32 -> int64` implementation can
universally preserve arbitrary instance IDs (non-injectivity).

## Execution semantics

- `identity`/`exact_crop`: preserve input dtype and values with no interpolation.
- `nearest` + integer/bool tensor: calculate source coordinates in float64,
  round the source *indices* to nearest-even, then use direct integer indexing
  to gather original label values. No label is converted to floating point.
- `trilinear` + continuous float32/float64 tensor: build a grid with the
  same dtype as the feature tensor and invoke `torch.grid_sample`.
- `trilinear` + integer/bool tensor: fail closed instead of inventing a
  continuous meaning for a discrete class identifier.

## Tie semantics

Nearest-neighbor interpolation has an ambiguity when the requested source
coordinate lies exactly halfway between two voxels. ClinGeom defines an
explicit, deterministic **round-half-to-even** index rule:

    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5] -> [0, 2, 2, 4, 4, 6, 6].

This is defined over float64 source-coordinate calculations. It need not
match a float32 normalized grid_sample's half-coordinate behavior, because
round-trip normalized coordinate quantization can nudge a mathematical
halfway position to one side. Outside exact ties, the two rules agree
up to the precision of the supplied coordinates.

## Reproducible evidence

Run:

    python -m pytest -q research/medical_geometry/test_torch_repair_dtype.py
    python research/medical_geometry/benchmark_lossless_labels.py

The benchmark explicitly executes a negative-control float32 transport
followed by the new lossless transport on the same high-value integer labels.
Any reported number of corrupted voxels pertains to the synthetic labels
used in the benchmark, not an estimate of clinical incidence.

## Scientific boundary

This is an exact, discrete data-type preservation property, not a new
segmentation architecture, zero-shot robustness result or medical validation.
Correctly retaining labels cannot compensate for incorrect segmentation
predictions or unverified patient-frame identity.