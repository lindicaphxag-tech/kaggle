# ClinGeom: execution-bound numerical proof, not only real-arithmetic geometry

## Counterexample found in the existing v3 inference boundary

The existing interpolation certificate is a mathematically valid **real-arithmetic**
bound conditional on an externally justified patient-world Hessian norm.
It is **not** by itself a bound on the finite-precision tensor returned by
PyTorch's 5-D `grid_sample`.

One self-contained example: sample the exactly linear scalar field
`f(i,j,k)=1000*i` at integer input grid centers. A target whose first
source index is `0.3` has true field value `300` and Hessian zero.
Analytical trilinear error bound: **0**. Actual float32 PyTorch interpolation
typically differs from 300 by a small nonzero value. If a program claims
"authorized maximum output error = 0" from only the Hessian bound, that
**overstates what was verified about the executed arithmetic**.

This is a first-party correctness counterexample to conflating two proof
obligations, **not** a cryptographic collision, a new interpolation theorem,
a MONAI/TorchIO defect, or clinical validation.

## Constructive method: exact-rational executed-op witness

`executed_interpolation_witness.py` performs a CPU-only, fail-closed
*a posteriori* audit of the actual PyTorch repair on selected output voxels:

1. Validate the existing lattice repair plan, semantic role and source shape.
2. Replay actual CPU `LatticeRepair3d` on exactly the supplied tensor.
3. Reject a caller-provided output that differs from the replay.
4. Use the actual 64-bit source indices computed by the wrapper, and regard
   each stored `float32` or `float64` input sample as an **exact binary rational**.
5. Independently form the exact tensor-product trilinear barycentric
   combination with Python `fractions.Fraction` (up to eight vertices).
6. Compare each exact rational result with the actual stored output bit value,
   and **round the maximum residual upward** to a finite IEEE double.
7. Optionally add a separately justified analytic interpolation bound and
   round that combined upper bound upward.
8. Bind the witness to the full source tensor, output tensor, plan digest,
   computed source-coordinate bytes, output dtype and audited element count.

This converts an observed mismatch into a replayable, execution-bound
certificate against a **specific mathematical reference**.

## Explicitly outside the claimed proof

- It does **not** certify the difference between the ideal patient coordinate
  and the stored float64 index map. A separate coordinate rounding / first
  derivative error bound is needed for an end-to-end patient-space value bound.
- It does **not** prove externally supplied Hessian/curvature premises.
- The witness is per-execution and CPU-specific; CUDA backends are rejected.
- Its exact Fraction implementation is expensive: a hard `max_elements`
  budget refuses unrestricted large-volume auditing. This first version is
  aimed at selected high-risk voxels, regression or independent review.
- SHA-256 source/output hashes are deterministic provenance identifiers,
  not cryptographic signatures or independent external attestation.
- No model-task Dice gain, clinical outcome or upstream adoption is asserted.

## Direct reproduction

```bash
python -m pip install numpy pytest torch
PYTHONPATH=. python -m pytest -q research/medical_geometry/test_executed_interpolation_witness.py
PYTHONPATH=. python -m research.medical_geometry.benchmark_executed_interpolation_witness
```

The benchmark emits JSON showing the ideal value, actual native FP32 value,
their discrepancy and the independent exact-rational numerical residual.
A dedicated public CI matrix runs this on two supported Python versions.

## Research direction if replicated externally

The genuine contribution worth pursuing is **proof-preserving numerical
execution of spatial feature alignment**, including certified coordinate-map
roundoff, sparse targeted auditing and separate runtime/error provenance.
The zero-Hessian counterexample motivates this work; it alone is not a
top-conference novelty claim.
