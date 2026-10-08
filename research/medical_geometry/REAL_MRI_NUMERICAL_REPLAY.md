# Genuine brain-MRI evidence for execution-bound spatial interpolation

Runs the **already merged** ClinGeom exact-rational CPU replay auditor on a
real NiBabel tutorial structural MRI, rather than on only a linear numeric
counterexample. The source NIfTI comes from immutable NiBabel commit
`d6a0ba0056445c06407afb421749a8ed5c939c56`, Git blob
`b7e917e3f891c74fd99cdf82ec579ee254a195d0`. The upstream NiBabel
documentation/examples are covered by its [COPYING terms](https://github.com/nipy/nibabel/blob/master/COPYING);
retain original attribution and do not make a clinical diagnosis claim.

## Execution

- Take a **predeclared central 8³** crop of the acquired 57×67×56 MRI.
- Preserve the actual NIfTI world affine with its explicit crop offset.
- Define the **same** 3³ target feature grid at source positions
  `[.3+2i,.3+2j,.3+2k]`.
- Execute real CPU PyTorch trilinear interpolation in `float32` and `float64`
  on identical stored MRI voxel values, without a trained model.
- Check each of the 27 source-index sampling outputs in each dtype against a
  separate **exact rational barycentric** reference constructed from stored
  IEEE-754 inputs and actual computed float64 coordinates.
- Bind fingerprints for source, output, index coordinates and repair plan;
  save raw numerical residuals, exact input patient affine and a genuine MRI
  triptych (input crop locator, FP32 output and FP32/FP64 discrepancy).

## What this is **not**

It is not a pretrained model task-performance experiment, clinical validation
or external-researcher reproduction. No externally verified patient-space
Hessian bound is known for this real image; therefore the report intentionally
**does not** produce a total bound on error relative to unknown ideal
continuous image intensities. It verifies the numerical execution residual of
two actual finite-precision operators relative to their exact stored-value
reference.

It also does not prove the ideal-to-computed physical coordinate transform is
free of rounding error. That is a separate outstanding obligation for complete
patient-space value correctness.

## Replay

```bash
python -m pip install numpy nibabel matplotlib torch
curl -fL "https://raw.githubusercontent.com/nipy/nibabel/d6a0ba0056445c06407afb421749a8ed5c939c56/doc/source/downloads/someones_anatomy.nii.gz" -o actual_brain_mri.nii.gz
test "$(git hash-object actual_brain_mri.nii.gz)" = b7e917e3f891c74fd99cdf82ec579ee254a195d0
PYTHONPATH=. python -m research.medical_geometry.benchmark_real_mri_numerical_replay --image actual_brain_mri.nii.gz --out artifacts/real_mri_execution
```

The public CI reproduces the full chain and makes the PNG and JSON available
as GitHub Actions artifacts after it passes.
