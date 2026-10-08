# Real three-dimensional MRI witness: ClinGeom

This experiment uses an **actual structural brain MRI** supplied with the
[NiBabel coordinate-system tutorial](https://nipy.org/nibabel/coordinate_systems.html).
It is a separate, modest evidence tier from synthetic-only tests.

- Official repository: `nipy/nibabel`
- Pinned commit: `d6a0ba0056445c06407afb421749a8ed5c939c56`
- Pinned file: `doc/source/downloads/someones_anatomy.nii.gz`
- Git blob: `b7e917e3f891c74fd99cdf82ec579ee254a195d0`
- Code/data attribution: NiBabel `COPYING` and source documentation; keep their
  original notices when reusing the image
- No model checkpoint, expert tumor mask, NIfTI patient ID or clinical diagnosis
  is used.

## What is real and what is controlled?

**Real:** acquired MRI voxel intensities, three-dimensional shape, affine
geometry, input voxel grid and the outputs of actual PyTorch operations
on these real voxel values.

**Controlled:** an explicitly synthetic *marker ROI* in the volume (not a
radiologist segmentation), an exact signed-axis array flip with intentionally
stale affine metadata, and a fixed non-learned 3-D convolutional box filter.

This avoids invented CT/MRI anatomy. It also avoids pretending that a controlled
marker is a real lesion.

## What will be generated?

`fig1_real_mri_patient_space.png` — three panels with the **same unchanged
MRI background and voxel reference grid**; expert/clinical terms must not be
applied to the synthetic marker. It reports overlap after a joint wrong reindex,
physical-space overlap on a fixed reference, actual centroid displacement and
the exact restoration from `A_out = A_in H`.

`fig3_real_mri_operator_lattices.png` — input real MRI crop, **actual**
`Conv3d(k=3,s=4,p=1)` and `AvgPool3d(k=4,s=4,p=0)` responses, plus their
different sampled centers on the same MRI grid. This is a deterministic
operator-level witness, **not** pretrained model inference.

`real_mri_witness.json` — source Git object, source SHA-256, affine, shape,
operator description, exact clinical claim boundary, all measured errors, and
output-array SHA-256 digests.

## How to reproduce

From this public repository's root:

```bash
python -m pip install "numpy<3" "nibabel>=5.2,<6" "scipy>=1.11" "matplotlib>=3.8"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
curl -fL --retry 3 "https://raw.githubusercontent.com/nipy/nibabel/d6a0ba0056445c06407afb421749a8ed5c939c56/doc/source/downloads/someones_anatomy.nii.gz" -o real_anatomy.nii.gz
test "$(git hash-object real_anatomy.nii.gz)" = "b7e917e3f891c74fd99cdf82ec579ee254a195d0"
python -m research.medical_geometry.real_mri_witness --image real_anatomy.nii.gz --out artifacts/real_mri
```

The [`clingeom-real-mri.yml`](../../.github/workflows/clingeom-real-mri.yml)
GitHub Actions workflow downloads and checks the same actual file, runs the
same code and uploads the PNG/JSON files as a **publicly downloadable artifact**
after successful completion.

## What this cannot claim

This is a real-image **representation experiment**, not a real-lesion repair,
frozen-model improvement, clinical safety certificate, external adoption,
independent third-party reproduction or MONAI/TorchIO bug report.

A separate public dataset with an expert clinical segmentation is necessary
before upgrading Fig. 1 to a clinical ROI experiment. Likewise a frozen
pretrained model and its actual predictions are prerequisites for Fig. 4.
