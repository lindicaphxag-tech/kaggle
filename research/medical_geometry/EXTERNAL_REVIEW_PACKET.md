# ClinGeom — external maintainer / independent reviewer packet

**Status: public first-party reproductions; not third-party adoption.**

This is a short, falsifiable way to review the strongest evidence. The
repository contains two independent research lines; ClinGeom studies
patient-space coordinate and feature-support semantics for medical AI.

## A. Natural external medical-library defects (highest review relevance)

### A1. `highdicom` / MONAI: missing DICOM physical frame identity

- Upstream: [`ImagingDataCommons/highdicom` PR #454](https://github.com/ImagingDataCommons/highdicom/pull/454),
  exact head `4524bb0b0c8b3141f6f5b26d047b8da43c69e61e`.
- Reproducer: [`highdicom_454_frame_identity.py`](upstream_repros/highdicom_454_frame_identity.py).
- Executable minimal patch: [`highdicom_454_frame_uid.patch`](upstream_repros/highdicom_454_frame_uid.patch).
- Frozen paired CI: [source before/after, public JSON](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37706342122).
- Falsifier: check whether two distinct FrameOfReferenceUID values are lost
  from `Volume → MetaTensor → Volume` despite preserved affine; applying
  the patch should preserve both, allow explicit UID override, and leave
  legacy missing UID metadata compatible.
- Maintainer question: should the bridge preserve DICOM UID provenance as a
  namespaced optional metadata field, with explicit arguments taking precedence?

### A2. `brain_mask`: real pretrained 3D U-Net nonaxial export error

- Upstream: [`ecalabr/brain_mask`](https://github.com/ecalabr/brain_mask),
  source `ca789fb26b70e359b61a8d52968813dd772fb4b1`.
- Pin public checkpoint, real anonymized T1 MRI and original native
  loader/patch inference/reconstruction/exporter by Git hashes.
- [Real pretrained source-head CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718977979)
  · [six-panel MRI and learned output artifact](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718977979/artifacts/11524838113)
  · [one-file exporter patch](upstream_repros/brainmask_export_plane.patch).
- **The same view's actual learned prediction** under original vs corrected
  export: axial Dice 1.000→1.000; coronal 0.004117→1.000;
  sagittal 0.363083→1.000. This is **same-prediction coordinate transport
  fidelity**, not clinical segmentation accuracy or cross-view equivariance.
- [Exact maintainer question, instructions and limits](upstream_repros/BRAINMASK_UPSTREAM_ISSUE_READY.md).
- Upstream has an unresolved license clarification; downstream redistribution
  and independent forks should first verify permissible terms.

## B. Constructive method: repair before irreversible fusion

- [Merged mechanism PR #44](https://github.com/lindicaphxag-tech/kaggle/pull/44):
  typed per-channel physical alignment **before** dense mixing.
- [Finite-precision correction PR #45](https://github.com/lindicaphxag-tech/kaggle/pull/45):
  on pinned MONAI DiNTS + native PyTorch dense mixing, different physical
  inputs yield **bitwise identical float32 post-mix tensors**, but have
  distinct physically intended target values. Therefore a post-mix-only
  inverse cannot generally exist.
- [Full green exact floating-point CI and JSON](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718214998).

```bash
PYTHONPATH=. python research/medical_geometry/benchmark_monai_bitwise_identical_fusion.py
```

Required: post-fusion difference exactly 0.0 (bitwise), target-world
difference 0.5; pre-fusion correction within frozen numerical tolerance.
Constructed analytic inputs and controlled weights—not a trained model
improvement or an upstream MONAI bug.

## How to submit a genuine independent reproduction

1. Fork at a named public SHA and use a fresh environment or your own CI.
2. Run **one** of the three experiments above without tuning criteria.
3. Publish source SHA, library versions, full raw outputs and all failed
   assertions (not just positive results).
4. Add your forked run to [replication task #9](https://github.com/lindicaphxag-tech/kaggle/issues/9)
   or [highdicom task #28](https://github.com/lindicaphxag-tech/kaggle/issues/28).

**Strict evidence separation:** green author-operated CI does not constitute
external adoption, and a patch file in this repo is **not** a merged upstream
contribution. External maintainers retain control of their API contracts.
Likewise, this repo's own licensing decision is still open;
do not assume source reuse permissions beyond applicable law.
