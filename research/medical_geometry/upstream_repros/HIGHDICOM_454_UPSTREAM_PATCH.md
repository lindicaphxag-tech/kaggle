# highdicom #454 — directly applicable two-site source patch

**Target:** [ImagingDataCommons/highdicom PR #454](https://github.com/ImagingDataCommons/highdicom/pull/454), exact commit `4524bb0b0c8b3141f6f5b26d047b8da43c69e61e`.

The standard unified diff [`highdicom_454_frame_uid.patch`](highdicom_454_frame_uid.patch) makes exactly two changes to `src/highdicom/volume.py`:

1. In `Volume.to_monai()`, preserve a known DICOM `FrameOfReferenceUID` as highdicom-owned MetaTensor metadata.
2. In `Volume.from_monai()`, use that metadata when **and only when** the caller has not supplied an explicit frame UID.

The proposal does **not** infer UIDs from voxel-to-world matrices, claim identity from equal affines, modify the DICOM/MONAI coordinate-frame transform, or extend MONAI's public metadata contract.

## Why this fixes something real

On the pinned public #454 source, the original paired validation recorded **0.0 affine error** while two distinct source FrameOfReferenceUID values were not recovered by implicit `Volume.from_monai` round-tripping. Explicit arguments did preserve identity.

Paired baseline/fix CI: [public workflow](https://github.com/lindicaphxag-tech/kaggle/actions/workflows/clingeom-highdicom-454-patched.yml).

The regression checks two distinct UIDs with equal affine, explicit override, missing legacy metadata, RAS/LPS geometry and absence of numerical affine change.

## Apply to the **exact source** for review

```bash
git clone https://github.com/ImagingDataCommons/highdicom.git upstream-highdicom
cd upstream-highdicom
git fetch origin 4524bb0b0c8b3141f6f5b26d047b8da43c69e61e
git checkout --detach 4524bb0b0c8b3141f6f5b26d047b8da43c69e61e
git apply --check /absolute/path/to/highdicom_454_frame_uid.patch
git apply /absolute/path/to/highdicom_454_frame_uid.patch
git diff --check
git diff -- src/highdicom/volume.py
```

This `.patch` is intended for the pinned audited branch, **not** an unverified later main. If the source has changed, re-review the two insertion sites rather than force-applying.

## Independent bytewise patch equivalence

The dedicated CI workflow `clingeom-highdicom-454-diff.yml`:

- fetches the exact upstream Git commit;
- runs `git apply --check` and applies the unified diff;
- independently applies the existing `patch_highdicom_454.py` to an unmodified second copy;
- compares the resulting production source **byte for byte**;
- compiles the patched Python source.

The previous paired MONAI/highdicom runtime workflow separately tests actual behavior. Neither check constitutes upstream acceptance.

## Upstream etiquette / status

GitHub App write access to third-party repositories has returned HTTP 403. This patch currently lives in the author's public research repository and has **not** been proposed or merged as a pull request to highdicom.

Highdicom maintainers decide whether frame UID transport belongs in this bridge and what naming/tests are appropriate. Please do not interpret author-operated CI as third-party endorsement.
