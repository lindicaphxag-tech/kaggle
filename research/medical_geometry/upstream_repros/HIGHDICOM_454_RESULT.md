# highdicom #454 — physical-frame identity witness

Status: **public execution witness; not upstream confirmation**.

## Audited runtime

- highdicom open PR: `ImagingDataCommons/highdicom#454`
- highdicom commit: `4524bb0b0c8b3141f6f5b26d047b8da43c69e61e`
- MONAI: `1.6.1`
- Python: `3.12`
- public workflow run: `37475115875`
- workflow result: **success**
- uploaded artifact digest:
  `sha256:7d5ff2768428049acf40f1e70292cbfb83de9cbbcb420b34b070ad91fd1297f4`

## Executed JSON result

```json
{
  "audited_highdicom_sha": "4524bb0b0c8b3141f6f5b26d047b8da43c69e61e",
  "automatic_roundtrip_uid_a": null,
  "automatic_roundtrip_uid_b": null,
  "automatic_uid_preserved": false,
  "explicit_roundtrip_uid_a": "1.2.826.0.1.3680043.10.511.3.12345",
  "explicit_uid_preserved": true,
  "geometry_roundtrip_max_abs_error": 0.0,
  "monai_version": "1.6.1",
  "same_meta_affine_for_distinct_source_uids": true,
  "source_uid_a": "1.2.826.0.1.3680043.10.511.3.12345",
  "source_uid_b": "1.2.826.0.1.3680043.10.511.3.54321"
}
```

## Interpretation

The audited conversion path preserves the numerical voxel-to-world affine on
this witness exactly (`max_abs_error = 0.0`). However, two source Volumes with
different DICOM FrameOfReferenceUID values can produce MetaTensors with the
same affine, and automatic `Volume -> MetaTensor -> Volume` reconstruction
returns `frame_of_reference_uid=None` for both.

Passing an explicit UID to `Volume.from_monai()` still preserves it, so the
issue is not that highdicom cannot represent the identity. The missing piece
is automatic transport of that highdicom-owned physical identity across the
new bridge.

This is the concrete distinction ClinGeom/MedPSC makes between:

1. **numerical geometry correctness**, and
2. **clinical physical-frame identity preservation**.

Equal affines do not prove equal DICOM Frames of Reference.

## Claim boundary

- This result is not a claim that the highdicom maintainers have accepted the diagnosis.
- It is not a clinical-safety claim.
- It is not a claim that MONAI itself owns DICOM FrameOfReferenceUID semantics.
- It is a reproducible execution result on one exact public upstream revision.

## Reproduction

Run:

```bash
python research/medical_geometry/upstream_repros/highdicom_454_frame_identity.py
```

under the pinned environment in `.github/workflows/clingeom-highdicom-454.yml`.
