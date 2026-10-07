# Minimal upstream patch proposal for highdicom #454

Target audited head: `4524bb0b0c8b3141f6f5b26d047b8da43c69e61e`.

Goal: preserve `Volume.frame_of_reference_uid` through the new
`Volume <-> monai.data.MetaTensor` bridge without changing existing affine or
RAS/LPS behavior.

## Minimal semantic rule

1. `Volume.to_monai()` writes the UID only when it is present.
2. The metadata key is namespaced to highdicom.
3. `Volume.from_monai(..., frame_of_reference_uid=...)` keeps an explicit
   caller argument authoritative.
4. Only when that argument is `None` does the method fall back to metadata.
5. Missing metadata remains backward-compatible and yields `None`.

Suggested metadata key:

```python
HIGHDICOM_FRAME_OF_REFERENCE_UID = "highdicom_frame_of_reference_uid"
```

Suggested `to_monai()` addition after `meta = {}` / spatial metadata setup:

```python
if self.frame_of_reference_uid is not None:
    meta[HIGHDICOM_FRAME_OF_REFERENCE_UID] = str(
        self.frame_of_reference_uid
    )
```

Suggested `from_monai()` resolution before constructing `Volume`:

```python
if frame_of_reference_uid is None:
    frame_of_reference_uid = metatensor.meta.get(
        HIGHDICOM_FRAME_OF_REFERENCE_UID
    )
```

## Required regressions

- automatic UID round-trip;
- explicit argument overrides metadata;
- missing metadata stays `None`;
- two equal-affine Volumes with distinct UIDs remain distinguishable after round-trip;
- existing RAS/LPS tests remain independent of UID identity tests.

## Why the patch stays narrow

It deliberately does **not** infer `PATIENT`/`SLIDE` semantics, change MONAI
channel layout, change affine convention handling, or attach DICOM semantics
to MONAI generally. It preserves one piece of identity owned by a highdicom
`Volume` across a highdicom-owned conversion API.
