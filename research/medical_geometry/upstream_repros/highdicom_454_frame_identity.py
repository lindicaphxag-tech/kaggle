"""Public executable witness for highdicom PR #454.

Audited upstream commit:
4524bb0b0c8b3141f6f5b26d047b8da43c69e61e

Claim boundary:
- geometry round-trip is expected to succeed;
- automatic DICOM FrameOfReferenceUID round-trip is expected to be absent;
- explicit caller-supplied UID restoration is expected to succeed;
- this is not a claim that the open upstream PR has accepted the diagnosis.
"""

from __future__ import annotations

import json

import numpy as np

from highdicom.volume import Volume


AUDITED_HIGHDICOM_SHA = "4524bb0b0c8b3141f6f5b26d047b8da43c69e61e"
MONAI_VERSION = "1.6.1"
UID_A = "1.2.826.0.1.3680043.10.511.3.12345"
UID_B = "1.2.826.0.1.3680043.10.511.3.54321"


def make_volume(uid: str) -> Volume:
    return Volume.from_attributes(
        array=np.zeros((3, 4, 5), dtype=np.float32),
        image_position=(10.0, -20.0, 30.0),
        image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        pixel_spacing=(0.7, 0.8),
        spacing_between_slices=1.5,
        coordinate_system="PATIENT",
        frame_of_reference_uid=uid,
    )


def uid_text(value) -> str | None:
    return None if value is None else str(value)


def main() -> None:
    left = make_volume(UID_A)
    right = make_volume(UID_B)

    left_meta = left.to_monai()
    right_meta = right.to_monai()

    left_roundtrip = Volume.from_monai(left_meta)
    right_roundtrip = Volume.from_monai(right_meta)
    explicit = Volume.from_monai(left_meta, frame_of_reference_uid=UID_A)

    affine_error = float(np.max(np.abs(left_roundtrip.affine - left.affine)))
    same_meta_affine = bool(
        np.allclose(
            left_meta.affine.cpu().detach().numpy(),
            right_meta.affine.cpu().detach().numpy(),
            atol=0.0,
            rtol=0.0,
        )
    )

    report = {
        "audited_highdicom_sha": AUDITED_HIGHDICOM_SHA,
        "monai_version": MONAI_VERSION,
        "geometry_roundtrip_max_abs_error": affine_error,
        "source_uid_a": uid_text(left.frame_of_reference_uid),
        "source_uid_b": uid_text(right.frame_of_reference_uid),
        "automatic_roundtrip_uid_a": uid_text(left_roundtrip.frame_of_reference_uid),
        "automatic_roundtrip_uid_b": uid_text(right_roundtrip.frame_of_reference_uid),
        "explicit_roundtrip_uid_a": uid_text(explicit.frame_of_reference_uid),
        "same_meta_affine_for_distinct_source_uids": same_meta_affine,
        "automatic_uid_preserved": uid_text(left_roundtrip.frame_of_reference_uid) == UID_A,
        "explicit_uid_preserved": uid_text(explicit.frame_of_reference_uid) == UID_A,
    }

    np.testing.assert_allclose(left_roundtrip.affine, left.affine, atol=1e-6, rtol=0.0)
    assert same_meta_affine
    assert left.frame_of_reference_uid is not None
    assert right.frame_of_reference_uid is not None
    assert left_roundtrip.frame_of_reference_uid is None
    assert right_roundtrip.frame_of_reference_uid is None
    assert uid_text(explicit.frame_of_reference_uid) == UID_A

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
