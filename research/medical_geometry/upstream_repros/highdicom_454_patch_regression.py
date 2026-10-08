"""Post-patch assertions for the exact upstream highdicom #454 source checkout."""

import json
import numpy as np

from highdicom.volume import Volume
from research.medical_geometry.upstream_repros.highdicom_454_frame_identity import (
    UID_A, UID_B, make_volume, uid_text,
)


def main() -> None:
    a, b = make_volume(UID_A), make_volume(UID_B)
    original_affines = [a.affine.copy(), b.affine.copy()]
    meta_a, meta_b = a.to_monai(), b.to_monai()
    round_a, round_b = Volume.from_monai(meta_a), Volume.from_monai(meta_b)
    explicit = Volume.from_monai(meta_a, frame_of_reference_uid=UID_B)

    # Remove only the highdicom-owned UID field, leaving geometry untouched.
    legacy_meta = a.to_monai()
    legacy_meta.meta.pop('highdicom_frame_of_reference_uid', None)
    legacy = Volume.from_monai(legacy_meta)

    # Exercise convention selection independently of DICOM frame identity.
    lps_meta = a.to_monai(convert_to_ras=False)
    lps_round = Volume.from_monai(lps_meta)

    error = max(
        float(np.max(np.abs(v.affine - affine)))
        for v, affine in zip((round_a, round_b), original_affines)
    )
    report = {
        'pinned_upstream_head': '4524bb0b0c8b3141f6f5b26d047b8da43c69e61e',
        'patch_applied': True,
        'affine_max_abs_error': error,
        'automatic_uid_a': uid_text(round_a.frame_of_reference_uid),
        'automatic_uid_b': uid_text(round_b.frame_of_reference_uid),
        'explicit_override_uid': uid_text(explicit.frame_of_reference_uid),
        'legacy_metadata_uid': uid_text(legacy.frame_of_reference_uid),
        'lps_roundtrip_uid': uid_text(lps_round.frame_of_reference_uid),
        'distinct_uid_preserved': uid_text(round_a.frame_of_reference_uid) != uid_text(round_b.frame_of_reference_uid),
    }
    np.testing.assert_allclose(round_a.affine, a.affine, atol=1e-6, rtol=0)
    np.testing.assert_allclose(round_b.affine, b.affine, atol=1e-6, rtol=0)
    np.testing.assert_allclose(lps_round.affine, a.affine, atol=1e-6, rtol=0)
    assert uid_text(round_a.frame_of_reference_uid) == UID_A
    assert uid_text(round_b.frame_of_reference_uid) == UID_B
    assert uid_text(explicit.frame_of_reference_uid) == UID_B
    assert legacy.frame_of_reference_uid is None
    assert uid_text(lps_round.frame_of_reference_uid) == UID_A
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
