"""Executable legacy semantic-hash collision vs physically bound v3 certificate.

This is a collision in the *omitted input fields* of the old fingerprint,
not a cryptographic SHA-256 collision or third-party clinical evidence.
"""

import json
import numpy as np
import torch

from research.medical_geometry.test_semantic_certificate_binding import run,case
from hashlib import sha256


def legacy_v2_digest(result,source_bundle,mixer):
    """Reconstruct the old v2 digest protocol for the same repair results."""
    digest=sha256(b"clingeom-pre-mix-1x1-support-envelope-v2")
    for p in (mixer.weight,mixer.bias):
        if p is None:
            digest.update(b"NONE")
        else:
            a=p.detach().cpu().contiguous().numpy()
            digest.update(str((tuple(a.shape),str(a.dtype))).encode("ascii"))
            digest.update(a.tobytes())
    for s in result.source_repair_digests:
        digest.update(s.encode("ascii"))
    for fiber in source_bundle.fibers:
        for support in fiber.supports:
            digest.update(np.asarray(support.receptive_field,dtype="<f8").tobytes())
    digest.update(np.asarray(result.output_receptive_field_envelope,dtype="<f8").tobytes())
    return digest.hexdigest()


def main():
    _,bundle,_,mix=case()
    a=run(spacing=(1,1,1),frame_uid="1.2.3")
    b=run(spacing=(1,1,5),frame_uid="1.2.3")
    c=run(spacing=(1,1,1),frame_uid="1.2.4")
    before_a=legacy_v2_digest(a,bundle,mix)
    before_b=legacy_v2_digest(b,bundle,mix)
    before_c=legacy_v2_digest(c,bundle,mix)
    result={
        "experiment":"same tensor bytes and model, different patient chart/frame",
        "identical_output_tensors_for_all_three":bool(torch.equal(a.tensor,b.tensor) and torch.equal(a.tensor,c.tensor)),
        "voxel_plan_digests_identical":bool(a.source_repair_digests==b.source_repair_digests==c.source_repair_digests),
        "mm_scale_case_a":[1,1,1],
        "mm_scale_case_b":[1,1,5],
        "value_error_bounds_a":list(a.per_output_value_error_bounds),
        "value_error_bounds_b":list(b.per_output_value_error_bounds),
        "legacy_v2_same_digest_across_all":bool(before_a==before_b==before_c),
        "v3_digest_varies_with_mm_scale":a.conv_weight_provenance_sha256!=b.conv_weight_provenance_sha256,
        "v3_digest_varies_with_UID":a.conv_weight_provenance_sha256!=c.conv_weight_provenance_sha256,
        "legacy_fingerprint_a":before_a,
        "v3_semantic_fingerprint_a":a.conv_weight_provenance_sha256,
        "v3_semantic_fingerprint_b":b.conv_weight_provenance_sha256,
        "v3_semantic_fingerprint_uid_change":c.conv_weight_provenance_sha256,
        "claim_boundary":"first-party semantic-configuration fingerprint; no signature or verification of source UID truth"
    }
    assert result["identical_output_tensors_for_all_three"]
    assert result["voxel_plan_digests_identical"]
    assert result["value_error_bounds_a"]!=result["value_error_bounds_b"]
    assert result["legacy_v2_same_digest_across_all"]
    assert result["v3_digest_varies_with_mm_scale"]
    assert result["v3_digest_varies_with_UID"]
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":
    main()