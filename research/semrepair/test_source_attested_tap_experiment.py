import unittest
from hashlib import sha256

from research.semrepair.source_attested_tap_experiment import (
    study,issue_measurement,authorized_on_verified_trace,
)
from research.semrepair.mujoco_native_authority import WORLDS,rollout
from research.semrepair.open_catalog_source_gap import known_source_digest


class AttestationAgainstForceSpoofTests(unittest.TestCase):
    def test_authentic_physics_and_post_signing_spoof_matrix(self):
        z=study()
        self.assertEqual(z["authentic_legitimate_authorized"],12)
        self.assertEqual(z["post_signing_force_substitution_refused"],6)
        self.assertEqual(z["stale_nonce_replays_refused"],12)

    def test_wrong_verifier_key_refuses(self):
        key=sha256(b"distinct-research-only-test-key-1").digest()
        wrong=sha256(b"distinct-research-only-test-key-2").digest()
        w=next(w for w in WORLDS if w.name=="gain_fault")
        sample=rollout(w)
        signatures={q.name:rollout(q) for q in WORLDS}
        record=issue_measurement(
            trusted_issuer_key=key,source_sha256=known_source_digest(w),
            observations=sample,nonce="current-challenge")
        failure=authorized_on_verified_trace(
            trusted_verifier_key=wrong,evidence=record,
            observations=sample,requested_nonce="current-challenge",
            catalog_templates=signatures)
        self.assertEqual(failure["answer"],"REFUSE")
        self.assertFalse(failure["attestation_verified"])

    def test_opaque_unauthorized_source_rejected(self):
        key=sha256(b"research-only-test-key-for-origin-check").digest()
        w=next(w for w in WORLDS if w.name=="gain_fault")
        sample=rollout(w)
        record=issue_measurement(
            trusted_issuer_key=key,source_sha256=sha256(b"unknown-pkg").hexdigest(),
            observations=sample,nonce="fresh-nonce")
        failure=authorized_on_verified_trace(
            trusted_verifier_key=key,evidence=record,
            observations=sample,requested_nonce="fresh-nonce",
            catalog_templates={q.name:rollout(q) for q in WORLDS})
        self.assertEqual(failure["answer"],"REFUSE")


if __name__=="__main__":
    unittest.main()
