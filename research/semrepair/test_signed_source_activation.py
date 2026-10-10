"""Adversarial cryptographic provenance and actual native robot patch activation."""
import copy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from research.semrepair.signed_source_activation import (
    _canonical,PROTOCOL,read_executable_control_config,
    verified_atomic_source_activate,execute_loaded_source_action,
)

OLD={"producer":[[1.,3.5],[0.,1.]],
     "controller":[[1.,-3.5],[0.,1.]],"clip":.8}
# Atomic repair corrects BOTH erroneous maps and the artificial inner
# source limit, restoring the official native Pusher action contract [-2,2].
NEW={"producer":[[1.,0.],[0.,1.]],
     "controller":[[1.,0.],[0.,1.]],"clip":2.}
ROBOT="Farama-Gymnasium-Pusher-v5-seven-torque"
DEPLOYMENT=sha256(b"precommitted Pusher action [-2,2]^7 and baseline source envelope").hexdigest()
NONCE="separate-no-replay-challenge-20261010-001"


class SignedSourceActivationTests(unittest.TestCase):
    def setUp(self):
        t=TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.path=Path(t.name)/"active-source.json"
        self.old=_canonical(OLD)
        self.new=_canonical(NEW)
        self.path.write_bytes(self.old)
        self.keys={name:Ed25519PrivateKey.generate() for name in ("owner-A","owner-B","auditor-C")}
        self.roots={name:k.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,format=serialization.PublicFormat.Raw)
            for name,k in self.keys.items()}
        self.manifest={
            "protocol":PROTOCOL,
            "current_source_sha256":sha256(self.old).hexdigest(),
            "replacement_source_sha256":sha256(self.new).hexdigest(),
            "current_epoch":1,"target_epoch":2,
            "deployment_sha256":DEPLOYMENT,
            "expected_nonce":NONCE,
            "authorized_repair":"atomic",
            "actor_family":ROBOT,
        }

    def signatures(self,owners=("owner-A","auditor-C"),manifest=None):
        blob=_canonical(self.manifest if manifest is None else manifest)
        return [{"issuer":who,"ed25519_hex":self.keys[who].sign(blob).hex()}
                for who in owners]

    def activate(self,signatures=None,manifest=None,replacement=None,nonce=NONCE):
        return verified_atomic_source_activate(
            active_config=self.path,
            replacement_bytes=self.new if replacement is None else replacement,
            manifest=self.manifest if manifest is None else manifest,
            signatures=self.signatures() if signatures is None else signatures,
            trusted_public_roots=self.roots,
            expected_current_epoch=1,
            expected_nonce=nonce,
            expected_actor_family=ROBOT,
            expected_deployment_sha256=DEPLOYMENT,
            threshold=2)

    def test_genuine_quorum_activates_actual_verified_disk_bytes(self):
        r=self.activate()
        self.assertEqual(r["activated_epoch"],2)
        self.assertEqual(r["endorsement_threshold"],2)
        self.assertEqual(set(r["verified_distinct_root_issuers"]),{"owner-A","auditor-C"})
        cfg=read_executable_control_config(self.path,sha256(self.new).hexdigest())
        self.assertEqual(cfg,NEW)
        self.assertEqual(self.path.read_bytes(),self.new)

    def test_one_signer_rejected_and_original_stays(self):
        with self.assertRaisesRegex(ValueError,"insufficient distinct"):
            self.activate(signatures=self.signatures(("owner-A",)))
        self.assertEqual(self.path.read_bytes(),self.old)

    def test_duplicate_signer_cannot_simulate_multi_source_quorum(self):
        sig=self.signatures(("owner-A",))[0]
        with self.assertRaisesRegex(ValueError,"insufficient distinct"):
            self.activate(signatures=[sig,sig,sig])
        self.assertEqual(self.path.read_bytes(),self.old)

    def test_forged_second_signature_rejected(self):
        sig=self.signatures(("owner-A","owner-B"))
        sig[1]["ed25519_hex"]="00"*64
        with self.assertRaisesRegex(ValueError,"insufficient distinct"):
            self.activate(signatures=sig)
        self.assertEqual(self.path.read_bytes(),self.old)

    def test_tampered_patch_bytes_fail_even_with_genuine_signatures(self):
        modified=copy.deepcopy(NEW)
        modified["producer"][0][1]=1.1
        with self.assertRaisesRegex(ValueError,"replacement bytes differ"):
            self.activate(replacement=_canonical(modified))
        self.assertEqual(self.path.read_bytes(),self.old)

    def test_nonce_replay_rejected_and_original_unmodified(self):
        with self.assertRaisesRegex(ValueError,"challenge"):
            self.activate(nonce="reused-message-with-different-id")
        self.assertEqual(self.path.read_bytes(),self.old)

    def test_rollback_epoch_refused(self):
        m=copy.deepcopy(self.manifest)
        m["target_epoch"]=1
        with self.assertRaisesRegex(ValueError,"epoch"):
            self.activate(manifest=m,signatures=self.signatures(manifest=m))

    def test_signed_issuer_cannot_expand_from_producer_only_to_atomic(self):
        m=copy.deepcopy(self.manifest)
        m["authorized_repair"]="producer-only"
        with self.assertRaisesRegex(ValueError,"outside signed allowed"):
            self.activate(manifest=m,signatures=self.signatures(manifest=m))

    def test_signed_controller_patch_cannot_change_hidden_saturation(self):
        # Two real issuer signatures do NOT authorize bypassing signed
        # component scope. A signed controller repair cannot silently increase
        # the producer's internal actuator authority clip.
        r=copy.deepcopy(OLD)
        r["controller"]=[[1.,0.],[0.,1.]]
        r["clip"]=2.
        replacement=_canonical(r)
        manifest=copy.deepcopy(self.manifest)
        manifest["authorized_repair"]="controller-only"
        manifest["replacement_source_sha256"]=sha256(replacement).hexdigest()
        with self.assertRaisesRegex(ValueError,"internal source limit changed"):
            self.activate(manifest=manifest,signatures=self.signatures(manifest=manifest),
                          replacement=replacement)
        self.assertEqual(self.path.read_bytes(),self.old)

    def test_vendor_env_or_deployment_contract_mismatch_refuses(self):
        m=copy.deepcopy(self.manifest)
        m["actor_family"]="AnotherRobot"
        with self.assertRaisesRegex(ValueError,"robot family"):
            self.activate(manifest=m,signatures=self.signatures(manifest=m))
        m=copy.deepcopy(self.manifest)
        m["deployment_sha256"]=sha256(b"other action contract").hexdigest()
        with self.assertRaisesRegex(ValueError,"deployment envelope"):
            self.activate(manifest=m,signatures=self.signatures(manifest=m))

    def test_source_tampering_before_install_refused(self):
        self.path.write_bytes(_canonical({
            "producer":[[1.,2.5],[0.,1.]],
            "controller":OLD["controller"],"clip":.8}))
        with self.assertRaisesRegex(ValueError,"original source bytes differ"):
            self.activate()

    def test_post_activation_tampering_detected_on_load(self):
        self.activate()
        self.path.write_bytes(self.old)
        with self.assertRaisesRegex(ValueError,"modified after activation"):
            read_executable_control_config(self.path,sha256(self.new).hexdigest())

    def test_real_official_pusher_mujoco_one_step_matches_after_actual_config_activation(self):
        import numpy as np
        import gymnasium as gym
        original=read_executable_control_config(self.path,sha256(self.old).hexdigest())
        requested=np.array([1.7,1.5,.3,-.4,.1,-.6,.2],dtype=np.float32)
        wrong=execute_loaded_source_action(requested,original)
        self.assertGreater(float(np.linalg.norm(np.clip(wrong,-2.,2.)-requested)),.1)
        self.activate()
        correct=read_executable_control_config(self.path,sha256(self.new).hexdigest())
        patched=execute_loaded_source_action(requested,correct)
        np.testing.assert_allclose(patched,requested,rtol=0,atol=1e-7)
        with gym.make("Pusher-v5") as env:
            env.reset(seed=23011)
            q_before=env.unwrapped.data.qpos.copy()
            _,reference_reward,_,_,_=env.step(requested)
            ref_state=env.unwrapped.data.qpos.copy()
            env.reset(seed=23011)
            _,repaired_reward,_,_,_=env.step(np.clip(patched,-2,2))
            np.testing.assert_allclose(ref_state,env.unwrapped.data.qpos,rtol=0,atol=1e-9)
            self.assertAlmostEqual(float(reference_reward),float(repaired_reward),places=9)
            env.reset(seed=23011)
            _,fault_reward,_,_,_=env.step(np.clip(wrong,-2.,2))
            self.assertGreater(abs(float(reference_reward)-float(fault_reward)),1e-5)


if __name__=="__main__":
    unittest.main()
