"""Dual signatory source+calibration trust domain adversarial tests.

All signers are generated INSIDE this test and are NOT field manufacturers.
"""
import copy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from research.semrepair.signed_source_activation import (
    PROTOCOL,_canonical,read_executable_control_config,
    execute_loaded_source_action,
)
from research.semrepair.robust_signed_source_activation import (
    PROFILE_TYPE,verify_calibration_and_activate
)

OLD={"producer":[[1.,3.5],[0.,1.]],
     "controller":[[1.,-3.5],[0.,1.]],"clip":.8}
NEW={"producer":[[1.,0.],[0.,1.]],
     "controller":[[1.,0.],[0.,1.]],"clip":2.01}
DRIFT=[[.001,.001],[.001,.001]]
ROBOT="Farama-Gymnasium-Pusher-v5-seven-torque"
DEPLOY=sha256(b"trusted-demo-deployment-Pusher-seven-actuator-box2-20261010").hexdigest()
NONCE="demonstration-unique-activation-nonce-20261010"


class TwoDomainRobustSignedSourceTests(unittest.TestCase):
    def setUp(self):
        td=TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.path=Path(td.name)/"actual-pusher-source.json"
        self.original=_canonical(OLD)
        self.new=_canonical(NEW)
        self.path.write_bytes(self.original)
        self.patch_keys={f"source-{i}":Ed25519PrivateKey.generate() for i in range(3)}
        self.cal_keys={f"cal-{i}":Ed25519PrivateKey.generate() for i in range(3)}
        def export(keys):
            return {n:k.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw) for n,k in keys.items()}
        self.patch_roots=export(self.patch_keys)
        self.cal_roots=export(self.cal_keys)
        self.manifest={
            "protocol":PROTOCOL,
            "current_source_sha256":sha256(self.original).hexdigest(),
            "replacement_source_sha256":sha256(self.new).hexdigest(),
            "current_epoch":1,"target_epoch":2,
            "deployment_sha256":DEPLOY,
            "expected_nonce":NONCE,
            "authorized_repair":"atomic",
            "actor_family":ROBOT,
            "max_abs_input_action":2.0,
        }
        self.profile={
            "protocol":PROFILE_TYPE,
            "actor_family":ROBOT,
            "deployment_sha256":DEPLOY,
            "replacement_source_sha256":sha256(self.new).hexdigest(),
            "target_epoch":2,
            "deployment_action_radius":2.0,
            "producer_abs_coeff_error":copy.deepcopy(DRIFT),
            "controller_abs_coeff_error":copy.deepcopy(DRIFT),
            "max_allowed_abs_output_action_error":.01,
            "max_authorized_internal_source_clip":2.05
        }

    @staticmethod
    def signatures(keys,bytes_,n=2):
        return [
            {"issuer":name,"ed25519_hex":keys[name].sign(bytes_).hex()}
            for name in list(keys)[:n]]

    def activate(self,*,cal_roots=None,cal_signatures=None,profile=None,
                 original_manifest=None,replacement=None,pin=None):
        p=self.profile if profile is None else profile
        m=self.manifest if original_manifest is None else original_manifest
        r=self.new if replacement is None else replacement
        blob=_canonical(p)
        mb=_canonical(m)
        return verify_calibration_and_activate(
            active_config=self.path,
            replacement_bytes=r,
            patch_manifest=m,
            patch_signatures=self.signatures(self.patch_keys,mb),
            trusted_patch_roots=self.patch_roots,
            calibration_profile_bytes=blob,
            calibration_signatures=self.signatures(self.cal_keys,blob) if cal_signatures is None else cal_signatures,
            trusted_calibration_roots=self.cal_roots if cal_roots is None else cal_roots,
            expected_calibration_profile_sha256=sha256(blob).hexdigest() if pin is None else pin,
            expected_current_epoch=1,expected_nonce=NONCE,
            expected_actor_family=ROBOT,expected_deployment_sha256=DEPLOY,
            expected_action_radius=2.0,threshold=2)

    def test_two_distinct_trusted_signature_domains_permit_robust_activation(self):
        result=self.activate()
        self.assertEqual(result["decision"],"DEMONSTRATION_SIGNED_ROBUST_SOURCE_CONFIG_INSTALLED")
        self.assertEqual(len(result["verified_source_signers"]),2)
        self.assertEqual(len(result["verified_independent_calibration_signers"]),2)
        self.assertEqual(result["cryptographic_signature_domains"],2)
        cert=result["worst_case_envelope_certificate"]
        self.assertTrue(cert["certified"])
        self.assertLess(cert["robust_internal_clip_abs_input_bound"],NEW["clip"])
        self.assertLess(cert["robust_infinity_output_error_bound"],.01)
        self.assertEqual(self.path.read_bytes(),self.new)

    def test_calibration_requires_distinct_issuer_public_keys(self):
        p=self.profile
        sig=self.signatures(self.cal_keys,_canonical(p),1)
        with self.assertRaisesRegex(ValueError,"insufficient distinct valid"):
            self.activate(cal_signatures=sig+sig+sig)
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_same_source_key_reused_as_calibration_root_rejected(self):
        bad=dict(self.cal_roots)
        bad["cal-0"]=next(iter(self.patch_roots.values()))
        with self.assertRaisesRegex(ValueError,"not independent"):
            self.activate(cal_roots=bad)
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_signed_nominally_correct_but_no_source_headroom_refused(self):
        p=copy.deepcopy(self.profile)
        new=copy.deepcopy(NEW)
        new["clip"]=2.0
        replacement=_canonical(new)
        p["replacement_source_sha256"]=sha256(replacement).hexdigest()
        m=copy.deepcopy(self.manifest)
        m["replacement_source_sha256"]=sha256(replacement).hexdigest()
        with self.assertRaisesRegex(ValueError,"cannot certify"):
            self.activate(profile=p,original_manifest=m,replacement=replacement)
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_valid_calibration_signature_with_too_strict_output_budget_refused(self):
        p=copy.deepcopy(self.profile)
        p["max_allowed_abs_output_action_error"]=.001
        with self.assertRaisesRegex(ValueError,"cannot certify"):
            self.activate(profile=p)
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_signed_two_domains_cannot_override_calibration_hard_clip_cap(self):
        p=copy.deepcopy(self.profile)
        new=copy.deepcopy(NEW)
        new["clip"]=2.11
        replacement=_canonical(new)
        p["replacement_source_sha256"]=sha256(replacement).hexdigest()
        m=copy.deepcopy(self.manifest)
        m["replacement_source_sha256"]=sha256(replacement).hexdigest()
        with self.assertRaisesRegex(ValueError,"separately signed calibrated hard cap"):
            self.activate(profile=p,original_manifest=m,replacement=replacement)
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_bad_signed_clip_cap_shape_fails_closed(self):
        p=copy.deepcopy(self.profile)
        p["max_authorized_internal_source_clip"]=float("nan")
        # Canonical serializer refuses NaN, even if supplied by trusted signers.
        with self.assertRaises(ValueError):
            self.activate(profile=p)
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_signed_cap_authorizes_actual_nominal_patch_only(self):
        report=self.activate()
        self.assertAlmostEqual(report["signed_hard_internal_source_clip_cap"],2.05)
        self.assertLessEqual(NEW["clip"],report["signed_hard_internal_source_clip_cap"])

    def test_pinned_outside_calibration_profile_sha_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError,"independently pinned digest"):
            self.activate(pin="0"*64)
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_replacement_bytes_not_bound_to_calibration_profile_refused(self):
        bad=copy.deepcopy(NEW)
        bad["producer"][0][1]=.2
        with self.assertRaisesRegex(ValueError,"does not bind exact"):
            self.activate(replacement=_canonical(bad))
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_calibration_epoch_replay_refused_despite_valid_signatures(self):
        p=copy.deepcopy(self.profile)
        p["target_epoch"]=1
        with self.assertRaisesRegex(ValueError,"does not bind exact"):
            self.activate(profile=p)
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_valid_calibration_signature_cannot_shrink_robot_actuator_contract(self):
        p=copy.deepcopy(self.profile)
        p["deployment_action_radius"]=.5
        with self.assertRaisesRegex(ValueError,"does not bind exact"):
            self.activate(profile=p)
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_approved_robust_installed_config_runs_original_pusher_physics(self):
        import numpy as np
        import gymnasium as gym
        self.activate()
        cfg=read_executable_control_config(self.path,sha256(self.new).hexdigest())
        desired=np.asarray((1.8,1.6,.3,-.2,0.,.5,-.5),dtype=np.float32)
        actual=execute_loaded_source_action(desired,cfg)
        np.testing.assert_allclose(actual,desired,atol=1e-7,rtol=0)
        with gym.make("Pusher-v5") as env:
            env.reset(seed=23011)
            obs_good,r_good,_,_,info_good=env.step(desired)
            env.reset(seed=23011)
            obs_installed,r_installed,_,_,info_installed=env.step(actual)
            np.testing.assert_allclose(obs_good,obs_installed,atol=1e-10,rtol=0)
            self.assertAlmostEqual(float(r_good),float(r_installed),places=10)
            self.assertAlmostEqual(info_good["reward_dist"],info_installed["reward_dist"],places=10)


if __name__=="__main__":
    unittest.main()
