import unittest

from research.semrepair.native_tap_integrity_attack import (
    ALIAS_PAIRS, HELD_EXCITATIONS, WORLDS, decide_with_source_assumption,
    experiment,
)


class NativeTapIntegrityAttackTests(unittest.TestCase):
    def test_spoofed_untrusted_deep_tap_changes_repair_authority(self):
        report=experiment()
        self.assertEqual(report["attack_cells"],6)
        self.assertEqual(report["naive_wrong_authorizations"],6)
        self.assertEqual(report["source_untrusted_refusals"],6)
        self.assertEqual(report["authentic_tap_correct"],6)

    def test_true_physics_in_alternative_worlds_alias_at_upstream_taps(self):
        from research.semrepair.mujoco_native_authority import (
            rollout,signature,max_error,MEASUREMENT_HALF_WIDTH)
        byname={w.name:w for w in WORLDS}
        for a,b in ALIAS_PAIRS:
            t1=rollout(byname[a],HELD_EXCITATIONS[0])
            t2=rollout(byname[b],HELD_EXCITATIONS[0])
            for channel in ("external","producer_to_controller_ctrl"):
                self.assertLessEqual(
                    max_error(signature(t1[channel]),signature(t2[channel])),
                    2*MEASUREMENT_HALF_WIDTH)

    def test_disallowing_unauthenticated_critical_tap_does_not_fake_success(self):
        byname={w.name:w for w in WORLDS}
        for truth,attacker in ALIAS_PAIRS:
            z=decide_with_source_assumption(
                byname[truth],HELD_EXCITATIONS[1],
                force_source_trusted=False,spoof_force_from=byname[attacker])
            self.assertEqual(z.answer,"REFUSE")
            self.assertEqual(z.readout_steps,(
                "external","producer_to_controller_ctrl"))
            self.assertNotEqual(z.answer,byname[truth].authority)


if __name__=="__main__":
    unittest.main()
