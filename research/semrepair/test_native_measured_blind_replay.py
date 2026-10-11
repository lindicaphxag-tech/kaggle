import unittest
from fractions import Fraction
from research.semrepair.native_measured_blind_replay import (
    study, HELD_EXCITATIONS, WORLDS, _score_one,
)

class Tests(unittest.TestCase):
    def test_original_physx_external_alias_from_actual_mujoco(self):
        r=study()
        self.assertEqual(r["measured_world_excitation_cells"], 12)
        self.assertEqual(r["full_tap"]["authorized"],12)
        self.assertEqual(r["average_measured_probe_cost"], "4")
        self.assertEqual(r["full_tap"]["wrong_authorizations"],0)
        self.assertLess(r["external_only"]["authorized"],12)
        self.assertLess(r["risk_without_force_tap"]["authorized"],12)

    def test_risk_cap_fails_closed_on_hidden_alias(self):
        gain=next(w for w in WORLDS if w.name=="gain_fault")
        r=_score_one(gain,0,HELD_EXCITATIONS[0],max_probe_risk=Fraction(1,2))
        self.assertEqual(r.authorization,"REFUSE")

    def test_reference_repair_requires_no_secret_metadata(self):
        ref=next(w for w in WORLDS if w.name=="reference")
        r=_score_one(ref,0,HELD_EXCITATIONS[0],max_probe_risk=Fraction(1))
        self.assertTrue(r.correct_authorization)
        self.assertLess(r.physical_closure_max_error,1e-8)

if __name__=="__main__":
    unittest.main()
