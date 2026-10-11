"""Official seven-DoF Pusher external learned-policy experimental controls."""
import unittest
import numpy as np
from research.semrepair.external_pusher_transfer import (
    _matrices,SCENARIOS,ARMS,SEEDS,FIRST_TWO_ACTIONS,
    blind_producer_controller_numeric_taps,altered_action,
    load_external_pusher_checkpoint
)
from research.semrepair.strong_clipped_actuator_oracle import output_reachability_polygons


class PusherTransferTests(unittest.TestCase):
    def test_predeclared_all_severities_seeds_and_policies(self):
        self.assertEqual(len(SCENARIOS),5)
        self.assertEqual(len(SEEDS),32)
        self.assertEqual(len(set(SEEDS)),32)
        self.assertEqual(len(ARMS),6)
        self.assertEqual(FIRST_TWO_ACTIONS,(0,1))
        self.assertEqual(SCENARIOS[-1],("shear-3point5-plus-clip",3.5,.8))

    def test_blind_frozen_probes_reconstruct_two_source_components(self):
        for name,k,L in SCENARIOS:
            r=blind_producer_controller_numeric_taps(k,L)
            self.assertEqual(r["candidate_patch"],"atomic",name)
            self.assertLess(r["heldout_producer_error"],1e-8)
            self.assertLess(r["heldout_controller_error"],1e-8)

    def test_strong_oracle_better_or_equal_to_unrepaired_in_action_error(self):
        for name,k,L in SCENARIOS:
            P,C=_matrices(k)
            regions=output_reachability_polygons(C,inner_bound=L,outer_bound=2.)
            for a,b in ((1.6,1.6),(-1.,-.9),(.1,.2),(-1.8,1.6)):
                source=np.array((a,b,.35,-.2,.6,0.,-.4),dtype=np.float32)
                faulty=np.clip(altered_action(source,"untouched-compensating-source-fault",P,C,L,regions),-2,2)
                oracle=np.clip(altered_action(source,"exact-two-clip-full-information-oracle",P,C,L,regions),-2,2)
                self.assertLessEqual(
                    float(np.linalg.norm(source-oracle)),
                    float(np.linalg.norm(source-faulty))+1e-5,(name,a,b))
                np.testing.assert_array_equal(source[2:],oracle[2:])

    def test_correct_source_matches_official_action(self):
        for name,k,L in SCENARIOS:
            P,C=_matrices(k)
            regions=output_reachability_polygons(C,inner_bound=L,outer_bound=2.)
            action=np.array((.9,.9,.3,.1,-.8,-.2,.7),dtype=np.float32)
            for arm in ("official-healthy","blind-numeric-source-candidate",
                        "source-known-atomic-upper-bound"):
                np.testing.assert_array_equal(altered_action(action,arm,P,C,L,regions),action)

    def test_official_checkpoint_genuine_seven_motor_shape(self):
        actor,digest=load_external_pusher_checkpoint()
        self.assertEqual(len(digest),64)
        self.assertEqual(tuple(actor.observation_space.shape),(23,))
        self.assertEqual(tuple(actor.action_space.shape),(7,))


if __name__=="__main__":
    unittest.main()
