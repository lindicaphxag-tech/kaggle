"""Frozen protocol checks without reading held-out episode outcomes."""
import unittest
from research.semrepair.external_reacher_sealed_holdout import (
    FAULTS,HOLDOUT_SEEDS,ARMS,_matrices,identify_blind_fault,
    _action
)
from research.semrepair.frozen_external_reacher_sac import ENV_SEEDS
from research.semrepair.strong_clipped_actuator_oracle import output_reachability_polygons
import numpy as np

class HeldoutExternalPolicyTests(unittest.TestCase):
    def test_strict_new_holdout_seed_isolation(self):
        self.assertEqual(len(HOLDOUT_SEEDS),32)
        self.assertEqual(len(set(HOLDOUT_SEEDS)),32)
        self.assertFalse(set(HOLDOUT_SEEDS)&set(ENV_SEEDS))
        self.assertEqual(len(ARMS),6)
        self.assertEqual(len(FAULTS),5)

    def test_blind_probe_selection_under_all_fault_magnitudes(self):
        for label,k,L in FAULTS:
            P,C=_matrices(k)
            r=identify_blind_fault(P,C,L)
            self.assertEqual(r["candidate_patch"],"atomic",label)
            self.assertLess(r["heldout_producer_error"],1e-8)
            self.assertLess(r["heldout_controller_error"],1e-8)

    def test_exact_two_clip_oracle_no_worse_in_action_error_than_unmodified(self):
        for label,k,L in FAULTS:
            P,C=_matrices(k)
            polygons=output_reachability_polygons(C,inner_bound=L,outer_bound=1.)
            for action in ((.5,.4),(-.4,-.5),(.8,.8),(-.8,.8),(0.,0.)):
                raw=np.asarray(action,dtype=np.float32)
                faulty=np.clip(_action(raw,"fault-unrepaired",P,C,L,polygons),-1,1)
                oracle=np.clip(_action(raw,"exact-all-clip-blackbox-oracle",P,C,L,polygons),-1,1)
                self.assertLessEqual(
                    float(np.linalg.norm(raw-oracle)),
                    float(np.linalg.norm(raw-faulty))+1e-6,
                    (label,action))

    def test_source_candidate_is_identity_mapping_upper_bound_only(self):
        for label,k,L in FAULTS:
            P,C=_matrices(k)
            poly=output_reachability_polygons(C,inner_bound=L,outer_bound=1.)
            a=np.asarray((.9,-.6),dtype=np.float32)
            np.testing.assert_array_equal(_action(a,"blind-source-patch-candidate",P,C,L,poly),a)
            np.testing.assert_array_equal(_action(a,"oracle-source-atomic-upper-bound",P,C,L,poly),a)

if __name__=="__main__":
    unittest.main()
