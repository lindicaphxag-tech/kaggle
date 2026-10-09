"""Exact geometric impossibility versus full-information black-box inverse."""
import unittest
from math import dist
from research.semrepair.internal_saturation_reachability import (
    LIMIT, CONTROLLER,PRODUCER,exact_blackbox_torque_oracle,
    run_reachability_assay
)
from research.semrepair.observable_interface_factorization import mv,mmul,I


class SaturationReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=run_reachability_assay()

    def test_compensated_net_linear_map_is_identity(self):
        pair=mmul(CONTROLLER,PRODUCER)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(pair[i][j],I[i][j])

    def test_omniscient_blackbox_cannot_reach_some_legal_torque(self):
        desired=(8.,8.)
        self.assertLessEqual(max(abs(v) for v in desired),LIMIT)
        oracle=exact_blackbox_torque_oracle(desired)
        self.assertFalse(oracle["feasible"])
        self.assertGreater(oracle["min_l2_torque_error"],1e-4)
        self.assertGreater(max(abs(z) for z in oracle["unconstrained_required_inner_torque"]),LIMIT)

    def test_matched_reachable_interior(self):
        desired=(2.,-2.)
        oracle=exact_blackbox_torque_oracle(desired)
        self.assertTrue(oracle["feasible"])
        self.assertEqual(oracle["min_l2_torque_error"],0.)

    def test_all_unreachable_2d_torques_have_physical_mismatch(self):
        r=self.result
        self.assertEqual(r["grid_conditions_not_independent"],49)
        self.assertGreater(r["unreachable_target_count_even_for_omniscient_blackbox"],0)
        self.assertGreater(r["max_blackbox_first_step_joint_velocity_gap"],0)
        self.assertEqual(r["max_source_repair_joint_velocity_gap"],0.0)
        for row in r["rows"]:
            if not row["blackbox_reachable"]:
                self.assertGreater(row["blackbox_velocity_gap"],1e-10)

    def test_projection_is_optimal_in_2d_against_independent_coarse_grid(self):
        # Coarse upstream actuator-space search cannot do BETTER than analytic
        # nearest-point projection on the entire parallelogram.
        desired=(8.,8.)
        oracle=exact_blackbox_torque_oracle(desired)
        for ia in range(-18,19):
            for ib in range(-18,19):
                z=(ia*.5,ib*.5)
                if max(abs(x) for x in z)>LIMIT:
                    continue
                torque=mv(CONTROLLER,z)
                self.assertGreaterEqual(
                    dist(torque,desired)+1e-9,oracle["min_l2_torque_error"])

    def test_invalid_limit_rejected(self):
        with self.assertRaises(ValueError):
            exact_blackbox_torque_oracle((2.,3.),limit=0.)

if __name__=="__main__":
    unittest.main()
