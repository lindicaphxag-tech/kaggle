"""Test actual MuJoCo execution instead of just closed-form expected outputs."""
import unittest
from fractions import Fraction
from research.semrepair.saturation_transfer_challenge import (
    frozen_saturation_challenge, effective_motor_force, exact_equivalence_domain,
    LOW_COMMANDS, HIGH_COMMANDS, ACTUATOR_CONTROL_LIMIT,
)
from research.semrepair.mujoco_native_authority import WORLDS, corrected
from research.semrepair.mujoco_wrong_repair_counterfactual import (
    unauthorized_compensating_repair,
)


class SaturationTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=frozen_saturation_challenge()

    def test_same_low_support_but_stress_diverges_for_all_fault_worlds(self):
        r=self.report
        self.assertEqual(r["world_count"],3)
        self.assertEqual(r["matched_development_wrong_patch_cells"],3)
        self.assertEqual(r["stress_wrong_patch_trajectory_divergence_cells"],3)
        self.assertEqual(r["stress_true_patch_trajectory_closure_cells"],3)
        self.assertLess(r["development_amplitude_max"],r["transfer_amplitude_max"])

    def test_true_patch_remains_physical_solution(self):
        for row in self.report["rows"]:
            self.assertLess(row["stress"]["canonical_vs_authorized_external_error"],2e-8)
            self.assertGreater(row["stress"]["canonical_vs_wrong_patch_external_error"],1e-5)

    def test_calibrated_equivalence_domain_and_boundary_witness(self):
        for world in WORLDS:
            if world.name=="reference":
                continue
            right=corrected(world)
            wrong=unauthorized_compensating_repair(world)
            result=exact_equivalence_domain(right.software_scale,right.gear,
                                            wrong.software_scale,wrong.gear,
                                            ACTUATOR_CONTROL_LIMIT)
            threshold=result["guaranteed_equivalent_control_abs_le"]
            self.assertGreaterEqual(threshold,max(map(abs,LOW_COMMANDS)))
            self.assertLess(threshold,max(map(abs,HIGH_COMMANDS)))
            for command in (0,threshold/4,threshold):
                a=effective_motor_force(producer_scale=right.software_scale,
                                        gear=right.gear,command=command)
                b=effective_motor_force(producer_scale=wrong.software_scale,
                                        gear=wrong.gear,command=command)
                self.assertAlmostEqual(a,b,places=10)
            outside=threshold+1.0
            a=effective_motor_force(producer_scale=right.software_scale,
                                    gear=right.gear,command=outside)
            b=effective_motor_force(producer_scale=wrong.software_scale,
                                    gear=wrong.gear,command=outside)
            self.assertGreater(abs(a-b),1e-7)

    def test_invalid_effect_equivalence_claim_refused(self):
        with self.assertRaises(ValueError):
            exact_equivalence_domain(1,1,2,2,10)
        with self.assertRaises(ValueError):
            exact_equivalence_domain(-1,1,2,.5,10)

    def test_real_model_amplitude_sweep_retains_failure_rows(self):
        r=self.report
        self.assertEqual(len(r["amplitude_sweep"]),3*len(set(x["magnitude"] for x in r["amplitude_sweep"])))
        for world in {x["world"] for x in r["amplitude_sweep"]}:
            rows=[x for x in r["amplitude_sweep"] if x["world"]==world]
            self.assertTrue(any(x["in_certified_equal_command_range"] for x in rows))
            self.assertTrue(any(not x["in_certified_equal_command_range"] for x in rows))
            self.assertTrue(any(x["wrong_patch_trajectory_error"]>1e-5 for x in rows))


if __name__=="__main__":
    unittest.main()
