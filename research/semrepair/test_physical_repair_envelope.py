"""Exhaustive finite analytical envelope checks and fault-patch counterexamples."""
import unittest
from math import sin
from research.semrepair.physical_repair_envelope import (
    MotorMap, exact_control_envelope_gap, normalized_drive,
    source_and_envelope_authority,
)
from research.semrepair.mujoco_native_authority import WORLDS, corrected
from research.semrepair.mujoco_wrong_repair_counterfactual import (
    unauthorized_compensating_repair,
)


def as_motor(world):
    return MotorMap(world.software_scale,world.gear,world.mass,world.damping,10.0)


class PhysicalEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.reference=as_motor(next(w for w in WORLDS if w.name=="reference"))
        self.faults=tuple(w for w in WORLDS if w.name!="reference")

    def test_nominal_compensation_and_high_command_failure(self):
        for fault in self.faults:
            good=as_motor(corrected(fault))
            wrong=as_motor(unauthorized_compensating_repair(fault))
            with self.subTest(fault=fault.name):
                self.assertLess(exact_control_envelope_gap(self.reference,good,13.)[
                    "maximum_normalized_drive_error"],1e-11)
                self.assertLess(exact_control_envelope_gap(self.reference,wrong,.75)[
                    "maximum_normalized_drive_error"],1e-11)
                self.assertGreater(exact_control_envelope_gap(self.reference,wrong,13.)[
                    "maximum_normalized_drive_error"],.01)

    def test_breakpoint_is_real_witness(self):
        for fault in self.faults:
            wrong=as_motor(unauthorized_compensating_repair(fault))
            cert=exact_control_envelope_gap(self.reference,wrong,13.0)
            u=cert["witness_command"]
            self.assertAlmostEqual(abs(normalized_drive(self.reference,u)-
                                       normalized_drive(wrong,u)),
                                   cert["maximum_normalized_drive_error"])

    def test_dense_independent_bruteforce_not_above_exact_bound(self):
        for fault in self.faults:
            wrong=as_motor(unauthorized_compensating_repair(fault))
            cert=exact_control_envelope_gap(self.reference,wrong,13.0)
            dense=max(abs(normalized_drive(self.reference,-13+26*i/1000)-
                          normalized_drive(wrong,-13+26*i/1000))
                      for i in range(1001))
            self.assertLessEqual(dense,cert["maximum_normalized_drive_error"]+1e-10)

    def test_source_remains_required_even_when_dynamic_candidate_unique(self):
        good=as_motor(corrected(self.faults[0]))
        missing=source_and_envelope_authority(trusted_source_patch=None,
            proposed_patch="correct",reference=self.reference,
            proposed_motor=good,maximum_abs_command=13)
        self.assertEqual(missing["decision"],"REFUSE_UNVERIFIED_SOURCE")
        wrong_issuer=source_and_envelope_authority(trusted_source_patch="other",
            proposed_patch="correct",reference=self.reference,
            proposed_motor=good,maximum_abs_command=13)
        self.assertEqual(wrong_issuer["decision"],"REFUSE_WRONG_SOURCE_PATCH")

    def test_source_and_high_amplitude_both_required(self):
        for fault in self.faults:
            good=as_motor(corrected(fault))
            wrong=as_motor(unauthorized_compensating_repair(fault))
            good_result=source_and_envelope_authority(
                trusted_source_patch=fault.authority,proposed_patch=fault.authority,
                reference=self.reference,proposed_motor=good,maximum_abs_command=13.)
            bad_result=source_and_envelope_authority(
                trusted_source_patch=fault.authority,proposed_patch=fault.authority,
                reference=self.reference,proposed_motor=wrong,maximum_abs_command=13.)
            self.assertEqual(good_result["decision"],"PERMIT_CONDITIONAL_PHYSICAL_AND_SOURCE_GATE")
            self.assertEqual(bad_result["decision"],"REFUSE_OUTSIDE_REPAIR_ENVELOPE")

    def test_velocity_normalization_mismatch_must_not_certify(self):
        wrong=MotorMap(1,1,1,0.3,10)
        with self.assertRaises(ValueError):
            exact_control_envelope_gap(self.reference,wrong,2.)

    def test_invalid_physical_params_fail(self):
        for motor in (MotorMap(0,1,1,.2,10),MotorMap(1,-1,1,.2,10),
                      MotorMap(1,1,0,.2,10),MotorMap(1,1,1,.2,-10)):
            with self.assertRaises(ValueError):
                exact_control_envelope_gap(self.reference,motor,1.)

    def test_zero_range_uses_zero_only(self):
        r=exact_control_envelope_gap(self.reference,as_motor(corrected(self.faults[0])),0.0)
        self.assertEqual(r["maximum_normalized_drive_error"],0.0)
        self.assertEqual(r["tested_all_piecewise_linear_breakpoints"],[0.0])


if __name__=="__main__":
    unittest.main()
