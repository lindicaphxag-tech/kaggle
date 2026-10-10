"""Plan minimal meaningful software intervention, not a post-hoc reward score."""
import unittest

from research.semrepair.finite_authorized_joint_repair_planner import (
    two_fault_source_and_history_cost_sweep,
    minimize_joint_authorized_repair,
)


class FiniteAuthorizedJointRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=two_fault_source_and_history_cost_sweep()

    def test_loose_tolerance_requires_no_source_write_or_history_access(self):
        r=self.r["results"]["loose"]
        self.assertEqual(r["selected_source_component"],"source_current")
        self.assertEqual(r["selected_information_component"],"current_only")
        self.assertEqual(r["minimum_catalog_cost"],0)
        self.assertLess(r["worst_case_action_error_upper_bound"],.8)
        self.assertFalse(r["source_configuration_installed"])

    def test_medium_tolerance_selects_cheap_clip_repair_but_not_unnecessary_history(self):
        r=self.r["results"]["medium"]
        self.assertEqual(r["selected_source_component"],"one_field_source_clip")
        self.assertEqual(r["selected_information_component"],"current_only")
        self.assertEqual(r["minimum_catalog_cost"],2)
        self.assertAlmostEqual(r["worst_case_action_error_upper_bound"],.2,delta=2e-7)

    def test_strict_tolerance_requires_both_source_and_history(self):
        r=self.r["results"]["strict"]
        self.assertEqual(r["selected_source_component"],"one_field_source_clip")
        self.assertEqual(r["selected_information_component"],"verified_epoch_anchor")
        self.assertEqual(r["minimum_catalog_cost"],3)
        self.assertLess(r["worst_case_action_error_upper_bound"],1e-7)
        self.assertEqual(r["number_of_catalog_plans"],4)
        selected=next(x for x in r["plan_proof_records"]
                      if x["decision"]=="CONDITIONALLY_CONTRACT_VALID")
        self.assertEqual(len(selected["observation_class_witnesses"]),2)

    def test_refuse_when_missing_actuator_source_permission(self):
        r=self.r["results"]["strict_source_not_licensed"]
        self.assertEqual(r["decision"],"REFUSE_NO_AUTHORIZED_FEASIBLE_REPAIR")
        self.assertIsNone(r["selected_source_component"])
        self.assertFalse(r["external_signer_and_telemetry_authentication"])
        self.assertTrue(any(x["decision"]=="REFUSE_MISSING_EXTERNAL_AUTHORITY"
                            for x in r["plan_proof_records"]))

    def test_invalid_evidence_partition_or_cost_is_rejected(self):
        square=[[(-2.,-2.),(2.,-2.),(2.,2.),(-2.,2.)]]
        base=dict(desired_actions=[[1.,1.],[0.,0.]],
                  action_software_catalog=[dict(
                      name="a",cost=0,authorized=True,
                      physical_reachable_polygons=square)],
                  evidence_catalog=[dict(
                      name="b",cost=1,authorized=True,
                      observation_world_partition=[[0],[1]])],
                  max_per_action_error=.1)
        cases=[
            dict(evidence_catalog=[dict(name="b",cost=1,authorized=True,
                  observation_world_partition=[[0],[0]])]),
            dict(evidence_catalog=[dict(name="b",cost=1,authorized=True,
                  observation_world_partition=[[0]])]),
            dict(action_software_catalog=[dict(
                 name="a",cost=-1,authorized=True,
                 physical_reachable_polygons=square)]),
            dict(max_per_action_error=float("nan")),
        ]
        for case in cases:
            with self.subTest(case=case),self.assertRaises(ValueError):
                minimize_joint_authorized_repair(**{**base,**case})

    def test_no_false_cryptographic_or_general_search_certificate(self):
        for key,r in self.r["results"].items():
            self.assertTrue(r["optimality_is_finite_catalog_only"])
            self.assertFalse(r["external_signer_and_telemetry_authentication"])
            self.assertFalse(r["source_configuration_installed"])

if __name__=="__main__":
    unittest.main()
