"""Witness-verified robust common-input source/info intervention decisions."""
import unittest
from research.semrepair.robust_observation_authority_optimizer import (
    run_robust_common_input_joint_repair,
    choose_joint_source_history_with_common_input
)


class RobustWorldSpecificActuatorRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=run_robust_common_input_joint_repair()["cases"]

    def test_loose_contract_needs_neither_history_nor_source(self):
        r=self.x["loose"]
        self.assertEqual(r["decision"],
                         "CONDITIONAL_FINITE_MIN_COST_SOURCE_REPAIR_SELECTED_NOT_INSTALLED")
        self.assertEqual(r["source_repair"],"unchanged_unknown_sign_clipped")
        self.assertEqual(r["historical_evidence"],"current_observation_only")
        self.assertEqual(r["minimum_declared_finite_cost"],0)

    def test_intermediate_contract_prefers_historical_evidence_not_source_edit(self):
        r=self.x["intermediate"]
        self.assertEqual(r["historical_evidence"],"version_bound_historical_state")
        self.assertEqual(r["source_repair"],"unchanged_unknown_sign_clipped")
        self.assertEqual(r["minimum_declared_finite_cost"],1)
        other=next(z for z in r["candidate_records"] if
                   z["source"]=="authorized_one_source_clip_expansion" and
                   z["evidence"]=="current_observation_only")
        self.assertEqual(other["decision"],"CERTIFIED_CONDITIONALLY_INFEASIBLE")

    def test_strict_contract_needs_both_and_uses_different_real_upstream_actions(self):
        r=self.x["strict"]
        self.assertEqual(r["historical_evidence"],"version_bound_historical_state")
        self.assertEqual(r["source_repair"],"authorized_one_source_clip_expansion")
        self.assertEqual(r["minimum_declared_finite_cost"],3)
        record=next(x for x in r["candidate_records"] if
                    x["decision"]=="CERTIFIED_CONDITIONALLY_FEASIBLE")
        self.assertEqual(len(record["fibers"]),2)
        x,y=record["fibers"]
        self.assertLessEqual(x["robust_upper"],.1)
        self.assertLessEqual(y["robust_upper"],.1)
        self.assertGreater(x["same_upstream_witness"][0],0)
        self.assertLess(y["same_upstream_witness"][0],0)
        self.assertFalse(r["signers_authenticity_established"])

    def test_no_source_edit_authority_means_no_unsafe_shortcut(self):
        r=self.x["strict_no_source_rights"]
        self.assertEqual(r["decision"],"REFUSE_NO_LICENSED_VERIFIED_SOURCE_ACTION_PLAN")
        self.assertIsNone(r["source_repair"])
        self.assertTrue(any(x["decision"]=="UNLICENSED" for x in r["candidate_records"]))

    def test_bad_observation_partition_source_cost_refuse(self):
        I=((1.,0.),(0.,1.))
        md=dict(P=I,C=I,L=(2.,2.),B=(2.,2.))
        base=dict(
            desired_actions=[(.2,.2),(.2,.2)],
            source_catalog=[dict(name="s",cost=1,authorized=True,
                                 per_world_source_model=[md,md])],
            information_catalog=[dict(name="e",cost=0,authorized=True,
                                      world_observation_partition=[[0],[1]])],
            error_tolerance=.01)
        for v in (
            dict(information_catalog=[dict(name="e",cost=0,authorized=True,
                     world_observation_partition=[[0],[0]])]),
            dict(source_catalog=[dict(name="s",cost=-1,authorized=True,
                     per_world_source_model=[md,md])]),
            dict(source_catalog=[dict(name="s",cost=0,authorized=True,
                     per_world_source_model=[md])]),
        ):
            with self.subTest(v=v),self.assertRaises(ValueError):
                choose_joint_source_history_with_common_input(**{**base,**v})

if __name__=="__main__":
    unittest.main()
