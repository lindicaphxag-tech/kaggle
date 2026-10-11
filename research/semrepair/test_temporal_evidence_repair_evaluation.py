import unittest
import torch
from research.semrepair.temporal_evidence_repair_evaluation import (
    run_temporal_source_evidence_benchmark,WORLD_PAIRS,CHUNK_HORIZON)
from research.semrepair.temporal_evidence_repair_operator import (
    select_information_authority,execute_original_lerobot_conversion)
from research.semrepair.lerobot_temporal_anchor_case import (
    load_actual_upstream_function_pair,MASK)


class ContractSoundnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results=run_temporal_source_evidence_benchmark()
    def test_real_unchanged_vendor_software_and_negative_vendor_bug_claim(self):
        r=self.results
        self.assertEqual(r["scenario_worlds"],WORLD_PAIRS*2)
        self.assertEqual(r["correlated_action_outputs_not_independent_trials"],
                         WORLD_PAIRS*2*CHUNK_HORIZON)
        self.assertEqual(r["real_physical_robot_evaluations"],0)
        self.assertFalse(r["original_vendor_bug_reported"])
        self.assertFalse(r["source_bytes_actually_modified"])
    def test_info_minimax_error_and_tight_interval_gate(self):
        r=self.results
        self.assertAlmostEqual(r["mean_current_only_max_error"],
                               r["mean_minimax_theory_lower_bound"],places=11)
        self.assertGreater(r["mean_minimax_theory_lower_bound"],.12)
        self.assertEqual(r["untrusted_action_refusals"],WORLD_PAIRS*2)
        self.assertEqual(r["stale_epoch_refusals"],WORLD_PAIRS*2)
        self.assertEqual(r["strict_unauthorized_false_action_count"],0)
    def test_history_enabled_blackbox_ties_source_repair(self):
        r=self.results
        self.assertLessEqual(r["max_authorized_source_error"],1e-12)
        self.assertLessEqual(r["max_history_enabled_blackbox_error"],1e-12)
        self.assertTrue(r["history_enabled_blackbox_matches_source"])
        self.assertLess(r["minimum_declared_cost_with_valid_cache"],
                        r["minimum_declared_cost_source_only"])
        self.assertTrue(r["source_repair_is_not_necessary_when_valid_cache_available"])
        self.assertTrue(all(x["historical_cache_operation"]=="USE_VERIFIED_HISTORY"
                            and x["source_operation"]=="AUTHORIZED_SOURCE_ANCHOR"
                            for x in r["rows"]))
    def test_two_worlds_same_current_different_hidden_reference(self):
        _,fn,_=load_actual_upstream_function_pair()
        relative=torch.zeros((1,4,7),dtype=torch.float64)
        current=torch.zeros((1,7),dtype=torch.float64)
        anchors=(torch.full((1,7),-.2,dtype=torch.float64),
                 torch.full((1,7),.2,dtype=torch.float64))
        d,ref=select_information_authority(
            relative=relative,current=current,mask=MASK,
            possible_historical_anchors=anchors,
            allowable_max_action_error=.2)
        self.assertEqual(d.operation,"CURRENT_ONLY_MINIMAX")
        self.assertAlmostEqual(d.worst_case_error_bound,.2)
        estimated=execute_original_lerobot_conversion(fn,relative,d,ref,MASK)
        self.assertAlmostEqual(float(estimated.abs().amax()),0.)
        d,ref=select_information_authority(
            relative=relative,current=current,mask=MASK,
            possible_historical_anchors=anchors,
            allowable_max_action_error=.19)
        self.assertEqual(d.operation,"ABSTAIN")
        with self.assertRaisesRegex(ValueError,"not authorized"):
            execute_original_lerobot_conversion(fn,relative,d,ref,MASK)
    def test_fake_verified_source_anchor_contradicts_existing_worlds(self):
        rel=torch.zeros((1,2,7),dtype=torch.float64)
        cur=torch.zeros((1,7),dtype=torch.float64)
        possible=(torch.ones((1,7),dtype=torch.float64)*.3,)
        with self.assertRaisesRegex(ValueError,"contradicts"):
            select_information_authority(
                relative=rel,current=cur,mask=MASK,
                possible_historical_anchors=possible,
                allowable_max_action_error=.001,
                source_anchor=torch.zeros((1,7),dtype=torch.float64),
                source_anchor_verified=True,source_epoch=5,chunk_epoch=5)
    def test_invalid_source_values_and_epochs_fail_safe(self):
        rel=torch.zeros((1,2,7),dtype=torch.float64)
        cur=torch.zeros((1,7),dtype=torch.float64)
        possible=(cur.clone(),)
        with self.assertRaisesRegex(ValueError,"invalid declared"):
            select_information_authority(
                relative=rel,current=cur,mask=MASK,
                possible_historical_anchors=possible,
                allowable_max_action_error=float("nan"))
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            other=rel.clone();other[0,0,0]=float("inf")
            select_information_authority(
                relative=other,current=cur,mask=MASK,
                possible_historical_anchors=possible,
                allowable_max_action_error=.01)

if __name__=="__main__":
    unittest.main()
