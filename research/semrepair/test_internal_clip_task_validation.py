"""Preserve full-task failure outcomes and strong black-box comparator."""
import unittest
from research.semrepair.internal_clip_task_validation import (
    run_clipped_task_assay,SOURCES,CONTROLLERS
)
from research.semrepair.two_link_cross_controller_benchmark import N_GOALS


class SaturatedTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=run_clipped_task_assay()

    def test_complete_paired_assay(self):
        r=self.out
        self.assertEqual(len(r["rows"]),N_GOALS*len(CONTROLLERS)*len(SOURCES))
        self.assertEqual(r["physical_families"],1)
        self.assertEqual(r["pretrained_vla_count"],0)

    def test_reference_and_source_correction_are_identical(self):
        for controller in CONTROLLERS:
            a=self.out["summary"][f"{controller}|reference"]
            b=self.out["summary"][f"{controller}|source-bound-atomic"]
            self.assertEqual(a["successes"],b["successes"])
            self.assertAlmostEqual(a["mean_final_error"],b["mean_final_error"],places=10)
            self.assertEqual(b["mean_torque_tracking_error"],0.)

    def test_blackbox_oracle_must_be_strong_pointwise(self):
        # The projection baseline is optimal per requested instantaneous
        # torque, not a straw-man matrix inverse.
        for controller in CONTROLLERS:
            oracle=self.out["summary"][f"{controller}|omniscient-one-step-blackbox-oracle"]
            naive=self.out["summary"][f"{controller}|linear-blackbox-inverse"]
            self.assertEqual(oracle["goals"],naive["goals"])
            self.assertGreaterEqual(oracle["mean_torque_tracking_error"],0.)

    def test_raw_trajectories_and_no_failed_trials_filtered(self):
        for row in self.out["rows"]:
            self.assertEqual(len(row["result"]["trajectory"]),480)
            self.assertTrue(all(len(x)==3 for x in row["result"]["trajectory"]))

if __name__=="__main__":
    unittest.main()
