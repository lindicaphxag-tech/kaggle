"""Robot task closure for MINIMAL authorized source repairs."""
import unittest
from research.semrepair.clip_only_pusher_repair import (
    run_clip_only_source_recovery,CLIP_ONLY,
    REPLAY_FAULT,REPLACEMENT,EXTERNAL_ALLOWED_MAX_SOFTWARE_CLIP_HERE
)

class MinimalSignedPusherRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=run_clip_only_source_recovery()

    def test_actually_installed_one_field_signed_source_not_two_matrices(self):
        r=self.r
        self.assertEqual(r["actual_official_task_replays"],64)
        self.assertEqual(r["new_holdout_seeds"],0)
        self.assertEqual(CLIP_ONLY["producer"],REPLAY_FAULT["producer"])
        self.assertEqual(CLIP_ONLY["controller"],REPLAY_FAULT["controller"])
        self.assertEqual(r["signed_approval_records"]["clip-only"]["source_repair_class"],"clip-only")
        self.assertEqual(r["summary"]["installed-signed-clip-only"]["declared_source_fields_edited"],1)
        self.assertEqual(r["summary"]["installed-signed-atomic-matrices"]["declared_source_fields_edited"],3)
        self.assertEqual(r["signed_approval_records"]["clip-only"]["activated_epoch"],2)

    def test_full_original_reward_and_actual_object_goal_improvement(self):
        r=self.r["summary"]
        source=r["installed-signed-clip-only"]
        broken=r["file-backed-fault"]
        healthy=r["official-health"]
        self.assertGreater(source["mean_original_reward"],broken["mean_original_reward"])
        self.assertLess(source["mean_actual_object_goal_distance"],
                        broken["mean_actual_object_goal_distance"])
        self.assertAlmostEqual(healthy["mean_original_reward"],
                               source["mean_original_reward"],delta=1e-3)

    def test_signed_minimal_clip_only_performs_equivalently_to_larger_atomic_repair(self):
        for row in self.r["seed_paired"]:
            self.assertLess(abs(row["clip_only_minus_atomic_reward"]),1e-3)
            self.assertLess(abs(row["clip_only_minus_healthy_reward"]),1e-3)

    def test_minimal_actuation_requires_external_clip_cap_permission(self):
        self.assertGreater(CLIP_ONLY["clip"],self.r["exact_full_action_box_clipping_threshold"])
        self.assertLessEqual(CLIP_ONLY["clip"],EXTERNAL_ALLOWED_MAX_SOFTWARE_CLIP_HERE)
        self.assertTrue(self.r["software_clip_cap_is_author_assumed_NOT_independently_signed"])
        self.assertFalse(self.r["calibration_and_external_signer_roots_verified"])

if __name__=="__main__":
    unittest.main()
