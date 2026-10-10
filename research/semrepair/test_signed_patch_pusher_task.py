"""End-to-end sign/activate actual file adapter and execute native external task."""
import unittest
from research.semrepair.signed_patch_reacher_pusher_task import (
    run_real_pusher_signed_patch_task,TEST_SEEDS,STEPS
)

class SignedFileActivatedFrozenPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=run_real_pusher_signed_patch_task()

    def test_actual_signed_installed_bytes_used_by_mujoce_real_robot_simulator(self):
        r=self.data
        self.assertEqual(r["episode_count"],len(TEST_SEEDS)*3)
        self.assertTrue(r["actually_swapped_on_disk_runtime_adapter_file"])
        self.assertFalse(r["actual_manufacturer_operating_system_software_installation"])
        self.assertEqual(len(r["source_file_installed_sha256"]),64)
        self.assertEqual(r["activation_record"]["endorsement_threshold"],2)
        self.assertEqual(len(r["activation_record"]["verified_distinct_root_issuers"]),2)

    def test_real_frozen_policy_physics_equals_healthy_for_all_16_previously_inspected_seeds(self):
        r=self.data["summary"]
        healthy=r["healthy-unmodified"]
        installed=r["real-signed-atomic-file-loaded"]
        self.assertEqual(healthy["n_episodes"],16)
        self.assertAlmostEqual(healthy["mean_official_return"],installed["mean_official_return"],places=8)
        self.assertAlmostEqual(healthy["mean_real_object_goal_distance"],
                               installed["mean_real_object_goal_distance"],places=10)

    def test_unrepaired_loaded_source_config_actually_harms_known_fault_episodes(self):
        r=self.data["summary"]
        self.assertGreater(r["healthy-unmodified"]["mean_official_return"],
                           r["faulty-actual-file-loaded"]["mean_official_return"])
        self.assertLess(r["healthy-unmodified"]["mean_real_object_goal_distance"],
                        r["faulty-actual-file-loaded"]["mean_real_object_goal_distance"])

    def test_source_provenance_scope_is_explicitly_nonvendor(self):
        r=self.data
        self.assertEqual(r["authentic_external_robot_signer_key_count"],0)
        self.assertEqual(r["test_only_author_generated_signing_key_count"],3)
        self.assertEqual(r["new_holdout_seeds"],0)
        self.assertEqual(r["already_used_seed_count"],len(TEST_SEEDS))

if __name__=="__main__":
    unittest.main()
