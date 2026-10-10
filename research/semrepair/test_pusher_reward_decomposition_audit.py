"""Original Pusher reward-object distance decomposition and multiplicity."""
import unittest
from research.semrepair.pusher_reward_decomposition_audit import (
    SEVERITIES,SEEDS,audit_pusher,bootstrap_paired,exact_sign_test,T
)
ARMS=("official-healthy","untouched-compensating-source-fault",
      "composite-inverse-blackbox","exact-two-clip-full-information-oracle",
      "blind-numeric-source-candidate","source-known-atomic-upper-bound")


def fixture():
    rows=[]
    for name in SEVERITIES:
        for seed in SEEDS:
            for arm in ARMS:
                is_oracle=arm=="exact-two-clip-full-information-oracle"
                object_distance=.18 if is_oracle else .15
                contact_distance=.1
                ctrl_reward=-.02
                reward=T*(-object_distance-.5*contact_distance+ctrl_reward)
                rows.append({
                    "scenario":name,"seed":seed,"arm":arm,"steps":T,
                    "official_episode_return":reward,
                    "mean_object_goal_distance":object_distance,
                    "final_object_goal_distance":object_distance,
                    "mean_fingertip_object_distance":contact_distance,
                    "mean_official_control_reward":ctrl_reward})
    return {
        "official_env":"Pusher-v5","native_action_dimension":7,
        "fixed_fault_grid":[(name,2.,.8) for name in SEVERITIES],
        "scenario_arms":ARMS,
        "raw_rows":rows}

class CrossRobotAuditTests(unittest.TestCase):
    def test_original_goal_dist_and_reward_both_improve(self):
        report=audit_pusher(fixture())
        self.assertEqual(report["familywise_sign_test_contrasts"],10)
        for scenario in SEVERITIES:
            pair=report["results"][scenario]
            self.assertAlmostEqual(pair["endpoints"]["official_return_gain"]["mean"],3.)
            self.assertAlmostEqual(pair["endpoints"][
                "actual_object_goal_distance_reduction"]["mean"],.03)
            self.assertEqual(pair["endpoints"]["official_return_gain"]["positive_seed_count"],32)
            self.assertLess(pair["endpoints"]["official_return_gain"][
                "ten_contrast_bonferroni_p"],1e-6)

    def test_reject_missing_seed(self):
        f=fixture()
        f["raw_rows"].pop()
        with self.assertRaisesRegex(ValueError,"960"):
            audit_pusher(f)

    def test_reject_mismatched_reward_components(self):
        f=fixture()
        r=next(r for r in f["raw_rows"] if r["arm"]=="blind-numeric-source-candidate")
        r["official_episode_return"]+=2.
        with self.assertRaisesRegex(ValueError,"reward gain"):
            audit_pusher(f)

    def test_sign_tests_with_ties_and_unfavorable_source(self):
        self.assertEqual(exact_sign_test([0.]*32)["p"],1.)
        self.assertEqual(exact_sign_test([1.]*30+[-1.]*2)["wins"],30)
        self.assertEqual(exact_sign_test([1.]*30+[-1.]*2)["losses"],2)

    def test_bootstrap_constant_signal(self):
        self.assertEqual(bootstrap_paired([2.]*32),[2.,2.])

if __name__=="__main__":
    unittest.main()
