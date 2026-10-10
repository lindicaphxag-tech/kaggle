"""State-changing probe semantics: independent world replay and hard refusals."""
import unittest
from fractions import Fraction
from .state_changing_probe_authority import (
    Hidden, Probe, StatefulAuthorityPlanner, witness, report
)


class StateChangingRepairTests(unittest.TestCase):
    def test_mutating_probe_requires_reset_before_patch_installation(self):
        planner, start = witness()
        plan = planner.solve(start)
        self.assertEqual(plan.worst_cost, Fraction(2))
        self.assertEqual(plan.action, "fast_trace")
        self.assertTrue(all(child.action == "reset_epoch"
                            for _, child in plan.branches))
        self.assertEqual(planner.replay_all(start, plan), (Fraction(2), 2))

    def test_static_baseline_is_optimistic_and_unsafe(self):
        baseline, start = witness(assumed_static=True)
        plan = baseline.solve(start)
        self.assertEqual(plan.worst_cost, Fraction(1))
        actual, _ = witness()
        for world in start:
            observed, after = actual.probes["fast_trace"].step(world)
            self.assertEqual(dict(plan.branches)[observed].action,
                             "INSTALL:" + world.patch_hash)
            self.assertTrue(after.dirty)

    def test_license_and_cost_caps_are_fail_closed(self):
        planner, start = witness()
        self.assertIsNone(planner.solve(start, cost_limit=Fraction(1)))
        untrusted, start2 = witness(trusted_fast=False)
        plan = untrusted.solve(start2)
        self.assertEqual(plan.action, "nonmutating_snapshot")
        self.assertEqual(plan.worst_cost, Fraction(3))
        self.assertEqual(untrusted.replay_all(start2, plan), (Fraction(3), 2))
        no_alternative, start3 = witness(trusted_fast=False, include_snapshot=False)
        self.assertIsNone(no_alternative.solve(start3))
        no_alternative.complete_model = False
        self.assertIsNone(no_alternative.solve(start3))

    def test_identical_transcripts_cannot_license_distinct_patches(self):
        start = frozenset((Hidden("A", False, "pA"),
                           Hidden("B", False, "pB")))
        def blind(world):
            return "same", Hidden(world.implementation, True, world.patch_hash)
        planner = StatefulAuthorityPlanner(
            (Probe("blind", Fraction(1), Fraction(0), True, blind),),
            Fraction(1), 3, complete_model=True)
        self.assertIsNone(planner.solve(start))

    def test_same_patch_still_requires_clean_epoch(self):
        start = frozenset((Hidden("A", True, "same"),
                           Hidden("B", True, "same")))
        def reset(world):
            return "ok", Hidden(world.implementation, False, world.patch_hash)
        planner = StatefulAuthorityPlanner(
            (Probe("reset", Fraction(1), Fraction(0), True, reset),),
            Fraction(0), 2, complete_model=True)
        plan = planner.solve(start)
        self.assertEqual(plan.worst_cost, Fraction(1))
        self.assertEqual(planner.replay_all(start, plan), (Fraction(1), 2))

    def test_report_preserves_negative_result(self):
        result = report()
        self.assertEqual(result["state_aware_minimax_cost"], "2")
        self.assertEqual(result["invalid_naive_authorizations"], 2)
        self.assertTrue(result["cost_cap_1_refusal"])


if __name__ == "__main__":
    unittest.main()
