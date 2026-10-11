"""No learning / high-risk oracle claims; exact 2D physical polytope contracts."""
import math
import unittest
from research.semrepair.joint_information_reachability_operator import (
    four_way_orthogonal_source_and_history_repair,
    robust_blackbox_action_error_floor,
)


class JointSourceEvidenceMinimaxTests(unittest.TestCase):
    def test_joint_obstruction_requires_two_non_interchangeable_repairs(self):
        x=four_way_orthogonal_source_and_history_repair()["all_modes"]
        none=x["neither_repaired"]["tight_numerical_worst_case_minimax_error_upper"]
        source=x["source_clip_only"]["tight_numerical_worst_case_minimax_error_upper"]
        info_a=x["historical_anchor_only_A"]["tight_numerical_worst_case_minimax_error_upper"]
        info_b=x["historical_anchor_only_B"]["tight_numerical_worst_case_minimax_error_upper"]
        both_a=x["both_source_and_history_A"]["tight_numerical_worst_case_minimax_error_upper"]
        both_b=x["both_source_and_history_B"]["tight_numerical_worst_case_minimax_error_upper"]
        self.assertAlmostEqual(none,.7,delta=1e-6)
        self.assertAlmostEqual(source,.2,delta=1e-6)
        self.assertAlmostEqual(info_a,.7,delta=1e-6)
        self.assertAlmostEqual(info_b,.7,delta=1e-6)
        self.assertLess(both_a,1e-7)
        self.assertLess(both_b,1e-7)
        self.assertGreater(none,source)
        self.assertGreater(info_a,both_a)

    def test_pointwise_witness_against_exact_minimax_in_rectangle(self):
        square=[[(-2.,-2.),(2.,-2.),(2.,2.),(-2.,2.)]]
        x=robust_blackbox_action_error_floor(
            indistinguishable_desired_actions=[(1.3,1.5),(1.7,1.5)],
            reachable_actuator_polygons=square,requested_error_tolerance=.21)
        self.assertAlmostEqual(x["tight_numerical_worst_case_minimax_error_upper"],.2,delta=1e-7)
        self.assertTrue(x["can_meet_requested_tolerance"])
        self.assertEqual(x["decision"],"CONDITIONALLY_ACTION_COMPENSABLE")
        y=robust_blackbox_action_error_floor(
            indistinguishable_desired_actions=[(1.3,1.5),(1.7,1.5)],
            reachable_actuator_polygons=square,requested_error_tolerance=.19)
        self.assertFalse(y["can_meet_requested_tolerance"])
        self.assertEqual(y["decision"],"REFUSE_UNREACHABLE_OR_UNIDENTIFIABLE_COMMAND")

    def test_reachable_set_unions_must_not_be_spuriously_convexified(self):
        # The black-box cannot deliver anything in (-1,1) horizontal gap.
        left=[(-2.,-1.),(-1.,-1.),(-1.,1.),(-2.,1.)]
        right=[(1.,-1.),(2.,-1.),(2.,1.),(1.,1.)]
        x=robust_blackbox_action_error_floor(
            indistinguishable_desired_actions=[(0.,0.)],
            reachable_actuator_polygons=[left,right],
            requested_error_tolerance=.9)
        self.assertAlmostEqual(x["tight_numerical_worst_case_minimax_error_upper"],1.,delta=1e-7)
        self.assertFalse(x["can_meet_requested_tolerance"])
        witness=x["reachable_physical_action_witness"]
        self.assertGreaterEqual(abs(witness[0]),1.-1e-7)

    def test_point_observation_and_point_reachability(self):
        x=robust_blackbox_action_error_floor(
            indistinguishable_desired_actions=[(3.,-2.)],
            reachable_actuator_polygons=[[(0.,0.)]])
        self.assertAlmostEqual(x["tight_numerical_worst_case_minimax_error_upper"],3.,delta=1e-7)
        self.assertEqual(x["ambiguity_information_only_lower_bound"],0.)
        self.assertEqual(x["reachable_physical_action_witness"],[0.,0.])

    def test_witness_validity_and_binary_search_width(self):
        square=[[(-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)]]
        worlds=[(-.8,-.5),(.9,.3),(.2,.6)]
        x=robust_blackbox_action_error_floor(
            indistinguishable_desired_actions=worlds,
            reachable_actuator_polygons=square,
            absolute_search_accuracy=1e-9)
        self.assertLessEqual(
            x["tight_numerical_worst_case_minimax_error_upper"]-
            x["tight_numerical_worst_case_minimax_error_lower"],1.1e-9)
        w=x["reachable_physical_action_witness"]
        err=max(max(abs(w[j]-p[j]) for j in range(2)) for p in worlds)
        self.assertLessEqual(err,
          x["tight_numerical_worst_case_minimax_error_upper"]+5e-9)

    def test_refuse_invalid_worlds_or_untrusted_polygons(self):
        square=[[(-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)]]
        args=dict(indistinguishable_desired_actions=[(0.,0.)],
                  reachable_actuator_polygons=square)
        for bad in (
            dict(indistinguishable_desired_actions=[]),
            dict(indistinguishable_desired_actions=[(math.nan,0.)]),
            dict(reachable_actuator_polygons=[]),
            dict(reachable_actuator_polygons=[[(0.,math.inf)]]),
            dict(requested_error_tolerance=-.1),
            dict(absolute_search_accuracy=-1.),
        ):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                robust_blackbox_action_error_floor(**{**args,**bad})

if __name__=="__main__":unittest.main()
