"""Adversarial robust physical-input recoverability under hidden source worlds."""
import math
import random
import unittest
from research.semrepair.robust_multiworld_upstream_control import (
    physical_command,solve_common_upstream_minimax,
    run_indistinguishable_interface_stress
)


class CommonUpstreamSourceProofTests(unittest.TestCase):
    def test_two_hidden_source_signs_individually_reachable_but_not_by_same_input(self):
        x=run_indistinguishable_interface_stress()
        bad=x["uncertain_source_worlds"]
        self.assertEqual(bad["decision"],"UNSAT_WITH_INTERVAL_LOWER_BOUND")
        self.assertAlmostEqual(bad["global_minimax_lower_under_assumed_exact_model"],.7,delta=1e-8)
        self.assertAlmostEqual(bad["global_minimax_upper_under_assumed_exact_model"],.7,delta=1e-8)
        self.assertFalse(bad["can_authorize_policy_command"])
        self.assertEqual(len(bad["per_world_ACTUAL_physical_commands"]),2)
        self.assertTrue(all(x["independently_reachable_targets_per_world"]))

    def test_known_same_source_is_repairable_with_actual_bounded_upstream_witness(self):
        x=run_indistinguishable_interface_stress()["source_worlds_verified_to_share_correct_sign"]
        self.assertEqual(x["decision"],"SAT_WITH_COMMON_UPSTREAM_WITNESS")
        self.assertLessEqual(x["actual_common_upstream_max_error"],.01)
        self.assertLessEqual(abs(x["unique_common_upstream_witness"][0]),1.)
        self.assertTrue(x["can_authorize_policy_command"])

    def test_exact_minimax_not_optimistically_relaxed_output_oracle(self):
        I=((1.,0.),(0.,1.))
        neg=((-1.,0.),(0.,1.))
        w=lambda P:dict(P=P,C=I,L=(2.,2.),B=(2.,2.),desired=(.7,0.))
        for P in (I,neg):
            result=solve_common_upstream_minimax(
                worlds=[w(P)],input_radii=(1.,1.),accepted_error=.02,
                max_nodes=20000,accuracy=.002)
            self.assertTrue(result["can_authorize_policy_command"])
        combined=solve_common_upstream_minimax(
            worlds=[w(I),w(neg)],input_radii=(1.,1.),accepted_error=.1,max_nodes=400)
        self.assertFalse(combined["can_authorize_policy_command"])
        self.assertGreater(combined["global_minimax_lower_under_assumed_exact_model"],.69)

    def test_lipschitz_interval_box_cover_refusal_is_sound_against_dense_grid(self):
        I=((1.,0.),(0.,1.))
        neg=((-1.,0.),(0.,1.))
        worlds=[
            dict(P=I,C=I,L=(.5,.5),B=(2.,2.),desired=(.8,-.4)),
            dict(P=neg,C=I,L=(.5,.5),B=(2.,2.),desired=(.8,-.4)),
        ]
        x=solve_common_upstream_minimax(
            worlds=worlds,input_radii=(1.,1.),
            accepted_error=.05,max_nodes=3500,accuracy=.0001)
        self.assertEqual(x["decision"],"UNSAT_WITH_INTERVAL_LOWER_BOUND")
        brute=min(max(max(abs(y-world["desired"][j])
                          for j,y in enumerate(physical_command(world,(a/20,b/20))))
                      for world in worlds)
                  for a in range(-20,21) for b in range(-20,21))
        self.assertLessEqual(x["global_minimax_lower_under_assumed_exact_model"],
                             brute+1e-9)
        self.assertGreaterEqual(x["global_minimax_upper_under_assumed_exact_model"],
                                x["global_minimax_lower_under_assumed_exact_model"]-1e-9)

    def test_low_branch_budget_cannot_fabricate_unproven_repair(self):
        I=((1.,0.),(0.,1.))
        w=dict(P=I,C=I,L=(2.,2.),B=(2.,2.),desired=(.7,.3))
        x=solve_common_upstream_minimax(worlds=[w],
                 input_radii=(1.,1.),accepted_error=.001,
                 max_nodes=1,accuracy=.00000001)
        self.assertEqual(x["decision"],"UNKNOWN_BUDGET_OR_PRECISION")
        self.assertFalse(x["can_authorize_policy_command"])
        self.assertGreater(x["optimality_bracket_width"],.001)

    def test_invalid_worlds_and_infinite_inputs_rejected(self):
        I=((1.,0.),(0.,1.))
        w=dict(P=I,C=I,L=(2.,2.),B=(2.,2.),desired=(.7,.3))
        cases=[
            dict(worlds=[]),
            dict(worlds=[dict(w,P=[[float("nan"),0],[0,1]])]),
            dict(input_radii=(-1.,1.)),
            dict(accepted_error=-1.),
            dict(max_nodes=0),
            dict(accuracy=0.),
        ]
        for v in cases:
            with self.subTest(v=v),self.assertRaises(ValueError):
                solve_common_upstream_minimax(**{**dict(worlds=[w]),**v})

if __name__=="__main__":
    unittest.main()
