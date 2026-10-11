"""Global source-world MILP versus analytic and brute-force original action laws."""
import random
import unittest

from research.semrepair.exact_common_source_world_milp import (
    solve_common_upstream_milp,two_source_identity_regression,
    actual_physical_command
)


class GlobalDoubleClipSourceWorldMILPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=two_source_identity_regression()

    def test_TWO_hidden_source_versions_have_separate_zero_error_but_common_minimum_point_seven(self):
        bad=self.x["two_individually_repairable_but_indistinguishable_versions"]
        self.assertEqual(bad["decision"],"UNSAT_GLOBAL_DUAL_BOUND")
        self.assertAlmostEqual(bad["global_mixed_integer_dual_lower"],.7,delta=1e-7)
        self.assertAlmostEqual(bad["global_primal_objective_upper"],.7,delta=1e-7)
        self.assertAlmostEqual(bad["replayed_true_max_error"],.7,delta=1e-7)
        self.assertEqual(bad["number_of_source_worlds"],2)
        self.assertEqual(bad["expanded_source_CLIP_binary_variables"],24)

    def test_known_version_has_true_same_input_authorization(self):
        good=self.x["version_identity_known"]
        self.assertEqual(good["decision"],"SAT_REPLAYED_COMMON_UPSTREAM_INPUT")
        self.assertAlmostEqual(good["global_mixed_integer_dual_lower"],0.,delta=1e-8)
        self.assertLess(good["replayed_true_max_error"],1e-8)
        self.assertLessEqual(abs(good["actual_shared_upstream_control"][0]),1.)

    def test_shear_clipping_exact_global_bound_better_than_unchecked_physical_output(self):
        shear=((1.,3.5),(0.,1.))
        inverse=((1.,-3.5),(0.,1.))
        world=dict(P=shear,C=inverse,L=(.4,.4),B=(1.,1.),
                   desired=(.8,.5))
        p=solve_common_upstream_milp(worlds=[world],tolerance=.01)
        self.assertEqual(p["decision"],"UNSAT_GLOBAL_DUAL_BOUND")
        self.assertAlmostEqual(p["global_mixed_integer_dual_lower"],
                               .4777777777777778,delta=2e-7)
        self.assertAlmostEqual(p["replayed_true_max_error"],
                               .4777777777777778,delta=2e-7)
        self.assertLessEqual(abs(p["actual_shared_upstream_control"][0]),1.+1e-9)
        self.assertLessEqual(abs(p["actual_shared_upstream_control"][1]),1.+1e-9)

    def test_all_solver_global_lower_bounds_do_not_exceed_direct_dense_physical_grid(self):
        rng=random.Random(20261010)
        for seed in range(6):
            k=rng.uniform(-3.,3.)
            L=rng.uniform(.2,1.0)
            world=dict(P=((1.,k),(0.,1.)),
                       C=((1.,-k),(0.,1.)),L=(L,L),B=(1.,1.),
                       desired=(rng.uniform(-1,1),rng.uniform(-1,1)))
            proof=solve_common_upstream_milp(worlds=[world],
                                             tolerance=.001,seconds=20)
            grid=min(max(abs(actual_physical_command(
                world,(i/20.,j/20.))[a]-world["desired"][a])
                for a in range(2))
                for i in range(-20,21) for j in range(-20,21))
            self.assertLessEqual(proof["global_mixed_integer_dual_lower"],
                                 grid+1e-6)
            self.assertGreaterEqual(proof["replayed_true_max_error"]+1e-6,
                                    proof["global_mixed_integer_dual_lower"])

    def test_unknown_at_exact_boundary_not_falsely_authorized(self):
        I=((1.,0.),(0.,1.))
        world=dict(P=I,C=I,L=(2.,2.),B=(2.,2.),
                   desired=(.7,0.))
        x=solve_common_upstream_milp(worlds=[world],
                                    tolerance=0.,numeric_margin=1e-6)
        self.assertEqual(x["decision"],"UNKNOWN_NO_AUTHORIZATION")
        self.assertLess(x["replayed_true_max_error"],1e-8)

    def test_bad_worlds_and_uncertainty_require_refusal(self):
        I=((1.,0.),(0.,1.))
        normal=dict(P=I,C=I,L=(2.,2.),B=(2.,2.),
                    desired=(.7,0.))
        for changes in (
            dict(worlds=[]),
            dict(worlds=[dict(normal,P=((float("nan"),0.),(0.,1.)))]),
            dict(input_radii=(-1.,1.)),
            dict(tolerance=-.5),
            dict(seconds=0),
            dict(numeric_margin=0),
            dict(solver_gap=-1),
        ):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                solve_common_upstream_milp(**{**{"worlds":[normal]},**changes})

if __name__=="__main__":
    unittest.main()
