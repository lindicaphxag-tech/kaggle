"""Native articulated arm benchmark regression and no-overclaim controls."""
import math
import unittest
from research.semrepair.two_link_cross_controller_benchmark import (
    CASES,CONTROLLERS,ARM_NAMES,N_GOALS,run_cross_controller_benchmark,
    make_model, frozen_targets,tip_xy,
)
import mujoco

class CrossControllerBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=run_cross_controller_benchmark()

    def test_genuinely_articulated_two_hinge_robot(self):
        model=make_model()
        self.assertEqual(model.nq,2)
        self.assertEqual(model.nu,2)
        self.assertEqual(model.nv,2)
        self.assertEqual(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,0),"shoulder")
        self.assertEqual(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,1),"elbow")

    def test_all_fair_arms_and_trials_kept(self):
        r=self.result
        self.assertEqual(len(r["raw"]),len(CASES)*len(CONTROLLERS)*N_GOALS*len(ARM_NAMES))
        self.assertEqual(len(r["diagnosis"]),len(CASES))
        self.assertTrue(r["matched_blackbox_composition_receives_same_training_probes"])
        self.assertFalse(r["source_provenance_authentication"])
        self.assertEqual(r["learned_policy_count"],0)

    def test_blackbox_composition_is_a_serious_equal_info_baseline(self):
        scores=self.result["summary"]
        for case,_,_,_ in CASES:
            for policy in CONTROLLERS:
                black=scores[f"{case}|{policy}|matched-blackbox-inverse"]
                source=scores[f"{case}|{policy}|oracle-source-owner"]
                self.assertEqual(black["successes"],source["successes"])
                # Equal semantics when all mappings are linear and no
                # intermediate saturations: any claimed strict task advantage
                # over this comparator would be dishonest.
                self.assertAlmostEqual(black["mean_terminal_error"],
                                       source["mean_terminal_error"],places=7)

    def test_numeric_taps_exclude_fault_labels(self):
        for x in self.result["diagnosis"]:
            self.assertTrue(x["source_candidate_matches_oracle"])
            self.assertLess(x["heldout_max_tap_error"],1e-9)

    def test_source_candidate_and_source_oracle_task_equivalence(self):
        scores=self.result["summary"]
        for case,_,_,_ in CASES:
            for policy in CONTROLLERS:
                inferred=scores[f"{case}|{policy}|numeric-source-candidate"]
                oracle=scores[f"{case}|{policy}|oracle-source-owner"]
                self.assertEqual(inferred["successes"],oracle["successes"])
                self.assertAlmostEqual(inferred["mean_terminal_error"],
                                       oracle["mean_terminal_error"],places=7)

    def test_error_metrics_finite_and_no_trials_omitted(self):
        for r in self.result["raw"]:
            e=r["result"]["terminal_error"]
            self.assertTrue(math.isfinite(e))
            self.assertGreaterEqual(e,0.)
        for val in self.result["summary"].values():
            self.assertEqual(val["goals"],N_GOALS)
            self.assertLessEqual(val["successes"],N_GOALS)

    def test_tip_fk_matches_forward_native_mechanics(self):
        model=make_model()
        for pair in frozen_targets()[:4]:
            d=mujoco.MjData(model)
            d.qpos[:2]=pair
            mujoco.mj_forward(model,d)
            s=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"tip")
            xy=tuple(float(x) for x in d.site_xpos[s][:2])
            self.assertLess(math.dist(xy,tip_xy(pair)),1e-8)

if __name__=="__main__":
    unittest.main()
