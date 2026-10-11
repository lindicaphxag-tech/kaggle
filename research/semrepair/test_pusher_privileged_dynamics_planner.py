"""An OFFICIAL simulator-level reward oracle on physically feasible actions."""
import math
import unittest
import numpy as np
import gymnasium as gym

from research.semrepair.pusher_privileged_dynamics_planner import (
    PrivilegedOneStepModelPlanner,_rollout,SEEDS,SEVERE,
    ORACLE_ACTION_LIBRARY_SPEC
)
from research.semrepair.external_pusher_transfer import (
    load_external_pusher_checkpoint, _matrices,ENV
)
from research.semrepair.strong_clipped_actuator_oracle import (
    output_reachability_polygons,optimal_clipped_output
)

class PrivilegedNativePlannerTests(unittest.TestCase):
    def test_initial_state_and_reward_is_actual_official_pusher_model(self):
        actor,sha=load_external_pusher_checkpoint()
        _,k,L=SEVERE
        _,C=_matrices(k)
        with gym.make(ENV) as env:
            planner=PrivilegedOneStepModelPlanner(
                env.unwrapped.model,env.unwrapped.frame_skip,C,L)
            initial,_=env.reset(seed=SEEDS[0])
            ideal,_=actor.predict(initial,deterministic=True)
            action,score=planner.choose_feasible_action(env.unwrapped.data,ideal)
            recomputed=planner.native_reward_of_action(env.unwrapped.data,action)
            self.assertAlmostEqual(float(score),float(recomputed),places=9)
            scratch_qpos=planner.scratch.qpos.copy()
            scratch_qvel=planner.scratch.qvel.copy()
            scratch_xpos=planner.scratch.xpos.copy()
            _,real_reward,_,_,info=env.step(action)
            if abs(float(recomputed)-float(real_reward))>1e-7:
                actual=env.unwrapped.data
                print("PHYSICS_MISMATCH_DIAGNOSTIC",{
                    "source_integrator_version":__import__("mujoco").__version__,
                    "clone_reward":float(recomputed),
                    "official_reward":float(real_reward),
                    "official_reward_parts":info,
                    "candidate_control":action.tolist(),
                    "qpos_max_abs_delta":float(np.max(np.abs(scratch_qpos-actual.qpos))),
                    "qvel_max_abs_delta":float(np.max(np.abs(scratch_qvel-actual.qvel))),
                    "derived_body_xpos_max_abs_delta":float(np.max(np.abs(scratch_xpos-actual.xpos))),
                    "cloned_predicted_native_action_reward":float(
                        -0.1*np.sum(np.square(action))),
                    "cloned_object_goal_distance":float(np.linalg.norm(
                        scratch_xpos[planner.obj]-scratch_xpos[planner.goal])),
                    "official_object_goal_distance":-float(info["reward_dist"]),
                    "cloned_fingertip_object_distance":float(np.linalg.norm(
                        scratch_xpos[planner.obj]-scratch_xpos[planner.fingertip])),
                    "official_fingertip_object_distance":-float(info["reward_near"])/.5,
                },flush=True)
            self.assertAlmostEqual(float(score),float(real_reward),places=7)
            self.assertEqual(action.shape,(7,))
            self.assertTrue(np.all(action<=2.00001))
            self.assertTrue(np.all(action>=-2.00001))
            # Candidates share output torques after the SECOND official
            # clamp; duplicate reachable outputs are deliberately deduplicated.
            # At this test state the fixed grid produces 19 distinct
            # physically executable controls, not >20.
            self.assertGreaterEqual(planner.reward_evaluations,19)

    def test_all_oracle_actions_from_exact_faulty_reachable_set(self):
        _,k,L=SEVERE
        _,C=_matrices(k)
        poly=output_reachability_polygons(C,inner_bound=L,outer_bound=2.)
        request=(1.6,1.5)
        p=optimal_clipped_output(request,poly)
        self.assertGreater(p["minimum_euclidean_error"],0.)
        self.assertTrue(all(math.isfinite(z) for z in p["action"]))

    def test_predeclared_scope_annotates_followup_not_new_holdout(self):
        self.assertEqual(len(SEEDS),16)
        self.assertEqual(SEVERE,("shear-3point5-plus-clip",3.5,.8))
        self.assertIn("projection",ORACLE_ACTION_LIBRARY_SPEC)

if __name__=="__main__":
    unittest.main()
