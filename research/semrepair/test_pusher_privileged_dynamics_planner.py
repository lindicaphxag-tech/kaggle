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
            _,real_reward,_,_,_=env.step(action)
            self.assertAlmostEqual(float(score),float(real_reward),places=7)
            self.assertEqual(action.shape,(7,))
            self.assertTrue(np.all(action<=2.00001))
            self.assertTrue(np.all(action>=-2.00001))
            self.assertGreater(planner.reward_evaluations,20)

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
