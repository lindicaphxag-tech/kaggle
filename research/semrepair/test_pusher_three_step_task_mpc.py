"""True Pusher three-step actor-in-loop native task reward oracle tests.

No assertion that our method must win; the task study records all failures.
"""
import unittest
import numpy as np
import gymnasium as gym
from research.semrepair.external_pusher_transfer import (
    load_external_pusher_checkpoint,_matrices,ENV
)
from research.semrepair.pusher_three_step_task_mpc import (
    TrueNativeThreeStepMPC,SEEDS,ARM_NAMES,HORIZON,FAULT_SHEAR,
    FAULT_INNER_CLIP,FUTURE_CONTROL_GRID
)


class RealTaskLookaheadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actor,cls.hash=load_external_pusher_checkpoint()

    def test_model_cloned_branch_next_reward_exactly_matches_official(self):
        _,C=_matrices(FAULT_SHEAR)
        with gym.make(ENV) as env:
            p=TrueNativeThreeStepMPC(env.unwrapped.model,
                                     env.unwrapped.frame_skip,
                                     self.actor,C,FAULT_INNER_CLIP)
            obs,_=env.reset(seed=SEEDS[0])
            desired,_=self.actor.predict(obs,deterministic=True)
            action,_,num=p.choose(env.unwrapped.data,desired)
            self.assertGreaterEqual(num,9)
            immediate=p.branch_value(env.unwrapped.data,desired,
                                     tuple(float(z) for z in action[:2]),horizon=1)
            _,real,_,_,_=env.step(action)
            self.assertAlmostEqual(immediate,float(real),places=7)

    def test_three_step_lookahead_does_not_mutate_original_env(self):
        _,C=_matrices(FAULT_SHEAR)
        with gym.make(ENV) as env:
            p=TrueNativeThreeStepMPC(env.unwrapped.model,
                                     env.unwrapped.frame_skip,
                                     self.actor,C,FAULT_INNER_CLIP)
            obs,_=env.reset(seed=SEEDS[1])
            d=env.unwrapped.data
            q_before=d.qpos.copy()
            v_before=d.qvel.copy()
            warm_before=d.qacc_warmstart.copy()
            time_before=float(d.time)
            desired,_=self.actor.predict(obs,deterministic=True)
            score=p.branch_value(d,desired,(0.,0.))
            self.assertTrue(np.isfinite(score))
            np.testing.assert_array_equal(q_before,d.qpos)
            np.testing.assert_array_equal(v_before,d.qvel)
            np.testing.assert_array_equal(warm_before,d.qacc_warmstart)
            self.assertEqual(time_before,float(d.time))
            self.assertGreater(p.policy_tail_predictions,0)

    def test_source_actions_are_within_native_physical_bounds(self):
        _,C=_matrices(FAULT_SHEAR)
        with gym.make(ENV) as env:
            p=TrueNativeThreeStepMPC(env.unwrapped.model,
                                     env.unwrapped.frame_skip,
                                     self.actor,C,FAULT_INNER_CLIP)
            obs,_=env.reset(seed=SEEDS[0])
            desired,_=self.actor.predict(obs,deterministic=True)
            samples=p._proposals(desired)
            self.assertGreaterEqual(len(samples),9)
            for s in samples:
                self.assertEqual(len(s),2)
                self.assertTrue(np.all(np.abs(s)<=2.0000001))
            self.assertEqual(HORIZON,3)
            self.assertEqual(len(ARM_NAMES),3)

if __name__=="__main__":
    unittest.main()
