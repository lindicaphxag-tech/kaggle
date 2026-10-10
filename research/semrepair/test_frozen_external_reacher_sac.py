"""Third-party policy integration tests. No new weights are trained."""
import unittest
import numpy as np
from research.semrepair.frozen_external_reacher_sac import (
    C,P,I,ARMS,ENV_SEEDS,HF_REVISION,INNER_CONTROL_LIMIT,
    _transform,frozen_external_checkpoint,execute_frozen_episode,
)
from research.semrepair.observable_interface_factorization import mmul
import gymnasium as gym


class ExternalReacherPolicyTests(unittest.TestCase):
    def test_predeclared_fixed_checkpoint_and_seeds(self):
        self.assertEqual(len(HF_REVISION),40)
        self.assertEqual(len(ENV_SEEDS),16)
        self.assertEqual(len(set(ENV_SEEDS)),16)
        self.assertTrue(all(seed>=0 for seed in ENV_SEEDS))
        self.assertEqual(len(ARMS),7)
        self.assertEqual(mmul(C,P),I)

    def test_low_amplitude_compensation_matches_healthy(self):
        for u in ((0.,0.),(.1,.2),(-.2,.1)):
            healthy=_transform(np.asarray(u),"healthy")
            faulty=_transform(np.asarray(u),"compensating-fault")
            np.testing.assert_allclose(healthy,faulty,rtol=0,atol=1e-7)

    def test_source_repair_recovers_off_support_commands(self):
        sample=np.array((.8,.8))
        fixed=_transform(sample,"correct-source-atomic-upper-bound")
        fault=_transform(sample,"compensating-fault")
        self.assertGreater(float(np.linalg.norm(fixed-fault)),.01)
        projected=_transform(sample,"full-information-inner-geometry-projection")
        exact=_transform(sample,"exact-two-clip-action-oracle")
        self.assertLessEqual(float(np.linalg.norm(sample-exact)),
                             float(np.linalg.norm(sample-np.clip(projected,-1.,1.)))+1e-6)
        self.assertLessEqual(float(np.linalg.norm(sample-exact)),
                             float(np.linalg.norm(sample-fault))+1e-6)

    def test_official_mujoco_external_task_and_pretrained_checkpoint(self):
        model,sha=frozen_external_checkpoint()
        self.assertEqual(len(sha),64)
        with gym.make("Reacher-v5") as env:
            a=execute_frozen_episode(env,model,ENV_SEEDS[0],"healthy")
            b=execute_frozen_episode(env,model,ENV_SEEDS[0],"correct-source-atomic-upper-bound")
            self.assertEqual(a["steps"],50)
            self.assertEqual(a["episode_return"],b["episode_return"])
            self.assertEqual(a["delivered_actions"],b["delivered_actions"])
            self.assertTrue(np.isfinite(a["mean_tip_target_distance"]))

if __name__=="__main__":
    unittest.main()
