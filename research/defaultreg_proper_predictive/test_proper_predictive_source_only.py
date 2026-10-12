"""Source-only smoke tests against the exact original tangent/covariance stack.
These contain NO target participant inputs or EEG outcomes.
"""
import unittest
import numpy as np
from run_proper_predictive_posthoc import select_proper_source_nll
import original_frozen_runner as old


class SourceOnlyPredictiveTest(unittest.TestCase):
    @staticmethod
    def fake_source(n_people=10, per_person=8, channels=3):
        rng=np.random.default_rng(20261012)
        C=[]; y=[]; g=[]
        for sid in range(1,n_people+1):
            for trial in range(per_person):
                M=rng.normal(size=(channels,channels))
                # All source participant trial SPD covariances.
                c=M@M.T+np.eye(channels)*(1.0+.05*sid+.2*(trial%2))
                C.append(c); y.append(trial%2);g.append(sid)
        return np.asarray(C),np.asarray(y),np.asarray(g)

    def test_real_original_fold_geometry_and_predictive_loss(self):
        C,y,g=self.fake_source()
        allowed=[.01,.05,.1,.5,1.]
        lam,path=select_proper_source_nll(C,y,g,allowed)
        self.assertIn(lam,allowed)
        self.assertEqual(len(path),len(allowed))
        self.assertTrue(np.isfinite(list(path.values())).all())
        lam2,path2=select_proper_source_nll(C,y,g,allowed)
        self.assertEqual(lam,lam2)
        self.assertEqual(path,path2)

    def test_validation_class_shift_is_not_silently_centered(self):
        C,y,g=self.fake_source()
        # A source participant with a shifted class distribution changes
        # a genuine predictive criterion; the old criterion suppresses
        # that class-mean shift, as proved independently in V16.
        grid=[.01,.05,.1,.5,1.]
        a,path1=select_proper_source_nll(C,y,g,grid)
        M=C.copy()
        # Preserve SPD but create systematic class-linked shifted covariance.
        for i in range(len(M)):
            if g[i]==10 and y[i]==1:
                M[i]=np.diag([10.,1.,1.])@M[i]@np.diag([10.,1.,1.])
        b,path2=select_proper_source_nll(M,y,g,grid)
        self.assertGreater(max(abs(path1[k]-path2[k]) for k in path1),1e-8)

    def test_not_enough_source_people_fails_closed(self):
        C,y,g=self.fake_source(7)
        with self.assertRaises((AssertionError,ValueError)):
            select_proper_source_nll(C,y,g,[.01,.1])


if __name__=="__main__":
    unittest.main()
