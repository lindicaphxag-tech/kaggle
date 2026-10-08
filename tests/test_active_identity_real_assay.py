"""Closed-loop trust query experiment tests (synthetic data only)."""
import unittest
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.eeg.active_identity_real_assay import allowed_assignments, evaluate_queries
from research.eeg.filterbank_logeuclidean_real import filter_bank_log_covariance

class ActiveRealAssayContracts(unittest.TestCase):
    def test_all_electrode_swap_scenarios_present(self):
        out=allowed_assignments(5,((0,1),(2,3)))
        self.assertEqual(len(out),4)
        self.assertEqual(len(set(out)),4)

    def test_nonseparable_spatial_model_resolves_all_in_scope_truths(self):
        rng=np.random.default_rng(109)
        names=('C1','C3','C2','C4','F3')
        X=rng.normal(size=(12,5,480))
        y=np.tile([0,1],6)
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,random_state=2))
        clf.fit(filter_bank_log_covariance(X),y)
        result=evaluate_queries(X[:4],y[:4],names,clf)
        self.assertEqual(result["n_test"],4)
        self.assertEqual(result["invariance_violations"],0)
        self.assertEqual(result["known_identity_bacc"],result["post_query_bacc"])

    def test_identity_without_grounded_electrode_names_refused(self):
        rng=np.random.default_rng(2)
        X=rng.normal(size=(4,5,480));y=np.array([0,1,0,1])
        clf=make_pipeline(StandardScaler(),LogisticRegression())
        clf.fit(filter_bank_log_covariance(X),y)
        with self.assertRaisesRegex(ValueError,"Missing predeclared"):
            evaluate_queries(X,y,("F1","F2","F3","F4","F5"),clf)

if __name__=="__main__":
    unittest.main()
