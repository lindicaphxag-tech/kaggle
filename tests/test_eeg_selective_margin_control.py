"""Strong comparator: validate calibration without any test-label-dependent threshold."""
import unittest
import numpy as np

from research.eeg.validation_only_margin_control import (
    fit_validation_margin_threshold, audit_test_predictions,
)
from research.eeg.eegmmidb_certificate_real import certify_cohort
from research.eeg.eegmmidb_real_pilot import SPLITS


class MatchedCoverageTests(unittest.TestCase):
    def test_validation_coverage_target(self):
        fit=fit_validation_margin_threshold([.1,.2,.3,.4,.5],[True,True,False,False,False])
        self.assertEqual(fit.requested_validation_coverage,.4)
        self.assertEqual(fit.achieved_validation_coverage,.4)
        self.assertAlmostEqual(fit.threshold,.35)
        self.assertFalse(fit.ties_prevent_exact_coverage)

    def test_ties_report_unmatchable_calibration(self):
        fit=fit_validation_margin_threshold([1,1,1,1],[True,False,False,False])
        self.assertTrue(fit.ties_prevent_exact_coverage)
        self.assertEqual(fit.achieved_validation_coverage,1.0)

    def test_zero_and_full_coverage(self):
        none=fit_validation_margin_threshold([.2,.3],[False,False])
        self.assertTrue(np.isinf(none.threshold))
        full=fit_validation_margin_threshold([.2,.3],[True,True])
        self.assertEqual(full.threshold,0.0)

    def test_never_use_labels_to_fit_threshold(self):
        fit=fit_validation_margin_threshold([.1,.2,.3,.4],[True,True,False,False])
        res1=audit_test_predictions([.7,.2],[1,0],[1,1],fit.threshold)
        res2=audit_test_predictions([.7,.2],[1,0],[0,0],fit.threshold)
        self.assertEqual(res1['accepted'],res2['accepted'])
        self.assertNotEqual(res1['accuracy_on_accepted'],res2['accuracy_on_accepted'])

    def test_end_to_end_disjoint_synthetic_cohort(self):
        split=SPLITS['pilot']
        ids=split['train']+split['val']+split['test']
        rng=np.random.default_rng(123)
        channels=('C1','C3','C2','C4','F3','F4')
        X=rng.normal(size=(len(ids)*6,6,2))
        y=np.tile([0,1,0,1,0,1],len(ids))
        sub=np.repeat(ids,6)
        x=certify_cohort(X,y,sub,channels)
        control=x['validation_only_confidence_baseline']
        self.assertEqual(control['calibration_subjects'],list(split['val']))
        self.assertTrue(control['no_test_threshold_tuning'])
        self.assertEqual(control['comparison']['n_test'],len(split['test'])*6)
        self.assertEqual(x['false_invariance_certificates'],0)


if __name__=='__main__':
    unittest.main()
