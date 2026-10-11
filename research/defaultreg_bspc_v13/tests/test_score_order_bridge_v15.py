"""No-target-label controls for the necessary ROC-AUC score order bound."""
import unittest
import numpy as np
from shin_score_order_bridge_v15 import changed_pairs


class TrialOrderBridgeTests(unittest.TestCase):
    def test_strictly_monotone_scores_cannot_change_roc_auc(self):
        rng=np.random.default_rng(8)
        a=rng.normal(size=60)
        k,total=changed_pairs(a,7*a+12)
        self.assertEqual(k,0)
        self.assertEqual(total,1770)

    def test_swapped_scores_change_order(self):
        a=np.array([1.,2.,3.])
        b=np.array([2.,1.,3.])
        self.assertEqual(changed_pairs(a,b),(1,3))

    def test_ties_are_a_distinct_pair_order(self):
        a=np.array([1.,1.,2.])
        b=np.array([1.,1.1,2.])
        self.assertEqual(changed_pairs(a,b)[0],1)

    def test_invalid_scores_fail(self):
        with self.assertRaises(ValueError):
            changed_pairs([1.,float("nan")],[1.,2.])


if __name__=="__main__":
    unittest.main()
