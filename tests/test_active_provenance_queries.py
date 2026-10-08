"""Fail-closed and exact minimax contracts for active electrode ID verification."""
import itertools
import unittest
import numpy as np
from research.eeg.active_provenance_queries import ProvenanceQueryPlanner

class ActiveProvenanceTests(unittest.TestCase):
    def test_all_permutations_robust_without_query(self):
        x=np.array([[1.],[2.],[3.]])
        w=np.ones_like(x)
        p=ProvenanceQueryPlanner(x,w,-2.,[(0,1,2)])
        r=p.propose()
        self.assertTrue(r.authorized)
        self.assertEqual((r.label,r.worst_case_queries),(1,0))

    def test_two_disjoint_ambiguities_query_then_certify(self):
        x=np.array([[1.],[-1.],[.5],[-.5]])
        w=np.array([[1.],[-1.],[.7],[-.7]])
        p=ProvenanceQueryPlanner(x,w,0.,[(0,1),(2,3)])
        before=p.propose()
        self.assertFalse(before.authorized)
        self.assertIn(before.worst_case_queries,(1,2))
        for a in itertools.permutations((0,1)):
            for b in itertools.permutations((2,3)):
                true_mapping=a+b
                final,answered=p.interact(true_mapping)
                self.assertTrue(final.authorized)
                self.assertLessEqual(len(answered),before.worst_case_queries)
                margin=sum(x[i,0]*w[true_mapping[i],0] for i in range(4))
                self.assertEqual(final.label,int(margin>0))

    def test_trusted_answer_outside_hypothesis_set_rejected(self):
        p=ProvenanceQueryPlanner(np.array([[1.],[-1.]]),
            np.array([[1.],[-1.]]),0.,[(0,1)])
        with self.assertRaisesRegex(ValueError,"contradicts"):
            p.propose([(0,9)])

    def test_budget_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"budget"):
            ProvenanceQueryPlanner(np.ones((8,1)),np.ones((8,1)),
                0.,[tuple(range(8))],max_hypotheses=720)

    def test_zero_margin_cannot_be_certified_even_when_all_known(self):
        p=ProvenanceQueryPlanner(np.ones((2,1)),np.zeros((2,1)),
                                 0.,[(0,1)])
        result=p.propose()
        self.assertFalse(result.authorized)
        self.assertIsNone(result.worst_case_queries)
        self.assertIsNone(result.next_observed_row)

    def test_single_pair_worst_case_one_query(self):
        p=ProvenanceQueryPlanner(np.array([[1.],[-1.]]),
            np.array([[1.],[-1.]]),0.,[(0,1)])
        self.assertEqual(p.propose().worst_case_queries,1)
        final,queries=p.interact((1,0))
        self.assertTrue(final.authorized)
        self.assertEqual(len(queries),1)
        self.assertEqual(final.label,0)

if __name__=="__main__":
    unittest.main()
