"""Independent brute-force optimality cross-check (NOT reuse policy recursion)."""
import itertools
import unittest

import numpy as np
from research.eeg.active_provenance_queries import ProvenanceQueryPlanner

class OptimalityOracleTests(unittest.TestCase):
    def test_minimax_depth_matches_100_independent_triplet_enumerations(self):
        rng=np.random.default_rng(20261009)
        for _ in range(100):
            x=rng.normal(size=(3,2))
            w=rng.normal(size=(3,2))
            b=float(rng.normal())
            perms=list(itertools.permutations(range(3)))
            margins=[b+sum(float(x[i]@w[p[i]]) for i in range(3)) for p in perms]
            if any(abs(m)<1e-8 for m in margins):
                continue
            labels=[m>0 for m in margins]
            if len(set(labels))==1:
                brute=0
            else:
                one_query=False
                for row in range(3):
                    buckets={}
                    for p,label in zip(perms,labels):
                        buckets.setdefault(p[row],set()).add(label)
                    one_query |= all(len(possible)==1 for possible in buckets.values())
                brute=1 if one_query else 2
            predicted=ProvenanceQueryPlanner(x,w,b,[(0,1,2)]).propose()
            self.assertEqual(predicted.worst_case_queries,brute)

if __name__=="__main__":
    unittest.main()
