"""Pure-stdlib finite catalog source-counterexample optimization regression."""
from itertools import combinations
import unittest

from research.semrepair.minimum_counterexample_probe_cover import (
    Probe,exact_min_cost_counterexamples,
    baseline_maniskill_grammar_probe_suite
)

class TestExactSourceProbeCover(unittest.TestCase):
    def test_symmetric_source_overfit_removed_by_one_asymmetric_probe(self):
        r=baseline_maniskill_grammar_probe_suite()
        self.assertEqual(r["selected_probes"],["cfg_1"])
        self.assertEqual(r["provable_minimum_cost"],1)
        self.assertEqual(r["candidate_count"],5)
        self.assertTrue(r["certificate_complete_against_catalog"])

    def test_incomplete_probe_bank_refuses(self):
        r=exact_min_cost_counterexamples(
            ("a","b"),[Probe("one",1,("a",))])
        self.assertEqual(r["decision"],
                         "REFUSE_NO_COMPLETE_DISTINGUISHING_PROBE_SET")
        self.assertIsNone(r["selected_probes"])

    def test_insufficient_probing_budget_refuses_even_valid_suite(self):
        r=exact_min_cost_counterexamples(
            ("a","b"),[Probe("one",2,("a",)),Probe("two",3,("b",))],
            max_budget=4)
        self.assertEqual(r["decision"],"REFUSE_INSUFFICIENT_PROBE_BUDGET")
        self.assertEqual(r["provable_minimum_cost"],5)
        self.assertIsNone(r["selected_probes"])

    def test_certified_weighted_exact_cost_equals_independent_bruteforce(self):
        bank=[
            Probe("p0",4,("a","b")),Probe("p1",2,("c","d")),
            Probe("p2",1,("a","c")),Probe("p3",1,("b","d")),
            Probe("p4",10,("a","b","c","d")),
        ]
        actual=exact_min_cost_counterexamples(("a","b","c","d"),bank)
        brute=[]
        for n in range(1,len(bank)+1):
            for subset in combinations(bank,n):
                if set().union(*(set(x.excludes) for x in subset))=={"a","b","c","d"}:
                    brute.append(sum(p.integer_cost for p in subset))
        self.assertEqual(actual["provable_minimum_cost"],min(brute))
        self.assertEqual(actual["selected_probes"],["p2","p3"])

    def test_untrusted_or_unbounded_probe_bank_rejected(self):
        with self.assertRaises(ValueError):
            exact_min_cost_counterexamples((),[Probe("a",1,("a",))])
        with self.assertRaises(ValueError):
            exact_min_cost_counterexamples(("a",),[Probe("q",-1,("a",))])
        with self.assertRaises(ValueError):
            exact_min_cost_counterexamples(("a",),[Probe("q",1,("other",))])
        with self.assertRaises(ValueError):
            exact_min_cost_counterexamples(
                ("a",),[Probe("q",1,("a",))],max_budget=-1)

if __name__=="__main__":
    unittest.main()
