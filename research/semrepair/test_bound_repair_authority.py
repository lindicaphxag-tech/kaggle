import unittest
from fractions import Fraction as F
from research.semrepair.bound_repair_authority import RepairAuthority as A, execute_bound_repair


def auth(p, dependency="f"):
    return A(p*64, "b"*64, "c"*64, dependency*64)


class TestBoundRepair(unittest.TestCase):
    def _run(self, repairs):
        return execute_bound_repair(
            worlds=("w0","w1"), authorities=repairs,
            probes={}, policy=lambda _: "reject",
            observe=lambda _: F(0),
            total_cost_budget=F(0), max_single_probe_risk=F(0))

    def test_same_patch_but_different_dependency_refused(self):
        x = self._run({"w0":auth("a"),"w1":auth("a","d")})
        self.assertEqual(x["decision"], "REFUSED_CONFLICTING_REPAIRS")
        self.assertIsNone(x["bound_repair"])

    def test_identical_complete_authority_needs_no_probe(self):
        x = self._run({"w0":auth("a"),"w1":auth("a")})
        self.assertEqual(x["decision"], "AUTHORIZED")
        self.assertEqual(x["bound_repair"].dependency_sha256, "f"*64)
        self.assertEqual(x["probe_cost"], F(0))

    def test_distinct_implementation_requires_new_evidence(self):
        first = auth("a")
        second = A("a"*64,"d"*64,"c"*64,"f"*64)
        self.assertEqual(self._run({"w0":first,"w1":second})["decision"],
                         "REFUSED_CONFLICTING_REPAIRS")

    def test_malformed_sha_refused(self):
        with self.assertRaises(ValueError):
            A("not a sha", "b"*64, "c"*64, "d"*64)


if __name__ == "__main__":
    unittest.main()
