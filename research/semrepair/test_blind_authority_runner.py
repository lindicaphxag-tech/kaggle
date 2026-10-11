import unittest
from fractions import Fraction as F
from blind_authority_runner import ProbeSpec, execute_blind


class Tests(unittest.TestCase):
    def setUp(self):
        self.w = ("w0", "w1")
        self.auth = {"w0": "patch:A", "w1": "patch:B"}
        self.p = {"internal": ProbeSpec(
            "internal", {"w0": F(0), "w1": F(2)}, F(0), F(1), F(1))}

    def test_blind_internal_tap_resolves(self):
        r = execute_blind(
            worlds=self.w, repair_by_world=self.auth, probes=self.p,
            policy=lambda _: "internal", observe=lambda _: F(2),
            total_cost_budget=F(1), max_single_probe_risk=F(1))
        self.assertEqual(r.authorized_repair, "patch:B")

    def test_risk_refused_without_probe(self):
        r = execute_blind(
            worlds=self.w, repair_by_world=self.auth, probes=self.p,
            policy=lambda _: "internal", observe=lambda _: F(0),
            total_cost_budget=F(1), max_single_probe_risk=F(0))
        self.assertIsNone(r.authorized_repair)
        self.assertEqual(r.probes_used, ())

    def test_external_alias_refused(self):
        p = {"external": ProbeSpec(
            "external", {"w0": F(1), "w1": F(1)}, F(0))}
        r = execute_blind(
            worlds=self.w, repair_by_world=self.auth, probes=p,
            policy=lambda _: "external", observe=lambda _: F(1),
            total_cost_budget=F(1), max_single_probe_risk=F(0))
        self.assertEqual(r.status, "REFUSED_UNINFORMATIVE_OBSERVATION")

    def test_invalid_hidden_response_refused(self):
        r = execute_blind(
            worlds=self.w, repair_by_world=self.auth, probes=self.p,
            policy=lambda _: "internal", observe=lambda _: F(9),
            total_cost_budget=F(1), max_single_probe_risk=F(1))
        self.assertEqual(r.status, "REFUSED_MODEL_MISSPECIFICATION")

    def test_common_repair_zero_probe(self):
        r = execute_blind(
            worlds=self.w,
            repair_by_world={"w0": "patch:X", "w1": "patch:X"},
            probes={}, policy=lambda _: "reject", observe=lambda _: F(0),
            total_cost_budget=F(0), max_single_probe_risk=F(0))
        self.assertEqual(r.status, "AUTHORIZED")
        self.assertEqual(r.probe_cost, F(0))


if __name__ == "__main__":
    unittest.main()
