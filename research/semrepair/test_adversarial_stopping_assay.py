import unittest
from fractions import Fraction as F

from research.semrepair.adversarial_stopping_assay import audit_minimax_blind_replay
from research.semrepair.noisy_authority_game import Probe


class AdversarialStoppingAssayTests(unittest.TestCase):
    def setUp(self):
        self.worlds = ("h0", "h1", "h2", "h3")
        self.authority = {w: "patch:"+w for w in self.worlds}
        self.probes = (
            Probe("A", tuple(zip(self.worlds, (F(0), F(3,2), F(3), F(9,2)))), F(1)),
            Probe("B", tuple(zip(self.worlds, (F(3,2), F(9,2), F(0), F(3)))), F(1), risk=F(1,4)),
        )

    def test_exhaustive_four_world_minimax_vs_execution(self):
        report = audit_minimax_blind_replay(
            worlds=self.worlds, authority=self.authority, probes=self.probes,
        )
        self.assertEqual(report["status"], "ALL_FINITE_OBSERVATION_CELLS_AUTHORIZED")
        self.assertEqual(report["certified_minimax_cost"], "2")
        self.assertEqual(report["observed_worst_case_cost"], "2")
        self.assertGreaterEqual(report["traces"], 4)

    def test_prohibited_tap_refuses(self):
        report = audit_minimax_blind_replay(
            worlds=self.worlds, authority=self.authority,
            probes=self.probes, risk_cap=F(0),
        )
        self.assertEqual(report["status"], "UNIDENTIFIABLE_REFUSAL")
        self.assertEqual(report["traces"], 0)

    def test_overlapping_noise_refuses(self):
        probes = tuple(Probe(p.name, p.center_by_world, F(3,2), p.cost, p.risk)
                       for p in self.probes)
        result = audit_minimax_blind_replay(
            worlds=self.worlds, authority=self.authority, probes=probes,
        )
        self.assertEqual(result["status"], "UNIDENTIFIABLE_REFUSAL")

    def test_same_authority_zero_probes(self):
        report = audit_minimax_blind_replay(
            worlds=("x", "y"), authority={"x": "same", "y": "same"},
            probes=(),
        )
        self.assertEqual(report["certified_minimax_cost"], "0")
        self.assertEqual(report["traces"], 2)

    def test_budget_abort_never_partial_success(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete coverage"):
            audit_minimax_blind_replay(
                worlds=self.worlds, authority=self.authority, probes=self.probes,
                maximum_traces=1,
            )


if __name__ == "__main__":
    unittest.main()
