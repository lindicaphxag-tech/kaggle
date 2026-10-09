import unittest
from fractions import Fraction as F
from research.semrepair.minimal_trusted_evidence import (
    EvidenceField,exact_minimum_provenance,
    finite_decision_region_theorem,native_mujoco_source_certificate,
)

class MinimumEvidence(unittest.TestCase):
    def test_actual_mujoco_issuer_provenance_is_necessary(self):
        r=native_mujoco_source_certificate()
        self.assertFalse(r["sensor_only"]["sensor_only_total_correct_authority_possible"])
        self.assertEqual(r["minimum_authority_evidence"]["cost_exact"],"1")
        self.assertEqual(r["full_identity_evidence"]["cost_exact"],"2")

    def test_untrusted_self_report_does_not_resolve_conflict(self):
        a={"x":"gear-fix","y":"different-code-owner"}
        fields=(EvidenceField("untrusted_claim",F(0),{"x":"x","y":"y"},False),)
        z=exact_minimum_provenance(
            repair_by_world=a,aliased_pairs=[("x","y")],fields=fields)
        self.assertFalse(z["feasible"])
        self.assertEqual(len(z["uncovered_conflicting_pairs"]),1)

    def test_complementary_fields_exact_weighted_cover(self):
        arms={"a":"A","b":"B","c":"A","d":"B"}
        pairs=[("a","b"),("c","d")]
        f1=EvidenceField("left",F(1),dict(a="0",b="1",c="0",d="0"))
        f2=EvidenceField("right",F(1),dict(a="0",b="0",c="0",d="1"))
        f3=EvidenceField("expensive",F(5),dict(a="a",b="b",c="c",d="d"))
        z=exact_minimum_provenance(
            repair_by_world=arms,aliased_pairs=pairs,fields=(f1,f2,f3))
        self.assertEqual(z["selected_fields"],["left","right"])
        self.assertEqual(z["cost_exact"],"2")

    def test_same_repair_needs_no_identity(self):
        r=exact_minimum_provenance(
            repair_by_world={"a":"same","b":"same"},
            aliased_pairs=[("a","b")],fields=())
        self.assertTrue(r["feasible"])
        self.assertEqual(r["cost_exact"],"0")

    def test_uncovered_conflict_is_explicit_impossibility(self):
        z=exact_minimum_provenance(
            repair_by_world={"a":"A","b":"B"},
            aliased_pairs=[("a","b")],
            fields=(EvidenceField("constant",F(2),{"a":0,"b":0}),))
        self.assertFalse(z["feasible"])
        self.assertEqual(z["uncovered_conflicting_pairs"],[["a","b"]])


if __name__=="__main__":
    unittest.main()
