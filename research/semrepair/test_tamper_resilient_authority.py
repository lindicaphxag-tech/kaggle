"""Independent enumerative and mutation tests for tamper-resilient authority."""
import itertools
import unittest
from fractions import Fraction as F

from research.semrepair.tamper_resilient_authority import (
    Channel, authority_distance, exact_robust_channel_design,
    selective_authorize, exact_binary_flip_assay,
)


class AuthorityByzantineEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.authority = {"a": "A", "b": "B"}
        self.c = tuple(Channel(f"s{i}", F(1), {"a": "a", "b": "b"}) for i in range(3))

    def test_three_witnesses_recover_one_adversarial_flip(self):
        r = exact_binary_flip_assay()
        self.assertEqual(r["robust_3_channel"]["correct"], 8)
        self.assertEqual(r["unsafe_1_channel_guess"]["wrong"], 2)
        self.assertEqual(r["selective_2_channel"]["refused"], 4)

    def test_distinct_repairs_need_three_for_one_corruption(self):
        r = exact_robust_channel_design(authority=self.authority, channels=self.c, corruption_budget=1)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["cost_exact"], "3")
        self.assertEqual(r["minimum_cross_authority_distance"], 3)
        self.assertEqual(r["selected_channels"], ["s0", "s1", "s2"])
        fail = exact_robust_channel_design(authority=self.authority, channels=self.c[:2], corruption_budget=1)
        self.assertFalse(fail["feasible"])
        self.assertEqual(fail["unresolvable_pairs"], [["a", "b"]])

    def test_same_authority_alias_is_not_an_obligation(self):
        authority = {"a1": "A", "a2": "A", "b": "B"}
        channels = (Channel("s1", F(1), {"a1": "same", "a2": "same", "b": "other"}),
                    Channel("s2", F(1), {"a1": "x", "a2": "y", "b": "z"}))
        self.assertEqual(authority_distance(authority, channels), {("a1", "b"):2, ("a2", "b"):2})
        r = exact_robust_channel_design(authority=authority, channels=channels, corruption_budget=0)
        self.assertEqual(r["cost_exact"], "1")
        self.assertEqual(r["selected_channels"], ["s1"])

    def test_exact_cost_and_ineligible_cheaper_channel(self):
        channels = (Channel("cheap-untrusted", F(0), {"a": "a", "b": "b"}, False),
                    Channel("costly", F(2,3), {"a": "a", "b": "b"}),
                    Channel("medium", F(3,4), {"a": "a", "b": "b"}))
        r = exact_robust_channel_design(authority=self.authority, channels=channels, corruption_budget=0)
        self.assertEqual(r["selected_channels"], ["costly"])
        self.assertEqual(r["cost_exact"], "2/3")
        self.assertRaises(ValueError, selective_authorize, observed={"cheap-untrusted": "a"},
                          authority=self.authority, channels=channels[:1], corruption_budget=0)

    def test_decoding_abstains_on_ambiguous_and_out_of_model(self):
        response = selective_authorize(observed={"s0":"a", "s1":"b"},
                                       authority=self.authority, channels=self.c[:2],corruption_budget=1)
        self.assertEqual(response["decision"], "REFUSE_AMBIGUOUS_AUTHORITY")
        impossible = selective_authorize(observed={"s0":"other"},
                                         authority=self.authority, channels=self.c[:1],corruption_budget=0)
        self.assertEqual(impossible["decision"], "REFUSE_OUTSIDE_ASSUMPTIONS")

    def test_exhaustive_hamming_separation_equivalence(self):
        # Independent brute-force oracle, 256 binary four-channel codeword pairs.
        for code_a in itertools.product("01", repeat=4):
            for code_b in itertools.product("01", repeat=4):
                distance = sum(x != y for x,y in zip(code_a,code_b))
                for f in (0,1):
                    disjoint = True
                    for observed in itertools.product("01", repeat=4):
                        da = sum(x!=y for x,y in zip(observed,code_a))
                        db = sum(x!=y for x,y in zip(observed,code_b))
                        if da<=f and db<=f:
                            disjoint=False
                            break
                    self.assertEqual(disjoint, distance >= 2*f+1)

    def test_corruption_cover_never_wrongly_authorizes(self):
        for true_world, symbol in (("a", "a"), ("b", "b")):
            for tampered in (None, 0, 1, 2):
                for wrong_symbol in ("a", "b", "outside"):
                    obs={c.name:(wrong_symbol if i==tampered else symbol)
                         for i,c in enumerate(self.c)}
                    answer=selective_authorize(observed=obs,authority=self.authority,
                                               channels=self.c,corruption_budget=1)
                    self.assertEqual(answer["decision"], "AUTHORIZE")
                    self.assertEqual(answer["repair"], self.authority[true_world])

    def test_missing_labels_and_invalid_costs_refuse(self):
        self.assertRaises(ValueError, exact_robust_channel_design,
                          authority=self.authority, channels=(Channel("m",F(1),{"a":"a"}),),
                          corruption_budget=0)
        self.assertRaises(ValueError, exact_robust_channel_design,
                          authority=self.authority, channels=self.c, corruption_budget=True)
        self.assertRaises(ValueError, exact_robust_channel_design,
                          authority=self.authority, channels=(Channel("bad", F(-1), {"a":0,"b":1}),),
                          corruption_budget=0)


if __name__=="__main__":
    unittest.main()
