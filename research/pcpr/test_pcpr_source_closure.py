"""Isolated synthetic regression tests; NOT independent held-out PR outcomes."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pcpr_source_closure import discover_fields,registration_runtime,assess_pair

BEFORE = """
import torch
from torch import nn
class WeaveLayer(nn.Module):
    def __init__(self):
        self.W_AA: torch.Tensor = init(torch.empty(self.n_atom_input_feat, self.n_hidden_AA))
        self.b_AA: torch.Tensor = torch.zeros((self.n_hidden_AA,))
    def forward(self, x):
        return x @ self.W_AA + self.b_AA
"""
AFTER = """
import torch
from torch import nn
class WeaveLayer(nn.Module):
    def __init__(self):
        self.W_AA: nn.Parameter = nn.Parameter(init(torch.empty(self.n_atom_input_feat, self.n_hidden_AA)))
        self.b_AA: nn.Parameter = nn.Parameter(torch.zeros((self.n_hidden_AA,)))
    def forward(self, x):
        return x @ self.W_AA + self.b_AA
"""
CONSUMER = "def create_optimizer(self): return torch.optim.Adam(self.model.parameters())"


class ScopeTests(unittest.TestCase):
    def test_source_read_discovery(self):
        self.assertEqual(set(discover_fields(BEFORE,"WeaveLayer")),{"W_AA","b_AA"})
    def test_original_unregistered(self):
        self.assertFalse(registration_runtime(discover_fields(BEFORE,"WeaveLayer")["W_AA"].rhs)["registered"])
    def test_new_registered_and_gradient(self):
        r=registration_runtime(discover_fields(AFTER,"WeaveLayer")["W_AA"].rhs)
        self.assertTrue(r["registered"] and r["parameter_gradient"])
    def test_bounded_positive(self):
        r=assess_pair(BEFORE,AFTER,CONSUMER,"WeaveLayer")
        self.assertEqual([x["status"] for x in r["fields"]],["BOUNDED_REGISTRATION_WITNESS"]*2)
    def test_missing_unchanged_consumer_refusal(self):
        self.assertEqual(assess_pair(BEFORE,AFTER,"def foo(): pass","WeaveLayer")["status"],
                         "ABSTAIN_NO_SOURCE_INDEPENDENT_OPTIMIZER_CONSUMER")
    def test_requires_grad_alone_no_registration(self):
        self.assertFalse(registration_runtime("torch.zeros((self.n_hidden_AA,))")["registered"])
    def test_partially_repaired_counterfeit_refuses(self):
        counterfeit=AFTER.replace("nn.Parameter(init(torch.empty(self.n_atom_input_feat, self.n_hidden_AA)))",
                                  "init(torch.empty(self.n_atom_input_feat, self.n_hidden_AA))")
        r=assess_pair(BEFORE,counterfeit,CONSUMER,"WeaveLayer")
        self.assertEqual({x["field"]:x["status"] for x in r["fields"]}["W_AA"],
                         "ABSTAIN_NO_REGISTRATION_GAIN")
    def test_sha_refuses_tampering(self):
        with self.assertRaises(ValueError):
            assess_pair(BEFORE,AFTER,CONSUMER,"WeaveLayer",expected_before_sha="bad")
    def test_unknown_function_refuses(self):
        with self.assertRaises(ValueError):
            registration_runtime("evil(torch.empty((3,3)))")
    def test_import_escape_refuses(self):
        with self.assertRaises(ValueError):
            registration_runtime("__import__('os')")
    def test_unread_field_excluded(self):
        s=BEFORE.replace("self.W_AA + self.b_AA","self.W_AA")
        self.assertNotIn("b_AA",discover_fields(s,"WeaveLayer"))
    def test_already_registered_not_new_witness(self):
        r=assess_pair(AFTER,AFTER,CONSUMER,"WeaveLayer")
        self.assertEqual([x["status"] for x in r["fields"]],["ALREADY_REGISTERED"]*2)


if __name__=="__main__":unittest.main()
