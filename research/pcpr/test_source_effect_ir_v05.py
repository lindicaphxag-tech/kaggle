import unittest
from source_effect_ir_v05 import scan_pair,tokenize,code_units,FACETS

BASE="""import torch
class Net:
 def forward(self,x):
  y = torch.zeros_like(x)
  return x + y
"""
AFTER="""import torch
class Net:
 def forward(self,x):
  y = torch.zeros_like(x,dtype=x.dtype)
  return x + y
"""
class SourceEffectIRTests(unittest.TestCase):
 def test_changes_discovered_from_complete_function(self):
  h,status=scan_pair("src/net.py",BASE,AFTER)
  self.assertEqual(status,"PARSED")
  self.assertEqual(len(h),1)
  self.assertEqual(h[0]["symbol"],"Net.forward")
 def test_dtype_family_only_hypothesis(self):
  h,_=scan_pair("f.py",BASE,AFTER)
  self.assertIn("precision",h[0]["facets"])
  self.assertEqual(h[0]["status"],"TYPED_EFFECT_HYPOTHESIS_NOT_VERIFIED")
 def test_before_function_argument_scope(self):
  h,_=scan_pair("f.py",BASE,AFTER)
  self.assertEqual(h[0]["before_public_arguments"],["x"])
  self.assertTrue(h[0]["before_has_return"])
 def test_no_claim_of_patch_correctness(self):
  h,_=scan_pair("f.py",BASE,AFTER)
  self.assertNotIn("patch_correct",h[0])
 def test_no_change_no_hypothesis(self):
  h,_=scan_pair("f.py",BASE,BASE)
  self.assertFalse(h)
 def test_unparsable_abstain(self):
  h,status=scan_pair("f.py","x = )",AFTER)
  self.assertEqual(status,"ABSTAIN_UNPARSEABLE")
  self.assertFalse(h)
 def test_lora_state_changes(self):
  a="""def export(model):\n return list(model.named_parameters())\n"""
  b="""def export(model):\n return model.state_dict()\n"""
  h,_=scan_pair("convert.py",a,b)
  self.assertIn("state",h[0]["facets"])
  self.assertIn("model.state_dict",h[0]["new_calls"])
  self.assertIn("model.named_parameters",h[0]["dropped_calls"])
 def test_gradient(self):
  a="def wrap(t):\n return nn.Parameter(t)\n"
  b="def wrap(t):\n return nn.Parameter(t, requires_grad=t.requires_grad)\n"
  h,_=scan_pair("m.py",a,b)
  self.assertIn("gradient",h[0]["facets"])
 def test_masking(self):
  a="def f(logits, lengths):\n return logits\n"
  b="def f(logits, lengths):\n return torch.where(lengths > 0, logits, 0)\n"
  h,_=scan_pair("m.py",a,b)
  self.assertIn("masking",h[0]["facets"])
 def test_finite_numeric(self):
  a="def f(x):\n return x\n"
  b="def f(x):\n return torch.nan_to_num(x)\n"
  h,_=scan_pair("m.py",a,b)
  self.assertIn("numerical_domain",h[0]["facets"])
 def test_shape(self):
  a="def f(x):\n return x\n"
  b="def f(x):\n return x.squeeze(-1)\n"
  h,_=scan_pair("m.py",a,b)
  self.assertIn("layout",h[0]["facets"])
 def test_optimizer(self):
  a="def f(opt):\n return opt\n"
  b="def f(opt):\n return opt.state\n"
  h,_=scan_pair("m.py",a,b)
  self.assertIn("optimizer",h[0]["facets"])
 def test_new_function_does_not_claim_old_anchor(self):
  h,_=scan_pair("new.py","def foo(x):return x\n","def foo(x):return x\ndef new(x):return x.detach()\n")
  self.assertEqual(h,[])
 def test_no_shallow_change_claims_for_decorator(self):
  a="@torch.no_grad()\ndef f(x):\n return x\n"
  b="def f(x):\n return x\n"
  h,_=scan_pair("m.py",a,b)
  self.assertTrue(h)
  self.assertIn("gradient",h[0]["facets"])
 def test_all_facets_named(self):
  self.assertEqual(len(FACETS),9)
 def test_bound_real_code_context(self):
  h,_=scan_pair("x.py",BASE,AFTER)
  self.assertEqual(len(h[0]["source_before_sha256"]),64)
  self.assertEqual(len(h[0]["source_after_sha256"]),64)
 def test_stable_facet_order(self):
  h,_=scan_pair("f.py",BASE,AFTER)
  self.assertEqual(h[0]["facets"],sorted(h[0]["facets"]))
if __name__=="__main__":unittest.main()
