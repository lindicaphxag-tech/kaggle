import unittest
from symbolic_native_operator_v08 import candidate_pairs,score_file

B="""import torch
class M(torch.nn.Module):
 def forward(self,x):
  T=x.size(0)
  weight=torch.zeros([T,T],device=self.weight.device)
  return x @ weight
"""
A=B.replace("torch.zeros([T,T],device=self.weight.device)",
           "torch.zeros([T,T],dtype=self.weight.dtype,device=self.weight.device)")

class SymbolicNativeV08Tests(unittest.TestCase):
 def test_module_weight_root_detected(self):
  a,s=candidate_pairs("m.py",B,A)
  self.assertEqual(s,"PARSED")
  self.assertEqual(len(a),1)
  self.assertEqual(a[0]["tensor_ref"],("module_field","weight"))
 def test_module_weight_native_dtype_gain(self):
  r=score_file("m.py",B,A)
  self.assertEqual(r["n_native_contexts"],4)
  self.assertEqual(r["n_dtype_gain_contexts"],2)
  self.assertEqual(r["n_preserving_contexts"],2)
 def test_module_different_attribute_not_overpromoted(self):
  a=A.replace("self.weight.device","self.another.device")
  r=score_file("m.py",B,a)
  self.assertEqual(r["n_native_contexts"],0)
 def test_module_unknown_field_refuses(self):
  a=A.replace("self.weight.dtype","self.magic.dtype")
  r=score_file("m.py",B,a)
  self.assertEqual(r["n_candidates"],0)
 def test_parameter_root_compatible(self):
  b="def f(x):return torch.eye(x.size(0),device=x.device)"
  a="def f(x):return torch.eye(x.size(0),dtype=x.dtype,device=x.device)"
  r=score_file("m.py",b,a)
  self.assertEqual(r["n_native_contexts"],4)
  self.assertEqual(r["n_dtype_gain_contexts"],2)
 def test_no_dtype_change_not_a_candidate(self):
  self.assertEqual(score_file("m.py",B,B)["n_candidates"],0)
 def test_non_matching_factory_args_refuses(self):
  a=A.replace("[T,T]","[T,T+1]")
  self.assertEqual(score_file("m.py",B,a)["n_candidates"],0)
 def test_forged_input_is_not_specification(self):
  o=score_file("m.py",B,A)["observations"][0]
  self.assertFalse(o["independent_requirement_gold"])
  self.assertFalse(o["whole_patch_authorized"])
 def test_unknown_tensor_dim_does_not_execute(self):
  b="def f(x):return torch.zeros((external_dim,4))"
  a="def f(x):return torch.zeros((external_dim,4),dtype=x.dtype)"
  r=score_file("m.py",b,a)
  self.assertEqual(r["n_native_contexts"],0)
 def test_unsafe_call_refuses(self):
  b="def f(x):return torch.zeros(eval('10'))"
  a="def f(x):return torch.zeros(eval('10'),dtype=x.dtype)"
  r=score_file("m.py",b,a)
  self.assertEqual(r["n_native_contexts"],0)
 def test_alias_ambiguity_refused(self):
  b="def f(x):return torch.eye(4)"
  a="def f(x):return torch.eye(4,dtype=self.other.dtype)"
  self.assertEqual(score_file("m.py",b,a)["n_candidates"],0)
 def test_unchanged_device_keyword_preserved(self):
  r=score_file("m.py",B,A)
  o=r["observations"][0]
  self.assertIn("device=self.weight.device",o["before_expression"])
  self.assertIn("device=self.weight.device",o["after_expression"])
 def test_malformed_source_abstains(self):
  self.assertEqual(score_file("m.py","class M(:",A)["status"],"ABSTAIN_PARSE")
 def test_source_digest(self):
  r=score_file("m.py",B,A)
  o=r["observations"][0]
  self.assertEqual(len(o["before_sha256"]),64)
  self.assertEqual(len(o["after_sha256"]),64)

if __name__=="__main__":unittest.main()
