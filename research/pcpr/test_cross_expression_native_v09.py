import unittest
from cross_expression_native_v09 import candidate_pairs,score_file
BASE="""import torch
def f(g,t):
 return g.reshape(t.shape)
"""
FIX=BASE.replace("g.reshape(t.shape)","g.reshape(t.shape).to(dtype=t.dtype)")
TB="""import torch
def jac_to_grad(weights,jacobian):
 return torch.tensordot(weights,jacobian,dims=1)
"""
TA=TB.replace("torch.tensordot(weights,jacobian,dims=1)",
              "torch.tensordot(weights.to(dtype=jacobian.dtype),jacobian,dims=1)")

class CrossExpressionNativeV09Tests(unittest.TestCase):
 def test_output_cast_candidate(self):
  a,s=candidate_pairs("grad.py",BASE,FIX)
  self.assertEqual(s,"PARSED")
  self.assertEqual(len(a),1)
  self.assertEqual(a[0]["pattern"],"OUTPUT_CAST")
 def test_output_cast_native_dtype(self):
  v=score_file("grad.py",BASE,FIX)
  self.assertEqual(v["n_native_contexts"],3)
  self.assertEqual(v["n_dtype_reference_gains"],2)
  self.assertEqual(v["n_preserving_controls"],1)
 def test_gradient_kernel_cast_candidate(self):
  a,s=candidate_pairs("jac.py",TB,TA)
  self.assertEqual(s,"PARSED")
  self.assertEqual(len(a),1)
  self.assertEqual(a[0]["pattern"],"ARGUMENT_CAST")
 def test_gradient_cast_recovers_or_abstains(self):
  a=score_file("jac.py",TB,TA)
  self.assertEqual(a["n_native_contexts"],3)
  self.assertGreaterEqual(a["n_recovered_execution"],1)
 def test_same_source_no_candidate(self):
  self.assertEqual(score_file("jac.py",TB,TB)["n_candidates"],0)
 def test_changed_kernel_not_a_false_match(self):
  x=TA.replace("torch.tensordot(","torch.mm(")
  self.assertEqual(score_file("jac.py",TB,x)["n_candidates"],0)
 def test_after_intent_not_truth(self):
  x=score_file("jac.py",TB,TA)
  self.assertFalse(x["whole_patch_authorized"])
  self.assertEqual(x["n_native_contexts"],3)
 def test_reject_unapproved_inner_method(self):
  b="def f(g,t):return g.erase_all(t.shape)"
  a="def f(g,t):return g.erase_all(t.shape).to(dtype=t.dtype)"
  self.assertEqual(score_file("evil.py",b,a)["n_native_contexts"],0)
 def test_excess_python_source_no_execution(self):
  b="def f(g,t):return eval(g.reshape(t.shape))"
  a=b.replace("g.reshape(t.shape)","g.reshape(t.shape).to(dtype=t.dtype)")
  v=score_file("x.py",b,a)
  self.assertGreaterEqual(v["n_candidates"],1)
  self.assertEqual(v["n_native_contexts"],3) # isolated AST no eval() sourced outside
 def test_non_tensor_reference_refused(self):
  x=FIX.replace("dtype=t.dtype","dtype=torch.float16")
  self.assertEqual(score_file("x.py",BASE,x)["n_candidates"],0)
 def test_dtype_source_exact(self):
  v=score_file("x.py",BASE,FIX)
  x=v["observations"][0]
  self.assertEqual(len(x["original_before_sha256"]),64)
  self.assertEqual(len(x["original_after_sha256"]),64)
 def test_changed_noncast_argument_not_match(self):
  a=TA.replace("dims=1","dims=2")
  self.assertEqual(score_file("x.py",TB,a)["n_candidates"],0)
 def test_multiple_casts_fail_closed(self):
  a=TB.replace("weights,jacobian","weights.to(dtype=jacobian.dtype),jacobian.to(dtype=weights.dtype)")
  self.assertEqual(score_file("x.py",TB,a)["n_candidates"],0)
 def test_malformed_source_abstain(self):
  self.assertEqual(score_file("x.py","def f(:",FIX)["status"],"ABSTAIN_PARSE")
 def test_nested_wrapper_match_code(self):
  x=score_file("x.py",BASE,FIX)
  self.assertEqual(x["observations"][0]["pattern"],"OUTPUT_CAST")
if __name__=="__main__":unittest.main()
