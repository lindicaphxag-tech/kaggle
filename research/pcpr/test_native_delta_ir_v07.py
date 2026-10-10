import unittest
from native_delta_ir_v07 import candidate_pairs,score_file,check_expression,replay

BEFORE="""import torch
def f(c):
 return c-torch.eye(c.size(0),device=c.device)
"""
AFTER="""import torch
def f(c):
 return c-torch.eye(c.size(0),dtype=c.dtype,device=c.device)
"""

class SourceExactNativeDeltaTests(unittest.TestCase):
 def test_candidates_on_same_factory_call(self):
  a,status=candidate_pairs("lib/loss.py",BEFORE,AFTER)
  self.assertEqual(status,"PARSED")
  self.assertEqual(len(a),1)
  self.assertEqual(a[0]["factory"],"torch.eye")
  self.assertEqual(a[0]["input_root"],"c")
 def test_genuine_dtype_contexts(self):
  r=score_file("loss.py",BEFORE,AFTER)
  self.assertEqual(r["n_native_contexts"],3)
  self.assertEqual(r["n_positive_dtype_contexts"],2)
  self.assertEqual(r["native_results"][0]["n_preserving"],1)
 def test_operator_scope_cannot_certify(self):
  r=score_file("loss.py",BEFORE,AFTER)["native_results"][0]
  self.assertFalse(r["whole_patch_authorized"])
  self.assertFalse(r["independent_requirement_validated"])
  self.assertEqual(r["scope"],"NATIVE_PYTORCH_OPERATOR_ONLY")
 def test_same_before_after_zero(self):
  self.assertEqual(score_file("x.py",BEFORE,BEFORE)["candidate_count"],0)
 def test_unrelated_code_shift_prevents_matching(self):
  a=AFTER.replace("torch.eye(c.size(0),dtype=c.dtype,device=c.device)",
                  "torch.eye(c.size(0)+1,dtype=c.dtype,device=c.device)")
  self.assertEqual(score_file("x.py",BEFORE,a)["candidate_count"],0)
 def test_zero_without_one_dtype(self):
  a=AFTER.replace("dtype=c.dtype,","")
  self.assertEqual(score_file("x.py",BEFORE,a)["candidate_count"],0)
 def test_non_tensor_dtype_ref_not_promoted(self):
  b="def f(c):\n return torch.eye(c.size(0))\n"
  a="def f(c):\n return torch.eye(c.size(0),dtype=torch.float32)\n"
  self.assertEqual(score_file("x.py",b,a)["candidate_count"],0)
 def test_nested_unsafe_call_refuses(self):
  b="def f(c):\n return torch.eye(exec('2'))\n"
  a="def f(c):\n return torch.eye(exec('2'),dtype=c.dtype)\n"
  r=score_file("x.py",b,a)
  self.assertEqual(r["candidate_count"],1)
  self.assertEqual(r["n_native_contexts"],0)
 def test_kw_rejects_unsafe_out(self):
  b="def f(c):\n return torch.full((4,4),1,layout='x')\n"
  a="def f(c):\n return torch.full((4,4),1,layout='x',dtype=c.dtype)\n"
  r=score_file("x.py",b,a)
  self.assertEqual(r["n_native_contexts"],0)
 def test_parser_abstains(self):
  r=score_file("x.py","class X(:",AFTER)
  self.assertEqual(r["parse_status"],"ABSTAIN_PARSE")
 def test_one_root_only(self):
  b="def f(c,x):\n return torch.eye(c.size(0))\n"
  a="def f(c,x):\n return torch.eye(c.size(0),dtype=x.dtype)\n"
  r=score_file("x.py",b,a)
  self.assertEqual(r["candidate_count"],1)
  self.assertEqual(r["n_native_contexts"],0) # separate c unbound in x context
 def test_factory_whitelist(self):
  b="def f(x):return os.system('true')\n"
  a="def f(x):return os.system('true',dtype=x.dtype)\n"
  self.assertEqual(score_file("x.py",b,a)["candidate_count"],0)
 def test_full_precision_preserving(self):
  out=score_file("x.py",BEFORE,AFTER)["native_results"][0]
  self.assertEqual([z["dtype"] for z in out["contexts"]],["float16","bfloat16","float32"])
  self.assertTrue(out["contexts"][2]["preserved"])
 def test_digests(self):
  out=score_file("x.py",BEFORE,AFTER)["native_results"][0]
  self.assertEqual(len(out["source_sha256_before"]),64)
  self.assertEqual(len(out["source_sha256_after"]),64)
 def test_multiple_calls_deduplicated(self):
  b="def f(c):\n a=torch.eye(c.size(0))\n return a+a\n"
  a="def f(c):\n a=torch.eye(c.size(0),dtype=c.dtype)\n return a+a\n"
  self.assertEqual(score_file("x.py",b,a)["candidate_count"],1)
 def test_unsupported_code_stays_abstain(self):
  b="def f(c): return torch.eye(c.size(0),device=c.device,layout=torch.strided)\n"
  a="def f(c): return torch.eye(c.size(0),dtype=c.dtype,device=c.device,layout=torch.strided)\n"
  r=score_file("x.py",b,a)
  self.assertEqual(r["candidate_count"],1)
  self.assertEqual(r["n_native_contexts"],0)

if __name__=="__main__":unittest.main()
