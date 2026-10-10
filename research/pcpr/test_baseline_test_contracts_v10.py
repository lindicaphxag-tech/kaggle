import unittest
from baseline_test_contracts_v10 import scan

BASE="""import torch
class TestThing:
 def test_shape(self,device,dtype,shape):
  img=torch.rand(shape,device=device,dtype=dtype)
  out=kornia.enhance.adjust_gamma(img,1.)
  assert out.shape==shape
 def test_dtype(self):
  x=torch.ones(2)
  result=toolkit.layer(x)
  assert result.dtype==x.dtype
 def test_value(self):
  image=torch.ones(2)
  out=toolkit.layer(image)
  self.assert_close(out,image)
"""
PATCH=BASE+"""
class TestNewClaim:
 def test_fabricated_new(self):
  out=kornia.enhance.adjust_gamma(torch.ones(2),1.)
  assert out.dtype==torch.float16
"""
class BeforeOnlyTests(unittest.TestCase):
 def test_preexisting_shape(self):
  result=scan(BASE,PATCH,"tests/test_thing.py")
  c=[c for c in result["contracts"] if c["source_test"].endswith("test_shape")]
  self.assertEqual(len(c),1)
  self.assertEqual(c[0]["property"],"shape")
  self.assertEqual(c[0]["target_api"],"kornia.enhance.adjust_gamma")
 def test_dtype_and_value(self):
  result=scan(BASE,PATCH,"tests/test_thing.py")
  self.assertEqual({x["property"] for x in result["contracts"]},{"shape","dtype","value_equality"})
 def test_new_patch_test_not_a_root(self):
  result=scan(BASE,PATCH,"tests/test_thing.py")
  self.assertTrue(all("fabricated" not in x["source_test"] for x in result["contracts"]))
 def test_modified_test_cannot_self_certify(self):
  after=BASE.replace("assert out.shape==shape","assert out.shape==torch.Size([2])")
  result=scan(BASE,after,"tests/test_thing.py")
  self.assertEqual(result["n_removed_or_changed_tests"],1)
  self.assertFalse(any(x["property"]=="shape" for x in result["contracts"]))
 def test_reversible_assert(self):
  b="def test_example():\n out=lib.f(x)\n assert x.dtype==out.dtype\n"
  a=scan(b,b,"tests/test_x.py")
  self.assertEqual(a["contracts"][0]["property"],"dtype")
 def test_test_name_replacement_is_not_unchanged(self):
  after=BASE.replace("test_shape","test_different")
  result=scan(BASE,after,"tests/test_thing.py")
  self.assertFalse(any("test_shape" in z["source_test"] for z in result["contracts"]))
 def test_assert_close_not_misnamed_dtype(self):
  result=scan(BASE,PATCH,"tests/test_thing.py")
  self.assertEqual([x["relation"] for x in result["contracts"] if x["property"]=="value_equality"],["VALUE_EQ"])
 def test_approvals_always_false(self):
  x=scan(BASE,PATCH,"tests/test_thing.py")
  self.assertEqual(x["execution_claims"],0)
  self.assertTrue(all(not q["whole_patch_correctness"] for q in x["contracts"]))
  self.assertTrue(all(not q["expected_value_validated"] for q in x["contracts"]))
 def test_test_digest(self):
  c=scan(BASE,PATCH,"tests/test_thing.py")["contracts"][0]
  self.assertEqual(len(c["test_ast_sha256"]),64)
  self.assertNotEqual(c["before_full_test_file_sha256"],c["after_full_test_file_sha256"])
 def test_same_baseline_source_allowed(self):
  self.assertEqual(scan(BASE,BASE,"tests/test_thing.py")["n_unchanged_preexisting_tests"],3)
 def test_malformed_test_fail_closed(self):
  self.assertEqual(scan("def o(:",BASE,"x.py")["status"],"ABSTAIN_UNPARSEABLE")
 def test_api_prefix_filter(self):
  o=scan(BASE,BASE,"x.py",api_prefix="kornia.enhance")
  self.assertEqual(o["n_contracts"],1)
 def test_different_api_not_hallucinated(self):
  o=scan(BASE,BASE,"x.py",api_prefix="nonexistent")
  self.assertEqual(o["n_contracts"],0)
 def test_simple_direct_calls_are_not_exec(self):
  o=scan(BASE,BASE,"x.py")
  self.assertEqual(o["not_supported"],"NO_DYNAMIC_FIXTURE_ALIASES_NO_TEST_EXECUTION")
 def test_stricter_than_v05_syntax_hints(self):
  o=scan("def test_foo():\n assert torch.ones(2).dtype==torch.float32\n",
         "def test_foo():\n assert torch.ones(2).dtype==torch.float32\n","x.py")
  self.assertEqual(o["n_contracts"],0)
 def test_no_opaque_assertion(self):
  o=scan("def test_foo():\n assert api.f(x)\n",
         "def test_foo():\n assert api.f(x)\n","x.py")
  self.assertEqual(o["n_contracts"],0)

if __name__=="__main__":unittest.main()
