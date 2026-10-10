import unittest
from semantic_interventions_v06 import mutate,valid_single_edit

class InterventionTests(unittest.TestCase):
 def test_dtype_control(self):
  source="def f(x):\n return torch.tensor([0.,1.], device=x.device,dtype=x.dtype)\n"
  out,record=mutate(source,"f","drop_tensor_factory_dtype")
  self.assertNotIn("dtype=x.dtype",out)
  self.assertEqual(record["n_operations"],1)
  self.assertTrue(valid_single_edit(source,out))
 def test_dtype_does_not_remove_device(self):
  source="def f(x):\n return torch.tensor([0.,1.], device=x.device,dtype=x.dtype)\n"
  out,_=mutate(source,"f","drop_tensor_factory_dtype")
  self.assertIn("device=x.device",out)
 def test_clone_removed(self):
  source="def f(x):\n y=x.clone()\n return y\n"
  out,_=mutate(source,"f","drop_clone")
  self.assertNotIn(".clone()",out)
 def test_parameter_requires_grad(self):
  source="def f(t):\n return nn.Parameter(t,requires_grad=t.requires_grad)\n"
  out,_=mutate(source,"f","drop_parameter_grad")
  self.assertNotIn("requires_grad=",out)
 def test_buffer_demotion(self):
  source="class M(nn.Module):\n def __init__(self,x):\n  self.register_buffer('weight',x)\n"
  out,_=mutate(source,"M.__init__","demote_buffer_registration")
  self.assertIn("self.weight = x",out)
 def test_no_matching_opcode_refuses(self):
  source="def f(x):\n return x+1\n"
  with self.assertRaises(ValueError):mutate(source,"f","drop_clone")
 def test_unknown_opcode_refuses(self):
  with self.assertRaises(ValueError):mutate("def f():pass","f","rewrite_entire_repo")
 def test_ambiguous_mutations_refuse(self):
  source="def f(x):\n a=x.clone()\n b=x.clone()\n return a+b\n"
  with self.assertRaises(ValueError):mutate(source,"f","drop_clone")
 def test_one_function_only(self):
  source="def f(x):\n return x.clone()\ndef g(x):\n return x.clone()\n"
  out,_=mutate(source,"f","drop_clone")
  self.assertIn("def g(x):\n    return x.clone()",out)
 def test_missing_class_method_refuses(self):
  with self.assertRaises(ValueError):mutate("def f(x):return x.clone()","Q.__init__","drop_clone")
 def test_manifest_sha_lengths(self):
  before="def f(x):\n return x.clone()\n"
  _,m=mutate(before,"f","drop_clone")
  self.assertEqual(len(m["input_sha256"]),64)
  self.assertEqual(len(m["output_sha256"]),64)
  self.assertEqual(m["kind"],"AUTHOR_SYNTHESIZED_SINGLE_AST_INTERVENTION")
 def test_no_plain_attribute_demoted(self):
  source="class M:\n def __init__(self,x):\n  self.state=x\n"
  with self.assertRaises(ValueError):mutate(source,"M.__init__","demote_buffer_registration")
 def test_nonliteral_buffer_refuses(self):
  source="class M:\n def __init__(self,x,k):\n  self.register_buffer(k,x)\n"
  with self.assertRaises(ValueError):mutate(source,"M.__init__","demote_buffer_registration")
 def test_nonparameter_requires_grad_retained(self):
  source="def f(x):\n return torch.tensor(x,requires_grad=True)\n"
  with self.assertRaises(ValueError):mutate(source,"f","drop_parameter_grad")
 def test_eye_dtype_counterfactual(self):
  src="def forward(self,x):\n return torch.eye(x.size(0),dtype=x.dtype,device=x.device)\n"
  mutated,report=mutate(src,"forward","drop_tensor_factory_dtype")
  self.assertNotIn("dtype=x.dtype",mutated)
  self.assertIn("device=x.device",mutated)
  self.assertEqual(report["operator"],"drop_tensor_factory_dtype")
if __name__=="__main__":unittest.main()
