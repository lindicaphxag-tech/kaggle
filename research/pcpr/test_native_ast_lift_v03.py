"""Unit tests of source-exact AST lifting, including deliberate fault controls.

All fixture snippets are synthetic. Real GitHub SHA-pinned before/after runs
are a separate CI step with their own machine-readable evidence.
"""
import unittest
import torch
from native_ast_lift_v03 import lift_function,lift_constructor,blob_id

OLD="""from typing import Union
class Quaternion(nn.Module):
 def __init__(self, data: Union[torch.Tensor, nn.Parameter]) -> None:
  super().__init__()
  if data.shape[-1] != 4: raise ValueError("shape")
  self._data = data
"""
NEW="""from typing import Union
class Quaternion(nn.Module):
 def __init__(self, data: Union[torch.Tensor, nn.Parameter]) -> None:
  super().__init__()
  if data.shape[-1] != 4: raise ValueError("shape")
  if isinstance(data,nn.Parameter):
   self._data=data
  else:
   self.register_buffer("_data",data)
"""

class NativeLiftTests(unittest.TestCase):
 def test_persistent_buffer_after(self):
  c=lift_constructor(NEW)
  obj=c(torch.ones(4))
  self.assertIn("_data",obj.state_dict())
  self.assertIn("_data",dict(obj.named_buffers()))
 def test_no_buffer_before(self):
  c=lift_constructor(OLD)
  self.assertNotIn("_data",c(torch.ones(4)).state_dict())
 def test_counterfeit_fails_persistence(self):
  c=lift_constructor(NEW,counterfeit=True)
  self.assertNotIn("_data",c(torch.ones(4)).state_dict())
 def test_parameter_keeps_optimizer_identity(self):
  c=lift_constructor(NEW)
  p=torch.nn.Parameter(torch.ones(4))
  self.assertIs(c(p)._data,p)
 def test_real_parent_cast(self):
  cls=lift_constructor(NEW)
  x=torch.nn.Module();x.child=cls(torch.ones(4))
  x.to(torch.float64)
  self.assertEqual(x.child._data.dtype,torch.float64)
 def test_cannot_forge_without_buffer(self):
  with self.assertRaises(ValueError):
   lift_constructor(OLD,counterfeit=True)
 def test_exact_original_function(self):
  fn=lift_function("def original(x: torch.Tensor)->torch.Tensor:\n return torch.where(x>0,x,torch.zeros_like(x))","original")
  self.assertTrue(torch.equal(fn(torch.tensor([-1.,2.])),torch.tensor([0.,2.])))
 def test_reject_function_with_decorator(self):
  with self.assertRaises(ValueError):
   lift_function("@cache\ndef foo(x):return x","foo")
 def test_blob_id_is_git_style(self):
  import hashlib
  payload=b"x\n"
  self.assertEqual(blob_id(payload),hashlib.sha1(b"blob 2\x00"+payload).hexdigest())
 def test_shape_contract_is_actual_constructor(self):
  with self.assertRaises(ValueError):
   lift_constructor(NEW)(torch.ones(3))
 def test_reject_import_side_effect(self):
  with self.assertRaises(ValueError):
   lift_function("def dangerous():\n import os\n return os.getcwd()","dangerous")
 def test_reject_dynamic_open(self):
  with self.assertRaises(ValueError):
   lift_function("def dangerous():\n return open('/tmp/foo','w')","dangerous")
 def test_reject_class_introspection(self):
  with self.assertRaises(ValueError):
   lift_function("def dangerous(x):\n return x.__class__.__mro__","dangerous")
 def test_reject_eval(self):
  with self.assertRaises(ValueError):
   lift_function("def dangerous(x):\n return eval(x)","dangerous")
 def test_reject_global_statement(self):
  with self.assertRaises(ValueError):
   lift_function("def dangerous():\n global payload\n return 3","dangerous")
if __name__=="__main__":unittest.main()
