import unittest
from typed_effects_v02 import classified_patch,PROBES

class TypedNativeEffects(unittest.TestCase):
 def pick(self,minus,plus):
  return classified_patch([{"filename":"src/module.py",
                            "patch":"@@ -1 +1 @@\n-"+minus+"\n+"+plus}])
 def test_buffer_registration(self):
  self.assertIn("registration_buffer",self.pick('self._data = data','self.register_buffer("_data", data)'))
 def test_requires_grad(self):
  self.assertIn("parameter_grad",self.pick('self.register_parameter(name, nn.Parameter(tensor))',
                                         'self.register_parameter(name, nn.Parameter(tensor, requires_grad=requires_grad))'))
 def test_state_serialization(self):
  self.assertIn("state_serialization",self.pick('m.named_parameters()','m.state_dict()'))
 def test_axis_broadcast(self):
  self.assertIn("axis_broadcast",self.pick('return self._z','return self._z.squeeze(-1) if self._z.dim()==2 else self._z'))
 def test_non_python_abstains(self):
  self.assertFalse(classified_patch([{"filename":"src/not_python.cpp","patch":"@@ -1 +1 @@\n-self._data=data\n+self.register_buffer('x',data)"}]))
 def test_only_tests_abstain(self):
  self.assertFalse(classified_patch([{"filename":"tests/test.py","patch":"@@ -1 +1 @@\n-self._data=data\n+self.register_buffer('x',data)"}]))
 def test_four_native_semantic_effects(self):
  expected={"registration_buffer":"state_key_added","parameter_grad":"delta",
            "state_serialization":"delta_includes_buffer","axis_broadcast":"delta"}
  for kind,k in expected.items():
   with self.subTest(kind=kind):self.assertTrue(PROBES[kind]()[k])
if __name__=="__main__":unittest.main()
