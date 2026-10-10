import unittest
from source_effect_path import ownership_route

W = """
class Weave(nn.Module):
 def __init__(self):
  self.layers: nn.ModuleList = nn.ModuleList()
  weave_layer = torch_layers.WeaveLayer()
  self.layers.append(weave_layer)
class WeaveModel(TorchModel):
 def __init__(self):
  self.model = Weave()
  super(WeaveModel,self).__init__(self.model)
"""
T = """
class TorchModel(Model):
 def _ensure_built(self):
  self._pytorch_optimizer = self.optimizer._create_pytorch_optimizer(self.model.parameters())
"""
class PathTests(unittest.TestCase):
 def test_valid(self):
  self.assertEqual(ownership_route(W,T)["status"],"STATIC_OWNERSHIP_ROUTE")
 def test_missing_super(self):
  self.assertIn("NO_MODEL_TO_PARENT_INIT",ownership_route(W.replace("super(WeaveModel,self).__init__(self.model)","pass"),T)["missing"])
 def test_wrong_inheritance(self):
  self.assertIn("NO_INHERITANCE",ownership_route(W.replace("WeaveModel(TorchModel)","WeaveModel(object)"),T)["missing"])
 def test_unregistered_module_container(self):
  self.assertIn("NO_MODULELIST_OWNER",ownership_route(W.replace("nn.ModuleList()","[]"),T)["missing"])
 def test_missing_append(self):
  self.assertIn("NO_WEAVELAYER_APPEND",ownership_route(W.replace("self.layers.append(weave_layer)","self.other.append(weave_layer)"),T)["missing"])
 def test_different_optimizer_route(self):
  self.assertIn("NO_ACTUAL_OPTIMIZER_CALL",ownership_route(W,T.replace("self.model.parameters()","[]"))["missing"])
if __name__=="__main__":unittest.main()
