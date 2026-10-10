import unittest
from unchanged_consumer_graph_v08 import dependency_graph,summarise_file

OLD="""import torch
def kernel(x):
 return torch.eye(x.size(0))
def run(x):
 return kernel(x).to(dtype=x.dtype)
def app(x):
 return run(x)
"""
AFTER=OLD.replace("torch.eye(x.size(0))","torch.eye(x.size(0),dtype=x.dtype)")
class ConsumerClosureTests(unittest.TestCase):
 def test_immediate_and_transitive_unchanged_callers(self):
  r=dependency_graph(OLD,AFTER)
  self.assertEqual(r["n_changed_existing_functions"],1)
  self.assertEqual(r["n_changed_producers_with_root"],1)
  self.assertEqual(r["n_syntactic_anchor_paths"],2)
  self.assertEqual({x["depth"] for x in r["roots"]},{1,2})
 def test_no_reference_patch_self_oracle(self):
  r=dependency_graph(OLD,AFTER)
  self.assertTrue(all(not x["semantic_requirement_adjudicated"] for x in r["roots"]))
  self.assertTrue(all(not x["whole_patch_authorized"] for x in r["roots"]))
 def test_changed_consumer_not_unchanged_root(self):
  b=AFTER.replace("return kernel(x).to(dtype=x.dtype)","return kernel(x)")
  r=dependency_graph(OLD,b)
  self.assertEqual(r["n_changed_existing_functions"],2)
  self.assertEqual(r["n_syntactic_anchor_paths"],1) # app -> run; not run -> kernel
 def test_dynamic_dispatch_refuses(self):
  b="def kernel(x):return x\ndef run(obj,x):return obj.kernel(x)\n"
  a=b.replace("return x","return x+1")
  r=dependency_graph(b,a)
  self.assertEqual(r["n_syntactic_anchor_paths"],0)
 def test_same_class_self_call(self):
  b="class C:\n def impl(self,x):return x\n def forward(self,x):return self.impl(x)\n"
  a=b.replace("return x","return x+1",1)
  r=dependency_graph(b,a)
  self.assertEqual(r["n_syntactic_anchor_paths"],1)
  self.assertEqual(r["roots"][0]["unchanged_consumer"],"C.forward")
 def test_class_qualified_call(self):
  b="class C:\n def impl(self,x):return x\n def forward(self,x):return C.impl(self,x)\n"
  a=b.replace("return x","return x+1",1)
  self.assertEqual(dependency_graph(b,a)["n_syntactic_anchor_paths"],1)
 def test_different_class_self_dispatch_no_wrong_edge(self):
  b="class A:\n def run(self,x):return x\nclass B:\n def run(self,x):return x\n def forward(self,x):return self.run(x)\n"
  a=b.replace("def run(self,x):return x","def run(self,x):return x+1",1)
  self.assertEqual(dependency_graph(b,a)["n_syntactic_anchor_paths"],0)
 def test_only_unchanged_code_mined(self):
  a="def f(x):return x\n"
  b="def f(x):return x+1\ndef g(x):return f(x)\n"
  self.assertEqual(dependency_graph(a,b)["n_syntactic_anchor_paths"],0)
 def test_parse_error_no_forged_anchor(self):
  self.assertEqual(dependency_graph("def x(:","def x():pass")["status"],"ABSTAIN_PARSE")
 def test_scope_declares_unsupported_alias(self):
  r=dependency_graph(OLD,AFTER)
  self.assertEqual(r["unsupported_scope"],"NO_CROSS_FILE_ALIAS_OR_DYNAMIC_DISPATCH")
 def test_original_digest_binding(self):
  r=dependency_graph(OLD,AFTER)
  self.assertEqual(len(r["roots"][0]["unchanged_consumer_ast_sha256"]),64)
 def test_monotone_depth(self):
  self.assertEqual(dependency_graph(OLD,AFTER,max_depth=1)["n_syntactic_anchor_paths"],1)
 def test_no_self_loop(self):
  b="def f(x):\n f(x)\n return x\n"
  a=b.replace("return x","return x+1")
  self.assertEqual(dependency_graph(b,a)["n_syntactic_anchor_paths"],0)
 def test_wrapper_exposes_counts(self):
  x=summarise_file("sample.py",OLD,AFTER)
  self.assertEqual(x["n_changed_producers_with_root"],1)
  self.assertFalse(x["whole_patch_authorized"])
if __name__=="__main__":unittest.main()
