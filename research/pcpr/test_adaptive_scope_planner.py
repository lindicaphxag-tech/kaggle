import itertools
import unittest
from adaptive_scope_planner import (
    Experiment,minimax_plan,follow,fixed_order_cost,compare,generate_finite_world
)

class Optimality(unittest.TestCase):
 def test_single_label_terminal_zero_cost(self):
  s,_=minimax_plan(["ok","ok"],[Experiment("a",2,("0","1"))])
  self.assertEqual(s((0,1)).upper_cost,0)
  self.assertIsNone(s((0,1)).next_experiment)
 def test_indistinguishable_hypotheses_refuse(self):
  s,_=minimax_plan(["ok","bad"],[Experiment("x",1,("0","0"))])
  self.assertIsNone(s((0,1)).upper_cost)
 def test_cheaper_informative_query(self):
  labels=["ok","bad"];x=[Experiment("costly",10,("a","b")),
                          Experiment("cheap",1,("a","b"))]
  s,_=minimax_plan(labels,x)
  self.assertEqual((s((0,1)).upper_cost,s((0,1)).next_experiment),(1,"cheap"))
 def test_no_reuse_of_query_on_identical_partition(self):
  s,_=minimax_plan(["ok","bad"],[Experiment("useless",1,("a","a")),
                                  Experiment("useful",2,("a","b"))])
  self.assertEqual(s((0,1)).next_experiment,"useful")
 def test_budget_refusal(self):
  s,_=minimax_plan(["ok","bad"],[Experiment("a",3,("a","b"))],max_cost=2)
  self.assertIsNone(s((0,1)).upper_cost)
 def test_duplicate_rejected(self):
  with self.assertRaises(ValueError):
   minimax_plan(["a","b"],[Experiment("a",1,("x","y")),
                            Experiment("a",3,("x","y"))])
 def test_zero_cost_rejected(self):
  with self.assertRaises(ValueError):
   minimax_plan(["a","b"],[Experiment("a",0,("x","y"))])
 def test_native_plan_never_whole_patch_authorization(self):
  r=follow(["ok","bad"],[Experiment("test",2,("x","y"))],0)
  self.assertIn("terminal_label",r)
  self.assertEqual(r["cost"],2)
  self.assertNotIn("whole_patch_authorized",r)
 def test_oracle_baseline_has_worse_worst_cost_in_constructed_world(self):
  labels=["ok","ok","bad","bad"]
  ex=[Experiment("expensive",9,("a","a","b","b")),
      Experiment("cheap",1,("a","a","b","b"))]
  self.assertEqual(compare(labels,ex)["minimax_worst_cost"],1)
  self.assertEqual(compare(labels,ex)["fixed_order_worst_cost"],9)
 def test_exact_exhaustive_small_full_binary_catalog(self):
  for values in itertools.product(("0","1"),repeat=6):
   labels=("good","bad","bad")
   ex=[Experiment("a",1,tuple(values[:3])),
       Experiment("b",2,tuple(values[3:]))]
   solve,_=minimax_plan(labels,ex)
   root=solve((0,1,2))
   if root.upper_cost is not None:
    self.assertTrue(root.upper_cost>=1)
    for h in range(3):
     ans=follow(labels,ex,h)
     self.assertEqual(ans["terminal_label"],labels[h])
 def test_fixed_random_simulation_48_seeds(self):
  for seed in range(48):
   r=compare(*generate_finite_world(seed))
   self.assertLessEqual(r["minimax_worst_cost"],r["fixed_order_worst_cost"])
   self.assertEqual(r["minimax_refusals"],0)

if __name__=="__main__":unittest.main()
