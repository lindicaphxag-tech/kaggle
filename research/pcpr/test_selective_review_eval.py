import unittest
from dataclasses import replace
from selective_review_eval import Judgment, evaluate, strict_paired, cluster_bootstrap,wilson

def cases(n=20):
 return [Judgment(f"repo{i%5}/pr{i}",f"repo{i%5}",
                  "incorrect" if i%4==0 else "correct",i%2==0,
                  1.0 if i%2==0 else 0.2,0.5,"rubric-v1")
         for i in range(n)]

class SelectiveRiskTests(unittest.TestCase):
 def test_empty_truth_no_fake_accuracy(self):
  self.assertEqual(evaluate([])["status"],"NOT_EVALUABLE_EMPTY")
 def test_always_abstain_fails_coverage(self):
  rows=[replace(x,approved=False) for x in cases()]
  o=evaluate(rows)
  self.assertEqual(o["reason"],"COVERAGE_BELOW_PREDECLARED_TARGET")
  self.assertIsNone(o["false_authorization_rate_known"])
 def test_true_gate_above_coverage(self):
  o=evaluate(cases())
  self.assertEqual(o["status"],"DESCRIPTIVE_ONLY_PENDING_STRONG_BASELINES")
 def test_unknown_gold_approved_refusal(self):
  rows=cases()
  rows[0]=replace(rows[0],truth="unadjudicated")
  o=evaluate(rows)
  self.assertEqual(o["reason"],"NO_COMPLETE_INDEPENDENT_GOLD_FOR_APPROVED_CASES")
 def test_unknown_gold_nonapproved_is_counted(self):
  rows=cases()
  rows[1]=replace(rows[1],truth="unadjudicated")
  o=evaluate(rows)
  self.assertEqual(o["n_unadjudicated"],1)
  self.assertEqual(o["n_approved_unadjudicated"],0)
 def test_per_pr_not_field_count(self):
  self.assertEqual(evaluate(cases())["n_pr"],20)
 def test_exact_wilson_0_0_unavailable(self):
  self.assertIsNone(wilson(0,0))
 def test_wilson_finite_zero_failures(self):
  low,high=wilson(0,10)
  self.assertEqual(low,0.0)
  self.assertGreater(high,0.2)
 def test_duplicate_pr_denied(self):
  a=cases();a.append(a[0])
  with self.assertRaises(ValueError):evaluate(a)
 def test_invalid_label_denied(self):
  a=cases();a[0]=replace(a[0],truth="merged-means-correct")
  with self.assertRaises(ValueError):evaluate(a)
 def test_missing_rubric_denied(self):
  a=cases();a[0]=replace(a[0],rubric_version="")
  with self.assertRaises(ValueError):evaluate(a)
 def test_wrong_cost_denied(self):
  a=cases();a[0]=replace(a[0],cost_seconds=-1)
  with self.assertRaises(ValueError):evaluate(a)
 def test_one_repo_no_generalization(self):
  a=[replace(x,repo="same") for x in cases()]
  self.assertEqual(evaluate(a)["reason"],"TOO_FEW_REPOSITORY_CLUSTERS_FOR_GENERALIZATION")
 def test_different_cohorts_refuse_paired(self):
  a=cases();b=a[:-1]
  with self.assertRaises(ValueError):strict_paired(a,b)
 def test_conflicting_truth_refuse_paired(self):
  a=cases();b=a.copy();b[0]=replace(b[0],truth="correct")
  with self.assertRaises(ValueError):strict_paired(a,b)
 def test_mismatched_coverage_refuses_significance(self):
  a=cases();b=a.copy();b[0]=replace(b[0],approved=False)
  self.assertEqual(strict_paired(a,b)["status"],"NOT_MATCHED_COVERAGE")
 def test_same_data_p_value_one(self):
  a=cases()
  self.assertEqual(strict_paired(a,a)["uncorrected_two_sided_exact_discordance_p"],1.0)
 def test_cluster_bootstrap_is_defined(self):
  a=cases()
  r=cluster_bootstrap(a,n_resamples=100)
  self.assertIsNotNone(r)
  self.assertEqual(r["unit"],"repository")
 def test_small_clusters_no_confidence(self):
  a=[replace(x,repo="one") for x in cases()]
  self.assertIsNone(cluster_bootstrap(a))
if __name__=="__main__":unittest.main()
