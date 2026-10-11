import unittest
from research.semrepair.paired_external_sac_audit import (
    audit_seeds_and_paired_effects,exact_two_sided_sign_p,paired_bootstrap_ci
)

class PairedExternalAuditTests(unittest.TestCase):
    def data(self):
        names=("pilot-severity","large-shear","very-large-shear",
               "shear-plus-extra-clip","large-shear-plus-extra-clip")
        rows=[]
        for name in names:
            for seed in range(23011,23043):
                for arm,improvement in (("blind-source-patch-candidate",1.),
                                         ("exact-all-clip-blackbox-oracle",0.)):
                    rows.append({"fault_family":name,"seed":seed,"arm":arm,
                                 "return":-5.+improvement,
                                 "mean_tip_distance":.2-improvement*.05,
                                 "final_tip_distance":.1})
        return {"rows":rows}

    def test_exact_sign_and_multiplicity(self):
        self.assertEqual(exact_two_sided_sign_p(30,0),2**(-29))
        self.assertEqual(exact_two_sided_sign_p(0,0),1.)
        report=audit_seeds_and_paired_effects(self.data())
        self.assertEqual(report["familywise_comparisons"],5)
        for data in report["comparisons_by_severity"].values():
            self.assertEqual(data["reset_count"],32)
            self.assertEqual(data["source_better"],32)
            self.assertEqual(data["source_worse"],0)
            self.assertAlmostEqual(data["mean_official_reward_gain"],1.)
            self.assertLess(data["bonferroni_adjusted_p_for_five_fixed_levels"],.001)

    def test_duplicate_episode_must_fail(self):
        test=self.data()
        test["rows"].append(test["rows"][0])
        with self.assertRaisesRegex(ValueError,"duplicate"):
            audit_seeds_and_paired_effects(test)

    def test_missing_seed_must_fail(self):
        test=self.data()
        test["rows"].pop()
        with self.assertRaisesRegex(ValueError,"missing"):
            audit_seeds_and_paired_effects(test)

    def test_bootstrap_zero_or_constant_variance(self):
        self.assertEqual(paired_bootstrap_ci([0.]*32),(0.,0.))
        self.assertEqual(paired_bootstrap_ci([.5]*32),(.5,.5))

if __name__=="__main__":
    unittest.main()
