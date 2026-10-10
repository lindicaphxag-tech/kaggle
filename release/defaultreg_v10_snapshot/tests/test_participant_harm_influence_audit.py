"""Contract tests for supplementary harm/influence analysis."""
from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from participant_harm_influence_audit import (
    audit, markdown_report, wilson_reference_interval
)

LEDGER = ROOT / "evidence" / "verified_three_cohort_paired_effect_vectors_v8.json"


class ParticipantHarmInfluenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(LEDGER.read_text())

    def test_frozen_denominators_and_harm_counts(self):
        results = audit(self.data)["datasets"]
        self.assertEqual(set(results), {
            "EEGMMIDB_S021_S040", "BNCI2014_002", "BNCI2015_001"
        })
        expected = {
            "EEGMMIDB_S021_S040": (5, 4),
            "BNCI2014_002": (3, 2),
            "BNCI2015_001": (2, 2),
        }
        for name, (cov, lw) in expected.items():
            self.assertEqual(results[name]["vs_covariance_nll"]["n_harmed_gt_0_02"], cov)
            self.assertEqual(results[name]["vs_ledoitwolf"]["n_harmed_gt_0_02"], lw)

    def test_mean_median_divergence_is_not_hidden(self):
        b = audit(self.data)["datasets"]["BNCI2015_001"]["vs_covariance_nll"]
        self.assertAlmostEqual(b["mean_effect"], 0.04328865740740737)
        self.assertAlmostEqual(b["median_effect"], 0.010525)
        self.assertGreater(b["mean_effect"], 3 * b["median_effect"])

    def test_individual_influence_cannot_reverse_sign_in_frozen_cohorts(self):
        d = audit(self.data)["datasets"]
        for x in d.values():
            for arm in x.values():
                self.assertTrue(arm["leave_one_participant_out_direction_all_positive"])
                self.assertGreater(arm["leave_one_participant_out_mean_range_posthoc"][0], 0)

    def test_wilson_reference_interval_sensible_extremes(self):
        self.assertEqual(wilson_reference_interval(0, 20)[0], 0.0)
        self.assertEqual(wilson_reference_interval(20, 20)[1], 1.0)
        lo, hi = wilson_reference_interval(5, 20)
        self.assertAlmostEqual(lo, 0.111861, places=5)
        self.assertAlmostEqual(hi, 0.468701, places=5)
        with self.assertRaises(ValueError):
            wilson_reference_interval(7, 5)

    def test_report_discloses_chronology_and_conditionality(self):
        report = markdown_report(audit(self.data))
        self.assertIn("after all target outcomes", report)
        self.assertIn("shared source-model", report)
        self.assertIn("BNCI2015_001", report)
        self.assertIn("+0.0105", report)

    def test_missing_cohort_and_ids_fail_closed(self):
        bad = copy.deepcopy(self.data)
        bad["cohorts"].pop("BNCI2014_002")
        with self.assertRaises(ValueError):
            audit(bad)
        bad = copy.deepcopy(self.data)
        bad["cohorts"]["BNCI2014_002"]["participant_ids"][0] = "2"
        with self.assertRaises(ValueError):
            audit(bad)

    def test_corrupted_vectors_fail_closed(self):
        bad = copy.deepcopy(self.data)
        bad["cohorts"]["BNCI2015_001"]["primary"]["vs_covariance_nll"][0] = float("nan")
        with self.assertRaises(ValueError):
            audit(bad)
        bad = copy.deepcopy(self.data)
        bad["cohorts"]["BNCI2015_001"]["primary"]["vs_covariance_nll"][0] = -0.5
        with self.assertRaises(ValueError):
            audit(bad)

if __name__ == "__main__":
    unittest.main()
