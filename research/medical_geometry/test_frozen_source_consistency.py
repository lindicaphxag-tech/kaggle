import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from research.medical_geometry.frozen_source_consistency import (
    verify_frozen_negative_source, FROZEN_REPORT,
)


class EvidenceBindingTests(unittest.TestCase):
    def test_frozen_real_report_still_supports_negative_conclusion(self):
        report=verify_frozen_negative_source()
        self.assertEqual(report["independent_subject_ids"],6)
        self.assertLess(report["historical_paired_roi_patient_delta_mean"],0)
        self.assertEqual(report["strong_canonical_first_table_mean_dice"],.186392)
        self.assertEqual(report["gated_table_mean_dice"],.186315)

    def test_inventing_positive_patient_is_rejected(self):
        source=FROZEN_REPORT.read_text(encoding="utf8")
        modified=source.replace("−0.0001833","+0.0001833",1)
        with TemporaryDirectory() as d:
            path=Path(d)/"source.md"
            path.write_text(modified,encoding="utf8")
            with self.assertRaisesRegex((ValueError,AssertionError), "count|negative"):
                verify_frozen_negative_source(path)

    def test_claim_cannot_discard_first_view(self):
        source=FROZEN_REPORT.read_text(encoding="utf8")
        modified=source.replace("Canonical-RAS unpermuted first-view model", "DISCARDED strong comparator",1)
        with TemporaryDirectory() as d:
            path=Path(d)/"source.md"
            path.write_text(modified,encoding="utf8")
            with self.assertRaisesRegex(ValueError,"strong original"):
                verify_frozen_negative_source(path)


if __name__=="__main__":
    unittest.main()
