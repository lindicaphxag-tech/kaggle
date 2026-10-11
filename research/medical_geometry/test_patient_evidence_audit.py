import unittest
from patient_evidence_audit import ArmOutcome as A, paired_patient_audit as audit


def row(p, arm, dice, scope="full_volume", mask="mask"):
    return A(p, "MRI", arm, mask, "ckpt", scope, dice, 10., 1.)


class Tests(unittest.TestCase):
    def test_paired_negative_outcome_retained(self):
        x = audit([row("a", "canonical_first", .7),
                   row("a", "pre_mix_repair", .6),
                   row("b", "canonical_first", .8),
                   row("b", "pre_mix_repair", .7)])
        self.assertEqual(x["patient_count"], 2)
        self.assertAlmostEqual(
            x["comparisons"]["pre_mix_repair"]["mean_patient_dice_delta"], -.1)

    def test_unmatched_refused(self):
        with self.assertRaisesRegex(ValueError, "unpaired"):
            audit([row("a", "canonical_first", .7),
                   row("a", "pre_mix_repair", .6),
                   row("b", "canonical_first", .8)])

    def test_mask_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            audit([row("a", "canonical_first", .7),
                   row("a", "pre_mix_repair", .6, mask="bad")])

    def test_roi_versus_full_refused(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            audit([row("a", "canonical_first", .7),
                   row("a", "pre_mix_repair", .6, scope="cropped_roi")])

    def test_duplicate_patient_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit([row("a", "canonical_first", .7),
                   row("a", "canonical_first", .7),
                   row("a", "pre_mix_repair", .6)])


if __name__ == "__main__":
    unittest.main()
