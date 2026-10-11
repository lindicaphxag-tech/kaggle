import unittest
from research.medical_geometry.paired_negative_analysis import (
    PairedPatientDelta as D, exact_exploratory_sign_summary as score,
)


class Tests(unittest.TestCase):
    def test_pinned_six_patient_roi_negative_direction(self):
        # Reported historical per-patient deltas, NOT independently rescored raw masks.
        vals = [-.0001833,-.0000630,-.0000485,-.0000237,-.0001002,-.0000446]
        z = score([D("s"+str(i), v) for i,v in enumerate(vals)])
        self.assertEqual(z["positive"],0)
        self.assertEqual(z["negative"],6)
        self.assertAlmostEqual(z["mean_dice_delta"],-.00007721666666666667)
        self.assertEqual(z["two_sided_exploratory_sign_p"],.03125)

    def test_ties_count_without_inflation(self):
        z = score([D("a",0.0),D("b",.1),D("c",-.1)])
        self.assertEqual(z["ties"],1)
        self.assertEqual(z["two_sided_exploratory_sign_p"],1.)

    def test_duplicate_patient_refused(self):
        with self.assertRaisesRegex(ValueError, "pseudoreplication"):
            score([D("a",.1),D("a",-.2)])

    def test_nonfinite_effect_refused(self):
        with self.assertRaises(ValueError):
            D("x",float("nan"))


if __name__=="__main__":
    unittest.main()
