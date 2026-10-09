import unittest
from research.semrepair.open_catalog_risk_coverage import audit_open_catalog_risk_coverage

class OpenCatalogRiskCoverage(unittest.TestCase):
    def test_conditional_safety_versus_coverage_not_only_accuracy(self):
        s=audit_open_catalog_risk_coverage()["score"]
        self.assertEqual(s["sensor_only"]["correct_authorizations"],12)
        self.assertEqual(s["sensor_only"]["wrong_authorizations"],3)
        self.assertEqual(s["sensor_only"]["refusals"],9)
        self.assertEqual(s["sensor_only"]["wrong_repair_risk_given_authorization"],.2)
        self.assertEqual(s["trusted_source_required"]["wrong_authorizations"],0)
        self.assertEqual(s["trusted_source_required"]["authorized_coverage"],.5)
        self.assertEqual(s["sensor_only"]["authorized_coverage"],.625)

if __name__=="__main__":
    unittest.main()
