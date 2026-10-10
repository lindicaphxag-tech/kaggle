"""Red/green checks for an explicitly finite, premerge-only LeRobot repair grammar."""
import unittest
from . import lerobot_crossfile_finite_synth as subject
from . import real_merged_lerobot_processor_contract as vendor


class PremergeCrossFileSynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {}
        for role, path, sha in (
            ("train", vendor.SCRIPT_PATH, vendor.BEFORE_SOURCE_BLOB),
            ("registry", vendor.REGISTRY_PATH, vendor.BEFORE_REGISTRY_BLOB),
            ("config", vendor.CONFIG_PATH, vendor.BEFORE_CONFIG_BLOB),
        ):
            cls.sources[role] = vendor._pinned_source(vendor.BEFORE_COMMIT, path, sha)[0]

    def test_exhaustive_minimum_with_full_source_ownership(self):
        selected, explored = subject.select(self.sources)
        self.assertEqual(len(explored), 16)
        self.assertIsNotNone(selected)
        cost, candidate, result = selected
        self.assertEqual(cost, 5)
        self.assertEqual(candidate, subject.Candidate(True, True, True, True))
        self.assertTrue(result["pass"])
        self.assertEqual(len(result["scenarios"]), len(subject.SCENARIOS))
        self.assertTrue(all(row[0] for row in result["scenarios"].values()))

    def test_cheap_partial_and_insufficient_authority_fail_closed(self):
        cheap, _ = subject.select(self.sources, cost_limit=4)
        no_config, _ = subject.select(self.sources, allowed_scopes=("train", "registry"))
        no_registry, _ = subject.select(self.sources, allowed_scopes=("train", "config"))
        self.assertIsNone(cheap)
        self.assertIsNone(no_config)
        self.assertIsNone(no_registry)

    def test_negatives_separately_break_contract(self):
        for candidate in (
            subject.Candidate(False, True, True, True),
            subject.Candidate(True, False, True, True),
            subject.Candidate(True, True, False, True),
            subject.Candidate(True, True, True, False),
        ):
            with self.subTest(candidate=candidate):
                result = subject.assess(self.sources, candidate)
                self.assertFalse(result["pass"])

    def test_tampered_vendor_source_refused(self):
        invalid = dict(self.sources)
        invalid["train"] = invalid["train"].replace(
            "processor_pretrained_path = None", "processor_pretrained_path = 0")
        with self.assertRaises(ValueError):
            subject.assess(invalid, subject.Candidate(True, True, True, True))


if __name__ == "__main__":
    unittest.main()
