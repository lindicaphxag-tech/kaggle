"""Real pinned vendor kernel plus authorized EXACT one-node authored wrapper repair."""
import unittest
from research.semrepair.lerobot_callsite_ast_repair import (
    ORIGINAL_CALLSITE,SHA,synthesize_authorized_minimal_temporal_patch,
    run_original_lerobot_ast_repair
)

class OriginalLeRobotAstSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=run_original_lerobot_ast_repair()
    def test_real_vendor_code_and_matched_pair_error(self):
        r=self.r
        self.assertEqual(r["source_repair_comparisons"],128)
        self.assertEqual(r["improved_case_count"],128)
        self.assertGreater(r["mean_faulty_max_abs_error"],.1)
        self.assertLess(r["maximum_fixed_max_abs_error"],1e-12)
        self.assertTrue(r["real_vendor_kernel_unchanged"])
        self.assertTrue(r["author_injected_fault"])
        self.assertEqual(r["actual_robot_episodes"],0)
    def test_exactly_one_authorized_AST_arg_edited(self):
        r=self.r
        m=r["author_callsite_edit_metadata"]
        self.assertEqual(m["changed_AST_argument_names"],1)
        self.assertEqual(m["author_callsite_file_changes"],1)
        self.assertEqual(m["real_source_file_byte_replacements"],0)
        self.assertIn("-    return decoder(chunk, current_state, mask)",m["patch_unified_diff"])
        self.assertIn("+    return decoder(chunk, initial_anchor, mask)",m["patch_unified_diff"])
    def test_reject_wrong_bytes_or_no_source_authority_or_stale_epoch(self):
        defaults=dict(original_source=ORIGINAL_CALLSITE,
                      expected_sha256=SHA,write_authorized=True,
                      anchor_epoch=8,live_chunk_epoch=8,anchor_available=True)
        for changes,reason in (
            ({"expected_sha256":"0"*64},"REFUSE_UNTRUSTED_SOURCE_BYTES"),
            ({"write_authorized":False},"REFUSE_UNLICENSED_SOURCE_EDIT"),
            ({"anchor_epoch":7},"REFUSE_MISSING_OR_STALE_CHUNK_ANCHOR"),
            ({"anchor_available":False},"REFUSE_MISSING_OR_STALE_CHUNK_ANCHOR"),
        ):
            with self.subTest(reason=reason):
                r,code=synthesize_authorized_minimal_temporal_patch(**{**defaults,**changes})
                self.assertIsNone(code)
                self.assertEqual(r["decision"],reason)
    def test_unexpected_source_structure_cannot_be_patched(self):
        bad=ORIGINAL_CALLSITE.replace(
            "return decoder(chunk, current_state, mask)",
            "return decoder(chunk, initial_anchor, mask)")
        with self.assertRaisesRegex(ValueError,"callsite"):
            synthesize_authorized_minimal_temporal_patch(
                original_source=bad,
                expected_sha256=__import__("hashlib").sha256(bad.encode()).hexdigest(),
                write_authorized=True,anchor_epoch=8,live_chunk_epoch=8,
                anchor_available=True)
if __name__=="__main__":unittest.main()
