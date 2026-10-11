import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from research.semrepair.bound_repair_authority import RepairAuthority
from research.semrepair.artifact_byte_authority import (
    AuthorityFiles, attest_current_artifacts, authorize_verified_files,
)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        paths = []
        hashes = []
        for name in ("patch", "impl", "evidence", "dependency"):
            p = root / (name + ".txt")
            p.write_bytes(("independent:" + name).encode())
            paths.append(p)
            hashes.append(sha256(p.read_bytes()).hexdigest())
        self.files = AuthorityFiles(*paths)
        self.auth = RepairAuthority(*hashes)

    def test_all_real_bytes_verified(self):
        self.assertTrue(attest_current_artifacts(self.auth, self.files)["valid"])
        self.assertTrue(authorize_verified_files(
            {"decision": "AUTHORIZED", "bound_repair": self.auth},
            self.files)["accepted"])

    def test_valid_format_but_wrong_bytes_fail_closed(self):
        self.files.patch.write_text("changed", encoding="utf-8")
        output = authorize_verified_files(
            {"decision": "AUTHORIZED", "bound_repair": self.auth}, self.files)
        self.assertFalse(output["accepted"])
        self.assertIn("patch_sha256:sha256_mismatch",
                      output["file_evidence"]["failures"])

    def test_missing_artifact_fails_closed(self):
        self.files.dependency.unlink()
        report = attest_current_artifacts(self.auth, self.files)
        self.assertFalse(report["valid"])

    def test_syntactically_valid_fake_digests_do_not_authorize(self):
        fake = RepairAuthority(*(["a"*64]*4))
        self.assertFalse(attest_current_artifacts(fake, self.files)["valid"])

    def test_unidentified_authority_cannot_authorize_bytes(self):
        out = authorize_verified_files(
            {"decision": "REFUSED_CONFLICTING_REPAIRS",
             "bound_repair": self.auth}, self.files)
        self.assertFalse(out["accepted"])


    def test_symlink_with_correct_expected_bytes_refused(self):
        real = self.files.patch.with_name("real-patch.txt")
        real.write_bytes(self.files.patch.read_bytes())
        self.files.patch.unlink()
        self.files.patch.symlink_to(real)
        report = attest_current_artifacts(self.auth, self.files)
        self.assertFalse(report["valid"])
        self.assertTrue(any("patch_sha256:unreadable_symlink" in x
                            for x in report["failures"]))

    def test_named_pipe_refused_without_blocking(self):
        import os
        if not hasattr(os, "mkfifo"):
            self.skipTest("POSIX-only regular-file safety check")
        self.files.evidence.unlink()
        os.mkfifo(self.files.evidence)
        report = attest_current_artifacts(self.auth, self.files)
        self.assertFalse(report["valid"])
        self.assertTrue(any("evidence_sha256:unreadable_symlink" in x
                            for x in report["failures"]))


if __name__ == "__main__":
    unittest.main()
