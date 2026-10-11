import unittest
from pathlib import Path
from hashlib import sha256
from tempfile import TemporaryDirectory
from research.semrepair.bound_repair_authority import RepairAuthority
from research.semrepair.artifact_byte_authority import AuthorityFiles
from research.semrepair.pinned_repair_payload import handoff_same_verified_bytes

class PinnedRepairSnapshot(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        paths=[]
        for label in ("patch","impl","evidence","dependency"):
            p=root/(label+".bin")
            p.write_bytes(("frozen-"+label).encode())
            paths.append(p)
        self.files=AuthorityFiles(*paths)
        self.auth=RepairAuthority(*(sha256(p.read_bytes()).hexdigest() for p in paths))
        self.decision={"decision":"AUTHORIZED","bound_repair":self.auth}

    def test_success_consumes_exact_verified_bytes_even_if_paths_change(self):
        captured=[]
        def consume(snapshot):
            # Deliberate path race AFTER verification: consumer still sees
            # the immutable *previously hashed* bytes, not the changed path.
            self.files.patch.write_bytes(b"substituted")
            captured.append(snapshot.patch)
        result=handoff_same_verified_bytes(decision=self.decision,
                  files=self.files,consumer=consume)
        self.assertEqual(result["status"],"VERIFIED_BYTES_HANDED_OFF")
        self.assertEqual(captured,[b"frozen-patch"])

    def test_tampered_path_cannot_reach_consumer(self):
        self.files.patch.write_bytes(b"altered-before-check")
        captured=[]
        result=handoff_same_verified_bytes(decision=self.decision,
                files=self.files,consumer=captured.append)
        self.assertEqual(result["status"],"REFUSED_SOURCE_BYTES")
        self.assertFalse(captured)

    def test_symlink_swap_refused(self):
        target=self.files.patch.with_name("elsewhere")
        target.write_bytes(self.files.patch.read_bytes())
        self.files.patch.unlink()
        self.files.patch.symlink_to(target)
        result=handoff_same_verified_bytes(decision=self.decision,
                files=self.files,consumer=lambda snap:self.fail("must refuse"))
        self.assertEqual(result["status"],"REFUSED_SOURCE_BYTES")

    def test_denied_decision_never_invokes_callback(self):
        result=handoff_same_verified_bytes(
            decision={"decision":"REFUSED","bound_repair":self.auth},
            files=self.files,consumer=lambda snap:self.fail("must refuse"))
        self.assertEqual(result["status"],"REFUSE")

if __name__=="__main__":
    unittest.main()
