import unittest
from fractions import Fraction as F
from tempfile import TemporaryDirectory
from pathlib import Path
from hashlib import sha256

from research.semrepair.bound_repair_authority import RepairAuthority
from research.semrepair.blind_authority_runner import ProbeSpec
from research.semrepair.artifact_byte_authority import AuthorityFiles
from research.semrepair.provenance_bound_authorizer import (
    execute_provenance_bound_authorization as run,
)


class CombinedProofCarryingTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.authorities={}
        self.files={}
        for world in ("w0","w1"):
            group=Path(self.temp.name)/world
            group.mkdir()
            paths=[]
            digests=[]
            for kind in ("patch","implementation","evidence","dependency"):
                path=group/(kind+".txt")
                path.write_bytes((world+":"+kind).encode())
                paths.append(path)
                digests.append(sha256(path.read_bytes()).hexdigest())
            a=RepairAuthority(*digests)
            self.authorities[world]=a
            self.files[a.binding]=AuthorityFiles(*paths)
        self.probes={"deep":ProbeSpec("deep",{"w0":F(0),"w1":F(2)},F(0),
                                      cost=F(1),risk=F(1,4))}
        self.kwargs=dict(worlds=("w0","w1"),authorities=self.authorities,
                        probes=self.probes,policy=lambda _: "deep",
                        candidate_files_by_binding=self.files,
                        total_cost_budget=F(1),max_single_probe_risk=F(1))

    def test_trusted_probe_and_actual_patch_bytes_both_needed(self):
        a=run(**self.kwargs,observe=lambda _: F(0),
              externally_trusted_probe_names=frozenset({"deep"}))
        self.assertEqual(a["decision"],"FILE_BYTES_AUTHORIZED_ONLY")
        self.assertEqual(a["authorization_digest"],
                         self.authorities["w0"].binding)
        self.assertFalse(a["source_trust_externally_proven"])

    def test_untrusted_probe_not_even_observed(self):
        calls=[]
        r=run(**self.kwargs,observe=lambda p:(calls.append(p),F(0))[1],
              externally_trusted_probe_names=frozenset())
        self.assertEqual(r["decision"],"REFUSED_UNTRUSTED_EVIDENCE")
        self.assertEqual(calls,[])

    def test_tampered_patch_refused_after_correct_world_identified(self):
        self.files[self.authorities["w0"].binding].patch.write_bytes(b"spoofed")
        r=run(**self.kwargs,observe=lambda _: F(0),
              externally_trusted_probe_names=frozenset({"deep"}))
        self.assertEqual(r["decision"],"REFUSED_INVALID_PATCH_ARTIFACT")

    def test_missing_actual_patch_refused(self):
        r=run(**{**self.kwargs,"candidate_files_by_binding":{}},
              observe=lambda _: F(0),
              externally_trusted_probe_names=frozenset({"deep"}))
        self.assertEqual(r["decision"],"REFUSED_MISSING_PATCH_ARTIFACT")

    def test_spoofed_trusted_observation_cannot_be_detected_by_this_module(self):
        # Negative control: attacker forges w1 reading while truth is w0.
        # This test documents inability to attest physical sensor identity.
        r=run(**self.kwargs,observe=lambda _: F(2),
              externally_trusted_probe_names=frozenset({"deep"}))
        self.assertEqual(r["decision"],"FILE_BYTES_AUTHORIZED_ONLY")
        self.assertEqual(r["authorization_digest"],
                         self.authorities["w1"].binding)
        self.assertNotEqual(r["authorization_digest"],
                            self.authorities["w0"].binding)

    def test_invalid_trusted_probe_inventory(self):
        with self.assertRaises(ValueError):
            run(**self.kwargs,observe=lambda _:F(0),
                externally_trusted_probe_names=frozenset({"not-declared"}))


if __name__=="__main__":
    unittest.main()
