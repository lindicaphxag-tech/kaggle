"""Independent corner enumeration and patch provenance refusal tests."""
import hashlib
import itertools
import unittest
from research.semrepair.authority_action_envelope_planner import (
    PATCH_COSTS,proof_for_patch,choose_source_authorized_patch,freeze_example_report,
)
from research.semrepair.observable_interface_factorization import I,mv,mmul

P=((1.,.8),(0.,1.))
C=((1.,-.8),(0.,1.))
DIGEST=hashlib.sha256(b"externally-asserted-source-identity").hexdigest()


class EnvelopePlannerTests(unittest.TestCase):
    def plan(self,radius,allowed=tuple(PATCH_COSTS),verified=True):
        return choose_source_authorized_patch(
            producer=P,controller=C,action_radius=radius,inner_clip=9.,
            allowed_by_external_source=allowed,
            external_source_verified=verified,
            source_digest=DIGEST if verified else None)

    def test_source_provenance_is_mandatory(self):
        self.assertEqual(self.plan(8,verified=False)["decision"],
                         "REFUSE_MISSING_EXTERNAL_SOURCE_PROVENANCE")

    def test_small_action_box_needs_no_mutation(self):
        r=self.plan(2)
        self.assertEqual(r["patch"],"noop")
        self.assertEqual(r["minimum_cost"],"0")

    def test_large_action_box_requires_atomic_source_repair(self):
        r=self.plan(8)
        self.assertEqual(r["patch"],"atomic")
        self.assertEqual(r["minimum_cost"],"2")
        d={x["patch"]:x for x in r["candidates"]}
        self.assertFalse(d["noop"]["certified"])
        self.assertFalse(d["noop"]["unbounded_upstream_action_reachability"])
        self.assertTrue(d["controller-only"]["unbounded_upstream_action_reachability"])
        self.assertFalse(d["controller-only"]["repair_preserves_identity_mapping"])
        self.assertTrue(d["atomic"]["certified"])

    def test_source_authority_disallows_fictitious_min_cost_patch(self):
        out=self.plan(8,allowed=["producer-only","controller-only"])
        self.assertEqual(out["decision"],"REFUSE_NO_PATCH_VALID_OVER_DEPLOYMENT_ENVELOPE")
        self.assertIsNone(out["patch"])

    def test_controller_cannot_fix_target_outside_source_limit(self):
        out=self.plan(10)
        self.assertEqual(out["decision"],"REFUSE_NO_PATCH_VALID_OVER_DEPLOYMENT_ENVELOPE")

    def test_exact_rectangular_command_box_vs_analytic_max_norm(self):
        for patch in PATCH_COSTS:
            info=proof_for_patch(
                producer=P,controller=C,patch=patch,action_radius=8,inner_clip=9)
            effect=I if patch in ("producer-only","atomic") else P
            corners=[(-8.,-8.),(-8.,8.),(8.,-8.),(8.,8.)]
            actual=max(abs(z) for u in corners for z in mv(effect,u))
            self.assertAlmostEqual(info["maximum_producer_magnitude_on_action_box"],actual)

    def test_frozen_record_exposes_unverified_source_flag(self):
        report=freeze_example_report()
        self.assertEqual(report["small_development_envelope"]["patch"],"noop")
        self.assertEqual(report["larger_deployment_envelope"]["patch"],"atomic")
        self.assertEqual(report["same_deployment_with_no_source_provenance"]["patch"],None)
        self.assertFalse(report["larger_deployment_envelope"][
            "external_source_authentication_performed_by_this_module"])

if __name__=="__main__":
    unittest.main()
