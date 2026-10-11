"""Independent JSS-style dimensional scalability and synthesis optimality gates."""
import itertools
import random
import unittest
from research.semrepair.authority_aware_repair_synthesis import (
    certify_multichannel_robust_repair,
    choose_authority_constrained_patch,
    _identity,example_native_seven_joint_catalog
)


class AuthoritySynthesisTests(unittest.TestCase):
    def validate(self,n,*,drift=0.,L=2.01,R=2.,eps=.01):
        mat=_identity(n)
        d=[[drift]*n for _ in range(n)]
        return certify_multichannel_robust_repair(
            producer=mat,controller=mat,
            action_radii=[R]*n,internal_clip_limits=[L]*n,
            producer_abs_error=d,controller_abs_error=d,
            output_tolerance=[eps]*n,roundoff_margin=0.)

    def test_multi_dim_2_7_16_32_identity(self):
        for n in (2,7,16,32):
            with self.subTest(n=n):
                r=self.validate(n,L=2.0,eps=0.)
                self.assertTrue(r["certified"])
                self.assertEqual(r["dimension"],n)
                self.assertTrue(all(x==0 for x in r["outgoing_abs_error_upper_bound"]))

    def test_authority_sensitive_optimality_in_real_robot_seven_controls(self):
        demo=example_native_seven_joint_catalog()
        self.assertTrue(demo["producer_controller_compose_to_identity"])
        selected=demo["licensed_all_costs"]
        self.assertEqual(selected["selected"],"E-clip-only")
        self.assertEqual(selected["minimum_declared_catalog_cost"],1.)
        self.assertTrue(selected["catalog_optimality_proven"])
        self.assertFalse(selected["trust_root_signatures_verified_here"])
        self.assertEqual(demo["clip_change_not_licensed"]["selected"],None)
        self.assertEqual(demo["strict_motor_clip_caps"]["selected"],None)

    def test_controllers_that_look_identity_but_clip_do_not_pass(self):
        n=7;P=_identity(n);C=_identity(n)
        P[0][1]=3.5
        C[0][1]=-3.5
        o=certify_multichannel_robust_repair(
            producer=P,controller=C,
            action_radii=[2]*n,internal_clip_limits=[.8]*n,
            producer_abs_error=[[0]*n for _ in range(n)],
            controller_abs_error=[[0]*n for _ in range(n)],
            output_tolerance=[1e-10]*n,roundoff_margin=0.)
        self.assertFalse(o["certified"])
        self.assertTrue(o["output_error_covered_over_full_box"])
        self.assertFalse(o["stage_saturation_absent_over_full_box"])
        self.assertEqual(o["stage_worst_case"][0],9.)

    def test_uncertainty_exactly_prevents_false_approval(self):
        n=7
        self.assertTrue(self.validate(n,drift=.00005,L=2.001,eps=.01)["certified"])
        self.assertFalse(self.validate(n,drift=.0002,L=2.001,eps=.01)["certified"])
        self.assertFalse(self.validate(n,drift=.00005,L=2.001,eps=.001)["certified"])

    def test_adversarial_product_error_bound_holds_for_real_7d(self):
        n=7
        rng=random.Random(73127)
        I=_identity(n)
        d=.00013
        tolerance=.04
        proof=self.validate(n,drift=d,L=2.002,eps=tolerance)
        self.assertTrue(proof["certified"])
        bound=proof["outgoing_abs_error_upper_bound"]
        for _ in range(100):
            P=[[I[i][j]+rng.uniform(-d,d) for j in range(n)] for i in range(n)]
            C=[[I[i][j]+rng.uniform(-d,d) for j in range(n)] for i in range(n)]
            u=[rng.choice((-2.,-1.5,0.,1.5,2.)) for _ in range(n)]
            inner=[sum(P[i][j]*u[j] for j in range(n)) for i in range(n)]
            self.assertTrue(all(abs(inner[i]) <= proof["max_allowed_internal_clip"][i] for i in range(n)))
            actual=[sum(C[i][j]*inner[j] for j in range(n)) for i in range(n)]
            self.assertTrue(all(abs(actual[i]-u[i]) <= bound[i]+1e-10 for i in range(n)))

    def test_source_licenses_prevent_cheap_unsafe_saturation_repair(self):
        n=2;I=_identity(n)
        P=[[1.,.8],[0.,1.]]
        C=[[1.,-.8],[0.,1.]]
        zero=[[0.,0.],[0.,0.]]
        patches=[
            dict(name="fault",producer=P,controller=C,
                 clip=[2.,2.],edits=[],cost=0),
            dict(name="unsafe-clip-only",producer=P,controller=C,
                 clip=[3.61,2.01],edits=["clip"],cost=1),
            dict(name="atomic",producer=I,controller=I,
                 clip=[2.01,2.01],edits=["producer","controller"],cost=3)
        ]
        res=choose_authority_constrained_patch(
            catalog=patches,action_radii=[2.,2.],producer_abs_error=zero,
            controller_abs_error=zero,output_tolerance=[1e-9,1e-9],
            licensed_component_edits=["producer","controller","clip"],
            hard_clip_caps=[3.,3.])
        self.assertEqual(res["selected"],"atomic")
        self.assertEqual(res["minimum_declared_catalog_cost"],3.)
        res=choose_authority_constrained_patch(
            catalog=patches,action_radii=[2.,2.],producer_abs_error=zero,
            controller_abs_error=zero,output_tolerance=[1e-9,1e-9],
            licensed_component_edits=["controller","clip"],
            hard_clip_caps=[3.,3.])
        self.assertIsNone(res["selected"])

    def test_bad_input_rejected_instead_of_faking_safety(self):
        n=7;I=_identity(n);zero=[[0.]*n for _ in range(n)]
        kw=dict(producer=I,controller=I,action_radii=[2.]*n,
                internal_clip_limits=[2.1]*n,producer_abs_error=zero,
                controller_abs_error=zero,output_tolerance=[1.]*n)
        for key,bad in (("producer",[[1.,0.]]),
                        ("controller",[[float("nan")]*n for _ in range(n)]),
                        ("action_radii",[2.]*6),
                        ("producer_abs_error",[[-.1]*n for _ in range(n)]),
                        ("internal_clip_limits",[0.]*n),
                        ("output_tolerance",[float("nan")]*n)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                certify_multichannel_robust_repair(**{**kw,key:bad})

    def test_duplicate_or_under_authorized_patch_cannot_win_catalog(self):
        n=2;I=_identity(n);z=[[0,0],[0,0]]
        c=[dict(name="x",producer=I,controller=I,
                clip=[2.01,2.01],edits=["controller"],cost=0)]
        def solve(candidates):
            return choose_authority_constrained_patch(
                catalog=candidates,action_radii=[2,2],
                producer_abs_error=z,controller_abs_error=z,
                output_tolerance=[1e-8,1e-8],
                licensed_component_edits=[],hard_clip_caps=[3,3])
        self.assertIsNone(solve(c)["selected"])
        with self.assertRaisesRegex(ValueError,"duplicate"):
            solve(c+c)

if __name__=="__main__":
    unittest.main()
