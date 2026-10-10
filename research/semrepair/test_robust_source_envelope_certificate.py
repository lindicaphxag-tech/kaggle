"""Safety-adversarial interval math, independent of published SAC checkpoints."""
import itertools
import math
import random
import unittest

from research.semrepair.robust_source_envelope_certificate import (
    robust_repair_certificate,I
)


class RobustSourceEnvelopeTests(unittest.TestCase):
    def choose(self,P=I,C=I,dP=((0.,0.),(0.,0.)),
               dC=((0.,0.),(0.,0.)),R=2.,L=2.01,
               epsilon=.01,margin=0.):
        return robust_repair_certificate(
            producer=P,controller=C,
            producer_abs_error_bounds=dP,
            controller_abs_error_bounds=dC,
            full_command_radius=R,internal_clip_limit=L,
            allowed_max_output_error=epsilon,numeric_margin=margin)

    def test_declared_zero_uncertainty_identity_passes_full_box(self):
        x=self.choose(L=2.,epsilon=0.,margin=0.)
        self.assertTrue(x["certified"])
        self.assertEqual(x["robust_infinity_output_error_bound"],0.)
        self.assertEqual(x["robust_internal_clip_abs_input_bound"],2.)

    def test_robust_legitimate_drift_and_finite_error_headroom(self):
        d=((.001,.001),(.001,.001))
        x=self.choose(dP=d,dC=d,L=2.01,epsilon=.01)
        self.assertTrue(x["certified"])
        self.assertGreater(x["robust_internal_clip_abs_input_bound"],2.)
        self.assertLess(x["robust_infinity_output_error_bound"],.01)

    def test_valid_nominal_patch_refused_when_real_controller_clip_is_tight(self):
        d=((.001,.001),(.001,.001))
        x=self.choose(dP=d,dC=d,L=2.,epsilon=.01)
        self.assertEqual(x["decision"],"REFUSE_UNPROVEN_UNCERTAIN_INTERFACE_ENVELOPE")
        self.assertFalse(x["no_internal_clipping_for_any_allowed_source_drift"])

    def test_invalid_accuracy_and_deliberate_compensating_offset_refused(self):
        x=self.choose(P=((1.,.1),(0.,1.)),
                      C=((1.,-.1),(0.,1.)),L=2.6,epsilon=.0001)
        # CP=I and big clip succeeds for the known exact linear pair.
        self.assertTrue(x["certified"])
        x=self.choose(P=((1.,.1),(0.,1.)),
                      C=((1.,-.09),(0.,1.)),L=2.6,epsilon=.0001)
        self.assertFalse(x["certified"])
        self.assertFalse(x["all_command_errors_within_declared_tolerance"])

    def test_permitted_uncertainty_grid_never_exceeds_stated_supremum(self):
        # Falsify by deterministic Monte Carlo with both coefficient-box
        # corners and interior values across every corner of command box.
        P=((1.,.0),(.0,1.))
        C=((1.,0.),(0.,1.))
        dP=((.001,.0007),(.0004,.0006))
        dC=((.0008,.0005),(.0004,.001))
        contract=self.choose(P=P,C=C,dP=dP,dC=dC,R=2.,L=2.01,epsilon=.01)
        self.assertTrue(contract["certified"])
        bound=contract["robust_infinity_output_error_bound"]
        rng=random.Random(20261010)
        for trial in range(240):
            Pa=[[P[i][j]+dP[i][j]*rng.uniform(-1,1) for j in range(2)]
                for i in range(2)]
            Ca=[[C[i][j]+dC[i][j]*rng.uniform(-1,1) for j in range(2)]
                for i in range(2)]
            for u in itertools.product((-2.,-.5,0.,.5,2.),repeat=2):
                raw=[sum(Pa[i][j]*u[j] for j in range(2)) for i in range(2)]
                self.assertLessEqual(max(abs(q) for q in raw),
                                     contract["robust_internal_clip_abs_input_bound"]+1e-12)
                output=[sum(Ca[i][k]*raw[k] for k in range(2)) for i in range(2)]
                observed=max(abs(output[i]-u[i]) for i in range(2))
                self.assertLessEqual(observed,bound+1e-12)

    def test_uncertainty_sign_and_value_validation_is_fail_closed(self):
        for kwargs in (dict(dP=((-.1,0.),(0.,0.))),
                       dict(epsilon=float("nan")),
                       dict(R=0),dict(L=0),
                       dict(P=((1.,0.),(0.,))),
                       dict(dC=((float("inf"),0.),(0.,0.)))):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                self.choose(**kwargs)

    def test_signed_range_cannot_be_downgraded_without_worsening_bound(self):
        d=((.003,.003),(.003,.003))
        a=self.choose(dP=d,dC=d,L=2.,R=2.,epsilon=.03)
        b=self.choose(dP=d,dC=d,L=2.,R=.5,epsilon=.03)
        self.assertFalse(a["certified"])
        self.assertTrue(b["certified"])
        # A signed deployment must bind R to real actuator contract; using
        # lower R without permission does NOT prove full Pusher [-2,2] safety.

if __name__=="__main__":
    unittest.main()
