"""Adversarial regression: model validity is not guaranteed by three taps."""
import itertools
import unittest
from research.semrepair.deployment_validity_shield import (
    EXTENDED_PROBES, I, CHECK, PROBE_E1, PROBE_E2,
    TapSample, nominal_probe, nonlinear_tap,
    validate_numeric_model, requested_patch_if_finite_compatible,
    cartesian_cover_radius, conditional_lipschitz_envelope,
    nonlinear_probe_challenge, cubic_ambush,
    finite_probe_interpolation_counterexample, bounded_noise_patch_contract,
    matched_seven_probe_baseline_challenge,
    cross_axis_equal_cost_falsification,
)


class DeploymentValidityTests(unittest.TestCase):
    def test_three_legacy_probes_are_identical_under_adversarial_cubic(self):
        for axis in (0, 1):
            for u in (PROBE_E1,PROBE_E2,CHECK):
                self.assertLess(max(abs(a-b) for a,b in zip(
                    cubic_ambush(u,strength=0.5,axis=axis),u)),1e-12)
            for u in EXTENDED_PROBES:
                if axis==0 and u[0]==0 or axis==1 and u[1]==0:
                    continue
                self.assertGreater(max(abs(a-b) for a,b in zip(
                    cubic_ambush(u,strength=0.5,axis=axis),u)),1e-4)

    def test_actual_native_mujoco_cubic_breaks_finite_linear_claim(self):
        r=nonlinear_probe_challenge()
        self.assertEqual(r["dependent_parameter_cases"],12)
        self.assertEqual(r["legacy_false_linear_acceptances"],12)
        self.assertEqual(r["extra_probe_detected_nonlinear"],12)
        self.assertEqual(r["nominal_probe_count"],3)
        self.assertEqual(r["extended_probe_count"],7)

    def test_extra_nominal_probe_rejects_patch_candidate(self):
        t=[nonlinear_tap(u,strength=.1,axis=0) for u in (PROBE_E1,PROBE_E2)]
        old=requested_patch_if_finite_compatible(
            t,[nonlinear_tap(CHECK,strength=.1,axis=0)])
        self.assertEqual(old["candidate_patch"],"noop")
        new=requested_patch_if_finite_compatible(
            t,[nonlinear_tap(u,strength=.1,axis=0)
               for u in (CHECK,)+EXTENDED_PROBES])
        self.assertIsNone(new["candidate_patch"])
        self.assertEqual(new["decision"],"REFUSE_LINEAR_MODEL_FALSIFIED")

    def test_true_linear_bounded_perturbations_do_not_falsify_null(self):
        eps=.0005
        def perturbed(u, jitter):
            clean=nominal_probe(I,I,u)
            return TapSample(u,tuple(v+jitter*eps for v in clean.producer),
                             tuple(v-jitter*eps for v in clean.controller))
        # Different fixed errors at training and validation are allowed.
        train=(perturbed(PROBE_E1,1),perturbed(PROBE_E2,-1))
        valid=[perturbed(CHECK,1),perturbed((1.2,-1.2),-1)]
        verdict=validate_numeric_model(train,valid,tap_noise_bound=eps)
        self.assertEqual(verdict["decision"],"FINITE_PROBES_COMPATIBLE_ONLY")
        wrong=validate_numeric_model(train,valid,tap_noise_bound=0)
        self.assertEqual(wrong["decision"],"REFUSE_LINEAR_MODEL_FALSIFIED")

    def test_exact_square_mesh_cover_radius(self):
        mesh=[(float(x),float(y)) for x,y in itertools.product((-1.,0.,1.),repeat=2)]
        self.assertEqual(cartesian_cover_radius(mesh,lower=-1,upper=1),0.5)
        with self.assertRaisesRegex(ValueError,"incomplete"):
            cartesian_cover_radius(mesh[:-1],lower=-1,upper=1)
        with self.assertRaisesRegex(ValueError,"outside"):
            cartesian_cover_radius(mesh,lower=-.5,upper=.5)

    def test_no_external_regularity_must_refuse_global_certificate(self):
        mesh=[(float(x),float(y)) for x,y in itertools.product((-1.,0.,1.),repeat=2)]
        result=conditional_lipschitz_envelope([0.]*9,mesh,lower=-1,upper=1,
                        externally_verified_lipschitz=None,tolerance=1e-4)
        self.assertEqual(result["decision"],"REFUSE_NO_EXTERNAL_REGULARITY_BOUND")
        self.assertFalse(result["certified"])

    def test_conditional_global_certificate_is_tight_for_linear_residual(self):
        mesh=[(float(x),float(y)) for x,y in itertools.product((-1.,0.,1.),repeat=2)]
        accepts=conditional_lipschitz_envelope([0.]*9,mesh,lower=-1,upper=1,
                    externally_verified_lipschitz=0.,tolerance=1e-9)
        self.assertTrue(accepts["certified"])
        refuses=conditional_lipschitz_envelope([0.]*9,mesh,lower=-1,upper=1,
                    externally_verified_lipschitz=2.,tolerance=.1)
        self.assertFalse(refuses["certified"])
        self.assertAlmostEqual(refuses["upper_residual"],1.)

    def test_even_all_seven_probes_cannot_globally_certify_arbitrary_nonlinearity(self):
        inputs=(PROBE_E1,PROBE_E2,CHECK)+EXTENDED_PROBES
        witness=finite_probe_interpolation_counterexample(inputs,evaluation=(2.,.4))
        self.assertEqual(witness["probed_commands"],7)
        self.assertEqual(witness["maximum_training_discrepancy"],0.)
        self.assertGreater(witness["unqueried_discrepancy"],1.)
        self.assertFalse(witness["novel_mathematical_priority"])

    def test_interpolation_witness_rejects_queried_location(self):
        with self.assertRaises(ValueError):
            finite_probe_interpolation_counterexample([PROBE_E1],evaluation=PROBE_E1)

    def test_noise_certified_patch_categories_with_native_simulator(self):
        SWAP=((0.,1.),(1.,0.))
        NEG=((-1.,0.),(0.,-1.))
        configurations=[
            (I,I,"noop"), (SWAP,I,"producer-only"),
            (I,NEG,"controller-only"), (SWAP,SWAP,"atomic"),
        ]
        for p,c,expected in configurations:
            training=[nominal_probe(p,c,u) for u in (PROBE_E1,PROBE_E2)]
            validation=[nominal_probe(p,c,u) for u in (CHECK,)+EXTENDED_PROBES]
            r=bounded_noise_patch_contract(training,validation,
                                           per_coordinate_error_bound=0.0005)
            self.assertEqual(r["decision"],"NOISY_LINEAR_PATCH_CANDIDATE_NOT_AUTHORIZED")
            self.assertEqual(r["candidate_patch"],expected)
            self.assertFalse(r["independent_source_verified"])

    def test_adversarial_coordinate_noise_within_declared_bound(self):
        eps=.00025
        def noisy(u,p,c,sign):
            clean=nominal_probe(p,c,u)
            return TapSample(u,tuple(v+sign*eps for v in clean.producer),
                             tuple(v-sign*eps for v in clean.controller))
        a,b=PROBE_E1,PROBE_E2
        training=[noisy(a,I,I,1),noisy(b,I,I,-1)]
        valid=[noisy(CHECK,I,I,1),noisy((1.2,0.),I,I,-1)]
        r=bounded_noise_patch_contract(training,valid,per_coordinate_error_bound=eps)
        self.assertEqual(r["candidate_patch"],"noop")

    def test_noisy_near_identity_must_abstain(self):
        training=[nominal_probe(I,I,u) for u in (PROBE_E1,PROBE_E2)]
        result=bounded_noise_patch_contract(training,[nominal_probe(I,I,CHECK)],
                         per_coordinate_error_bound=.03)
        self.assertEqual(result["decision"],"REFUSE_INSUFFICIENT_PRECISION")

    def test_noisy_near_singular_producer_must_abstain(self):
        weak=((1.e-5,0.),(0.,1.))
        training=[nominal_probe(weak,I,u) for u in (PROBE_E1,PROBE_E2)]
        result=bounded_noise_patch_contract(training,[nominal_probe(weak,I,CHECK)],
                         per_coordinate_error_bound=.0001)
        self.assertEqual(result["decision"],"REFUSE_NOISY_RANK_AMBIGUITY")

    def test_nonlinear_alias_is_rejected_before_noisy_classification(self):
        train=[nonlinear_tap(u,strength=.05,axis=0) for u in (PROBE_E1,PROBE_E2)]
        validation=[nonlinear_tap(u,strength=.05,axis=0)
                    for u in (CHECK,)+EXTENDED_PROBES]
        result=bounded_noise_patch_contract(train,validation,
                                            per_coordinate_error_bound=.0001)
        self.assertEqual(result["decision"],"REFUSE_LINEAR_MODEL_FALSIFIED")

    def test_equal_cost_generic_probe_control_must_be_reported(self):
        r=matched_seven_probe_baseline_challenge()
        self.assertEqual(r["constructed_nonlinear_parameter_cases"],12)
        self.assertEqual(r["detection_counts"]["historical_3"],0)
        self.assertEqual(r["detection_counts"]["extremal_7"],12)
        self.assertEqual(r["detection_counts"]["generic_7"],12)
        self.assertEqual(r["probe_budgets"]["extremal_7"],
                         r["probe_budgets"]["generic_7"])

    def test_cross_axis_fault_defeats_axis_only_probe_expansion(self):
        r=cross_axis_equal_cost_falsification()
        self.assertEqual(r["constructed_conditions"],6)
        self.assertEqual(r["detection_counts"]["three"],0)
        self.assertEqual(r["detection_counts"]["axial_seven"],0)
        self.assertEqual(r["detection_counts"]["generic_seven"],6)
        self.assertEqual(r["detection_counts"]["corner_seven"],6)
        self.assertEqual(r["costs"]["corner_seven"],r["costs"]["generic_seven"])

    def test_linear_classifier_also_requires_predeclared_fault_model(self):
        t=[nominal_probe(I,I,u) for u in (PROBE_E1,PROBE_E2)]
        r=requested_patch_if_finite_compatible(t,[nominal_probe(I,I,CHECK)])
        self.assertEqual(r["candidate_patch"],"noop")
        self.assertEqual(r["decision"],"PATCH_CANDIDATE_ONLY_NOT_AUTHORIZED")
        self.assertFalse(r["model_validity"]["global_validity_certified"])


if __name__=="__main__":
    unittest.main()
