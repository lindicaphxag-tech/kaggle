import sys
import unittest
from pathlib import Path
import numpy as np

SRC=Path(__file__).resolve().parents[1]/"src"
sys.path.insert(0,str(SRC))

from regularisation_path_geometry import (
    analyse_path_geometry,
    covariance_predictive_nll,
    finite_difference_check,
    standardized_margin,
)


class RegularisationPathGeometryTests(unittest.TestCase):
    def setUp(self):
        self.S=np.array([
            [2.0,0.35,0.10],
            [0.35,0.8,0.06],
            [0.10,0.06,0.25],
        ])
        self.Sigma=np.array([
            [1.7,0.25,0.08],
            [0.25,1.1,0.12],
            [0.08,0.12,0.45],
        ])
        self.ds=np.array([0.9,-0.45,0.25])
        self.de=np.array([0.65,-0.15,0.55])

    def test_exact_derivatives_match_central_finite_difference(self):
        out=finite_difference_check(
            self.S,self.ds,self.Sigma,self.de,0.37,h=1e-6
        )
        self.assertLess(out["covariance_abs_error"],2e-7)
        self.assertLess(out["margin_abs_error"],2e-7)

    def test_rotation_is_sigma_orthogonal_to_discriminant(self):
        g=analyse_path_geometry(
            self.S,self.ds,self.Sigma,self.de,0.42
        )
        scale=max(1.0,g.w_sigma_norm*g.rotation_sigma_norm)
        self.assertLess(
            abs(g.sigma_orthogonality_residual),
            5e-12*scale,
        )

    def test_covariance_objective_ignores_class_contrast(self):
        a=analyse_path_geometry(
            self.S,self.ds,self.Sigma,self.de,0.31
        )
        b=analyse_path_geometry(
            self.S,-3*self.ds,self.Sigma,7*np.array([0.1,0.8,-0.2]),0.31
        )
        self.assertAlmostEqual(
            a.covariance_nll_derivative,
            b.covariance_nll_derivative,
            places=13,
        )
        self.assertAlmostEqual(a.covariance_nll,b.covariance_nll,places=13)
        self.assertNotAlmostEqual(
            a.standardized_margin_derivative,
            b.standardized_margin_derivative,
            places=6,
        )

    def test_flipping_evaluation_contrast_flips_margin_derivative_only(self):
        a=analyse_path_geometry(
            self.S,self.ds,self.Sigma,self.de,0.28
        )
        b=analyse_path_geometry(
            self.S,self.ds,self.Sigma,-self.de,0.28
        )
        self.assertAlmostEqual(
            a.covariance_nll_derivative,
            b.covariance_nll_derivative,
            places=13,
        )
        self.assertAlmostEqual(
            a.standardized_margin_derivative,
            -b.standardized_margin_derivative,
            places=13,
        )
        self.assertAlmostEqual(
            a.standardized_margin,
            -b.standardized_margin,
            places=13,
        )

    def test_single_eigendirection_changes_scale_not_margin_direction(self):
        S=np.diag([0.3,1.0,2.7])
        Sigma=np.diag([0.5,1.3,2.0])
        ds=np.array([1.0,0.0,0.0])
        de=np.array([0.7,0.0,0.0])
        g=analyse_path_geometry(S,ds,Sigma,de,0.55)
        self.assertLess(g.rotation_sigma_norm,1e-12)
        self.assertLess(abs(g.standardized_margin_derivative),1e-12)
        self.assertNotAlmostEqual(g.scale_rate,0.0,places=8)

    def test_isotropic_source_covariance_has_zero_path_motion(self):
        S=2.4*np.eye(4)
        Sigma=np.diag([1.0,1.2,1.4,1.6])
        ds=np.array([1.0,-0.4,0.3,0.2])
        de=np.array([0.4,0.3,-0.2,0.1])
        g=analyse_path_geometry(S,ds,Sigma,de,0.73)
        self.assertLess(abs(g.scale_rate),1e-14)
        self.assertLess(g.rotation_sigma_norm,1e-14)
        self.assertLess(abs(g.standardized_margin_derivative),1e-14)
        self.assertLess(abs(g.covariance_nll_derivative),1e-14)

    def test_public_functions_reject_non_spd_input(self):
        bad=np.array([[1.0,2.0],[2.0,1.0]])
        with self.assertRaises(ValueError):
            covariance_predictive_nll(bad,np.eye(2),0.3)
        with self.assertRaises(ValueError):
            standardized_margin(
                np.eye(2),np.ones(2),bad,np.ones(2),0.3
            )

    def test_status_forbids_selector_interpretation(self):
        g=analyse_path_geometry(
            self.S,self.ds,self.Sigma,self.de,0.2
        )
        self.assertFalse(g.target_information_used_for_selection)
        self.assertIn("ANALYSIS_ONLY",g.status)


if __name__=="__main__":
    unittest.main()
