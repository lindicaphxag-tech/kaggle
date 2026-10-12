"""V23 adversarial tests for analytical continuous-path enclosure."""
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from spectral_path_certificate_v23 import certified_path_rank_stability


class SpectralRankCertificateTests(unittest.TestCase):
    def test_single_direction_rank_never_changes(self):
        state = (np.array([2.]), np.eye(1), np.array([1.]),
                 np.zeros(1), 3.)
        X = np.array([[2.], [3.], [0.], [1.]])
        y = np.array([1, 1, 0, 0])
        r = certified_path_rank_stability(state, X, y, 0., 1., segments=1)
        self.assertEqual(r["certified_stable_fraction"], 1.)
        self.assertEqual(r["uniform_path_auc_change_bound"], 0.)
        self.assertEqual(r["observed_endpoint_auc_delta"], 0.)

    def test_root_crossing_detected_and_ambiguous(self):
        # pair contrast v=(1,-2); z(lambda)=1/(1+2*lambda)-2/(5-2*lambda)
        # crosses zero exactly at lambda=.5
        state = (np.array([1., 5.]), np.eye(2), np.ones(2),
                 np.zeros(2), 3.)
        X = np.array([[1., -2.], [0., 0.]])
        y = [1, 0]
        d = certified_path_rank_stability(state, X, y, .1, .9, segments=16)
        self.assertEqual(d["certified_stable_fraction"], 0.)
        self.assertEqual(d["observed_endpoint_pair_sign_changes"], 1)
        self.assertAlmostEqual(d["observed_endpoint_auc_delta"], -1.)

    def test_scalar_positive_rescaling_and_common_translation(self):
        state = (np.array([2., 4.]), np.eye(2), np.array([1., 0.]),
                 np.zeros(2), 3.)
        X = np.array([[2., 2.], [3., 1.], [0., 7.], [1., 4.]])
        y = [1, 1, 0, 0]
        a = certified_path_rank_stability(state, X, y, 0, 1, segments=8)
        b = certified_path_rank_stability(state, 3 * X + 50, y, 0, 1, segments=8)
        self.assertEqual(a["stable_pairs_certified"], b["stable_pairs_certified"])
        self.assertEqual(a["stable_pairs_certified"], 4)

    def test_random_continuous_grid_enclosed_by_bound(self):
        rng = np.random.default_rng(13)
        for p in (2, 3, 8, 16):
            e = np.exp(rng.normal(size=p))
            nu = e.mean()
            U, _ = np.linalg.qr(rng.normal(size=(p, p)))
            d = rng.normal(size=p)
            X = rng.normal(size=(17, p))
            y = np.r_[np.ones(8, int), np.zeros(9, int)]
            state = (e, U, d, np.zeros(p), nu)
            out = certified_path_rank_stability(
                state, X, y, .02, .93, segments=13, block_size=3
            )
            anchor = out["observed_endpoint_auc_low"]
            Z = X @ U
            zp, zn = Z[y == 1], Z[y == 0]
            zd = U.T @ d
            for lam in np.linspace(.02, .93, 71):
                w = zd / ((1-lam)*e + lam*nu)
                sc = Z @ w
                dif = sc[y == 1, None] - sc[y == 0][None, :]
                auc = np.mean((dif > 0) + .5 * (dif == 0))
                self.assertLessEqual(
                    abs(auc-anchor), out["uniform_path_auc_change_bound"]+1e-10
                )
            self.assertLessEqual(
                abs(out["observed_endpoint_auc_delta"]),
                out["uniform_path_auc_change_bound"]+1e-12
            )

    def test_more_subdivision_can_certify_more_nonroots(self):
        rng = np.random.default_rng(5)
        state = (np.array([.2, 5., 10.]), np.eye(3),
                 np.array([1., 1., 1.]), np.zeros(3), 5.)
        X = rng.normal(size=(22, 3))
        y = np.r_[np.ones(11, int), np.zeros(11, int)]
        coarse = certified_path_rank_stability(state, X, y, .1, .9, segments=1)
        fine = certified_path_rank_stability(state, X, y, .1, .9, segments=32)
        self.assertGreaterEqual(fine["certified_stable_fraction"],
                                coarse["certified_stable_fraction"] - 1e-14)

    def test_fail_closed_invalid_and_singular(self):
        state = (np.array([0., 1.]), np.eye(2),
                 np.ones(2), np.zeros(2), .5)
        X = np.array([[1., 0.], [0., 1.]])
        with self.assertRaises(ValueError):
            certified_path_rank_stability(state, X, [1, 0], 0., .4)
        with self.assertRaises(ValueError):
            certified_path_rank_stability(state, X, [1, 0], .1, .4, segments=0)
        with self.assertRaises(ValueError):
            certified_path_rank_stability(state, X, [1, 0], .4, .1)
        with self.assertRaises(ValueError):
            certified_path_rank_stability(state, X, [1, 1], .1, .4)

if __name__ == "__main__":
    unittest.main()
