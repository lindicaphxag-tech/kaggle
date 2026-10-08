"""Adversarial exactness and classifier parity tests for electrode ambiguity."""
import itertools
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.eeg.electrode_ambiguity_certificate import (
    certified_linear_prediction,
    sklearn_pipeline_raw_weights,
)


class TestAssignmentCertificate(unittest.TestCase):
    def test_exact_bounds_match_brute_force(self):
        rng = np.random.default_rng(99)
        for _ in range(30):
            X = rng.normal(size=(6, 2))
            W = rng.normal(size=(6, 2))
            groups = ((0, 1, 2), (4, 5))
            cert = certified_linear_prediction(X, W, -0.13, groups)
            margins = []
            for p in itertools.permutations(groups[0]):
                for q in itertools.permutations(groups[1]):
                    z = X.copy()
                    z[list(groups[0])] = X[list(p)]
                    z[list(groups[1])] = X[list(q)]
                    margins.append(-0.13 + float(np.sum(z * W)))
            self.assertAlmostEqual(cert.lower_margin, min(margins), places=11)
            self.assertAlmostEqual(cert.upper_margin, max(margins), places=11)
            self.assertEqual(cert.permutations_examined_implicitly, 12)

    def test_unchanged_identity_singleton(self):
        X = np.array([[1.], [2.]])
        W = np.array([[5.], [3.]])
        cert = certified_linear_prediction(X, W, 0, ())
        self.assertTrue(cert.authorized)
        self.assertEqual(cert.label, 1)
        self.assertEqual(cert.lower_margin, 11)

    def test_impossible_certification_abstains(self):
        X = np.array([[1.], [-1.]])
        W = np.array([[1.], [-1.]])
        cert = certified_linear_prediction(X, W, 0, ((0, 1),))
        self.assertFalse(cert.authorized)
        self.assertLess(cert.lower_margin, 0)
        self.assertGreater(cert.upper_margin, 0)

    def test_full_unconditional_certification(self):
        X = np.array([[3.], [2.]])
        W = np.array([[1.], [1.]])
        cert = certified_linear_prediction(X, W, -1, ((0, 1),))
        self.assertTrue(cert.authorized)
        self.assertEqual(cert.label, 1)

    def test_reject_invalid_ambiguity(self):
        X = np.ones((4, 2))
        for groups in (((0, 1), (1, 2)), ((0, 5),), ((2, 2),)):
            with self.assertRaises(ValueError):
                certified_linear_prediction(X, X, 0, groups)

    def test_fitted_sklearn_model_equivalence(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 6))
        y = (X[:, 0] - X[:, 3] > .1).astype(int)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, random_state=7))
        model.fit(X, y)
        w, b, labels = sklearn_pipeline_raw_weights(model, 3, 2)
        for x in X[:20]:
            cert = certified_linear_prediction(x.reshape(3, 2), w, b, ())
            self.assertAlmostEqual(
                cert.lower_margin,
                float(model.decision_function([x])[0]),
                places=9,
            )
            self.assertEqual(labels[cert.label], model.predict([x])[0])


if __name__ == '__main__':
    unittest.main()
