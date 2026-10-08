"""Minimal synthetic electrode-ID contract witness; NOT a real EEG accuracy result."""
import numpy as np

def align_channels(x, incoming, expected):
    incoming, expected = tuple(incoming), tuple(expected)
    if len(incoming) != x.shape[0] or len(set(incoming)) != len(incoming):
        raise ValueError("Input electrode identities missing or duplicated")
    if len(set(expected)) != len(expected) or set(incoming) != set(expected):
        raise ValueError("Expected and observed electrode sets differ")
    return x[[incoming.index(k) for k in expected]]

def score(x, weights):
    return float(np.dot(weights, x.mean(axis=1)))

def run():
    rng = np.random.default_rng(314159)
    labels = ("F3", "C3", "P3", "O1")
    x = rng.normal(size=(4, 256))
    x += np.array([2., -1., 0.5, -2.])[:, None]
    weights = np.array([0.7, -0.9, 1.2, 0.4])
    permutation = np.array([3, 0, 2, 1])
    shuffled = x[permutation]
    shuffled_labels = tuple(labels[j] for j in permutation)
    baseline = score(x, weights)
    shape_only = score(shuffled, weights)
    corrected = score(align_channels(shuffled, shuffled_labels, labels), weights)
    assert not np.isclose(baseline, shape_only)
    np.testing.assert_allclose(corrected, baseline, atol=1e-12, rtol=0)
    for bad in (("F3", "F3", "P3", "O1"), ("F3", "C3", "P3", "F7")):
        try:
            align_channels(x, bad, labels)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid channel provenance accepted")
    return dict(baseline=baseline, shape_only=shape_only, corrected=corrected)

if __name__ == "__main__":
    print(run())
