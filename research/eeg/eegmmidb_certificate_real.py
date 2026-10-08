"""Real-data secondary assay for exact linear electrode-identity uncertainty.

Reuses the SAME preregistered subject splits and spectral features as
eegmmidb_real_pilot.py. Anatomical ambiguity pairs frozen before outcomes.
Certifies invariance to specified identity assignments, NOT task correctness.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .eegmmidb_real_pilot import load_real_data, SPLITS, SEED
from .electrode_ambiguity_certificate import (
    certified_linear_prediction, sklearn_pipeline_raw_weights,
)

AMBIGUITY_PAIRS = (("C1", "C3"), ("C2", "C4"))


def certify_cohort(X, y, subjects, names, *, cohort='pilot'):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    subjects = np.asarray(subjects, dtype=int)
    split = SPLITS[cohort]
    train = np.isin(subjects, split['train'])
    test = np.isin(subjects, split['test'])
    if set(subjects) != set(split['train'] + split['val'] + split['test']):
        raise ValueError('Missing preregistered subjects; abort')
    if not train.any() or not test.any() or set(y[train]) != {0, 1}:
        raise ValueError('Bad split or labels')
    if X.shape != (len(y), len(names), 2):
        raise ValueError('Unexpected feature array')
    name_ix = {name: i for i, name in enumerate(names)}
    if any(name not in name_ix for pair in AMBIGUITY_PAIRS for name in pair):
        raise ValueError('Missing predeclared anatomical ambiguity electrode')
    groups = tuple(tuple(name_ix[name] for name in pair) for pair in AMBIGUITY_PAIRS)
    if len({i for group in groups for i in group}) != 4:
        raise ValueError('Ambiguity groups overlap unexpectedly')
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    )
    classifier.fit(X[train].reshape(int(train.sum()), -1), y[train])
    w, b, classes = sklearn_pipeline_raw_weights(classifier, len(names), 2)
    x_test = X[test].copy()
    rng = np.random.default_rng(SEED + 101)
    flip_bits = []
    for row in x_test:
        flips = []
        for group in groups:
            flip = bool(rng.integers(0, 2))
            if flip:
                row[list(group)] = row[list(group)[::-1]].copy()
            flips.append(flip)
        flip_bits.append(flips)
    truth = y[test]
    naive = classifier.predict(x_test.reshape(len(x_test), -1))
    known_identity = classifier.predict(X[test].reshape(len(x_test), -1))
    certificates = [
        certified_linear_prediction(row, w, b, groups) for row in x_test
    ]
    accepted = np.array([c.authorized for c in certificates], dtype=bool)
    predictions = np.array(
        [classes[c.label] if c.authorized else -1 for c in certificates]
    )
    # A valid mapping certificate must not disagree with known correct ordering;
    # this says NOTHING about task labels being predicted correctly.
    if np.any(predictions[accepted] != known_identity[accepted]):
        raise AssertionError('Certified mapping disagreement with known-identity oracle')

    def acc_on(mask, pred):
        return float(np.mean(pred[mask] == truth[mask])) if np.any(mask) else None

    by_subject = []
    for sid in split['test']:
        selected = subjects[test] == sid
        by_subject.append({
            'subject': int(sid), 'n': int(selected.sum()),
            'coverage': float(np.mean(accepted[selected])),
            'accepted_accuracy': acc_on(selected & accepted, predictions),
            'naive_accuracy': acc_on(selected, naive),
            'known_identity_accuracy': acc_on(selected, known_identity),
        })
    all_test = np.ones(len(truth), dtype=bool)
    return {
        'cohort': cohort,
        'ambiguity_pairs': AMBIGUITY_PAIRS,
        'n_test': int(test.sum()),
        'coverage': float(accepted.mean()),
        'certified_accepted_accuracy': acc_on(accepted, predictions),
        'naive_accuracy': acc_on(all_test, naive),
        'known_identity_accuracy': acc_on(all_test, known_identity),
        'false_invariance_certificates': int(np.sum(predictions[accepted] != known_identity[accepted])),
        'abstentions': int(np.sum(~accepted)),
        'per_subject': by_subject,
        'conditional_on': 'All true channel identities lie within stated disjoint pairs',
        'per_trial': {
            'labels': truth.tolist(),
            'subjects': subjects[test].tolist(),
            'flips': flip_bits,
            'accepted': accepted.tolist(),
            'prediction': predictions.tolist(),
            'naive': naive.tolist(),
            'known_identity': known_identity.tolist(),
            'margin_lower': [c.lower_margin for c in certificates],
            'margin_upper': [c.upper_margin for c in certificates],
        },
        'limits': [
            'Certified assignment-invariance, NOT correct task prediction',
            'Not a new theorem; exact Hungarian assignment is established',
            'Invalid if the ambiguity set misses a possible true mapping',
            'Selective risk may exceed naive full-coverage risk: report both',
            'This is not a pretrained EEG foundation model',
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', choices=SPLITS.keys(), default='pilot')
    ap.add_argument('--data-dir', type=Path, default=Path('./mne_data'))
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    cohort = SPLITS[args.cohort]
    X, y, s, names, xyz, files = load_real_data(
        cohort['train'] + cohort['val'] + cohort['test'], args.data_dir
    )
    result = certify_cohort(X, y, s, names, cohort=args.cohort)
    result['input_edf_sha256'] = files
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('ELECTRODE_CERTIFICATE_REAL_EEG', json.dumps({
        'coverage': result['coverage'],
        'accepted_accuracy': result['certified_accepted_accuracy'],
        'naive_accuracy': result['naive_accuracy'],
        'known_identity_accuracy': result['known_identity_accuracy'],
        'abstentions': result['abstentions'],
    }))


if __name__ == '__main__':
    main()
