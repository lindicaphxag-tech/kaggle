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
from .validation_only_margin_control import (
    fit_validation_margin_threshold, audit_test_predictions,
)
from .electrode_ambiguity_certificate import (
    certified_linear_prediction, sklearn_pipeline_raw_weights,
)

AMBIGUITY_PAIRS = (("C1", "C3"), ("C2", "C4"))


def apply_predeclared_pair_swaps(source, groups, *, seed):
    """Fixed-seed uncertain observed slots; no access to labels or model output."""
    out=np.asarray(source,dtype=float).copy()
    rng=np.random.default_rng(seed)
    traces=[]
    for row in out:
        flipped=[]
        for group in groups:
            change=bool(rng.integers(0,2))
            if change:
                row[list(group)]=row[list(group)[::-1]].copy()
            flipped.append(change)
        traces.append(flipped)
    return out, traces


def certify_cohort(X, y, subjects, names, *, cohort='pilot'):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    subjects = np.asarray(subjects, dtype=int)
    split = SPLITS[cohort]
    train = np.isin(subjects, split['train'])
    val = np.isin(subjects, split['val'])
    test = np.isin(subjects, split['test'])
    if set(subjects) != set(split['train'] + split['val'] + split['test']):
        raise ValueError('Missing preregistered subjects; abort')
    if not train.any() or not val.any() or not test.any() or set(y[train]) != {0, 1}:
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
    x_test, flip_bits = apply_predeclared_pair_swaps(X[test], groups, seed=SEED + 101)
    x_val, val_flip_bits = apply_predeclared_pair_swaps(X[val], groups, seed=SEED + 202)
    # For fairness the confidence-only acceptance threshold is selected using
    # validation subjects and validation certificate coverage ONLY.
    val_certificates = [certified_linear_prediction(row, w, b, groups) for row in x_val]
    val_accepted = np.array([c.authorized for c in val_certificates],dtype=bool)
    val_margins = np.abs(classifier.decision_function(x_val.reshape(len(x_val),-1)))
    fit = fit_validation_margin_threshold(val_margins, val_accepted)
    test_margins = np.abs(classifier.decision_function(x_test.reshape(len(x_test),-1)))
    truth = y[test]
    naive = classifier.predict(x_test.reshape(len(x_test), -1))
    confidence_control = audit_test_predictions(test_margins, naive, y[test], fit.threshold)
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
        'validation_only_confidence_baseline': {
            'threshold': fit.threshold if np.isfinite(fit.threshold) else None,
            'threshold_is_infinite': not np.isfinite(fit.threshold),
            'target_validation_coverage': fit.requested_validation_coverage,
            'actual_validation_coverage': fit.achieved_validation_coverage,
            'ties_prevent_exact_match': fit.ties_prevent_exact_coverage,
            'comparison': confidence_control,
            'calibration_subjects': list(map(int,split['val'])),
            'no_test_threshold_tuning': True,
        },
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
            'confidence_margin': test_margins.tolist(),
            'confidence_baseline_accepted': confidence_control['accepted'],
            'margin_lower': [c.lower_margin for c in certificates],
            'margin_upper': [c.upper_margin for c in certificates],
        },
        'limits': [
            'Certified assignment-invariance, NOT correct task prediction',
            'Not a new theorem; exact Hungarian assignment is established',
            'Invalid if the ambiguity set misses a possible true mapping',
            'Selective risk may exceed naive full-coverage risk: report both',
            'Confidence baseline threshold calibrated on validation subjects only',
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
