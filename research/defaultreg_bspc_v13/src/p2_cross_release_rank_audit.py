#!/usr/bin/env python3
"""Post-discovery numerical PRODUCT-association audit; no person provenance claims.

Reads the frozen 6x6 scalar table from the existing public experimental run.
The table derives from fit-half / score-half same-trial comparisons, not from
independent acquisitions, and is ineligible for FPR/PPV or clinical identity.
"""
import argparse
import hashlib
import json
from pathlib import Path

IDS = (4, 5, 11, 15, 47, 49)
EXPLORATORY_EDF_PAIRS = ((4, 5), (11, 15), (47, 49))
SCORE = 'excluded_stored_index_32_median_of_31_channel_heldout_medians'


def audit(source):
    rows = source['six_by_six_pairs']
    if len(rows) != len(IDS)**2:
        raise ValueError('Incomplete six-by-six table')
    matrix = {}
    for row in rows:
        k = (row['source_mat_public_id'], row['processed_edf_public_id'])
        if k in matrix or k[0] not in IDS or k[1] not in IDS:
            raise ValueError('Duplicate or out-of-scope source/destination ID')
        x = float(row[SCORE])
        if not (0 <= x <= 1) or row['excluded_stored_index_32_sum_ge_0p9999'] != 0:
            raise ValueError('Invalid score or exclusion sensitivity')
        matrix[k] = x
    if set(matrix) != {(s, d) for s in IDS for d in IDS}:
        raise ValueError('Incomplete source/destination Cartesian product')
    results = []
    for dest in IDS:
        ordered = sorted(((src, matrix[src, dest]) for src in IDS),
                         key=lambda item: (-item[1], item[0]))
        results.append({
            'edf_public_id': dest,
            'largest_candidate_mat_public_id': ordered[0][0],
            'largest_posthoc_channel_median_abs_r': ordered[0][1],
            'runnerup_candidate_mat_public_id': ordered[1][0],
            'runnerup_posthoc_channel_median_abs_r': ordered[1][1],
            'posthoc_rank_margin': ordered[0][1] - ordered[1][1],
            'all_source_candidates_in_rank_order': [
                {'mat_public_id': src, 'channel_median_abs_r': score}
                for src, score in ordered],
        })
    winners = {r['edf_public_id']: r['largest_candidate_mat_public_id'] for r in results}
    pair_agreements = [
        {'edf_public_ids': list(pair), 'winner_source_ids': [winners[x] for x in pair],
         'same_best_candidate': winners[pair[0]] == winners[pair[1]]}
        for pair in EXPLORATORY_EDF_PAIRS
    ]
    return {
        'status': 'POST_DISCOVERY_WITHIN_SIX_SOURCE_NUMERICAL_RANK_DIAGNOSTIC',
        'all_36_source_by_edf_pairs_checked': True,
        'exclude_stored_channel_32': True,
        'same_trial_time_split_not_independent_acquisition': True,
        'selected_sources_posthoc_not_unbiased_null_controls': True,
        'not_MAT_to_EDF_transform_proof': True,
        'not_verified_raw_signal_ancestry': True,
        'not_patient_or_acquisition_identity': True,
        'not_calibrated_classification_FPR_or_PPV': True,
        'results': results,
        'edf_pair_agreements': pair_agreements,
    }


def self_test():
    rows = []
    for j, dest in enumerate(IDS):
        for i, source in enumerate(IDS):
            rows.append({'source_mat_public_id': source,
                         'processed_edf_public_id': dest,
                         SCORE: 0.8 if i == j else (i + 1) / 100,
                         'excluded_stored_index_32_sum_ge_0p9999': 0})
    assert len(audit({'six_by_six_pairs': rows})['results']) == 6
    for bad in (rows[:-1], rows + [rows[-1]]):
        try:
            audit({'six_by_six_pairs': bad})
        except ValueError:
            pass
        else:
            raise AssertionError('Fail-closed input validation did not trigger')
    print('P2_POSTHOC_RANK_SELFTEST_PASS')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.input:
        if args.output is None:
            parser.error('--output required with --input')
        data = args.input.read_bytes()
        result = audit(json.loads(data))
        result['input_sha256'] = hashlib.sha256(data).hexdigest()
        serialized = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print('P2_POSTHOC_RANK_SHA256', hashlib.sha256(serialized.encode()).hexdigest())
        for r in result['results']:
            print('EDF', r['edf_public_id'], 'top MAT', r['largest_candidate_mat_public_id'],
                  'margin', f"{r['posthoc_rank_margin']:.6f}")


if __name__ == '__main__':
    main()
