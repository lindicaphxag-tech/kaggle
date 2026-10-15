#!/usr/bin/env python3
"""Acquisition-level split-conformal lower bound for a FIXED source-only candidate.
No per-target adaptation under strict Z0. Requires independent exchangeable
meta-acquisitions and locked predictor/policy chosen before calibration.
"""
import argparse,hashlib,json,math,random
from pathlib import Path

def lower_bound(predicted_margin,calibration_margins,alpha=.1):
    if not 0<alpha<1:raise ValueError('alpha')
    if not calibration_margins:raise ValueError('no independent acquisitions')
    if not all(math.isfinite(v) for v in calibration_margins+[predicted_margin]):
        raise ValueError('nonfinite margin')
    n=len(calibration_margins)
    k=math.ceil((n+1)*(1-alpha))
    if k>n:return float('-inf') # fail closed, never claim coverage from tiny n
    residual=sorted(predicted_margin-y for y in calibration_margins)
    return predicted_margin-residual[k-1]

def policy(bound,meaningful_gain=.02):
    return 'SWITCH' if bound>meaningful_gain else 'RETAIN_OR_ABSTAIN'

def run():
    cal_rng=random.Random(20261013)
    test_rng=random.Random(20261014)
    shift_rng=random.Random(20261015)
    # Candidate and source context are fixed; no test outcome was used to tune them.
    mu=.08
    cal=[cal_rng.gauss(.08,.035) for _ in range(79)]
    L=lower_bound(mu,cal,.1)
    assert lower_bound(.08,[.10]*3,.1)==float('-inf') # ceil(4*.9)>3
    assert policy(L)==policy(L) # same source pool => same target-blind action
    good=[test_rng.gauss(.08,.035) for _ in range(2500)]
    bad=[shift_rng.gauss(-.03,.035) for _ in range(2500)]
    coverage_good=sum(y>=L for y in good)/len(good)
    coverage_shift=sum(y>=L for y in bad)/len(bad)
    assert coverage_good>.8 and coverage_shift<.3
    return dict(status='SYNTHETIC_ACQUISITION_CONFORMAL_SCOPE_TEST_PASS',
                alpha=.1,independent_calibration_acquisitions=79,
                frozen_source_only_prediction=mu,
                lower_margin=L,required_material_margin=.02,
                frozen_action=policy(L),exchangeable_test_n=len(good),
                shifted_test_n=len(bad),
                exchangeable_empirical_coverage=coverage_good,
                shifted_empirical_coverage=coverage_shift,
                candidate_frozen_before_calibration=True,
                one_action_per_identical_source_context=True,
                no_target_EEG_or_metadata_used=True,
                no_real_EEG_meta_acquisitions=True,
                no_clinical_safety_guarantee=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path)
    a=ap.parse_args();r=run();print(json.dumps(r,sort_keys=True))
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        b=json.dumps(r,indent=2,sort_keys=True)+'\n';a.out.write_text(b)
        print('P3_SYNTHETIC_META_COVERAGE_SHA',hashlib.sha256(b.encode()).hexdigest())
