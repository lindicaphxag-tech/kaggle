#!/usr/bin/env python3
"""P3 candidate: conservative conditional positive-transfer source-group evidence gate.
This is a descriptive 2-standard-error source-only rule, NOT a target-risk guarantee.
"""
import numpy as np
def source_gain_gate(paired_independent_source_group_auc_lifts):
    a=np.asarray(paired_independent_source_group_auc_lifts,float).reshape(-1)
    if len(a)<6 or not np.isfinite(a).all() or np.any(abs(a)>1):raise ValueError('6+ independent paired source groups required')
    est=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)))
    lower=est-2*se
    return dict(source_gain=est,source_gain_se=se,descriptive_lower=lower,
       enable_transfer=bool(lower>0),n_source_groups=len(a),
       caveat='no target distribution-free bound; meta-gate remains unvalidated in real EEG')
if __name__=='__main__':
    assert source_gain_gate([.08,.06,.07,.09,.08,.07])['enable_transfer']
    assert not source_gain_gate([-.07,.04,.01,-.06,.02,.01])['enable_transfer']
    print('SOURCE_TRANSFER_GATE_SMOKE_PASS (synthetic)')
