# P1 exact finite-sample score-versus-AUC counterexample — EXECUTED
- CI success: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38312790835
- Immutable generated JSON: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38312790835/artifacts/11775128649
- Source: `research/defaultreg_bspc_v13/src/p1_fixed_auc_predictive_penalty_counterexample.py`
- Exact per-person linear score AUC stays identical under common and class-specific shifts in a score-orthogonal direction, while proper predictive minus validation-class-centred score increases.
- Nonnegative three-factor source score difference verified algebraically to 1e-10 in the fixed 2D construction.
- NOT a new EEG acquisition, target generalization, a corrected source selector win, or a clinical finding.
- Publication role: rigorous didactic counterexample and executable mechanism test for P1's already empirically audited EEG case. JNE author and scientific release gates still apply.
