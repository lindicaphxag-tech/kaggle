# V22 real-source partial execution receipt — BNCI2014_002

Date: 2026-10-12.
Scope: ONE outer held-person pool (`outer_person=1`) on real BNCI2014_002, with 13 other people used for source selection and grouped source-person validation.
This is retrospective source-only *mechanism* evidence, NOT target AUC, NOT an independent prospective confirmation, and NOT a target risk prediction.

- Workflow run: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38272783451
- Job: 114890868562; conclusion: success
- Artifact ID: 11757826713, name: defaultreg-v22-real-source-rank-bnci002-outer1-posthoc
- Reproduced held-source macro AUC delta (source-discrimination-selected minus historical class-centred-covariance-selected lambda): **+0.046250000000000006**
- Per-held-source participant mean gross pair-state churn: **0.15346153846153845**
- JSON report SHA-256: `e7a2c814d944eb91716d61ed984b6aec484981b5669ace85b11144a640e37988`
- 8 source-rank mechanistic/provenance unit tests: passed.
- Exact ROC-AUC reconstruction checked against untouched `source_path` of the old EEG runner.
- Outer target labels were not read by the diagnostic, though the old data loader processes the full cohort before source slicing. Strict physical separation is not claimed.

**Do not quote these source-fold metrics as improvements on an unseen target participant.** A full 14-person source-pool audit has its own run and result gate. The code has no new generalisation selector or target AUC promotion here.
