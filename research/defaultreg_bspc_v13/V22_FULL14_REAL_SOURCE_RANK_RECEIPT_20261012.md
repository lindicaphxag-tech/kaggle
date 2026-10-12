# V22 audited all-outer genuine EEG source-only receipt (BNCI2014_002)

Date: 2026-10-12. Status: **Executed successfully**, **retrospective**, **source-validation only**.

- [Full GitHub Actions workflow run](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38273307665): completed successfully.
- Source experiment: **14 outer held-person source pools**, 13 source persons in each outer pool, grouped fivefold source-person validation and exact original covariance/tangent LDA.
- Historical class-centred covariance-score lambda -> source macro AUC-selected lambda:
  - **+0.026796875** AUC averaged over 14 source pool reports;
  - mean gross positive-negative pair sign-state transitions **0.15314474587912086**;
  - 14-fold first outer source pool: +0.046250000000000006 AUC, gross churn 0.15346153846153845.
- Outcome JSON SHA-256: `15452a65fe43bb6f8e26f9f87deb65d73cec3a605966fa0060ee49805d72992f`.
- Run artifact ID: `11756734894` (`defaultreg-v22-real-source-rank-bnci002-all14-posthoc`).
- Frozen original source score path macro AUC verified to 1e-10 within each outer fold; pair credit/AUC identity verified to 1e-12; source/target scope guard and unit tests passed.

## Scientific and chronology limitations

The 14 source pools share training people; **they are not independent studies and no independent-fold p-value is calculated**. Source labels are used (permissibly) for held-SOURCE AUC estimates. Held outer target labels were not used in the rank diagnostic, which is a retrospective post-V21 mechanism experiment; the original raw loader may initially read entire EEG sessions before source slices are constructed. It is not a new target-future AUC improvement, not a newly blinded external acquisition, and not a demonstration that a novel regularisation selector works.

No claims of generalised superiority, formal preregistration or future-person risk certificates may be inferred from this receipt.
