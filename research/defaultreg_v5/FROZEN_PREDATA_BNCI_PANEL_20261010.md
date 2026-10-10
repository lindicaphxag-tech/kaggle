# DefaultReg v5 true PRE-DATA panel freeze

Frozen: 2026-10-10.

This branch was created after repository search returned no DefaultReg access
to either BNCI2014_002 or BNCI2015_001. The scientific protocol was frozen
first in the cs research repository as
CONFIRMATORY_PROTOCOL_V5_TRUE_PREDATA_20261010.md.

No result from either dataset was inspected before this freeze.

## Datasets

BNCI2014_002
- all 14 subject IDs;
- right_hand vs feet;
- one session / eight runs;
- native 15-EEG custom Laplacian montage;
- official task window length 5 s.

BNCI2015_001
- all 12 subject IDs;
- right_hand vs feet;
- all sessions returned by MOABB 1.7.2;
- native 13-EEG montage;
- official task window length 5 s.

No participant is excluded for performance.

## Preprocessing

- MOABB 1.7.2;
- 8-30 Hz;
- t=0..5 s relative to the dataset task interval;
- resample=160 Hz;
- reject_by_annotation=False;
- all native EEG channels declared by pinned dataset metadata;
- trial time de-meaning;
- covariance with trace-proportional numerical SPD floor;
- source-only arithmetic SPD tangent reference.

## Evaluation

Full outer leave-one-participant-out.

Every outer source pool uses exact source-participant LOPO for the scalar
criteria. Participant is the unit; no trial-pooled selector.

Historical FULL_GRID is the method-selection grid.

A dense 0.0025 [0,1] path plus near-zero anchors is also frozen for validity
frontiers. Dense target-path scores are audit outputs and may not select a
method from target performance.

## Frozen method set

1. LedoitWolf_LDA
2. OAS_LDA
3. source_cov_nll_full_domain_LDA
4. source_macro_full_domain_LDA
5. discriminability_first_LDA
6. mean_tail_fidelity_LDA
7. TS_LogisticRegression
8. Riemannian_MDM

Mean-tail-fidelity:
- source macro margin 0.01 AUC;
- source tail = worst 25% participant AUC mean (CVaR25);
- tail near-optimal margin 0.005 AUC;
- covariance predictive NLL is tertiary tie-breaker.

## Two-phase target audit

Phase A:
- fit/select from outer source participants only;
- generate every frozen method score;
- generate every feasible dense-lambda target score;
- write score bundle;
- SHA256 score bundle.

Phase B only after score bundle exists:
- load target labels;
- compute participant AUC / path AUC / inference.

This is an auditable execution ordering, not a claim that target labels were
absent from the downloaded dataset bytes.

## Primary inference

H2: source_macro_full_domain_LDA vs source_cov_nll_full_domain_LDA.
H3: source_macro_full_domain_LDA vs LedoitWolf_LDA.

Per dataset:
- exact sign flip (n<=20);
- participant bootstrap 20,000;
- Holm H2/H3;
- mean/median/Q25/Q10/Q05/CVaR25/CVaR10/worst/harm rates.

All negatives remain.

## Claim boundary

A positive result is evidence for the studied MI cohorts only.
No universal EEG, clinical, online or TNSRE claim is licensed.

Status at freeze: TRUE PRE-DATA / NO OUTCOME INSPECTION.
