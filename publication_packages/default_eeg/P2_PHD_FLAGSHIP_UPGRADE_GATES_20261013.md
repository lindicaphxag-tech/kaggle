# P2 / BSPC — release lineage and content-independent validation: doctoral-level workplan (NOT executed)
Date: 2026-10-13 | Proposed mechanism only; no publication, custodian reply or clinical patient identification.

## What is actually verified
- Liu2024 processed MOABB ID4/5: all 40 labelled epochs match; released 50 IDs, 2,000 epoch instances -> 1,960 unique exact contents, 49 *digital-content* groups.
- Publisher-supplied preprocessed EDF digital streams ID4/5: 32/33 complete channels identical. ID47/49: 6,957/9,280 exactly matching named-EEG channel-record blocks; ID11/15: 171/9,280. None proves clinical identity.
- Separately released sourcedata.zip MAT records: 0/33 entire numeric channels equal in the three candidate pairs; naive flattened r~.99995 dominated by between-channel means. Post hoc trial matching suggests a release-layer asymmetry but DOES NOT reconstruct conversion operations.
- 49-matched group AUC: nominal 50-ID .594000 -> reweighted 49 .588673 -> purged source refit 49 .582551; full-change local to affected component; NOT a causal patient-level leakage penalty.
- Four further public acquisitions: no detected exact cross-ID labelled epochs under specified pinned pipelines; negative screens do not prove universal independence.

## Doctoral problem statement
Infer or REFUSE to infer numerical signal ancestry across source MAT, preprocessed EDF, transformed epochs and evaluation records, while quantifying missed aliases and false links under realistic EEG transforms. Crucially the inferred graph is a **signal product graph**, not a graph of people.

## Proposed mechanism: provenance-lattice with abstention
H0 independent numerical signal products; H1 exact common digital payload; H2 time-local reused digital blocks; H3 related under a specified, documented operation family (units, affine amplitude, channel reorder, reference, filtering, epoch time shift / permutation); HU unresolved.
For each candidate edge construct a sequence of proofs/tests:
(1) canonical byte identity and exact channel-block identity;
(2) pre-registered transform-family alignment with *held-out time spans*, not evaluation on the same samples optimized for alignment;
(3) time-frequency residual on held-out windows and competing nulls;
(4) traceable published conversion metadata or custodian confirmation.
An edge is promoted H3 only if independent held-out residuals and calibrated error pass; otherwise HU, not forced identity.
Use negative pairs from independently sampled recordings as well as difficult positives synthesised from known-source transforms. Never train/evaluate a near-duplicate threshold using ID4/5, 47/49, 11/15 both for discovery and validation.
Record transformations as versioned operators and immutable digests, not as conjectured real clinical processing history.

## Required experiments — none added by this note
1. Reconstruct and cross-check source MAT -> EEGLAB -> EDF -> MOABB using documented metadata or public scripts; do not invent missing steps. Resolve reference, channel identity, label order and original trial time bases. Ask data stewards for nonidentifying technical clarification; draft only, human approval before sending.
2. Controlled synthetic positive pairs: exact copy, channel permutation, affine sign/scale, rereference, filtered/resampled, truncated, shuffled trial order, partially overlapping time, plus independent EEG with similar rhythms. Stratify by real recording/session as grouping unit.
3. Calibration: leave-dataset-out separation of fitted alignment/threshold from held-out recordings. Report FPR on independent negative pairs, FNR/recall by transform, PPV conditional on clearly stated synthetic prevalence (not pretend clinical prevalence), and coverage/abstention.
4. Measure practical downstream effect on shared tangent-LDA/logistic/ridge/one lightweight deep decoder and at least 2 unrelated EEG acquisitions where signal ancestry can actually be verified.
5. Separate three estimands: ID-weight change, same-digital-content-group refit, and post hoc near-link exclusion sensitivity; only an actual verified mapping supports conclusions about acquisition-level independence.
6. Robustness: arbitrary electrode mapping, shifts, data corruption, negative controls matched for event labels, nonstationary covariates, ablation of each transform; prevent outcome-guided grouping and post-selection multiplicity.
7. Independent analyst rebuild of one small public provenance graph and a stable code+metadata release with careful clinical data rights.

## Go / no-go gate
A: documentary or independently verifiable source conversion map; if not possible, manuscript remains a narrow public-release anomaly study, NOT a verified origination/causal investigation.
B: near-alias detection beats simple hash, raw block equality, normalised Pearson and time-series cross-correlation on genuinely held-out transformed recordings at matched FPR and time.
C: stable held-out PPV/recall and an explicit abstention set. Do NOT announce high-similarity pairs as identical persons.
D: contribution generalises beyond this single EDF conversion; report negative datasets and hard false matches.
E: supervised downstream improvement is secondary; reproducible split-integrity and benchmark-conclusion changes are primary.

## Published comparators and scope
Gil & Hernandez-Sabate BSPC 2026 DOI 10.1016/j.bspc.2026.110020 (six architectures / validation levels); Qin et al. BSPC 2026 DOI 10.1016/j.bspc.2026.109535 (nine paradigms and spatial normalisation). Proposed method must add cross-layer provenance inference not already covered by these validation studies. Do not claim journal top percentile from a one-release anomaly.