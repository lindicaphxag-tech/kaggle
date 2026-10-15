# PAPER 2 — Exact EEG recording-content identity and matched fitting-selection impacts
Status: INDEPENDENT full LaTeX manuscript, NOT submitted.
Original matched 49-record experimental workflow: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38290349868
Verified title manuscript: `research/defaultreg_bspc_v13/main_p2_eeg_record_provenance_EVIDENCE_GATED.tex`
Table: `research/defaultreg_bspc_v13/125_v30_actual_crossed_49_result.tex`
Literature source: `research/defaultreg_bspc_v13/references_v10.bib`

## Primary question and independent knowledge
Can distinct published EEG participant IDs carry exact numerical
EEG TRIAL CONTENTS in the same fixed processing pipeline, and how
does removing content-overlap affect held-record performance via
reporting denominator, SOURCE MODEL REFITTING and already
SOURCE SELECTED shrinkage?
This paper is about the unit of signal-data independence. NOT the
Gaussian scoring estimand or 3-factor AUC-invariance from P1.

## Quantified material (already generated; do not embellish)
- Liu2024: 50 released IDs, 2000 processed trials, but 1960 unique
  exact EEG trial/label contents. IDs 4/5 share complete same 40
  processed trials, creating 49 recording-content groups.
- EEG controls: Weibo2014 10 IDs/1580 trial instances and
  BNCI2014_002 14 IDs/2240 trial instances have zero exact
  cross-ID overlaps in the specified processing pipeline.
  Zero exact matches does NOT rule out similar or window-overlap.
- Corresponding raw EDF channel evidence: 32 of 33 numerical
  channels match between IDs 4 and 5 across the corresponding
  signal, channel 32 unnamed and different. This is signal
  evidence, NOT an assertion that two clinical patients
  share identity. A data-custodian confirmation is outstanding.
- Original 50-ID reported macro AUC = 0.594000. Keeping
  49 record representatives under ORIGINAL fit/choice =
  0.588673. Matched fully purged 49 content-disjoint refits =
  0.582551.
- 2x2 matched held-record AUC matrix:
  original fit lambdaA .588673, lambdaB .587347;
  purged fit lambdaA .580765, lambdaB .582551.
  Source-macro lambda changed for 37/49 groups.
  Conditional selection effects -.001327 and +.001786;
  interaction +.003112; total matched -.006122.
- These contrasts are RETROSPECTIVE; they neither identify
  individual patient IDs nor causal 'leakage penalty'.
- Original/purged 49-identity outcome vectors were independently
  compared to immutable archives with ~1e-10 numerical parity.

## Non-overlap publication firewall
This is the ONLY paper allowed to headline the ID4/5 content
overlap and 49-group 2x2 fitted-model-by-selector interaction.
P1 may mention datasets are provenance-checked but must
not introduce these quantitative results as a P1 discovery.
P3 must not reuse Liu2024 data overlap as an independent
statistical replicate, nor turn paper2's 49 groups into
49 independent patients.

## Publication positioning
Candidate BSPC: EEG signal quality/reliability, benchmark data
audit, and actionable split hygiene within journal scope:
https://shop.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094
Candidate Brain Topography if supported by broader EEG
measurement/reliability and physiologic validation. Do not
claim acceptance odds or 2026 non-CAS sector ranking.
Check official journal scope, subscriptions, publishing fees
and institution's official 2025 CAS partition at submission.

## Required gates
[ ] Contact/notify appropriate EEG dataset custodians before
    a public claim about a clinical dataset's record duplication,
    offer a response, and use privacy-preserving numerical
    characterization instead of speculation about patients.
[ ] Validate duplicate evidence under raw EDF and independent
    preprocessing / second software path; report negative controls.
[ ] Full pipeline release with original and purged source
    train/reference fits and all 49 per-record receipts.
[ ] Explain why the two sources' target performance differ
    without causal language.
[ ] Verify intended journal welcomes a dataset/data-integrity
    note supported by ONE affected acquisition plus 2 controls.
[ ] Final official publisher format, figures, author declarations,
    funding/ethics/rights/AI use approved by humans.
NO simultaneous submission as P1 text or rebranded duplicate.


## CURRENT PRIMARY TARGET (2026-10-13): Biomedical Signal Processing and Control — BSPC
Publisher Elsevier; 2025 legacy CAS large category Medicine 2.
Official aims explicitly accept signal quality, artifact/robustness
and benchmark or dataset papers:
https://shop.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094
This is a hybrid subscription journal. Subscription publishing
usually permits **no APC**; only OA choice incurs APC. Verify
the live author-guide for any unexpected charge at submission:
https://www.elsevier.com/about/policies-and-standards/pricing
Novelty must remain EEG recording CONTENT identity
and a paired 49-group fitting-vs-parameter mechanism—not a generic
textbook warning about data leakage.
Non-negotiable: notify data custodians on clinical release ID4/ID5;
verify raw/independent processing and avoid patient-identity claims.
**BSPC is the one selected PREPARATION target, NOT a submission.**

## Additional public EEG identity controls — executed 2026-10-13
- [SUCCESS: BNCI2014_001 and BNCI2015_001](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38326859467): respectively 9 IDs / 2,592 trial instances and 12 IDs / 5,600 trial instances; no exact cross-ID epoch/label content hashes under specified pipelines.
- BNCI2014_001: 0/36 comparable ID pairs aligned whole-record rho>=0.95. BNCI2015_001: 0/34 shape-compatible; **32 of 66 pairs not evaluated by this metric** due unequal shapes.
- These are POST-DISCOVERY, previously available public controls, not preregistered independent cohorts; exact negatives do not prove absence of reordered or time-shifted near duplicates. The study has five released acquisitions in its content audit, but one identified exact-alias acquisition.
- [New raw EDF digital exact-moment affine test](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38325795037) independently verifies a substantial difference between exact 4/5 alias and high processed similarity of 11/15 and 47/49; neither source data identity nor clinical patient identity can be inferred solely from similarity.
- Still BLOCKED: data-steward consultation; independent clinical release provenance; final PDF at current HEAD; authorship/ethics/rights approvals. No formal journal submission occurred.


## 2026-10-13 source MAT recheck — new decisive release-layer qualification

- Dataset release is multi-layer: MOABB loads publisher-preprocessed `edffile.zip`, while Figshare v5 separately distributes `sourcedata.zip` MAT files. Independent readers of the **same EDF** are not independent original acquisition verification.
- Source MAT files for ID 4/5, 47/49, 11/15 were extracted with strict HTTP 206 ranges, per-file ZIP CRC validation and waveform-free output. Successful run [38333284368](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38333284368).
- Each selected source MAT contained a numeric array with shape **40 x 33 x 4000**. None of the corresponding whole arrays were numerically exactly equal and **0/33 complete source numeric channels** were identical for all three pairs. These are source release artifacts, not verified unprocessed sensor log files.
- The original cross-MAT whole-array correlations about 0.99996 were misleading. Independent channel/trial and covariance decomposition run [38334394283](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38334394283) gave channel-median Pearson **0.625329, 0.818365, 0.763512**, and pooled *within-channel centred* correlations **0.728604, 0.849534, 0.932343** (4/5,47/49,11/15). Between-channel means explain >99.9% of whole-array variance; the apparent pooled correlations **must not** be used as source EEG waveform near-identity evidence.
- As-stored **publisher-preprocessed EDF** digital channel-record block comparison [38332413760](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38332413760) still confirms for 29 named EEG channels **9,280/9,280** 4/5, **6,957/9,280** 47/49 and **171/9,280** 11/15 aligned nonconstant exact blocks. The overlap layer and upstream source MAT relationship are NOT yet causally traced.
- Primary 49 **exact processed trial-content groups** remain valid only as the specified downstream-software evaluation unit. The optional 47-group exclusions are a **scenario**, not a true independent-person count. No original recruitment, acquisition session, export-code, patient-identity or clinical metadata claim.
- New MATLAB-to-EDF **channel/time correspondence check** workflow [source/EDF CI](https://github.com/lindicaphxag-tech/kaggle/actions/workflows/p2-source-mat-vs-edf-signal-correspondence.yml) is experimental and must not be cited as a positive result until successful scalar-only output has been checked.
- PDF evidence: abstract tightened to **210 words** with the explicit opposing source-MAT evidence, successful build [38335751719](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38335751719). This compilation preceded short-link typesetting revision; at least one remaining LaTeX overfull warning was observed, so last-mile formatting is not signed off.
- **Still BLOCKED**: dataset custodian nonidentifying explanation, verified source MAT-to-EDF lineage, exact artifact/DOI freeze, author identities/affiliations/declarations and rights/ethics approvals, final strict page visual check, and target-journal acceptance of study format. Draft PR is not a submitted manuscript.
