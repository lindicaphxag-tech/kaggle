# Four Default EEG papers — substantial manuscript and experimental upgrades
Date 2026-10-13. Status: **research upgrades committed; none approved for immediate journal submission**.

## Primary reproducible discovery this round (P2)
From ORIGINAL immutable V29 49-record artifact `CROSSED_MODEL_SELECTION_49_GROUPS.json`:
SHA-256 `0e7bfdb73970fcff9231a51954bd2d548f472cb296624967afc80d9d241ce2be`.
Source macro-AUC original fit/selector AA = .5886734693877551, purged
model/purged selection BB=.5825510204081633; 49-group effect -0.0061224489795918.
The exact digital-overlap-affected group (released representative ID4) AA=.8550,
AB=.8400, BA=.6075, BB=.5925: matched delta **-0.262500**.
REMAINING 48 groups' original-vs-purged target mean delta **-0.00078125**.
Therefore the aggregate is dominated by ONE known overlapping digital content
group: (-.2625 + 48*(-.00078125))/49 = -.0061224489795918.
Proper predictive source NLL selector ID4=-.2450, remaining48=+.0013020833,
all49=-.0037244898; its lambda never changes across all49.
Historical class-centred CovScore ID4=-.2600, remaining48=-.00078125,
all49=-.0060714286; its lambda never changes across all49.
Source macro lambda changed in 37/49; no proof of universal benefit.
The nominal-50-ID published average=.594000; same original
models restricted to 49 unique content-group representatives=.588673469,
and the implied extra duplicated held ID's AUC=.855. This is the
separate reporting denominator effect (minus .00532653), not model refit.
Subject IDs/recorded signal contents ARE NOT biological patient identities.
The measured contrast is software pipeline and reporting, NOT a
patient-level causal leakage effect. Most training pools overlap; no new
confirmatory independent n=49 population p-value.
Fully independently recomputed immutable source original 49 matrix:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38312796869
New localized exact-source influence auditor CI:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38315269388
P2 changed manuscripts include new methods, analysis-unit / content graph,
negative controls, publisher comparison, per-group impact sensitivity and
real 49-group plot. All methods use full original archive, not invented
large gains; author-approved data-custodian clarification not yet sent.

## P1 JNE methodological revision
- Original JNE manuscript wrongly included a BSPC-specific
  in-this-journal reference. This has been removed.
- Exact JNE peer comparisons were substituted: Eder et al. 2024
  DOI 10.1088/1741-2552/ad6793 and Seibel et al. 2026
  DOI 10.1088/1741-2552/ae453d.
- JNE structured Objective/Approach/Main results/Significance preserved.
- Proper SOURCE-trained Gaussian NLL's REAL post-outcome comparator
  is correctly reported: Shin n29 macro .785900 vs proper .786207
  (delta -.000307); BNCI2014_002 n14 macro .778705 vs proper .771886
  (delta +.006819 with exploratory interval crossing zero).
  No claim of stable superior algorithm over a corrected score.
- New executable exact fixed-decoder AUC-invariant density-drift toy
  passed CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38312790835
  (this is NOT an independently new EEG dataset).
- Official IOP JNE guidance for reviewers at least 12pt used.
- JNE nontarget-estimand scientific novelty/independent results still
  require full author statistical and editorial validation.

## P3 conditional source-selection study
- Added a reproducible exactly reversed synthetic
  two-target-world AUC counterexample for same fixed source data,
  illustrating *unconditional* source-only target-sign
  unidentifiability; CI passed
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38315923630
- Added explicit identical-acquisition model-selection regret protocol
  and a >.02 held-person harm endpoint for any future VALID predeclared
  comparative study, strictly distinct from P1's Gaussian scoring science.
- No new real participant experiment under a shared candidate family;
  current old Weibo/Cho/Dreyer/Lee tests remain different operators,
  do not pool p-values and do not reclassify Lee-B failed SHA as outcome.
- P3 remains not submission eligible.

## P4 separate numerical methods note
- Previously exposed real Weibo source groups:
  whole path old 94.121% -> Bernstein 94.267% certified stable source
  pair fraction, +.146 percentage points, not new-person AUC.
- Executed synthetic p2/3/6/9/12, three seeds each, 201 direct
  AUC samples per case, 15 deterministic SPD cases. At same
  12-segment budget Bernstein vs termwise/Taylor mean coverage
  +0/+4.321/+13.169/+16.667/+16.152 pp by p:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38312807723
- NEW segment-refinement source-only test on same 15 synthetic
  cases comparing 12,24,48 segments: 48 vs 12 merely +.0412 pp mean,
  improved 2/15, no numerical nonmonotonicity observed:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38315929126
- Means highlight numerator cancellation is more important than
  simple interval segmentation on tested synthetic cases. Not novel
  polynomial theory priority, formal machine-directed-rounding
  validation, speedup, higher-dimensional real EEG generalization,
  or independent dataset replication.
- P4 remains HOLD.

## Submission-go gates, DO NOT SUBSTITUTE formatting for evidence
**P1 JNE**: reproducible source/target per-person results + math proof
claims rechecked on final paper; author-approved JNE format, accessible
code/data references, conflicts/AI/ethics/funding and all author signatures;
independent scientific referee check of narrow novelty vs broad accepted
JNE benchmarks. Source-only geometry factors are not algorithmic target gains.

**P2 BSPC**: independent raw EDF-reader implementation and public data
steward clarification about exact ID4/5 signal contents (no inference
about patient identity), final figure and three source policy methods,
error-free compiled PDF, legal data release and author declarations.
Zero direct independent external dataset with another duplicate event;
report specificity and lack of prevalence estimate.

**P3 and P4** are research reserves and cannot be submitted
as recycled outcome papers from P1/P2. A full independent
same-policy EEG benchmark (P3) and stronger real numerics plus
formally checked FP bounds if claimed (P4) remain outstanding.

No acceptance rate or better-than-all-previous-accepted-papers claim is
justified. All manuscripts are GitHub Draft PRs, not journal submissions.
