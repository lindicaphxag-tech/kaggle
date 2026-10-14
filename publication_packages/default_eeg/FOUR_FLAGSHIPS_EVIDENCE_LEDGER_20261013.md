# Default EEG four flagship research tracks — evidence ledger (2026-10-13)

**All papers remain DRAFT / NOT SUBMITTED.**
A manuscript standard or a locally passing program is not evidence
of independent transfer benefit, journal acceptance or novelty superiority.

## Branch heads reviewed and pushed
- P1 JNE PR #143, head bc692d166be48e52d7f1c8be7422ebcd54850849
- P2 BSPC PR #144, head 5ebc7ae8b476400f168c89dcd17bd0eabe94b580
- P3 PLOS ONE PR #145, head 5d4aeb0618dc2a4965db0315ab28d4dd125f965d
- P4 Numerical Algorithms PR #146, head 10f6bad2d7482b4d42a3efa2bd385cbd275b4d81

## What this review actually adds

### P1: a hard trivial-baseline falsification on EXPOSED REAL EEG\n- Manuscript Results now explicitly replace misleading unqualified\n  covariance-NLL references with historical class-centred CovScore;\n  this is NOT a proper source-mean predictive Gaussian likelihood.
- From frozen BNCI2014_001 9-person second-scorer JSON, all nine
  source forecasts were positive; 7/9 observed deltas positive.
- The advertised 7/9 sign accuracy **equals always-positive**.
- MAE forecast .015380524477 vs delta=0 baseline .017698688272:
  descriptive reduction .002318163795 AUC.
- Paired exact two-sided sign-flip enumeration p=.3203125,
  exploratory ONLY due outcome exposure and shared LOPO source folds.
- Script + source-frozen result path:
  research/defaultreg_bspc_v13/src/p1_exposed_forecast_trivial_baseline_audit.py
- **OPEN:** preregistered data-unseen acquisition level prediction and
  second independent model scoring; strong source-only baseline benefit.

### P2: signal-product transform stress, NOT patient identity
- 120 synthetic null calibration pairs and 120 disjoint synthetic test
  acquisitions for each of six known transformations.
- New directed channel-rowspace containment: 120/120 channel
  permutation/gain positive associations vs 9/120 flat Pearson
  after fitted synthetic null thresholds. In test nulls 2/120 false
  links for BOTH metrics; no claimed zero-FPR detector.
- Real 6x6 MAT-to-EDF source ranks are still post-discovery:
  EDFs 4/5 choose MAT4, 11/15 choose MAT11, 47/49 choose MAT47,
  but no source conversion log or independent acquisition genealogy.
- Identified NEMAR nm000158 v1.0.3 (2026-08-18) as a separately
  hosted version of the same Liu2024 source, NOT new EEG people:
  publication_packages/default_eeg/P2_NEMAR_FIGSHARE_CROSS_DISTRIBUTOR_GATE_20261013.md
- **OPEN:** MAT->EDF documented pipeline, channel/time map, independent
  held-out known-origin positive/negative cohorts, data steward clarification.

### P3: assumption-explicit acquisition-level conformal prototype
- A FIXED policy and source-only margin forecast are chosen before
  calibration. The one-sided split-conformal residual rank gives a
  marginal lower bound under exchangeability of whole acquisitions.
- Synthetic 79-calibration / 2500 fresh exchangeable trial: empirical
  90.64% bound coverage at alpha=.1. Deliberately nonexchangeable
  test-law shift: only 3.0% coverage. Neither result concerns real EEG.
- Under strict Z0, the SAME known source pool implies the SAME policy
  for unknown targets: no target-specific adaptivity.
- **OPEN:** sufficient genuinely independent meta-acquisition source
  calibration units and at least two unopened target acquisitions.
  The four original operators are not yet a common confirmatory trial.

### P4: fixed-pole rational coefficient-uncertainty certificate
- For fixed exact positive poles, bounded score coefficient error
  yields robust sign polynomials N-E and -(N+E). Strict positive
  Bernstein coefficient vectors are sound sufficient certificates;
  any undecidable sign returns UNKNOWN.
- Local test: 96 fixed-seed synthetic rational cases, 31 POS,
  39 NEG, 26 UNKNOWN, 3,120 sampled uncertainty-corner checks.
  Corner checks are regressions; mathematical soundness follows from
  convex-hull enclosure, not sampling.
- The exact finite-rational prototype is pushed with a GitHub CI
  workflow and integrated P4 manuscript section.
- **OPEN:** IEEE754 outward rounding for the full signal pipeline,
  eigenvalue and eigenvector errors, strong end-to-end comparison on
  independent high-dimensional source tasks, real decision value.

## CI evidence restrictions
- Older generic GitHub PDF workflows may complete; they do NOT by
  themselves verify each newly created method workflow or prove its
  data acquisition independence.
- New method/CI file commits are verified by PR branch heads; new
  method CI SUCCESS is NOT claimed unless its own completed run is
  separately confirmed.
- Local P2/P4 tests are executed; P1 frozen-scalar calculations
  are independently reproduced from GitHub JSON; P3 synthetic
  coverage numbers are independently checked.
- No full latest HEAD PDF layout inspection, new held-out EEG dataset,
  custodian reply, coauthor approval, or real journal submission.

## Submission decisions (2026-10-13)
- P2: the closest paper to a careful limited-scope data-integrity
  submission, but it does not yet support a general automatic
  provenance classifier or identity claim.
- P1: valid estimand-accounting manuscript direction, predictive
  algorithm headline not justified.
- P4: defensible conditional mathematical prototype; not yet strong
  Numerical Algorithms empirical/numerical assurance.
- P3: exploratory negative studies and a synthetic assumption-check
  prototype, not yet an independent full PLOS ONE study.

All four have a separate research endpoint. Outcome reuse across
papers and acquisition-level dependencies MUST be disclosed.


## 2026-10-13 subsequent joint-uncertainty and external-manifest audit
- P4 #146 head 10f6bad2d7482b4d42a3efa2bd385cbd275b4d81.
  New exact-rational *joint* positive spectral-eigenvalue BOX and
  score-coefficient BOX Bernstein certificate, not simply fixed spectral poles.
  A local standalone regression checked 126 synthetic boxes, outcomes
  52 POS / 50 NEG / 24 UNKNOWN, with 208,560 exact rational
  uncertainty-corner/parameter point checks. The mathematics is a
  sufficient interval enclosure; corner samples are not proof.
  New script: research/defaultreg_bspc_v13/src/
  p4_joint_spectral_coefficient_box_certificate.py
  and new TeX:
  P4_joint_spectral_coefficient_box_scope_20261013.tex.
  Real EEG input-error bounds and external numerical superiority
  are NOT verified. The exact new CI job is NOT individually
  confirmed just because generic repository workflows succeeded.
- P2 #144 head 5ebc7ae8b476400f168c89dcd17bd0eabe94b580.
  Audited version-pinned NEMAR nm000158 JSON manifest at external
  commit 11e3eeb4b439d5f8ee1b5e3b8e6032824bc2de83.
  It states 50 whole EDF files imported via MOABB 1.5.0 on
  2026-08-11 and 50 distinct recorded SHA-256 whole-file digests.
  Published pairs 04/05, 11/15, 47/49 have distinct
  full-file digests. This does not contradict their numeric
  within-file channel similarity, does not independently
  hash raw downloaded file bytes, and does not establish
  raw MAT ancestry or clinical identity.
  Metadata-only audit:
  research/defaultreg_bspc_v13/src/p2_nemar_release_manifest_audit.py.
  Embedded P2 manuscript paragraph:
  P2_nemar_whole_EDF_manifest_scope_20261013.tex.
- P1 #143 and P3 #145: previous limitations unchanged,
  and no new prospective independent-acquisition data.
- All four remain Draft; author declarations, PDF HEAD/layout
  verification and release decisions are not closed.


## 2026-10-13 conclusive CI receipts from this work session

All findings below have SPECIFIC completed GitHub Actions receipts, not merely
source-code commits or a pending hypothetical experiment.

### P2 -- six actual independent-hosted public EDF downloads, SHA & digital content
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38384783079
Artifact 11804714208; status SUCCESS. All six NEMAR v1.0.3 public EDF files
independently downloaded and complete-file SHA256 recomputed, all matching
the fixed distributor manifest. Each is 10,568,704 bytes, 33 channels,
320 EDF records. Entire digital channel streams identical (same position
and permutation-invariant): public released ID 04/05 = 32/33;
11/15 = 1/33; 47/49 = 1/33. The six whole-file digests differ.
THIS PROVES REPRODUCIBLE DIGITAL CONTENT ON TWO DISTRIBUTION PATHS;
IT DOES NOT PROVE DUPLICATE PATIENTS, RAW MAT EXPORT LINEAGE,
INDEPENDENT CLINICAL ACQUISITIONS OR CAUSAL TRAIN/TEST LEAKAGE.

### P4 -- completed exact-rational certificates and perturbation controls
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38384340046
Artifacts 11804563432 (222 full synthetic-box decisions with
36 exact rational counterexample receipts) and 11804168928
(covariance-coordinate perturbation results). 222 constructed
rational spectral/score boxes: whole Bernstein certifies 173,
direct rational interval bound 174; lambda splitting 184;
deep spectral rescue two more; 186 certified signs
(95 POS, 91 NEG), and remaining 36 exact admissible
falsifications; zero unknown for THIS FIXTURE SET only.
A distinct covariance-coordinate (no fixed eigenspace) resolvent
guarantee certified 46 POS + 65 NEG of 120 rational SPD matrices;
9 UNKNOWN; 2,220 exact rational perturbation point oracles agree.
Input error norms are EXTERNALLY ASSUMED, not measured real-EEG
estimation error or IEEE754 outward-rounded guaranteed bounds.

### P3 -- original real-EEG outcome artifacts independently replayed
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38385255017
Artifact 11805555945. Original two acquisitions, three previously
EXPOSED person outcomes, five fixed within-acquisition policies.
Three-record descriptive mean paired AUC delta:
continuous 0, SCE -0.011218492798354,
anisotropic -0.0026266718106996434,
group-centred -0.004235468106995895.
Max nonbaseline advantage across original records was
+0.0004012345679011986, while SCE worst loss -0.021875.
NO PROSPECTIVE / INDEPENDENT population superiority/null test.

### P1 -- decision utility of nine exposed source-forecast scalar results
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38385539468
Previously exposed real EEG nine people: all source forecasts positive
and observed signs seven positive, so 7/9 directional accuracy is
identical to always-positive. Source predictor MAE .015380524477
vs zero forecast .017698688272; exploratory signflip p=.3203125.
Source macro compared to proper predictive NLL averaged +.012608239026 AUC.
Post-outcome threshold .005 selected macro 5/9 and performed
-.002545224623 AUC worse than always macro.
No new risk calibration, dataset or prospectively validated action benefit.

### Exact HEAD PDFs
A single branch-specific fail-closed TeX build exists in ALL FOUR paper
branches, producing a PDF with a SHA256 receipt, reference diagnostics,
page count and overfull/underfull warning count. At the verified time:
P1 latest head build success, https://github.com/lindicaphxag-tech/kaggle/actions/runs/38385789367
P2 latest actual-byte-result head build is triggered,
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38386085148
P3 current head build is triggered,
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38385798263
P4 current head build is triggered,
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38385803415
Do not promote QUEUED/RUNNING to SUCCESS. Successful LaTeX is NOT
camera-ready graphical inspection or authorship/ethics signoff.

The four PRs all remain DRAFT / NOT JOURNAL SUBMITTED.
True doctoral flagship acceptance additionally requires independent
validation not yet run, fair cross-journal accepted-paper benchmarking,
author-approved declarations, exact latest PDF proofing, and
external publication/maintainer evidence.


### P2 crucial channel-variation and physiological-reference negative control
A separate full-EDF rehash / digital channel *location* run completed
SUCCESS https://github.com/lindicaphxag-tech/kaggle/actions/runs/38386864421,
artifact 11807041377. As-stored EDF IDs 04/05 share 32/33
whole digital channel time series at the same positions, of which
**31 are time-varying waveforms** and one is a static **CPz
reference** channel (zero-based channel 17). ID pairs 11/15
and 47/49 each share **ONLY the constant CPz channel (index 17)**
and NO full time-varying exactly identical digital channel.
Therefore one-channel matches in those near-pairs are trivial
reference artifacts and cannot serve as exact physiological
duplicate evidence; their approximate digital-waveform similarity
remains a separate unverified correspondence question.
The new independent channel-localization script and its result
are incorporated in P2 source and the journal draft. The claims
remain restricted to *digital signal products*, not patients.


## 2026-10-13 end-of-session second experimental wave and PDF visual proofing

All four PRs remain DRAFT; journal submission, acceptance, true external
preregistration and held-acquisition-level generalisation remain OPEN.

### P2 -- full public-EDF census is now executed, not planned
SUCCESS run https://github.com/lindicaphxag-tech/kaggle/actions/runs/38388066957.
All 50 NEMAR v1.0.3 public preprocessed EDF files independently re-fetched
and SHA-256-matched against fixed distributor JSON; all
(50 choose 2) = 1,225 ID pairs checked for equal 29 labelled EEG
whole time-varying channels and each of 320 aligned native storage
blocks. ID 04/05 has 29/29 complete variable EEG channels and
9,280/9,280 exact blocks. ID 47/49 has 0/29 complete and
6,957/9,280 aligned blocks. ID 11/15 has 0/29 complete and
171/9,280 aligned blocks. **All remaining 1,222 pairs = zero exact
matching whole EEG channels and zero corresponding aligned blocks.**
Common constant CPz reference not counted. This is digital product
reproduction, not source MAT lineage, clinical acquisition identity,
patient identity, a causal performance penalty or population FPR.
A scalable 50-ID triangular grayscale vector map was integrated in
P2 PDF. Exact 22-page PDF run success
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38393701265
(head a959d53c...) with 1 residual ~0.75pt PGFPlots axis overfull.
Final PGFPlots axis shrunk and color replaced with a print-safe square
on head 322ed0a4...; subsequent exact-HEAD PDF run
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38394187739
was pending last check. Do not assume zero warnings before that run.

### P1 -- first-repo, source-locked AlexMI real held subjects
SUCCESS run https://github.com/lindicaphxag-tech/kaggle/actions/runs/38389770062.
AlexMI public right-hand-versus-feet EEG, 3 electrodes C3/Cz/C4,
source IDs 1--6 frozen in a SHA-256 selection receipt before reading
held IDs 7--8, 40 trials each. Independent shrinkage-LDA scorer with
a distinct EIGHT-point lambda grid (NOT the main 27-grid method):
held AUC macro 1.000/.995, proper predictive NLL .995/.845,
fixed .1 LDA .9875/.735, fixed tangent logistic C1 .970/.6425.
Macro-minus-proper average +.0775 AUC across ONLY two held people
in ONE named acquisition. Both signed source-only forecasts positive,
and always-positive trivial forecast also 2/2: novel individual
direction forecasting NOT proved. No independent external registration.
Its independent TeX subsection is integrated. Exact-HEAD article
27 pages, **zero overfull lines**, success
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38393587140.

### P3 -- new mental-task and MI acquisitions; both positive and negative outcomes
BNCI2015_004 *math versus letter* (NOT motor imagery):
SUCCESS https://github.com/lindicaphxag-tech/kaggle/actions/runs/38390289317,
8 source people source-locked before target ID9 (160 trials).
Macro AUC .51796875; SCE .527500 (+.00953125),
group-centred .52546875 (+.0075), anisotropic -.00015625.
This is a one-person new-task exploratory improvement, not a
population proof or confirmation of MI-specific transfer.

Dreyer2023B left/right motor imagery (same broader Dreyer study as
an earlier explored A subset):
SUCCESS https://github.com/lindicaphxag-tech/kaggle/actions/runs/38390737635.
Eight source public IDs61--68 fully frozen before held IDs69,70,
240 trials each. Continuous and macro identical, SCE source-macro
mean paired delta -.0145138888889, anisotropic -.0159027777778,
group-centred +.0009027777778. New public cohort but one collection;
no policy superiority. Both results integrated in P3 TeX.
Zhou2016 attempted download failed on upstream Zenodo HTTP read
timeout and was already method-infeasible under the originally
selected 3-source design (<8 required by these operators);
no target opened, no evidence from it.
P3 exact-HEAD 10-page PDF with zero overfull and zero underfull:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38393639245.

### P4 -- conditionally exact numerical theorem, real final EEG features
Previous results: 186/222 rational synthetic joint spectral/score
boxes have strict sign certificates and remaining 36 each have
an admissible exact rational falsifying witness. An orientation-free
covariance resolvent sufficient bound certifies 111/120 rational SPD
fixtures; 9 unknown. The final-feature rounding-to-pooled-covariance
chain certifies 64/64 intentionally well-separated synthetic cases
(48 positive, 16 negative); a separate constructed admissible
opposite-sign feature-error example correctly returns UNKNOWN.
Executed CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38389194383.

Real previously exposed BNCI2014_001 public EEG person1 numerical
fixed-point demonstration: final tangent feature *binary64 values*
rounded on a rational grid 2^-18 with verified entrywise eta=2^-19,
12 preselected positive/negative sample pairs: 8 strict positive,
4 strict negative signs certified over ALL lambda in [0,1].
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38391746843.
This is NOT a proof of IEEE-754 error propagation through MNE,
covariance/tangent operations, physical acquisition error,
cross-subject AUC benefit or superiority over mature numerical methods.
P4 15-page latest-head build zero overfull/underfull:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38393607317.

### Deliverable QA
All four exact-GitHub-head PDF ZIPs have been fetched via authenticated
GitHub connector, unzipped, and page-count/text-bound sanity checked.
Old and current PDF snapshots are versioned by their GitHub SHA receipts.
P1/P3/P4 zero overfull, P2 final after print-safe PGFPlots microedit
PENDING run; no file can be called final publisher proof unless this
latest build and human page-by-page inspection both pass.

### Remaining hard barriers
1. P2 released digital content vs ORIGINAL MAT->EDF provenance records and
   nonidentifying owner clarification remain unsolved. Avoid accusation.
2. P1 forecast advantage over always-positive and source-only naive
   policies is not established, despite strong AlexMI AUC contrast.
3. P3 candidate utility on genuinely independent uninspected acquisitions
   with calibrated acquisition-level uncertainty is not established.
4. P4 source-to-binary IEEE-754 error accounting, dimension-scalable
   strong external numeric comparator and mathematical novelty remain
   insufficient for confident Numerical Algorithms acceptance.
5. Human coauthors must verify manuscript ownership, independent ethical
   data use, conflict/funding statements, citations, final figure layout
   and corresponding-author-controlled editorial submission.


## Verified actual-author-review PDFs — post-layout correction (2026-10-13)

The following is a NEWER audit than prior pending-run descriptions.
All four actual GitHub Actions artifact ZIPs were downloaded; four
individual PDF SHA256 sums matched the in-artifact EXACT_HEAD_PDF_RECEIPT
JSON; PDFs opened successfully, page counts matched, and pdf text-blocks
were checked not to extend outside the physical paper media box.
All four TeX logs have ZERO overfull hboxes.

| Paper | Exact branch head | Pages | Overfull hboxes | Successful run |
| --- | --- | ---: | ---: | --- |
| P1 JNE | 1b65914cdd1e443d8fca2303375e5bea9b148df2 | 27 | 0 | https://github.com/lindicaphxag-tech/kaggle/actions/runs/38395087320 |
| P2 BSPC | 322ed0a4d6fd1a2ce0502a37b1d7c136d558f225 | 22 | 0 | https://github.com/lindicaphxag-tech/kaggle/actions/runs/38394187739 |
| P3 PLOS ONE | f54312402281509d82a51ed852528ef4e7c3a7a5 | 10 | 0 | https://github.com/lindicaphxag-tech/kaggle/actions/runs/38394850027 |
| P4 Numerical Algorithms | 08cd3dfe0ae67778b70e09b110dd6bdc7b991554 | 15 | 0 | https://github.com/lindicaphxag-tech/kaggle/actions/runs/38393607317 |

Substantive P1 diagram correction: historic class-centred CovScore is
no longer inaccurately labelled covariance-NLL or mistaken for true
source-mean predictive Gaussian likelihood. An unsupported "TRUE
PRE-DATA external cohort" assertion was removed, and the grayscale
participant forest plot was actually rendered and inspected; an
overlapping graph subtitle detected on the rendered page was deleted
in the above P1 exact-HEAD run. All outcomes from those cohorts remain
retrospective at project level.

Substantive P3 chronology correction: "target was acquired" replaced by
the accurate "previously publicly released target EEG was first
LOADED after source-choice SHA lock." This is not a newly recruited
clinical acquisition nor external preregistration.

P2's 50-ID complete 1,225-pair triangular matrix was rendered and
inspected. The immutable provenance URL was converted to a clickable
descriptive label for print, and the separate 49-content-group figure
now uses a black square rather than a red mark; last PGFPlots width
overrun was corrected. P1/P3/P4 have no overfull line warnings, and P2
has zero in its last completed build.

**Zero overfull lines and media-box containment DO NOT establish
submission-ready scientific design, journal template compliance,
patient source identity, independent prospective validation,
all-author signoff, or paper acceptance. All four remain DRAFT.**


## 2026-10-14: publication-integrity audit and external maintainer engagement

This section supersedes stale "pending" statements elsewhere in this
append-only ledger. All four PRs are still DRAFT and NOT journal-submitted.

### P1: source-person CV re-entry found and fixed
- Primary \`choice()\` source-person forecast audit revealed that excluding
  an outer source person's *validation score* still permitted that person's
  data in models trained to generate OTHER source-person validation rows.
  Calling that procedure strict outer-person crossfit was not justified.
- Recomputed nested source-person outer folds: for each of AlexMI SOURCE
  1--6, removed the held person from EVERY inner fit/geometry/λ selection,
  trained two LDA models only on five remaining people, tested outer h.
  Nested paired six-source-person ΔAUC:
  +.0925,+.185,-.04,+.02,-.05,+.15 (mean +.0595833).
  Previous weak source forecast +.0541667, separately labelled not strict.
  Executed source exclusion workflow SUCCESS:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38539884099
- The same ALREADY EXPOSED two AlexMI held people 7,8 under original
  27-grid yield +.005 and +.1875 AUC macro minus proper NLL, mean +.09625.
  Both strictly nested forecast and trivial always-positive predict
  their *signs* equally; no novel targeting accuracy established.
  Original 27-grid successful replay:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38536408103
- Newly corrected methods paragraph in
  P1_alexmi_strict_nested_source_forecast_correction_20261014.tex.
  Latest P1 source c591a32511e0cb29521d18044ba41a63446fb39b:
  28-page exact-head PDF SUCCESS
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38540178044
  0 overfull hboxes; the SHA256 of actual downloaded PDF matches CI JSON.

### P2: author code ZIP searched; official MOABB acknowledgment
- Original Figshare v5 original author \`code.zip\` read only, no script
  execution, successful metadata/static code ZIP CI:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38535556189
  21 MATLAB sources, two electrode location files, README.
  Existing published code does not document a complete and validated
  acquisition MAT -> manually preprocessed EDF export mapping.
  This limits provenance rather than contradicts exact EDF content evidence.
  New article subsection:
  P2_figshare_v5_author_code_static_provenance_gap_20261014.tex.
- Public MOABB upstream issue submitted, with all 50 file SHA and 1225
  digital comparisons:
  https://github.com/NeuroTechX/moabb/issues/1248
  External maintainer bruAristimunha responded
  "We're gonna fix this! Thank you for the report." on 2026-10-14.
  A maintainer's acknowledgment/intent to fix is NOT independent
  scientific verification of original acquisitions, merged patch,
  publication, or proof of patient duplication.
- Latest fetched P2 PDF source
  3fb97c79212af4459311f9c9a06e77b9a08cbd06:
  22 pages, 0 overfull, SHA256 matches the exact-head CI receipt.

### P3: six new outcomes, source SHA drift explicitly preserved
- Previously frozen policy-CHOICES (not byte-identical fitted models) from
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38390737635
  used on six further Dreyer2023B held IDs71--76, n=240 each.
  Fixed choices, *refitted* source model, real public data run:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38537617142
- Descriptive paired mean ROC-AUC vs source macro:
  continuous 0, SCE -.006898148148, anisotropic -.003611111111,
  group-centred -.0009375. No method establishes benefit.
- Original full-source covariance SHA
  7146797ee8d6caf71c57ee141604102b2d61217456cbfe3e00820b83e172b2a7,
  newly read source SHA
  9f650460378e695189c3fa145bf0a29fe010b24cd92ba4aa1196e7439ada7f65,
  DIFFER. Strict byte-identical source replay therefore FAILED CLOSED
  before any new target read. The lower-tier rerun uses same source
  IDs, sample counts and previous fixed CHOICES, but refitted source model.
  Do NOT pool old and new as same fitted model.
- The manuscript directly documents both result and defect in
  P3_dreyerb_six_new_locked_choices_source_drift_20261014.tex.
  The 11-page PDF retrieved from successful exact-main build
  af6f92c7dfe94ba42c87eab4d908bc5301852499 passes
  0 overfull + SHA256 matching its receipt.
  A target-free two-reload source-drift diagnostic was launched
  on https://github.com/lindicaphxag-tech/kaggle/actions/runs/38540426367
  but its completion was not confirmed when this line was authored.
  Do NOT count that diagnostic as successful until the receipt is checked.

### P4: still conditional despite strong arithmetic proofs
- Existing source code and peer review draft remain: exact rational
  interval certificates, admissible opposite-sign counterexamples,
  covariance-coordinate resolvent bounds, bounded final feature
  quantisation conditions and a small fixed-point real EEG demonstration.
- No validated acquisition / preprocessing / floating IEEE-754
  forward error bound and no independent algorithm advantage was
  produced on 2026-10-14. Do NOT manufacture a stronger conclusion.
- Most recent confirmed PDF source
  ce67a7478750b177720b0ab4dec8afc1c7e34b8e:
  15 pages, 0 overfull, SHA256 matching CI.

### Publication readiness / not reached
The current work materially improves *research integrity* and external
triage. It does NOT yet meet the user's stricter condition that ALL FOUR
are top-tier doctoral research, outperform accepted-paper controls,
and are ready for immediate submission. In particular:
- P1 still lacks useful source-only action prediction beyond always-macro
  / always-positive; crossfit correction alone is not novelty.
- P2 still lacks recorded source MAT->EDF conversion, external custody
  confirmation of acquisition identities, independent discovery cohort,
  and causal link between release overlaps and reported model accuracy.
- P3 has heterogeneous negative and tiny positive examples across
  non-independent acquisitions; there is no registered acquisition-level
  meta-test and source SHA drift remains under diagnosis.
- P4 theory guarantees are conditional on externally validated input
  error bounds; end-to-end EEG and independent strong numerical
  competitor advantage unverified.
- All four must pass coauthor signoff, declarations, references,
  venue-specific requirements and editorial submission.


### 2026-10-14 postscript: P3 intermittent source digest and final current PDF
- The *target-free* Dreyer2023B source-only twice-reload diagnostic
  succeeded: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38540426367.
  Both reads of public IDs61--68 produced the identical original source
  SHA \`7146797ee8d6caf71c57ee141604102b2d61217456cbfe3e00820b83e172b2a7\`;
  all eight labelled participant ordered-trial and permutation-invariant
  per-trial hashes matched between those reads. Earlier independent CI
  reload had \`9f650460378e695189c3fa145bf0a29fe010b24cd92ba4aa1196e7439ada7f65\`.
  Therefore source-feature byte instability is intermittent between
  runs and has NO proven cause.
- A new strict eight-person *retrospective* original-SHA gate failed
  CLOSED again before ANY target was read:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38541224660.
  Do not claim the eight people were successfully evaluated on identical
  original source-feature bytes. Do not repeat selectively until one
  run happens to pass.
- Both results and technical research scope are included in P3
  \`P3_dreyerb_source_only_two_reload_fingerprint_20261014.tex\`.
  The latest complete, actually fetched P3 EXACT-HEAD PDF:
  branch SHA \`09dd26f804bba87a99c86a3f22503f905d3adae5\`,
  12 pages, 0 overfull, SHA256
  \`e4f9051876e0decd9244521d1d4c4a43c42d354fc987c8691db0be51ac47e5e7\`,
  run https://github.com/lindicaphxag-tech/kaggle/actions/runs/38541671186.
  All four 2026-10-14 PDF exact-head CI ZIPs were downloaded,
  PDF SHA reverified against the contained JSON receipt and staged
  together in a reproducible author-review ZIP. Author sign-off,
  external preregistration and journal acceptance remain OPEN.


## 2026-10-14 independent real-data receipts and MOABB upstream draft

- P1: SUCCESS full nested source-person training and selection exclusion
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38539884099.
  Six AlexMI sources, inner grouped CV excludes outer source person
  from ALL inner training and validation. Outer-person paired AUC
  differences +.0925,+.185,-.04,+.02,-.05,+.15, mean +.05958333.
  Old source forecast +.05416667 had improper indirect training re-entry;
  it MUST NOT be represented as strict LOPO. Previously exposed target
  IDs7/8 +.005 and +.1875; trivial always-positive remains 2/2.
  Corrected mathematical/validation scope integrated in P1 manuscript.
- P3: SUCCESS two new source-only reads, NO held people:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38542627607.
  Dreyer2023B source IDs61..68 have identical individual ordered hashes,
  sorted labelled-epoch hashes, features and whole-source SHA in two
  same-environment reads. The new SHA 9f650460...9ada7f65 differs from
  original 7146797e...3e172b2a7. Original source-model bit identity is
  UNPROVEN; six-person target study is strictly prior-choice refit.
  P3 latest TeX integrated receipt. Do not claim old exact source replay.
- P2 external engineering: Official MOABB Issue #1248
  https://github.com/NeuroTechX/moabb/issues/1248 has maintainer reply
  "We're gonna fix this! Thank you for the report." This is an external
  commitment to look at a data-loader issue, not independent patient
  identity proof, not academic acceptance. Documentation-only DRAFT
  upstream PR https://github.com/NeuroTechX/moabb/pull/1249 was opened;
  no merge yet.
- P2 full-release, erasure-tolerant real digital-content identity:
  All-50 full-byte EDF SHA256 and 1225-pair real census SUCCESS
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38388066957.
  Separate independent six-EDF byte rehash and nonconstant-native-record
  topology SUCCESS
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38543795987.
  All 9280 matching digital EEG channel blocks in 4/5 are nonconstant.
  47/49 has 6957 exact nonconstant blocks, and 245 of 320 EDF storage
  records have 28 of 29 matching named EEG channels; the longest
  uninterrupted >=28-channel span covers zero-based records 128..306,
  length 179. P4 channel does NOT have any exact matching native
  time-varying block. 11/15 exact common P3 digital blocks 171,
  never >1 EEG channel per native record.
  Two immutable independent result artifacts were independently
  reconciled SUCCESS in
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38544354555.
  On 1225 pairs: full EDF checksum linkage 0; full 29-channel
  time-varying EEG linkage 1 (4/5); posthoc >=28 of29 nonconstant
  EEG channels for >=100 consecutive native storage records linkage 2
  (4/5 and 47/49); other 1223 pairs have no link under that rule.
  The graph has 48 published DIGITAL-PRODUCT content components,
  NOT 48 unique patients, recordings or original clinical acquisitions.
  Thresholds were selected after the dataset was examined;
  do not report prospective classifier specificity/FPR/generalization.
  Formal definition, source-code and table integrated into P2 manuscript.
- P4 distinct eight-person AlexMI (one already-project-examined public
  data collection) conditional final binary64 feature-quantisation
  numerical audit requested in
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38544936551.
  Read actual completed artifact before inserting ANY numerical result.
  Input error bounds concern final tangent-feature quantisation only,
  NOT end-to-end EEG preprocessing or acquisition/machine-error proof.
- All four author manuscript PRs remain DRAFT. This ledger describes
  proven results and research gaps; it does not constitute accepted
  peer review, journal submission or clinical identification.


## 2026-10-14 final verified second-data P4 result and PDFs

- P4 second real public named EEG acquisition numerical test SUCCESS:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38544936551.
  AlexMI eight persons, two electrodes, three final binary64 tangent
  coordinates, eight fixed positive-negative pair comparisons per person
  across four prespecified fractional-bit grids (18/10/6/4).
  For EACH precision: 64 correlated sample pairs.
  Whole-lambda conditional positive-or-negative certificates vs UNKNOWN:
  18 bits 63/1 (48 POS,15 NEG), 10 bits 61/3 (47 POS,14 NEG),
  6 bits 25/39 (23 POS,2 NEG), 4 bits 0/64.
  These 256 pair-by-precision cases are NOT independent EEG acquisitions.
  Distinct named public EEG data from prior BNCI2014_001 numerical input,
  but AlexMI people previously examined by this project for P1.
  Scope is exact rational quantisation of machine-produced FINAL features
  and conditional covariance/resolvent score sign only; NOT end-to-end
  MNE/tangent machine-error bound, prospective BCI safety or AUC gain.
  Article subsection integrated in P4 current head
  dc6c9dfe2baf04d761afa4407ea0ba6a6f5dfc31.
- FINAL verified current manuscript PDF artifacts,
  each copied from its GitHub exact-HEAD build and SHA-256 checked
  against the contained receipt (0 overfull hboxes throughout):
    P1 JNE: 28 pages, c591a32511e0cb29521d18044ba41a63446fb39b,
      https://github.com/lindicaphxag-tech/kaggle/actions/runs/38540178044
    P2 BSPC: 23 pages, 5ba20e21c52f41174cec2fd87a7c9506b2454027,
      https://github.com/lindicaphxag-tech/kaggle/actions/runs/38544611226
    P3 PLOS ONE: 12 pages, 06b2a33f4030ddb9c13121939700df28c254f4df,
      https://github.com/lindicaphxag-tech/kaggle/actions/runs/38543145267
    P4 Numerical Algorithms: 16 pages, dc6c9dfe2baf04d761afa4407ea0ba6a6f5dfc31,
      https://github.com/lindicaphxag-tech/kaggle/actions/runs/38545449519
- Local author-review evidence ZIP was made with all four exact SHA-checked
  PDFs and the distinct P1/P2/P3/P4 machine-readable real-execution
  artifacts. Actual human journal compliance, true independent
  confirmatory validation, patient identity verification,
  full float pipeline error proof, submission and acceptance
  remain UNFINISHED, as do any external upstream PR merges.


## 2026-10-14 late evidence gate: DreyerB cross-run exact-source recovery

Three independent clean GitHub Actions runners, each twice retrieving
ONLY DreyerB source IDs 61--68, completed:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38548682677

- All three replicas' **exact source aggregate SHA-256** were equal
  to the first originally frozen successful source recording:
  \`7146797ee8d6caf71c57ee141604102b2d61217456cbfe3e00820b83e172b2a7\`.
- All six within-run source re-reads matched. All 8 source people had
  exact same ordered byte-level covariance-epoch digests and
  label-aware unordered multiset digests across all 3 clean runners,
  plus the diagnostic 9-decimal relative covariance and
  8-decimal log-trace fingerprint equality.
- The prior anomalous independent run's SHA
  \`9f650460378e695189c3fa145bf0a29fe010b24cd92ba4aa1196e7439ada7f65\`
  must NOT be erased. No evidence yet establishes whether it reflects
  raw source file retrieval changes, processing state, a feature
  array ordering change, or another transient numeric effect.
- No target ID 69--76 was accessed during this diagnostic.
  Original trained classifier parameter bytes were not archived,
  so recovering the original source input DOES NOT retroactively make
  the six-person re-fitted sensitivity an exact first-run classifier
  reproduction.
- New source-replay code and paper paragraph on P3 branch.
  Draft P3 still needs genuinely new acquisition-level validation.


## 2026-10-14 independent-scope gate corrections

**P1**: strict source-person outer CV on AlexMI (real previously exposed
source IDs 1--6 and target 7,8) revealed that the old forecast construction
was not a mathematically strict leave-one-source-person-out predictor:
other folds could have trained on h while h was nominally out-of-fold.
A corrected full NESTED inner CV and tangent reference fit removes each
outer source person from ALL inner models; successful
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38539884099.
Six truly out-of-source-person paired AUC margins:
+.0925,+.1850,-.0400,+.0200,-.0500,+.1500; mean +.0595833.
Earlier leaky-crossfit mean +.0541667 must be separately labelled.
The two already-exposed AlexMI targets yielded +.0050 and +.1875
(27-grid). Trivial always-positive direction correct both: no new
source-only decision forecast benefit. Article subsection updated.
**P1 scientific blocker remains** robust independently held acquisition,
nontrivial action utility, and strong venue comparisons.

**P2 external scientific/software follow-up**: upstream MOABB issue
https://github.com/NeuroTechX/moabb/issues/1248 received maintainer
bruAristimunha response 2026-10-14
"We're gonna fix this! Thank you for the report."
This is external maintainer acknowledgement/intention; NOT a merged
fix, independently approved provenance conclusion, or peer-reviewed
paper. Public whole-EDF byte content still proves 29/29 EEG channel
identity for published 4/5; 47/49 partial EDF storage overlap and
MAT ancestry unresolved.

**P3: cross-run source-SHA failure refined.** Two source-only diagnostic
CI runs
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38548578442 and
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38548578441
each read source DreyerB IDs61--68 TWICE. Within each runner 8/8
subject epoch SHA fingerprints remain strictly identical, but between
runners 0/8 bytewise per-subject digests agree. BOTH independently
rounded shape (covariance divided by trace to 1e-9) and log10-trace
(to 1e-8) fingerprints match for 8/8 persons across runners.
Maximum cross-run source-person mean covariance trace discrepancy
1.4211e-14; maximum logdet mean discrepancy 1.7764e-15.
This is consistent with precision-level numerical variations,
NOT an exact upper bound on entries or proof of root cause.
A truly byte-identical ORIGINAL source model was not reproduced;
the six-person follow-up remains old choices, newly refitted source.
A new exact-vs-quantized independent-run audit and P3 LaTeX paragraph
are implemented; CI rerun after fixing a GitHub Actions credential
literal:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38552831272
STATUS WAS QUEUED on ledger update; DO NOT mark SUCCESS without
a completed execution receipt.

**P4**: previously completed rational certificate remains conditional
on exact or bounded final-feature inputs; 12-pair previously exposed
real EEG demonstrates only stored tangent value rounding,
not preprocessing-level or IEEE754 source error certification.
No new prospective verification or journal acceptance implied.

The four paper PRs remain **DRAFT** and not editorially submitted.
This ledger separates achievements, executed evidence and
scientific/ethical/submission hard gates without claiming all have
been fulfilled.


### Source-drift CI status correction (2026-10-14 10:55 UTC)
The previously QUEUED P3 two-original-artifact forensic rerun is now
**SUCCESS**:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38552831272
Artifact `11887098613`, original source runs
`38548578442` and `38548578441`, with all eight per-person
rounded shape and amplitude digests identical across runners while
zero of eight exact float64 labelled-epoch digests agree.
Cross-run mean covariance-trace max abs difference:
1.4210854715202004e-14; logdet mean max:
1.7763568394002505e-15. Quantised fingerprints are an
approximate observation, **NOT** a rigorous perturbation
certificate, nor byte-equal original source model replay.
The new paper paragraph includes its completed reproducibility URL.

Current P1 #143 and P2 #144 exact HEAD manuscripts show SUCCESS;
P4 #146 exact HEAD shows SUCCESS. The P3 #145 latest
HEAD (post-forensic-paragraph) PDF build was QUEUED at verification:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38553132304.
Do not call the newest P3 author-review PDF checked until it succeeds.
All four paper PRs remain DRAFT. MOABB upstream acknowledgement is
not a merged correction, scientific peer-review or acceptance.
