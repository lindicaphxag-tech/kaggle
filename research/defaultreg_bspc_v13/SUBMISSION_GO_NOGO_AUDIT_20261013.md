# DEFAULT EEG — SUBMISSION GO/NO-GO audit (13 October 2026)

**Decision: NO-GO for direct publisher submission**.
This is an actionable readiness audit of the *actual* existing BSPC
manuscript and verified journal policies, not an acceptance forecast.
It does not mean the empirical science must start over.

## 1. What is already strong enough
- Main claim has been narrowed to a real, falsifiable SCIENTIFIC
  issue: validation-class-centred covariance score versus proper
  source-train predictive Gaussian score versus per-person AUC.
- Exact three-factor identity, fixed-discriminant AUC invariances,
  separate real-data source group audit across Cho2017, Dreyer2023A,
  and BNCI2014_002 (38 groups), independent artifact checks.
- Original EEGMMIDB precommitted 20-person comparison of source-macro
  AUC selector vs historical CovScore is +0.0561 ROC-AUC, not 5.61
  percentage points of classification accuracy, and includes harms.
- Null/negative Cho, Dreyer, Weibo and source-envelope analyses retained.
- Existing 18-page main PDF previously compiled in Github Actions.
- Related work now includes directly relevant published BSPC
  methodology comparators (Gil et al 2026, DOI
  10.1016/j.bspc.2026.110020; Qin et al 2026, DOI
  10.1016/j.bspc.2026.109535), domain adaptation with different
  target information budgets and 2025 cross-dataset GCN with
  target-labelled fine-tuning. This is SUFFICIENT for literature
  positioning, NOT for an equal-protocol experimental win claim.

## 2. HIGH-PRIORITY hard publication blockers in actual source
[BLOCKER 1] main_bspc_v31_EVIDENCE_GATED_CANDIDATE.tex still says:
  AUTHOR IDENTITIES AND AFFILIATIONS TO BE APPROVED
  "Original participant score receipts, repository commit hashes and
  code availability must be confirmed in the final submission package."
  "Author contribution, funding, competing interests, ethics and
  AI-assisted writing: To be completed and confirmed by all authors..."
These are unresolved; DO NOT fabricate authors, disclosures, IRB,
funding, software license, permission or provenance.

[BLOCKER 2] The live selected journal is not frozen.
The current typesetting is generic LaTeX article, not the final
journal's checklist/template. No confirmed title page, line/page
number settings, cover letter, figures/screenshots inspection,
correspondence author or ORCID metadata.

[BLOCKER 3] Methods-only scientific claim not sufficiently
operationalized across independent data. The three percentages
98.59 / 99.29 / 99.77% refer to SOURCE NLL correction share that
does not affect *fixed linear classifier's* within-person AUC.
They are retrospective source-validation analyses on previously
explored datasets, NOT new target gains, and weighted algebra is
prior art. A historical 20-person significant advantage versus
misnamed CovScore is NOT equivalently a new method outperforming
correct source predictive NLL / MDM / CSP / FBCSP.
For a journal requiring method performance gains, DO NOT SUBMIT
as a new performant selector. For a validation-estimand or audit
paper, quantify additional falsifiable decision consequences from
ALREADY legally observed receipts, show source/target boundary, and
label all exploratory and original inferential families.

[BLOCKER 4] External Lee2019 target A experiment was completed at
source SHA 27efc2565d5bc2bc22344b67ee994e42b985c16c34cabf6cdf55136270fa18f3,
12 persons: original macro AUC .75410, GID .75017, paired
delta -.003933, exact sign-flip p=.12890625.
Target B failed BEFORE ANY TARGET IO due 12 of 12 source
person covariance SHA not matching original run, although
the reported scalar nuisance-cov difference was ~2.2e-16.
The independent byte-level provenance audit explicitly failed:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38307723649
Do NOT rerun while relaxing first-source SHA without a
documented invariant, byte-wise source feature artifact strategy.
This side-study must not be described as two-block validation.
It does not invalidate the three earlier source-diagnostic analyses,
but shows cross-run numerical reproducibility is a LIVE risk.

[BLOCKER 5] Supporting information and result extraction:
pin release SHA for CODE plus raw human-readable numerical
source/group AUC receipts, executable notebooks/entry commands,
data download policy, release IDs, exact origin and hashes,
statistical preregistration/event chronology, dependency lock,
and discoverable repository access. GitHub PR DRAFT branches
with workflow artifacts are not a durable archival release/DOI.

## 3. What "enough comparisons" means
Literature: YES, for concise introduction and fair methodological
claim boundaries. Existing comparator matrix has multiple
PUBLISHED closely related papers and distinct target budget
category; don't add papers by quantity. Audit each actual DOI
and verify references with journal's required style.
Empirical equal-protocol head-to-head: NO, if claiming new
algorithm superiority or practical target gain.
For an evidence-based *measurement-validity* paper instead,
a field-standard strong comparator/trial-level ledger and
proof of consequence is still needed; performance wins
need not be the headline, but the numerical scientific
consequence must be precise and nontrivial.

## 4. Journal-specific editorial go/no-go
**BSPC / new method**: NO-GO. Strong SOTA improvements and distinct
novel solver are not established; negative SCE/GID evidence.
**Brain Sciences / computational validation Original Research**:
POSSIBLE after revising to estimand/measurement narrative, expanding
independent-data interpretation & disclosures; current version NO-GO.
Confirm fees and APC funding before author approval. Official APC as
of Jul 2026: CHF2400.
https://www.mdpi.com/journal/BrainSci/apc
**Frontiers in Neuroinformatics / Original Research or Methods**:
POSSIBLE if computational-public-data validation fits editorial
standards; novel pipeline and data/code evidence are necessary.
Article type options Original Research, Methods, Technology & Code;
author guidelines require metadata, data availability, correct
citations, journal style, scope statement & disclosure of AI use.
https://www.frontiersin.org/journals/neuroinformatics/about
https://www.frontiersin.org/journals/neuroinformatics/for-authors/author-guidelines
**PLOS ONE / Research Article**: accepts valid negative studies
but still requires meaningful original knowledge, exact protocols,
robust statistics, originality, public underlying data and honest
conclusions. Editorial soundness threshold is NOT zero.
https://journals.plos.org/plosone/s/criteria-for-publication
https://journals.plos.org/plosone/s/submission-guidelines

Do not assert any acceptance probability or automatic
journal ranking/quartile. Check latest Chinese Academy of Sciences
partition policy via institution when relevant.

## 5. Strict, shortest release gates
G1. Select ONE target journal and article type; format the primary
paper to that journal (not all journals at once).
G2. Confirm all human-authored title page, contributions, ethics of
secondary deidentified public EEG analysis, data use, funding,
conflicts, AI writing, code/data statement and cover letter.
G3. Independent numerical claim-to-artifact check for every
abstract/table AUC, paired confidence interval and p; distinguish
frozen vs later/post-outcome and historical CovScore vs PROPER NLL.
G4. Reduce Related Work to focused 4–8 directly relevant sources
and a clear comparison table: cohort, labels, target-X access,
classification/validation metric, fair source budget, effect.
Do NOT claim SOTA direct accuracy comparisons across mismatched
tasks or metrics.
G5. Include explicitly limited novelty and negative outcomes.
If a truly novel algorithm success is the primary story, a NEW
nondegenerate solver and external precommitted positive outcome
are mandatory, not wording edits.
G6. Full paper human technical review, valid references & figures,
fresh final PDF and supplementary code/data release inspection.
Only AFTER all six gates pass: journal upload for author approval.

## 6. Files and links reviewed
BSPC focused manuscript:
  research/defaultreg_bspc_v13/main_bspc_v31_EVIDENCE_GATED_CANDIDATE.tex
  GitHub Draft PR #140.
Related-work comparison paper section:
  research/defaultreg_bspc_v13/134_bspc_accepted_head_to_head_evidence_and_budgets.tex
  This branch has edits not necessarily merged into Draft PR #140.
Separate exploratory Lee GID study:
  GitHub Draft PR #141 (explicitly NEGATIVE).
Separate V26–V30 recording-identity paper:
  research/defaultreg_bspc_v13/main_bspc_v30_RECORD_IDENTITY_EVIDENCE.tex

**Current answer: Literature comparison sufficient for positioning;
head-to-head outcome evidence and official submission package
insufficient for an immediate submission.**
