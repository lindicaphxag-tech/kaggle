# Default EEG V28: editorial separation and submission promotion gate
_Date: 2026-10-13. Living internal review memo; NOT a journal endorsement or acceptance estimate._

## The most important editorial decision

Do NOT pack V22-V28 as equally strong main-paper contributions.
The accepted 2026 BSPC article Gil & Hernandez-Sabate (BSPC 120:110020,
doi:10.1016/j.bspc.2026.110020) already treats participant hierarchy,
leakage-safe validation and mixed-effects analysis. The 2026 Cortex
Zhang et al. paper (201:171–189, doi:10.1016/j.cortex.2026.05.005)
already studies EEG regularization and cross-validation sensitivity.
Generic split rules, simple alpha tradeoffs and shrinkage search are
PRIOR ART.

**Primary candidate (BSPC methods/scientific-measurement paper):**
"When Regularisation Selection Matters: Predictive Validation versus
Discriminative Transfer in Cross-Subject EEG."

Three central, independently understandable claims:

1. A validation-class-centred covariance likelihood uses held-source
   class means and is NOT a source-mean predictive Gaussian likelihood.
   Exact omitted nonnegative mean-shift term, source-only implementation
   and retrospective correct comparisons are central.
2. Different validation objectives can change selected lambda, but not
   reliably improve unknown-person rank ordering or adverse-subject risk.
   Include the frozen positive EEGMMIDB result, its harmed participants,
   the non-significant BNCI families, the negative CVaR comparison and
   independent-acquisition Weibo2014 exact equality (10/10 choices).
3. Finite-source continuous optimization can be numerically bounded,
   yet optimizing source AUC more precisely is not a target-generalisation
   theorem. Present the bounded spectral path briefly; complete
   Bernstein algebra remains in auditable online companion.

**Liu2024 V25-V28 must NOT be pooled into main EEG evidence as 50
independent people.** The raw release contained duplicated processed
record arrays under IDs 4/5. V27 independently scanned all 2,000
epochs, finding 1,960 unique trial contents and no additional exact
cross-ID partial overlap; V28 reconstructs 49 safe groups.
Until the full 49-group refit has finished and passed independent
cross-artifact checks, mark the Liu outcome as *data-integrity-qualified
retrospective sensitivity*, not another confirmatory algorithm gain.

Potential SECOND technical/data-provenance note:
"EEG Subject Identifiers Are Not Sufficient Split Units:
An Exact Recording/Epoch Provenance Audit of a Public Dataset".
This paper would require genuinely reusable provenance checks across
at least TWO additional data releases or independent sites, impact
quantification from full data/group safe re-fits, and a clear distinction
between released identifiers, recording content and actual patients.
A one-dataset hashing demonstration is NOT enough for a top-tier
original full article.

## Strongest rejection risks and STOP rules

- **Novelty:** The exact mean-shift decomposition is elementary algebra
  and the continuous bound is classical interval/spectral analysis.
  Originality must rest on the new measurement *consequence*,
  not a claim of discovering regularisation or graph clustering.
- **Outcome inflation:** Keep no target-AUC gains from V24 continuous
  search (Weibo gain over its original grid = precisely 0).
  Never quietly substitute class-centred covariance for correctly
  predictive source-mean Gaussian baseline.
- **Information isolation:** A pre-outcome source-choice SHA is NOT
  independent preregistration. Any study that globally reads target
  EEG to discover content identity is NOT physical target-isolated Z0.
- **Cross-study dependence:** Overlapping LOPO training pools and
  10 or 49 held observations do not become statistically independent
  studies. Do not print confirmatory p values for retrospective repairs.
- **Clinical/transfer relevance:** EEG AUC alone cannot support
  patient-safety, calibration-time savings, clinical utility or broad
  transfer-rule guarantees. Report signed individual harms, trial
  counts, data source and deployment-information budget.
- **Manuscript scope:** 13p primary and expanded companion preferable
  to a 25+p theory/forensics dump. Prior published BSPC 2026
  participant-hierarchy paper was 12 pages; page length by itself
  is not evidence of novelty.
- **Engineering reproducibility:** Main manuscript only quotes
  output from green CI run + immutable artifact + full source
  selection hash; any red/unfinished run remains explicitly pending.

## Current real, post-audit verified receipts

- Weibo2014 V24, 10 held people/1580 trials: both source macro-grid
  and anchored continuous target AUC 0.7314052933673468; same lambda
  in 10/10, correct Gaussian predictive 0.7293584183673468.
- BNCI2014_002 first-source-pool V24: finite source grid and
  repaired continuous AUC both 0.7939663461538462; residual conditional
  SOURCE optimality upper gap 0.016225961539461542. Not target AUC.
- Liu2024 V27: exact epoch overlap run 38287715932,
  50 IDs, 2,000 instances, 1,960 unique epoch contents, one
  shared 40-trial edge between IDs 4,5.
- V28 independent reconstruction run 38288719062:
  49 groups, all 49 trial-disjoint split contracts pass, JSON
  SHA256 5b50866f464a4d8f272ed3f90e0e6faac29c479f58008fbef983448242e8d484.
- V26 full re-fit on 49 unique records: PENDING verified completion
  and artifact reconciliation as of this memo. Do not quote gains.
