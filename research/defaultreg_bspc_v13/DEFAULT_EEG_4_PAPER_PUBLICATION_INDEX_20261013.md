# Default EEG journal portfolio — separated evidence ledger
Date: 2026-10-13 | GitHub repository: lindicaphxag-tech/kaggle
**Four independent DRAFTS, NOT four publication-ready articles.
Two realistically standalone completed research stories (P1/P2);
two conditional research/technical notes (P3/P4).**
Each is on an independent branch and a separate draft pull request.


## Selected journal assignment (2026-10-13; 2025 ORIGINAL CAS ranking)
| Track | Primary target | 2025 CAS major area | Fee route | Status |
|---|---|---|---|---|
| P1 [#143](https://github.com/lindicaphxag-tech/kaggle/pull/143) | **Journal of Neural Engineering (IOP)** | Medicine 3 | Explicitly free for subscription-access | Selected for manuscript preparation; NOT submitted |
| P2 [#144](https://github.com/lindicaphxag-tech/kaggle/pull/144) | **Biomedical Signal Processing and Control (Elsevier)** | Medicine 2 | Subscription route ordinarily avoids OA APC | Selected for manuscript preparation; NOT submitted |
| P3 [#145](https://github.com/lindicaphxag-tech/kaggle/pull/145) | Brain Topography (Springer) | Medicine 3 | Subscription generally no OA APC, possible extras | PROVISIONAL ONLY; major new independent physiology/evidence needed |
| P4 [#146](https://github.com/lindicaphxag-tech/kaggle/pull/146) | Digital Signal Processing (Elsevier) | Engineering & Technology 3 | Subscription route ordinarily avoids OA APC | PROVISIONAL ONLY; independent numerical/algorithmic gains needed |

*2026 'new journal partition' is NOT treated as the last CAS journal
partition. The university's current promotion regulations govern any
official classification. Fee statements mean OA APC only, not a promise
that no submission, page, or colour charge applies in all cases.*

Publisher references:
- JNE scope and FREE subscription publishing:
  https://publishingsupport.iopscience.iop.org/journals/journal-of-neural-engineering/about-journal-neural-engineering/
- BSPC scope benchmark and data/signal reliability:
  https://shop.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094
- Brain Topography:
  https://link.springer.com/journal/10548
- Digital Signal Processing:
  https://shop.elsevier.com/journals/digital-signal-processing/1051-2004

**A chosen journal is not a scientific pass.** P1 still needs
result-to-artifact full checks and author-approved data/declarations;
P2 additionally needs clinical data-custodian outreach and a
raw-data identity reliability review. P3 and P4 remain HOLD;
never submit redundant P1 negative case-control results as P3.

## Direct reader and corresponding-author navigation

| Paper | Main scientific object | Independent research branch / Draft PR | PDF Actions evidence | Stage |
|---|---|---|---|---|
| P1 | Historical CovScore vs proper Gaussian SOURCE predictive NLL vs per-person AUC; exact three-factor source score anatomy | [PR #143](https://github.com/lindicaphxag-tech/kaggle/pull/143); `publication/default-eeg-p1-estimand-20261013` | [P1 PDF successful](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38311636570/artifacts/11774099782) | Full stand-alone draft; author/claims/official style gate |
| P2 | EEG cross-ID EXACT content identity; 49 recording-content groups; frozen 2x2 source-fit x source-lambda comparison | [PR #144](https://github.com/lindicaphxag-tech/kaggle/pull/144); `publication/default-eeg-p2-record-identity-20261013` | [P2 PDF successful](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38311640037/artifacts/11775101562) | Full stand-alone draft; custodian/ethics/repro and journal fit gate |
| P3 | Different source-only algorithms fail to demonstrate repeatable new-person benefit; trial-by-trial negative accountability | [PR #145](https://github.com/lindicaphxag-tech/kaggle/pull/145); `publication/default-eeg-p3-transfer-negative-20261013` | [P3 PDF successful](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38311644823/artifacts/11774762329) | **Research hold**: non-identical candidate methods and overlapping P1 cohorts |
| P4 | Whole-numerator Bernstein polynomial certificate of fixed SOURCE pair ordering on shrinkage intervals | [PR #146](https://github.com/lindicaphxag-tech/kaggle/pull/146); `publication/default-eeg-p4-rank-certificates-20261013` | [P4 PDF successful](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38311883793/artifacts/11774294715) | **Technical note hold**: only source coverage +0.146pp and +0.064pp |

Existing principal research PR [#140](https://github.com/lindicaphxag-tech/kaggle/pull/140) contains shared reproducible original experiments; peer-review comparison [#142](https://github.com/lindicaphxag-tech/kaggle/pull/142) is editorial context rather than a fifth paper.

## Hard science ownership: one primary conclusion per manuscript
**P1 — Predictive likelihood vs source validation:**
Original EEGMMIDB S021-S040 prospective 20-person macro ROC-AUC
difference +0.0561 vs HISTORICAL class-centred CovScore with
Holm p=0.00597, FIVE harmed people. Source-only THREE-factor
diagnostics on exposed Cho2017, Dreyer2023A, BNCI2014_002:
98.59%, 99.29%, 99.77% of ADDITIONAL mean-shift score penalty
in fixed-linear-AUC-invariant components; 38 source validation
groups, independent SHA-audit. This is NOT an accuracy lift.
No cross-ID clinical record claims.

**P2 — Recording-content identity and held-record estimands:**
Liu2024 released 50 IDs, 2000 trial instances but 1960 distinct
exact processed contents; ID4/ID5 overlap all 40 processed epochs
under a specified pipeline, 49 record-content groups.
Weibo2014/BNCI2014_002 are zero-exact-match controls, not
separate positive replication of identity effects.
Matched 49-record source macro 2x2:
.588673(original fit old lambda), .587347(original fit new lambda),
.580765(purged fit old lambda), .582551(purged fit new lambda).
Nominal 50-ID .594000 is a DIFFERENT reporting denominator.
No human-patient identity claim; no causal leakage-size claim.
Notify/check original dataset custodians before publication.

**P3 — Algorithm falsification suite:**
Weibo2014 continuous vs grid tie at .731405; Cho2017 SCE 40-person
−.00005264 and lambda1 degeneration (second half sequential);
Dreyer 12-person anisotropy alpha0, delta −.001088; Lee2019
source forecast NEGATIVE, held A 12-person delta −.003933
(p=.1289), held B halted BEFORE target IO on source-covariance
SHA mismatch. These trials use DIFFERENT new candidate
methods and are NOT a single precommitted algorithm family.
P1 used some Cho/Weibo negative data as contextual controls,
so a stand-alone P3 would need NEW DISTINCT primary outcomes
and clear overlap disclosures before submission.

**P4 — Numerical source rank certification:**
On exposed three-channel Weibo2014 source validation,
nine source groups, p6 tangent geometry, V23 Taylor
certificate 94.121% stable source pairs full interval
vs V25 Bernstein 94.267%; between selectors 96.274%
vs 96.338%. This is certificate coverage,
not held-target AUC and not machine-verified
floating-point proof. Publication requires independent
numerical tests and an impact endpoint beyond +0.146pp.

## Overlap/anti-salami ledger
| Shared acquired EEG | P1 scientific use | P2 scientific use | P3 scientific use | P4 use |
|---|---|---|---|---|
| EEGMMIDB | historical held participant AUC + mechanism | none | background only | none |
| Cho2017 | retrospective source score anatomy; historical SCE negative control | none | would reuse same 40 Cho target rows; MUST resolve overlap | none |
| Dreyer2023A | source score anatomy and alpha0 negative | none | would reuse alpha0 results; MUST resolve overlap | none |
| BNCI2014_002 | source score anatomy | negative processed cross-ID identity check | background only | none |
| Weibo2014 | historical continuous negative validity control | negative processed cross-ID identity check | same held 10 target rows; MUST resolve overlap | same nine held SOURCE validation groups, DIFFERENT certificate endpoint |
| Liu2024 | NO data overlap claim in P1 | sole positive data integrity finding and 49 fit x parameter outputs | none | none |
| Lee2019_MI | no new abstract performance claim in P1 | none | GID frozen source and held A only; B FAILED | none |

P1+P2 may be separate because hypotheses, primary endpoints and
affected SIGNAL provenance are distinct, but must STILL fully
declare use of Weibo2014/BNCI as shared public cohort controls
and cite any relevant publicly disclosed companion manuscripts.
P3 is NOT automatically independent merely because it has a
different title: it would need new outcome evidence.
P4 methods technical result must disclose Weibo source fold reuse.
Never count overlapping people as separate independent discoveries.

## Publication priority and editorial placement (not an acceptance promise)
**Priority 1 P1:** exact source criterion estimand EEG methodology;
initial JNE / Brain Topography journal suitability inspection.
JNE subscription-access publication has no fee:
https://publishingsupport.iopscience.iop.org/journals/journal-of-neural-engineering/about-journal-neural-engineering/
JNE structured abstract was generated for P1. Journal specific
scope, impact, 2025 official CAS tier and institutional rules must
be checked afresh before submission.
**Priority 2 P2:** EEG record-identity measurement integrity,
BSPC publisher scope includes signal quality,
benchmark and dataset studies:
https://shop.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094
Approval from relevant dataset source team and strong independent
raw/processed duplicate validation are essential.
**P3/P4:** retain as rigorous research inventory; no simultaneous
redundant submissions; candidate workshops or independent full
methods notes only after substantive new experiments.

## Common author release gate
Author identity/order/affiliations/CRediT, funding, human subjects
secondary-use ethics, publication permissions, licenses,
competing interests, AI use declarations, code/data availability
with immutable versions, source hashes, analysis version and
journal-specific final PDF MUST be reviewed by actual
authors. GitHub Draft PR and CI PDF are NOT journal submission
or a journal review and cannot establish likelihood of acceptance.
