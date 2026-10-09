# Frozen prospective EEG active-query audit — 2026-10-09

## Original evidence (not a claimed SOTA method)
- Run: https://github.com/lindicaphxag-tech/kaggle/actions/runs/37834788421 — SUCCESS.
- Original artifact: https://github.com/lindicaphxag-tech/kaggle/actions/runs/37834788421/artifacts/11576114948 .
- Raw source: 48 EEGMMIDB EDF recordings with SHA-256 hashes; subjects 1–8 train, 9–12 validation, 13–16 test, task runs 4/8/12, 180 test epochs.
- Pre-committed split and code: https://github.com/lindicaphxag-tech/kaggle/blob/research/eeg-cert-matched-selective-baseline-20261009/research/eeg/PROSPECTIVE_ACTIVE_PROVENANCE_V1.md .
- The one-off random query comparator in the final run had been fixed so it only asks *still-informative* questions. Do not mix the previous comparator run #37834541388 with the final #37834788421.

## Exact descriptive real-data outcomes
| Outcome | Observed |
|---|---:|
| No-query authorization | 146/180 (81.11%) |
| Minimax active provenance queries | 45 total (0.2500 per trial) |
| One seeded fair random informative query policy | 50 total (0.2778 per trial) |
| Net fewer active queries | 5 across 180 trials (0.02778 / trial) |
| Trials with active fewer / equal / more queries | 9 / 167 / 4 |
| Active vs random query pairs | (0,0) 146; (1,1) 14; (1,2) 9; (2,1) 4; (2,2) 7 |
| Frozen model BAcc with known identity | 0.66667 |
| Frozen model BAcc on randomly swapped input | 0.67222 |
| Frozen model BAcc after active queries | 0.66667 |
| Violations of claimed *prediction invariance within the enumerated mapping set* | 0 |

### Important limitations on evidence strength
- Ordinary iid **trial-level** bootstrap (20,000 fixed-seed draws) descriptive 95% CI on mean saved queries: **[-0.01111, +0.06667] per trial**, includes zero.
- An exploratory per-trial sign test on 9 vs 4 non-tied trials gives two-sided p=**0.2668457**. Trials from one participant are correlated, so **this is NOT a valid independent-subject significance test**.
- Frozen per-trial artifact does **not** contain explicit per-trial subject IDs; subject-clustered uncertainty therefore cannot be verified from this artifact. Do not infer a subject-stratified confidence interval from the test run or artificially treat 180 trials as independent people.
- Only one seeded random-control schedule, two narrow two-electrode ambiguity pairs, and a simulated oracle supplying true physical electrode IDs; no physical electrode swap, no actual interaction with technicians.
- The output sign is guaranteed to match the fully known-mapping *same frozen model*, not the true motor imagery label. Not clinical safety certification.
- Directly corrupted BAcc **exceeded** known-identity BAcc on this test by 0.56 percentage points; there is **no task accuracy gain**.
- Dynamic programming for finite decision tables/minimax decision trees is not itself an original theorem. No external replication of this new algorithm or peer-reviewed acceptance.

## Research go/no-go recommendation
**NO-GO as a flagship claim of improved EEG classification or query efficiency.** The realized five saved queries are weak, and the within-subject covariance means the trial-wise CI is optimistic as independent-subject evidence.

**GO as a reproducible negative/diagnostic benchmark and infrastructure** linked to established Braindecode model integrations. The next meaningful original study needs a pretrained EEG foundation-model representation, wider and physically grounded identity uncertainty, a *prospective* unseen cohort, strong query-cost/coverage competitors with fixed information, and independent outside reproduction.

Do not re-label this already-inspected cohort as prospective for a post-hoc method change. New model choices must be chosen on validation/other development data and tested on a fresh holdout.
