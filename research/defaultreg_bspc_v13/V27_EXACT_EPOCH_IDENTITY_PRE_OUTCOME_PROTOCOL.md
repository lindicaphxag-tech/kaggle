# V27: Exact EEG epoch-overlap integrity audit (pre-outcome protocol)

Date: 2026-10-13. Retrospective integrity audit on already exposed public Liu2024 and on already exposed V26 record groups. NOT a prospective clinical experiment or an independent pre-registration. **No classifier targets, AUC, or outcome-driven heuristics are used to declare duplicates.**

## Why full-record hashes are insufficient

V26 confirmed one identical 40-trial processed EEG record group across subject IDs 4/5, retaining 49 unique full processed records. It does not logically rule out partial repetition: two different whole-record SHA256 digests may still contain one or more identical EEG epochs (possibly shared with different subject IDs). Such overlap, if used across source-training/outer-held boundaries, would invalidate a strict *epoch-independent* transfer claim.

## Frozen primary audit

- Public `MOABB==1.7.2` Liu2024, subjects 1..50; `MotorImagery` events left/right; 8–30 Hz, 160 Hz, no baseline, no artifact rejection by annotation; require EXACT 40 trials per ID, 29 EEG channels, 20 trials/class.
- For each trial, compute SHA256 over the full float64 contiguous trial array **including channel/sample dimensions**; the event label is stored separately. Confirm equality by exact NumPy element comparison when hashes collide.
- If the same EEG trial content has different class labels in any subject, fail closed and retain anomaly report; never silently use a duplicate.
- Examine every one of the 2,000 trials with equal inclusion and no AUC-based subset. Report subject-pair shared-trial counts and within-ID repetitions separately. No similarity threshold, time shift, selective filtering, Pearson/cosine score, or manual exclusion is allowed in the primary audit.
- Build connected components of SUBJECT IDs linked by ≥1 exactly repeated cross-ID EEG trial; groups need not be identical full 40-trial records. This is the minimal subject-group split to avoid cross-ID exact-epoch overlap. Count all unique exact epoch hashes and all repeated cross-ID trial matches.
- Compare the epoch-overlap connected components to the original V26 full-record grouping. Differences are **integrity findings**, not automatically improvements in AUC. Preserve known V26 group [4,5] and determine whether other partial-overlap groups exist; do NOT assume either answer.
- Keep raw EEG arrays off public outputs: only aggregates, subject IDs, shape, class counts and cryptographic report digests. The existence and extent of such overlaps do not establish actual patient identity.

## Independent validation and promotion gate

1. Unit tests must detect a *partial* overlap across otherwise different whole records, transitive components, a contradictory class label, within-subject exact repetition, full-record duplicates, and hash collisions.
2. Download released signals only in pinned official MOABB pipeline, run public GitHub Actions, SHA-freeze JSON before interpreting findings.
3. If partial overlaps outside [4,5] are present, the previously announced 49-record identity reanalysis is insufficient: require a source-only, re-fitted grouped evaluation before quoting an epoch-independent transfer result. That would be an additional retrospective sensitivity analysis, not an untouched cohort.
4. If no extra partial overlap is found, report this precise **negative integrity result**; it strengthens V26 record-group separation but does not prove absence of near-duplicates, overlapping *time windows*, participant correlations, or patient identity reuse.
5. No target EEG outcomes from this task are to be used for model selection. Group discovery uses held target EEG content, so this study is **not strict physical target-isolated Z0**.
