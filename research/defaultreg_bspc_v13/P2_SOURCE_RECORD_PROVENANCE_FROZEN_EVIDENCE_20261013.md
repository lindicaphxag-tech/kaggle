# P2 public Liu2024 source-record provenance — audited evidence freeze, 2026-10-13

**Study:** *When EEG Participant IDs Share Recording Contents* (BSPC candidate, Draft PR #144, NOT SUBMITTED).

## Five distinct claims / evidence layers

1. **Processing-dependent epoch identity.** With pinned MOABB 1.7.2/8–30 Hz/160 Hz, public release IDs 4 and 5 share 40/40 processed EEG epoch and label contents. Of 50 nominal public IDs, exact-content groups = 49, not 49 confirmed independent people.
2. **Publisher-supplied preprocessed EDF digital payload.** Exact, uncalibrated int16 channel streams across full EDF: 4/5 32/33 same; 47/49 and 11/15 have only 1/33 same. This is `edffile`, a publisher-preprocessed release layer, **NOT acquisition-original source MAT**.
3. **Native EDF temporal digital blocks.** [Completed run 38332413760](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38332413760). 29 labelled EEG channels * 320 native EDF record indices = 9,280 correlated-position comparisons per pair. Excluding constant blocks, aligned exact EEG blocks: 4/5 9,280/9,280; **47/49 6,957/9,280**; 11/15 171/9,280; 2/3 and 6/7 zero. For 47/49, 245/320 EDF record indices had >=90% of named EEG channels exactly matched; the longest consecutive run was 179 EDF records. These are storage blocks, not independent physiological trials.
4. **Independently published public source MAT products.** Figshare v5 file 38516555 `sourcedata.zip` (1,874,824,439 bytes), with 50 MAT members, including exactly one named corresponding public source file for each of 4,5,11,15,47,49. [Directory index 38332830221](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38332830221). [Six MAT numeric array range-only comparison SUCCESS 38333234894](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38333234894) read 228,125,389 HTTP partial-content body bytes and verified per-member CRC without downloading the full ZIP or publishing signals. For each of three pairs, a 40x33x4000 float64 numeric candidate array existed, nonexact yet extremely correlated across ALL coordinates: 4/5 r=.9999645144209475; 47/49 r=.9999576227072666; 11/15 r=.9999536828668547. This does **not establish that EEG channels are independently highly similar** (one dominant channel may inflate r), nor that the MAT array corresponds to acquisition-original unprocessed signals; channel-wise anti-confounding test is separately required.
5. **Clinical acquisition identity:** **UNKNOWN**. No inference about the number or identity of people, clinical acquisition sessions, intentional copying or publication error is licensed by public numerical similarity alone. A respectful data-custodian technical clarification is drafted but NOT SENT; no external contact on the authors' behalf was taken.

## Scientific implications (limits matter)

- The main **49 content group** exact equality and matched software-factorial source fitting remain the primary result. The **47 assumed group** exclusion is only an exploratory, post-discovery scenario; it cannot be called 47 independent patient recordings.
- 47/49 is a **negative control for whole-record exact-byte identity only**; it is **NOT** a negative control for near duplicates. The 2/3 and 6/7 comparisons are nonrandom post-discovery diagnostic controls.
- Five public EEG acquisitions were checked by specific processing pipelines; four had no detected exact cross-ID epoch overlaps under that analysis. This cannot estimate population-wide alias prevalence or exclude unequal-length shifted/reordered recordings.
- Numeric array r≈1 across a large source MAT may be driven by one high-energy channel; do not assert acquisition-level duplication before the separate channel/trial decomposition result.
- Source EDF signals and the separately distributed source MAT files are **different data products**, not necessarily independent primary acquisitions; MAT-to-EDF channel mapping, label alignment and export pipeline have not been verified.
- The AUC contrasts are retrospective software comparisons on overlapping source refits and cannot be interpreted as causal effects on independent human patients.

## Submission gates
- [x] Source MAT ZIP central directory validated without full archive download
- [x] Six source MAT individual members selectively retrieved with CRC validation and numeric candidate analysis
- [x] 29 named EEG channel EDF temporal-localization diagnostic executed
- [ ] Verify per-source-MAT-channel and per-trial correlations to reject common-amplitude / reference-channel confounding
- [ ] Confirm MAT-to-EDF trial/channel/index relation and source-stage processing history
- [ ] Receive/verifiably cite release custodian's public nonidentifying provenance clarification
- [ ] Freeze immutable code/JSON checksums and source package with responsible data-use disclosures
- [ ] Inspect final *HEAD matching* PDF's physical page borders, ref/bib, source captions and unresolved TeX warnings
- [ ] Human author, ethics, declarations and publication rights sign-off
- [ ] Actual BSPC submission only after author-controlled approval

**No custodian reply, publisher acceptance or true clinical patient identity is claimed.**
