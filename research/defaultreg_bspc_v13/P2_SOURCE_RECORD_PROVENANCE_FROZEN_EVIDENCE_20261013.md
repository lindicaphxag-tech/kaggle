# P2 public Liu2024 source-record provenance — audited evidence freeze, 2026-10-13

**Study:** *When EEG Participant IDs Share Recording Contents* (BSPC candidate, Draft PR #144, NOT SUBMITTED).

## Five distinct claims / evidence layers

1. **Processing-dependent epoch identity.** With pinned MOABB 1.7.2/8–30 Hz/160 Hz, public release IDs 4 and 5 share 40/40 processed EEG epoch and label contents. Of 50 nominal public IDs, exact-content groups = 49, not 49 confirmed independent people.
2. **Publisher-supplied preprocessed EDF digital payload.** Exact, uncalibrated int16 channel streams across full EDF: 4/5 32/33 same; 47/49 and 11/15 have only 1/33 same. This is `edffile`, a publisher-preprocessed release layer, **NOT acquisition-original source MAT**.
3. **Native EDF temporal digital blocks.** [Completed run 38332413760](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38332413760). 29 labelled EEG channels * 320 native EDF record indices = 9,280 correlated-position comparisons per pair. Excluding constant blocks, aligned exact EEG blocks: 4/5 9,280/9,280; **47/49 6,957/9,280**; 11/15 171/9,280; 2/3 and 6/7 zero. For 47/49, 245/320 EDF record indices had >=90% of named EEG channels exactly matched; the longest consecutive run was 179 EDF records. These are storage blocks, not independent physiological trials.
4. **Independently published public source MAT products.** Figshare v5 file 38516555 `sourcedata.zip` (1,874,824,439 bytes), with 50 MAT members, including exactly one named corresponding public source file for each of 4,5,11,15,47,49. [Directory index 38332830221](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38332830221). [Six MAT numeric array range-only comparison SUCCESS 38333234894](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38333234894) read 228,125,389 HTTP partial-content body bytes and verified per-member CRC without downloading the full ZIP or publishing signals. For each of three pairs, a 40x33x4000 float64 numeric candidate array existed, nonexact yet extremely correlated across ALL coordinates: 4/5 r=.9999645144209475; 47/49 r=.9999576227072666; 11/15 r=.9999536828668547. **Critical falsification:** [Completed source MAT channel/trial audit 38333754177](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38333754177) found median corresponding full-channel correlations 0.625329 (4/5), 0.818365 (47/49), and 0.763512 (11/15), **zero** complete channels with r>=0.9999 for all pairs, and only 15/1,280, 14/1,280, and 10/1,280 exactly equal nonconstant nominal trial-by-channel blocks. Source MAT index 17 is constant with RMS ~187500, revealing a serious cross-channel offset confound in naively pooled r≈.99995. The naive source MAT pooled r does **not** confirm near-equality of source EEG waveforms. MAT-to-EDF numerical/channel mapping remains unverified.
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
- [x] Verify per-source-MAT-channel and nominal trial-block correlations; a severe global-correlation artefact was identified and reported
- [ ] Confirm MAT-to-EDF trial/channel/index relation and source-stage processing history
- [ ] Receive/verifiably cite release custodian's public nonidentifying provenance clarification
- [ ] Freeze immutable code/JSON checksums and source package with responsible data-use disclosures
- [ ] Inspect final *HEAD matching* PDF's physical page borders, ref/bib, source captions and unresolved TeX warnings
- [ ] Human author, ethics, declarations and publication rights sign-off
- [ ] Actual BSPC submission only after author-controlled approval

**No custodian reply, publisher acceptance or true clinical patient identity is claimed.**


## Independently rehashed complete public-EDF cohort (2026-10-13 follow-up)
- SUCCESS exact GitHub Actions https://github.com/lindicaphxag-tech/kaggle/actions/runs/38388066957
  (run artifact 11806394377, 50 original NEMAR EDF full-file SHA256 digests
  recomputed from downloaded public bytes, all matching NEMAR immutable
  provenance manifest at 11e3eeb4b439d5f8ee1b5e3b8e6032824bc2de83).
- All **1,225 unordered different released-ID pairs**, not a hand-picked
  subset, tested at 29 EEG channel labels and 320 native EDF records:
  - 4/5: 29/29 complete variable EEG channels identical; 9,280/9,280
    corresponding EEG-labelled EDF digital blocks exactly identical.
  - 47/49: 0/29 complete variable EEG channels identical, but
    6,957/9,280 corresponding native storage blocks exactly identical.
  - 11/15: 0/29 complete variable EEG channels identical,
    171/9,280 native storage blocks identical.
  - **Remaining 1,222 ID pairs**: zero full-channel and zero
    same-position native EEG-block byte-exact comparisons.
- The separate exact single-constant-CPz reference identity in all
  three nominated pairs has **no physiological fingerprint value**;
  CPz is excluded throughout the 29-channel census.
- The public EDF product was retrieved originally through MOABB 1.5.0
  in NEMAR; our primary 40-epoch alias comparison uses MOABB 1.7.2.
  The two tests are different representation levels.
- 1,225 pair comparisons are not 1,225 independent human patients.
  Storage blocks are not independent trials. The empirical zero
  count is conditional on this release and exact alignment,
  not a calibrated clinical false-positive probability.
- The source MAT->EDF export pipeline and original physical acquisition
  ancestry remain UNVERIFIED. No claims of duplicated clinical
  patient identity or publisher culpability.
- New executable:
  `research/defaultreg_bspc_v13/src/p2_nemar_complete_fifty_pairwise_digital_scan.py`.
- Manuscript subsection and full pair table:
  `research/defaultreg_bspc_v13/P2_nemar_complete_all_1225_pairwise_native_EEG_20261013.tex`.
- Exact source-head article build SUCCESS
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38388523969
  (21 pages, 6 overfull hbox warnings; not visually publication-cleared).
