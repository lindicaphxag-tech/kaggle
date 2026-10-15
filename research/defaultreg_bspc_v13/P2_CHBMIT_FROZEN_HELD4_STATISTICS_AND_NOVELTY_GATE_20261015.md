# P2 external CHB-MIT held-new four-source statistical and novelty gate

Status: **real publicly released CHB-MIT EDF digital samples, controlled transformations, 2026-10-15; not clinical duplicates; not journal-submitted**.

## Actual measurements / exact CI receipts

- [First externally sourced CHB01 and CHB02](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38666039518) SUCCESS, digital header-validated EDF partial HTTP 206, 2.4 million bytes per source, 8 bipolar EEG channels, 768 sample values. Frozen from prior EEGMMIDB: Pearson absolute 0.985, ≥4 distinct source-target sensor correspondences, common absolute lag ≤100, overlap ≥230 samples.
- [CHB01--06 expanded benchmark](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38666330505) SUCCESS, 6 real source EDF prefix SHA256 receipts, 78 class-labelled controlled transformation cases in independently uploaded JSON.
- Development/previously inspected CHB01--02: coherent-clock 16/16 positives, 0/10 constructed negatives; channelwise independent-lag Hungarian 16/16, 2/10.
- **New subjects held CHB03--06:** coherent-clock **32/32** positives, **0/20** constructed negatives; independently best-lag Hungarian **32/32** positives, **4/20** constructed negatives.
- The 4 held discrepancies occur one per new source, **all** on the hand-constructed asynchronous SAME SOURCE adversarial case. They would become true positives for a different estimand: any reused single-channel samples. For coherent synchronized multichannel recording ancestry, they are false positives.
- Across all six CHB sources the counts are 48/48, 0/30 versus 48/48, 6/30. Whole-file SHA for source EDFs was NOT verified: raw prefix URL HTTP 206, header SHA256 and 2.4MB prefix SHA256 *are* logged.

## Statistical scope

Do NOT use the 32 manipulated variants as 32 independent new clinical subjects. Independent CHB source units in the frozen new-held contrast: four.
- Subject-level difference in coherent-clock false matches: shared 0/4 affected sources, independent-lag 4/4 affected sources. With four discordant subjects favoring shared clock and zero opposing, the conventional two-sided exact sign/McNemar conditional test is **p = 0.125**, not less than 0.05. The test is descriptive and artificial by construction.
- The apparent 0/20 constructed-case false alarm count is NOT an estimated population-wide false positive rate; within-person negative manipulations are correlated.
- Positive outcome agreement is 32/32 versus 32/32, so there is **no improvement in sensitivity** shown over the fully informed strong comparator.
- This is cross-task/corpus feasibility corroboration on an independent EDF collection. It is NOT a validated live clinical record-provenance false alarm profile or an experiment on unmodified naturally repeated release files.

## Originality and promotion gate

Shared time synchronization is an established general signal-processing concept. Hungarian channel assignment and Pearson correlation are classical. Claiming PhD-level algorithmic novelty requires a genuinely new mechanism (e.g. model of shared low-frequency drift under channel loss and exact auditable provenance witnesses) that beats published synchronizers / approximate fingerprints / time warp baselines on independent, blind real signal corpora. No such result has been established yet.

This author-review method extension should not erase the original real Liu2024 1,225-pair released-EDF audit and its patient/acquisition-identity scope restriction. Original MAT-to-EDF ancestry is unresolved, author declarations unapproved, no BSPC journal submission executed.
