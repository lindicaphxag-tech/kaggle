# P2 shared-clock EEG release provenance — completed second held-subject experiment (2026-10-15)

**The P2 constructively validated endpoint is coherent multichannel recording ancestry, not individual-channel reuse or clinical identity.**
**This is an experimentally supported research candidate, not a journal-established original algorithm or a clinical screening accuracy guarantee.**

## First benchmark: strong comparator **tied**
- [GitHub run 38661594302](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38661594302) SUCCESS. Development S001/S002, held S003/S004 from original publicly released PhysioNet EEGMMIDB v1.0.0 EDF data.
- For eight controlled positive transformations per source and three original controlled negatives per source, shared-clock matcher **16/16 positives, 0/6 negatives falsely matched** on S003/S004.
- Fully informed **channel-permutation + independent-best-lag Hungarian** baseline also **16/16 positives, 0/6 false matches**, disproving a superior-detection claim based solely on weaker baselines.

## Critical adversarial comparison on real PhysioNet EEG waveform substrate
- [GitHub run 38662768329](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38662768329) SUCCESS, source version `8a65d5281527dee8352701871368aaaac2ec9bd2`.
- Added one `asynchronous_same_source` case per person: every candidate sensor uses genuine digital samples from the SAME PhysioNet source EDF, but distinct controlled delays `[7,23,39,55,71,87,13,29]`. This is a NEGATIVE only for **the endpoint of one coherent globally synchronized multichannel record**, not for the broader endpoint of *any* numerical channel content reuse.
- S003/S004: shared-clock matcher **16/16 positive transformations**, **0/8 false matches**; independent-lag Hungarian **16/16**, **2/8 false matches**. Each discordance is exactly the constructed asynchronous-same-source condition; all remaining three negative types are passed by both methods.

## New subjects, previously not inspected in earlier method-selection runs
- [GitHub run 38663141073](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38663141073) SUCCESS, source version `3b418b58a12fc6192af964a5bf065c89ad152961`.
- Clean revised implementation exhaustively searches candidate global lags rather than top-14 post-hoc candidates. Frozen numeric gates: abs Pearson threshold 0.985, >=4 distinct sensor assignments, >=230 overlapping samples and |lag|<=100. New held S005/S006; development S001/S002; public EEGMMIDB signals, 8 sensors, 768 samples per record. Input EDF SHA256 is saved in the JSON evidence artifact.
- S005/S006: **shared-clock matcher 16/16 positive, 0/8 negatives**; **independent-lag Hungarian 16/16 positive, 2/8 negatives**, again both discordances only for the constructed asynchronous-same-source case.
- Other tested controlled baselines: same-labelled shifted Pearson 12/16, zero-lag/channel-permutation Hungarian 10/16, zero-lag named Pearson 8/16, exact full numeric equality 2/16; none has a false match in this small constructed negative set.

## Limits that must stay visible
- The new held set adds two subjects **within the same PhysioNet source collection**, not a new instrument/hospital/acquisition platform.
- The negative design explicitly tests the chosen **coherent-clock unit** and cannot support superior generic sample reuse detection. Some applications may deliberately consider the asynchronous channel copies positive.
- Correlation matching and Hungarian assignment are established methods; the contribution of jointly constraining source clock must be demonstrated over existing synchronization/DTW/landmark-fingerprint baselines before any original-algorithm claim.
- Four constructed errors across two held pairs of subjects are NOT a population-level FPR estimate. Even 0/8 has a one-sided 95% independent-Bernoulli upper bound of about 31.2% under assumptions this clustered test does not satisfy.
- No formal clinical person identity, original MAT-to-EDF lineage, certified physiological source localization or proven cross-hospital prevalence.
- The current BSPC release-layer study and its full 1,225-pair NEMAR EDF audit remain valid as distinct release-data measurements. Adding this algorithm to the main claims requires further original validation and manuscript/ethics/author approval; **no journal submission performed**.

## Deliverables
- Run-linked Python source + frozen input checksums in two immutable GitHub Action artifacts.
- Downloadable ChatGPT experimental package with reproduced 4-panel data figure, conditional theorem caveat, JSON results and runnable code; binaries NOT on this branch.
- Initial descriptive NEMAR 1,225-pair digital EDF audit at [CI 38388066957](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38388066957).

**Statistical confidence and original contribution are open review gates.**
