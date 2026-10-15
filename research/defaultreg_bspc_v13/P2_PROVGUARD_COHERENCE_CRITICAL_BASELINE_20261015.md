# P2 — shared-clock provenance as a narrow, falsifiable positive algorithmic hypothesis

**Status 2026-10-15: NOT scientific superiority and NOT clinical patient identification.**

## Resolved first real-EEG benchmark
- Public PhysioNet EEGMMIDB EDF v1.0.0, S001–S002 development, S003–S004 held.
- Fixed 8 EEG channels, 768 samples per record and 8 transformations per source; labels are artificially constructed **controlled transformations** of public EEG, not verified naturally recurring clinical episodes.
- Fully informed comparator `permutation+independent lag Hungarian` at old held protocol: **16/16 TPs, 0/6 FPs**, exactly matching ProvGuard **16/16, 0/6**. The older claim of improvement against weaker labelled/zero-shift/whole-signal exact baselines does not establish superiority over this strong comparator.
- CI: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38661594302 . Source and JSON artifact are versioned.
- Do not headline stronger-than-Hungarian gain from this result.

## Distinct hypothesis: common-clock digital provenance
The original content matcher allows source-channel and target-channel names to differ; the **new claim** is instead that at least k independent EEG sensor matches jointly admit a single time offset `ell`. Contrast this with an unconstrained per-channel best-offset Hungarian assignment, which can produce k individual matches at incompatible offsets.

The test deliberately introduces a new constructed negative, `asynchronous_same_source`: all eight sensor traces come from **the SAME public EEG source** but each is shifted by a different known amount. It is negative **ONLY** for the specifically defined endpoint "a common multichannel source segment with one global clock". It is positive for the broader endpoint "any per-channel numerical content reuse". Both definitions must be shown so readers cannot mistake a change of estimand for a universal detection gain.

- Frozen sensor delays [7,23,39,55,71,87,13,29] samples.
- Positive cases (8 types) unchanged.
- Frozen source signal, threshold, minimum matched sensors and lag horizon before held validation; P2 CI:
  https://github.com/lindicaphxag-tech/kaggle/actions/workflows/p2-provenance-consensus-physionet-heldtest.yml
- Once run completed, compare the real-EEG held results, not solely the synthetic smoke test.

## Elementary conditional false-alarm bound — **standard probability, not original theorem**
For C independent null sensors whose apparent best lags are uniform over L discrete lags, the chance that any k sensors share one lag is at most `L*choose(C,k)/L**k = choose(C,k)/L**(k-1)` by a union bound. For C=8,k=4,L=241 this is ~5.00e-6 **only under the idealized independence-uniform-lag null model**. EEG interchannel correlation violates these assumptions and the number must not be reported as an empirical clinical false positive rate. This is not a new mathematical result.

## Next actual publication-quality demonstration
1. Independent public EEG families and naturally observed release reuse with validated ground truth, not only constructed shifts on one dataset.
2. Strong baseline comparison with multichannel synchronization, robust cross-correlation, approximate fingerprinting and signal alignment/DTW.
3. Frozen threshold calibration, independent across-acquisition test, confidence intervals / high-error challenge cases; report loss under independent per-sensor clock warps.
4. Runtime, memory, error localization and abstention curves, no patients inferred from signal equality.
5. If evidence remains limited to a novel labelling of well-known signal synchrony, position as a useful evaluation tool or release integrity study, **not** a high-level new EEG decoding algorithm.

**Related contemporary work**: Olsen et al., EEG data alignment across devices, EMBC (2025), DOI 10.1109/EMBC58623.2025.11254069; Rong et al., electrode-configuration robustness in EEG source imaging, Clinical Neurophysiology (2025), DOI 10.1016/j.clinph.2025.04.009.
