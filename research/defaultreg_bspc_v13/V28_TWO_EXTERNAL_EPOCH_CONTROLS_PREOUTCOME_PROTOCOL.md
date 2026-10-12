# V28 cross-acquisition exact-epoch integrity negative controls

Protocol freeze: 2026-10-13. These cohorts and results of their EEG decoding tasks were previously exposed; this is **retrospective data-integrity testing, not a fresh performance test, new population validation, independent trial registration, or new Z0 selector**.

Exactly two prechosen public acquisitions:
1. Weibo2014, subjects 1..10, left/right MI, electrodes C3 Cz C4, MOABB MotorImagery: fmin 8, fmax 30, resample 160, baseline None, reject_by_annotation False.
2. BNCI2014_002, subjects 1..14, right-hand/feet MI, all native EEG channels under the original frozen runner; same MOABB filter/time settings. This uses the original `dataset_spec` and native original `MotorImagery`, not target-optimised preprocessing.

For every participant retain all trials from the fixed two-class paradigm; one binary class must not be missing, and all signals must be finite, equal in EEG channel/sample geometry within acquisition, with at least 20 trials of each class. Full contiguous float64 trial signal and dimensions are hashed; collisions are confirmed by exact numerical equality; discordant labels fail closed. Expose no waveforms or subject-level clinical/personal identifiers beyond already public dataset IDs.

Primary results: total epoch instances, unique exact epoch contents, within-ID repeats, exact inter-ID overlap edges, graph-connected content groups. Compare with Liu2024 V27 exact released-record finding; neither assume positive nor negative controls. Do not choose sample-similarity thresholds, shift tolerance or subject subsets after inspection. Do not combine these cohorts in a joint AUC analysis.

Stop/go: If unexpected exact cross-ID repetitions occur, block reuse of that dataset as a strict record-independent LOSO control until a content-safe regroup/refit. If no repetitions, report precisely that negative exact-content result but not absence of approximate or overlapping time-window reuse.
