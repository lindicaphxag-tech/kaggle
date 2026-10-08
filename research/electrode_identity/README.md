# Source-anchored EEG Electrode Identity: selective repair vs. refusal

**Research question, not an already validated invention:** A pretrained EEG foundation model such as [NeuroRVQ](https://github.com/braindecode/braindecode/pull/1218) may require channel names because its spatial positional embedding slots are channel-specific. If the **metadata names are wrong** (not merely in a different order), a shape-valid signal may be semantically wrong. Under what assumptions can a runtime detect, authorize correction, or abstain?

## Claim boundary

This **prototype is source-conditioned, no-pretrained-model** research on the sensor *metadata* layer. It is NOT a NeuroRVQ frozen-weights experiment, NOT actual EEG miswired sensor observation, NOT a new EEG clinical diagnostic or a formal statistical risk guarantee, and NOT an independently validated publication. The original NeuroRVQ architecture/paper/weights belong to the upstream researchers; [merged Braindecode #1218](https://github.com/braindecode/braindecode/pull/1218) and [merged #1223](https://github.com/braindecode/braindecode/pull/1223) are model/tooling integrations only.

This differs from **known-coordinate interpolation and channel permutation-equivariance** because the *declared identity itself* is treated as uncertain and corrections require evidence rather than being assumed safe. Assignment via Hungarian optimization, correlations and ambiguity margins are **standard tools**, not claimed as original theorems. Relevant nearest prior art requiring comparison: DIVER-0 (channel equivariance), HEAR (coordinate-conditioned embeddings), cross-montage EEG adapter methods, classical electrode-identity localization and multisensor assignment.

## Fixed exploratory method

- Requires trusted physical channel-anchor identities and a correct **same-subject reference recording**. These are strong assumptions that must be made explicit to any reviewer.
- From **two temporally disjoint source/target EEG segment pairs**, calculate each uncertain channel's Pearson-correlation vector against the trusted anchor signals. Each produces a quadratic cost between source *anatomical roles* and target *raw channel rows*.
- Solve a bijective minimum-cost assignment with SciPy Hungarian optimization. For every chosen channel mapping, **forbid it and solve again**; the closest globally different assignment gives an ambiguity margin. This is not a greedy local runner-up.
- Authorize a changed channel assignment only when BOTH temporally separated pairs return exactly the same assignment, and each global gap/channel is at least the **fixed heuristic 0.012**. If they disagree or the best alternative is indistinguishable, return `refuse`. If both identify no change, return `retain_identity`, not an unnecessary correction.
- Reject NaNs, zero-variance channels/anchors, wrong anchor count and post-hoc copying of assignments without two-witness authorization.
- The evidence score is **not a posterior probability, error bound, conformal guarantee, or safety certificate**. Trusted anchor corruption can invalidate all conclusions.

## Real-data source and frozen evaluation before results

Public open-access [PhysioNet EEG Motor Movement/Imagery Dataset, v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/); credit Schalk et al., DOI [10.13026/C28G6P](https://doi.org/10.13026/C28G6P), published 2009.

- Source subjects: `S001`, `S002`. Original files `S001R01.edf`, `S001R02.edf`, `S002R01.edf`, `S002R02.edf`. R01 = eyes open, R02 = eyes closed. These are two independent individuals and four real 64-electrode recordings, **not sixteen independent biological replicates**.
- Sampling 160 Hz, zero-phase 4th-order Butterworth 1–35Hz, first 60s, all source physical channel labels canonicalized via lowercase/trailing period removal. Fix 12 uncertain-channel role indices from every fifth alphabetical electrode name, plus the first 16 other names as trusted anchors. These exact roles are frozen before outcome.
- Each independent *same-baseline* witness pairs R01 0–12s vs 30–42s and 12–24s vs 42–54s (no same-waveform oracle). A second condition compares R01 0–12/12–24s to R02 0–12/12–24s (true resting EEG state shift).
- At target, inject **only known numerical metadata channel permutations**: no change, a one-pair swap, two disjoint swaps, four disjoint swaps among the 12 uncertain channels; anchors never corrupted. Ground-truth permutations are used for scoring **after** the algorithm returns, not its decision.
- 2 subjects × 2 conditions × 4 corruption severities = 16 tests. Primary source-level metrics: authorization/refusal counts, **correct** versus **wrong** authorized corrections, and **false correction under no corruption**; always record all 16 cells including failures. Failures are evidence of insufficiency, never selectively dropped. No hyperparameter search or threshold adaptation in this suite.
- Synthetic unit tests independently ensure positive recovery on well-conditioned distinct anchor signatures, no-op on clean inputs, refusal when two roles are identical in all observed anchor fingerprints, refusal when segments conflict and nonfinite evidence rejection. **The synthetic tests establish implementation sanity, not real-data efficacy.**
- GitHub Actions fetches public EDF files at runtime but **never commits, uploads or shares raw patient recordings**. Only public file SHA-256, fixed method and aggregate per-condition decisions are saved.

## What would be needed for a strong original method manuscript?

(1) Additional independent individuals, recordings, ages, hardware, references, electrode montages and naturally occuring metadata errors (rather than controlled synthetic reindex); (2) compare strict always-refuse, ordinary known-coordinate interpolation, unguarded assignment, and an expert physical montage oracle; (3) real **unchanged pretrained NeuroRVQ** representation/prediction impact under correct vs corrupted vs selectively repaired metadata on new subjects; (4) an independent investigator test plus true calibrated selective-risk/coverage bounds with a held-out calibration set; (5) robust failure/identifiability analysis when physical anchors are also uncertain; (6) comparison with sensor permutation registration, source localization, channel adaptation and channel-equivariant models from existing literature.

**Preliminary go/no-go:** If real R01/R02 distributions do not support nonzero correct-repair coverage with near-zero false autorization, record the negative and do not upgrade it into a novelty or safety claim. No external endorsement is asserted just because this artifact was created in an author-owned repository.

## Reproduce

```bash
python -m pip install numpy scipy mne pytest
python -m pytest -q research/electrode_identity/test_anchor_identity.py
python -m research.electrode_identity.public_eeg_pilot --cache /tmp/physionet-edf --output /tmp/eeg_identity_report.json
```

[Public reproducibility workflow](../../.github/workflows/eeg-source-anchored-identity.yml). The intentionally awkward research control is the key: if two source electrode roles are observationally identical relative to every trusted anchor, the true physical swap cannot be recovered from this evidence, and the system MUST refuse.
