# Frozen pretrained NeuroRVQ: real EDF electrode-binding mechanism control

**Experiment completed 2026-10-09. Exploratory mechanism evidence, not a method-level performance result.**

## Immutable external evidence
- GitHub Actions [success run #37896626800](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37896626800).
- [Original JSON result, real EDF hash and full execution log (artifact #11599934891)](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37896626800/artifacts/11599934891).
- Original-source Braindecode pinned commit: `a619f4b38333de6060811ffb8a5a4b1104551ff9`.
- Pretrained published model: `braindecode/neurorvq-eeg-pretrained`, resolved model revision `1678a36cd1cdf6c81e097f4b635f87519c0ec366`, CC BY-NC 4.0 (research/non-commercial).
- PhysioNet EEGMMIDB S001R04, raw EDF SHA-256 `3d161f88e1c00632585287d2ce584c2bc0f08862438eb255ea8723e00fac693d`.
- 15 real time-disjoint 3-second motor imagery EEG epochs, 4 physical channels (`C3,C4,Cz,F3`), CAR, .5–45 Hz, resampled to 200 Hz, amplitude in µV clipped to ±500, official frozen pretrained encoder **9,600-dimensional model features**.

## Explicit three-arm mechanism control
1. Original signal: canonical EEG channel order and corresponding pretrained spatial embedding slots.
2. Incorrect channel binding: swap C3↔C4 signal rows, **retain old spatial embedding slots**. The model is frozen; after encoder output, reindex channel-major token blocks to original canonical order to remove the trivial representation reorder.
3. Corrected metadata: the same swapped EEG signal, but simultaneously swap the **spatial slot IDs** used by pretrained token embeddings. Afterwards canonically reindex token blocks as in (2).

All conditions run the same frozen pretrained checkpoint and the same waveforms. No randomly initialized task-head logits enter the evaluated quantity.

## Real results
| Readout | Observed |
|---|---:|
| Real test EEG windows | 15, from one subject |
| Token feature dimensions (4 channels × 3 patches × multi-scale token embedding) | 9,600 |
| Raw wrong-input 9,600-D feature relative L2 drift (includes channel order) median | 0.48049415 |
| **Wrong binding, token order restored: relative L2 drift median** | **0.12892091** |
| **Correct metadata + token order restored: relative L2 drift median** | **0.00000203706** |
| Correct-metadata max absolute token-entry deviation | 4.4331e-7 |
| Exact inverse signal reordering back to canonical: model feature max absolute deviation | 0.0 |
| Wrong-binding drift larger than corrected-metadata drift | 15/15 windows |

This narrows the empirical explanation from "channel permutation changes flattening order" to a **spatial identity/montage-slot binding dependency in an actual third-party-accepted pretrained foundation model**. The distinction is mechanistic, not a new classification score.

## Limits and next hard gate
- This pilot has *one participant*; the 15 epochs are **not 15 independent participants**. No inferential p-value or generalization guarantee is claimed.
- Swapped electrode identities are **software-injected**, not measured physical electrode replacement.
- In this setup correct channel identities are available to the external oracle; deterministic reindexing/rebinding is bookkeeping, not an original algorithm.
- Model task head is randomly initialized; therefore **no classification BAcc, clinical safety or new SOTA** claim is supported here.
- The model architecture and pretrained weights are authored by the original NeuroRVQ team; Braindecode acceptance of the model port is different from outside validation of a newly invented method.
- Critical next step: real task *or* meaningful downstream error cost across disjoint subjects, missing/uncertain identity groups and a strong spatial/uncertainty estimator; calibrate on development subjects and reserve truly unseen subjects before new empirical claims.

## Related independently credited upstream adoption
- [NeuroRVQ model #1218](https://github.com/braindecode/braindecode/pull/1218): accepted integration and maintainer-run replication.
- [NeuroRVQ tokenizer #1223](https://github.com/braindecode/braindecode/pull/1223): accepted.
- [Maintainer-authored NeuroRVQ multimodal continuation #1254](https://github.com/braindecode/braindecode/pull/1254): **merged** third-party follow-on, *not credited as this author's PR*.
- [TFMTokenizer #1202](https://github.com/braindecode/braindecode/pull/1202) and [TMSA-Net #1209](https://github.com/braindecode/braindecode/pull/1209): separately accepted with maintainer-approved source fidelity and qualified model/benchmark validation.
