# EEG electrode identity: completed real-EDF pilot (preliminary, 2026-10-09)

**Status:** Completed native EDF task pilot and audit artifacts. **Negative/insufficient original-method performance**, not a flagship result.

## Independent reproducible source
- [Public GitHub Actions run #37829611465](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37829611465): **success** including real EDF and certificate steps.
- [Original per-trial JSON, EDF SHA-256 fingerprints and logs (artifact #11573787954)](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37829611465/artifacts/11573787954).
- Source commit: `1b737c0af910a3e126b1a2758d0ac83b79e5898a`.
- Subject-disjoint PhysioNet EEGMMIDB left/right imagery, EDF runs 4/8/12; subjects 1–6 train, 7–9 validation, 10–12 test; **135 held-out trials** in the test subjects.
- Log PSD 8–13 Hz and 13–30 Hz from 64 channels; fitted StandardScaler + LogisticRegression. This is **not** EEGCLIP or NeuroRVQ training/inference.

## Task-balanced accuracy (original GitHub logs)
| Condition | Balanced accuracy |
|---|---:|
| Correct channel identities | **0.5414** |
| Deliberately permuted channel order, shape-only | **0.4214** |
| Known-identity exact reindexing | **0.5414** |
| Feature mean-fill 10/25/40% | 0.5126 / 0.4905 / 0.5109 |
| Inverse-distance log-feature interpolation 10/25/40% | 0.5122 / 0.5121 / 0.5042 |

Reindexing known electrodes is a **trivial data contract operation**, not an innovative mechanism or performance gain. Interpolation here is performed on log spectral features, **not** proper spatial interpolation of raw EEG voltages.

## Identity-ambiguity certificate
- Under two *explicitly restricted* two-electrode ambiguity groups, assignment minimax authorization coverage: **121/135 = 89.6296%**, abstentions **14**.
- Accuracy among authorized trials: **0.520661** (plain accuracy, not balanced accuracy); directly predicted faulted-input trial accuracy **0.518519**; known-identity trial accuracy **0.540741**.
- No compelling classification advantage; threshold comparison is **not** matched in this first EDF run.
- Mathematical guarantee is **prediction-sign invariance under the stated bounded identity permutations only**. It cannot guarantee the true task label or correctness if the true electrode mapping lies outside those groups.
- Other concurrent PhysioNet pilot [#37830653308](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37830653308) ran 2 independent subjects and 16 injected cases, but **refused all 16**; zero false authorizations there is NOT useful selective repair.

## Updated science gate
1. Stop treating this weak fixed-feature baseline as evidence of a novel EEG method.
2. Freeze prior test labels; build **spatial-filter/riemannian** pretrained-free strong MI baseline on the *same* subject identities with validation-only tuning.
3. Compare a strong pretrained foundation-model feature head only once controller/channel preprocessing and pretrained electrode IDs are correctly specified; no random task head described as pretrained downstream accuracy.
4. Any robustness method must beat matched-information interpolation, channel dropout, and validation-calibrated selective confidence with a nonzero useful risk/coverage tradeoff, on multiple independent held-out subjects/datasets.
5. Treat these preliminary results as **exploratory**; new hypothesis needs a fresh disjoint prospective subject holdout, not test-set-informed threshold selection.

**Do not claim** L8/L9, external replication of this method, NeurIPS-level novelty, or improved clinical safety from these observations.
