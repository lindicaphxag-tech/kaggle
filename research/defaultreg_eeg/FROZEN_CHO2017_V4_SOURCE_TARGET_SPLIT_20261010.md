# Default Regularisation — Cho2017 v4 untouched external split freeze

Frozen 2026-10-10 before any Cho2017 EEG outcome was inspected under the
post-S021 v4 protocol.

## Chronology

EEGMMIDB S021-S040 has already prospectively confirmed the earlier
full-domain source-participant discriminability consequence. The later
Regularisation Validity Frontier / FCTR constructs were developed after that
block was opened.

Cho2017 is therefore the **first untouched external cohort** eligible to
confirm the post-S021 v4 claims.

This file freezes participant roles and preprocessing BEFORE the source-only
workflow is allowed to touch data. The confirmatory target workflow must not
run until the source artifact is frozen.

## Deterministic participant split

Dataset subject IDs are 1..52.

The split is generated from subject IDs only using:

```python
np.random.default_rng(20261010).permutation(np.arange(1,53))
```

No EEG data, labels, demographics, bad-trial indices or prior subject
performance enter the split.

### Source development pool (12)

`[3, 5, 19, 26, 31, 32, 36, 37, 44, 45, 46, 47]`

### Untouched confirmatory target block (20)

`[4, 6, 8, 10, 11, 12, 14, 15, 17, 18, 20, 28, 33, 34, 39, 40, 43, 48, 50, 52]`

### Reserve block (20)

`[1, 2, 7, 9, 13, 16, 21, 22, 23, 24, 25, 27, 29, 30, 35, 38, 41, 42, 49, 51]`

The reserve block may not replace an unfavorable target participant/block.
It is reserved for later independent stress/replication only.

## External common motor-channel set

Cho2017 contains FCz, but Lee2019_MI does not. To avoid changing feature
dimension between the two untouched external cohorts, the v4 external panel
uses the following 20-channel central motor set in both Cho2017 and
Lee2019_MI:

`FC5, FC3, FC1, FC2, FC4, FC6,
C5, C3, C1, Cz, C2, C4, C6,
CP5, CP3, CP1, CPz, CP2, CP4, CP6`

Tangent dimension:

[
20(20+1)/2 = 210.
]

This differs from the already-frozen EEGMMIDB S021-S040 21-channel / 231D
representation. S021-S040 is not retroactively changed.

The external replication claim is about objective alignment under participant
shift, not equality of absolute AUC across cohorts.

## Frozen preprocessing

MOABB version: **1.7.2**.

Dataset: Cho2017 / GigaDB DOI 10.5524/100295.

Download provider: **upstream**, so each selected subject is fetched from the
dataset's original per-subject distribution rather than silently switching
providers.

Paradigm:
- left_hand vs right_hand only;
- 8–30 Hz;
- task-relative 0–3 s;
- 20 fixed channels above;
- resample to 250 Hz;
- no baseline correction;
- `reject_by_annotation=False`.

The last rule is deliberate: the primary denominator retains all decoded
left/right trials rather than retrospectively selecting subjects/trials by
quality annotations. Any dataset-provided bad-trial metadata may be reported
as sensitivity metadata later, but cannot change the frozen primary
denominator after outcome inspection.

Each trial is time-demeaned before covariance estimation. A tiny
trace-proportional diagonal floor is added only to maintain strict SPD.

## Source-only stage

The source workflow is allowed to load ONLY the 12 source subjects.

It must produce:
- raw source file SHA256 identities;
- exact trial/class count per source participant;
- source covariance NPZ;
- 27-point historical FULL_GRID source path;
- dense frontier source path using 0.0025 spacing plus near-zero anchors;
- source macro-AUC and covariance predictive-NLL objectives;
- selected lambda_D and lambda_C;
- analytic final-source Ledoit-Wolf/OAS shrinkages;
- source artifact hashes.

The source workflow must assert that no confirmatory/reserve subject appears
in loaded metadata.

## Target lock

The 20 target IDs are metadata only in the source artifact.

No target file may be downloaded or opened by the source workflow.

A later target workflow may be created only after the source workflow
succeeds and its artifact/head/hash are recorded.

That target workflow must freeze all method score vectors—including all dense
frontier lambda scores—before target labels are used for AUC/inference.

**Status at freeze: NO Cho2017 EEG data inspected in this v4 protocol.**
