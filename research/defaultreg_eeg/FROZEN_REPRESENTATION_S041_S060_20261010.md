# Default Regularisation — prospective representation x regularisation block S041–S060

Frozen 2026-10-10 before any S041–S060 target outcome is opened.

## Purpose

This block tests the manuscript's most scientifically valuable historical
observation: whether changing the regularisation objective changes the
relative ranking of wide-band and filter-bank EEG representations.

It is separate from S021–S040:
- S021–S040: prospective consequence of full-domain objective mismatch.
- S041–S060: prospective representation x regularisation interaction.

The target block must remain untouched until the representation definitions,
regularisation path and primary interaction are frozen.

## Dataset and split

Dataset: PhysioNet EEGMMIDB v1.0.0.

Source participants: S001–S008 only.

Previously exposed / reserved elsewhere:
- S009–S020: development/falsification.
- S021–S040: full-domain mismatch confirmation.

Prospective representation targets:
**S041–S060 (20 participants).**

Runs: R04/R08/R12.
Task: T1 versus T2 motor imagery.
Epoch: 0.5–3.5 s after cue.
Channels: fixed 21 FC/C/CP channels.
Every original task cue must be retained.

## Frozen representations

### R1 WIDE
8–30 Hz, one covariance/tangent block.
Nominal tangent dimension: 231.

### R2 FB8
Eight fixed bands:
8–12, 12–16, 16–20, 20–24, 24–28, 28–32, 32–36, 36–40 Hz.
Each band uses the same 21 channels, epoch and source-only geometry.
Concatenate eight 231-D tangent blocks: 1848 dimensions.

### R3 FB8_SOURCE_PCA231
Take the exact R2 source features.
Fit PCA to 231 components using SOURCE trials only.
Apply the frozen source PCA transform to each target trial.
Target data do not enter centering, components or component count.

### R4 WIDE_NUISANCE1848
Take R1 and append deterministic label-independent nuisance coordinates until
total dimension equals 1848.
Noise scale comes from source-feature dispersion only.
Labels are not used to generate nuisance coordinates.

R4 is a negative control for "high dimension alone creates the effect."

## Frozen regularisation conditions

For R1–R4:
1. analytic Ledoit-Wolf;
2. OAS;
3. full-domain source-participant macro-AUC scalar shrinkage;
4. full-domain source covariance-predictive-NLL scalar shrinkage.

Full-domain grid is identical to S021–S040:
{0, .0005, .001, .002, .003, .005, .0075, .01, .015, .02, .03,
 .05, .075, .10, .15, .20, .30, .40, .50, .60, .70, .80,
 .90, .95, .98, .99, 1.0}.

No target statistic or outcome may choose representation, PCA, nuisance scale,
lambda, or method.

## Primary estimand

Let A(r,m,s) be target-participant AUC for representation r, regularisation
method m and subject s.

Primary participant-level interaction:

I_s =
  [A(FB8, source-discriminability, s) - A(WIDE, source-discriminability, s)]
  -
  [A(FB8, covariance-default, s) - A(WIDE, covariance-default, s)]

Primary covariance-default is Ledoit-Wolf.

A positive mean interaction supports the claim that the regularisation choice
changes the apparent spectral-representation benefit.

## Predeclared mechanism contrasts

C1 PRIMARY:
FB8 x regularisation interaction I_s above.

C2 dimension control:
under source-discriminability tuning,
A(FB8,s) - A(WIDE_NUISANCE1848,s).

C3 information-at-matched-dimension:
under source-discriminability tuning,
A(FB8_SOURCE_PCA231,s) - A(WIDE,s).

C4 objective comparison:
repeat C1 replacing analytic LW with source covariance-NLL optimum.

Holm correction across C1–C4.

## Falsification rules

- If C1 is near zero across participants, remove the "regularisation reverses
  spectral representation ranking" headline.
- If R4 reproduces R2, the effect is likely a dimensionality/conditioning
  artifact rather than spectral information.
- If R3 loses the R2 benefit, high-dimensional multi-band expansion may be a
  necessary part of the effect; do not claim pure spectral-information gain.
- If FB8 is inferior under every regulariser, the historical ranking reversal
  does not replicate and must remain historical discovery only.
- No representation/method pair may be dropped because it is unfavorable.

## Statistics

Independent unit: target participant, N=20.
For C1–C4:
- exact paired sign-flip where computationally feasible;
- participant bootstrap CI;
- Holm family-wise correction;
- mean, median, lower quartile, worst participant and harmed/improved counts.

No trial-level pseudo-replication.

## Venue interpretation

A replicated C1 plus dimension controls would materially strengthen a BSPC
paper because it shows that a routine signal-processing design decision
(regularisation) can change conclusions about spectral representation.

For TNSRE, this remains insufficient without practical BCI calibration /
cross-session/device implications.


## Frozen extension before reading any S041-S060 target outcome: source sample budget

This extension was committed before the model opened any S041-S060 result.
A prior workflow attempt may have started from the earlier protocol commit; any
artifact from that pre-extension attempt is **superseded** for confirmatory
interpretation and must not be used to choose the extension.

On R1 WIDE only, repeat source-only selection and final target scoring after
deterministically retaining the following fraction within every SOURCE
participant x class cell:

`{1.00, 0.75, 0.50, 0.25}`.

Rules:
- target trials are never subsampled and never enter selection;
- the random seed is fixed at 20261010 plus a deterministic fraction offset;
- every source participant and both classes remain represented;
- the tangent reference is refitted using only the retained source trials;
- the full [0,1] shrinkage grid is unchanged;
- report the source-discriminability optimum, source covariance-NLL optimum,
  analytic Ledoit-Wolf, and per-target AUC under each budget.

This is a controlled sample-size intervention with the sensors, frequency
band, epoch, target participants and feature definition held fixed. Together
with WIDE_NUISANCE1848 it separates two mechanisms the historical 21-vs-64
channel comparison could not isolate:
1. increased feature dimension without new neural signal;
2. reduced source trial budget without changing feature dimension.

The budget series is a **mechanism sensitivity analysis**, not an additional
primary outcome family. It cannot be used to choose which budget to headline.
