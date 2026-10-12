# Lee2019 new acquisition: first target A negative; B source-SHA fail-closed
2026-10-13. This is NOT a two-block completed confirmation, not
"independent GID transfer improvement", and not an L8/L9 milestone.

## Immutable first source before any held target
GitHub Actions 38305991164, source IDs1..12 (100 trials/ID).
Exact source JSON SHA256:
27efc2565d5bc2bc22344b67ee994e42b985c16c34cabf6cdf55136270fa18f3.
Original pooled LDA source lambda=0.4,
GID source lambda=0.15, source person paired GID-minus-original
macro AUC = -0.000800, source SIGNED improvement prediction NEGATIVE.
Original/GID direction cosine = 0.9883871642569202;
source between-group covariance trace fraction 0.81055301637.
All source-only preprocessing and numerical state locked.

## Held A: actual new acquisition, original SOURCE SHA reproduced exactly
GitHub Actions 38306653777, job 115002235613, SUCCESS.
Held IDs 13..24, n12, 100 MI trials each = 1,200 target trials.
Original source macro LDA AUC 0.7541000,
GID AUC 0.7501667, GID minus ORIGINAL = -0.00393333.
Exact two-sided all-4096-person-sign-flip p=0.12890625;
no individual harmed >.02 AUC. This is NOT a positive method result.
AUC by method on exactly same 12 held persons:
 original 0.7541000;
 balanced-class-contrast-only 0.7541000;
 GID 0.7501667;
 within-person-class-centred-covariance-only 0.7501667;
 three-channel CSP(3)+logistic regression 0.7517667;
 Riemannian MDM 0.7362667;
 tangent logistic regression 0.7455000;
 Ledoit-Wolf 0.7447000;
 OAS 0.7447667;
 fixed lambda1 0.7260000.
Interpretation: with these frozen source selections and held people,
the apparent GID decrement is reproduced by the covariance
component alone; the equal-source-class-contrast component is NOT
necessary to explain it. No superiority or GID synergistic benefit.

## Held B: NO OUTCOME; original source SHA failed
The same workflow 38306653777 job 115002235241 read the
12 SOURCE groups only, regenerated source JSON and STOPPED before
reading ANY held B ID25..36:
AssertionError SOURCE_CHOICE_SHA_NOT_IDENTICAL_STOP_BEFORE_TARGET.
Original SHA 27efc2565d5bc2bc22344b67ee994e42b985c16c34cabf6cdf55136270fa18f3.
Failed-B rebuilt SHA:
0b2869570c6ba4c466a46e9a7d9ea335c195cb9a64292e2b6de9cdcad167e7ff.

Independent no-EEG forensic artifact comparison workflow:
38307723649. Compared ORIGINAL immutable source json against the
failed-B pre-target artifact. The labels, source group count, target
group declarations and numerical selected policies were consistent
across the file; a 2.220446049250313e-16 difference appeared in
source between-group covariance fraction. However ALL TWELVE raw
covariance-bytes SHA fingerprints in source_trial_receipts differed.
The protocol fingerprint comparison was not completed because this
source-level failure triggered fail-closed. We cannot infer raw
recording identity or benign machine-rounding from these hashes
without saving/comparing processed covariance tensors and environment
fingerprints in a future new protocol.

The B run is not a statistical null or confirmatory negative.
It is a NUMERICAL/SOURCE-PROVENANCE REPRODUCIBILITY BLOCKER.
Do NOT relax SHA checking post held-A outcome to manufacture
an apparent two-block confirmation. Diagnose separately on SOURCE
data, preserve both original artifacts, and require a legitimately
new pre-outcome protocol/held acquisition before any promotion.

## Signed forecast and limit
Source forecast NEGATIVE and held A mean paired difference NEGATIVE
are directionally consistent in one study block, but p=.1289
and subject n=12 mean this does NOT validate prediction calibration,
forecast coverage or universal source-target monotonicity.
The independent target-B validation remains NOT COMPLETED.
Our first robust external GID method promotion FAILED by both the
source forecast and held A gain gate. A new method or selector
cannot be posthoc tuned on exposed Lee A persons then described
as prospectively successful.

## Publication separation
Paper A: BSPC primary validation-estimand audit PR #140.
Paper B: Liu 49 recording-content-group study separate.
GID/Lee PR #141: NEGATIVE research sidecar, not a submitted article.
