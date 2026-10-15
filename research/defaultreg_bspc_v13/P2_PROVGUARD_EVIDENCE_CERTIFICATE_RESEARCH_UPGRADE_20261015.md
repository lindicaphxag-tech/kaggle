# ProvGuard candidate upgrade: evidence-carrying shared-clock matching (2026-10-15)

## Scope
This is a proposed research mechanism, not verified new science. Goal: output an independently checkable witness (source/target channel bijection, global lag, overlapping source/target sample indices, per-channel train and independent verification similarity), with selective abstention. Separate (A) synchronous multichannel source reuse from (B) any-channel content reuse. Positive controlled-data results on CHB-MIT in earlier run are not evidence that this new variant performs better.

## Core algorithm
For lag l and injective channel assignment pi, compute training similarity s_ij(l) and held-out verification v_ij(l) for aligned samples. Keep an edge only if s>=tau_train and v>=tau_verify. Solve maximum-weight bipartite assignment at each candidate common lag; prefer number of verified distinct channels, weakest channel evidence, then total evidence. Return the jointly feasible witness or ABSTAIN, never assert patient identity.

## Soundness boundary
A witness certifies only that a specified deterministic numerical correlation-and-alignment predicate holds on the supplied arrays (when recomputed independently). It is not proof of physical acquisition identity or a calibrated probability. Temporal holdout should use blocked samples under realistic EEG autocorrelation, not interleaving. A safe train/test split is defined relative to all *confirmed* content-sharing graph edges and their connected components; unidentified overlaps cannot be ruled out.

## Required comparisons
- Unconstrained independent-lag Hungarian, shared-lag Hungarian, FFT cross-correlation with robust peak scoring.
- Published device synchronization approaches, DTW/multi-resolution alignment, approximate fingerprints.
- New algorithm ablation: heldout verification, abstention, channel loss, temporal drift, partial overlap.
- Evaluations grouped by source recording and study, with fixed thresholds and confidence intervals.
- Report paired results and compute cost; distinguish evaluation endpoint A vs B.

## Evidence taxonomy
(1) executed real EEG experiment for old shared-clock matcher; (2) executable candidate prototype and synthetic functional check; (3) new real-data validation pending; (4) theoretical, domain-independent source identity proof not established.

## Related work
Olsen et al. (EMBC 2025), doi:10.1109/EMBC58623.2025.11254069. Acan et al. (Behavior Research Methods 2026), doi:10.3758/s13428-026-03097-8. The novelty claim must be benchmarked against these and other published synchronization techniques.
