# The exact identifiability boundary of source-anchored sensor metadata

## Scope, not a sweeping EEG theorem

Assume a reliable source recording supplies **known physical identities** for
a reference set of (n) EEG channels and for (a) trusted anchors.
The detector receives two independent-window **anchor-correlation
fingerprints**, not privileged observed electrode wiring. An unknown
permutation maps raw target channel rows to source physical electrode roles.
Neither actual firmware wiring nor a patient's task label is observed.

For each evidence window (s), let (u_i^{(s)}\in\mathbb R^a)
represent reference physical role (i)'s Pearson correlation to trusted
anchors and (v_j^{(s)}\in\mathbb R^a) represent target row (j)'s
corresponding anchor fingerprint. Define the role-assignment cost

```text
C_s(j,i) = (1/a) * ||v_j^(s)-u_i^(s)||_2^2
π_s = argmin_{π ∈ S_n} Σ_j C_s(j, π(j)).
```

The implementation solves `π_s` as a bipartite **global** minimum-cost
bijection, then solves (n) additional constrained assignments to find
the exact cheapest alternative `π'_s≠π_s`. The per-channel gap
`Δ_s=(cost(π'_s)-cost(π_s))/n` is deterministic given the observations.

The proposed gate acts only if both time-separated estimates agree exactly
and both gaps are above a predeclared positive numerical cutoff. Otherwise
it abstains. It leaves a proven no-change mapping unmodified. Nothing in
this construction turns `Δ` into a statistically calibrated error
probability.

### Small lower bound within the *observed fingerprint model*

Suppose physical roles (i) and (k) have identical anchor fingerprints
in both windows. Swapping their hidden identities leaves **every observed
reference and target fingerprint** unchanged. Hence an estimator that
uses only those fingerprints cannot distinguish the two possible source
identity worlds. Under an equal prior on those two hidden worlds, the
best achievable identification probability is at most (1/2). Without
an equal prior, no single action can be guaranteed correct in both worlds.

This is a standard indistinguishability argument over the *limited
observation model* and **NOT** a new general limit on EEG decoding, a proof
that genuinely different full EEG recordings are indistinguishable, or a
certificate that the procedure is safe whenever a numerical margin exceeds
the chosen cutoff. The identical-fingerprint unit test forces zero
global-alternative gap and refusal.

### Conditions which invalidate the repair assumption

- A trusted anchor's own label is wrong: the physical reference chart is
  corrupted, so a stable but wrong assignment may look unique.
- The source and target are different people: electrical signatures
  and scalp geometry drift across individuals, so an apparently strong
  assignment need not identify the same electrode.
- Task state changes, especially resting eyes-open to eyes-closed: spectra,
  connectivity and the anchor correlations differ even if no label changed.
- Referencing, filtering, montage type or hardware moves differ across
  recordings. A fixed observed correlation graph is not guaranteed.
- Two neighboring channels have near-symmetric correlation signatures:
  uncalibrated uniqueness may be fragile to small sampling fluctuations.

Our real PhysioNet experiment deliberately includes the eyes-open/eyes-closed
condition as a **distribution shift** and includes a no-corruption control.
No threshold is tuned after these outcomes; any wrong authorized repair is
counted openly. The reference is only two participants with a source session
available for each, which is insufficient for a paper's clinical claims.

### Forward path to a genuinely stronger contribution

A useful original method would need calibrated selective risk/coverage under
trusted- and corrupted-anchor interventions, patient-disjoint validation,
controlled comparisons with ordinary channel graph registration and
spherical coordinate interpolation, and actual pretrained NeuroRVQ
tokenizer/model performance after correction. These requirements are
not established by the present algorithmic pilot.

Do not claim that Hungarian assignment, anchor covariance, dual-window
agreement, or refusal on observational symmetry is alone a novel invention.
