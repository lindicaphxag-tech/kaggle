# V25 retrospective spectral rank-cell falsification protocol

Status: 2026-10-13. This is an **already-exposed Weibo2014 dataset**;
there is no new target validation or prospective target-gain claim.

## Question and fixed comparison

Does exploiting whole numerator cancellation in a Bernstein polynomial
yield tighter *source-person* continuous-lambda rank stability bounds than
V23 independent-term/Taylor bounds on the **same** original 3-channel
Weibo2014 tangent-LDA source folds?

Use exactly the original 8–30 Hz/160 Hz/predeclared C3,Cz,C4 sensor
set and leave outer subject 1 out; source-only inner source folds must
contain the other 9 people. Each held source person is an analysis
unit; no pseudo-replication of trial-pair counts.

- Fixed p=6 covariance tangent features; numerator polynomial max degree 5.
- Exact code input: original scalar shrinkage spectrum, projected
  mean contrast, held source features/labels. No hyperparameter reselect.
- Compare V23 source certificate vs V25 Bernstein over [0,1] with 12
  uniform subintervals. Separately compare the interval between
  historical class-centred covariance-selected lambda and source macro
  AUC-selected lambda, if unequal. No cherry-picking intervals.
- Exact pairwise endpoint AUC agreement between implementations to
  1e-12 required; failures invalidate the comparison.
- Report mean source stable fraction, 9 source participant deltas,
  uncertain count and continuous AUC envelope.
- Bernstein may be worse due to cancellation/numerical conditioning.
  Retain regressions and inconclusive cases. Do not claim a theorem
  that Bernstein dominates Taylor; neither interval enclosure
  universally dominates the other.
- Synthetic tests: linear-product Bernstein identity, root at .5,
  1D no crossing, zero class separation, affine score invariance,
  dense intermediate AUC coverage and p>12 fail-closed.
- Never claim formally certified FP arithmetic: NumPy does not provide
  outward directed rounding, and threshold-slack is only a
  conservative heuristic conditional on the fitted eigensystem.
- Proposed paper impact: exact rank-cell *mechanistic interpretation*
  of why shrinkage parameter disagreement may be AUC-neutral.
  Even an improvement in SOURCE certainty does NOT validate a novel
  unseen-target selector. New external subjects in a separate pre-data
  protocol would be needed to promote it as a clinical/predictive method.

## Stop/go

GO for technical companion if seven synthetic tests and the real
source EEG run pass and comparisons reproduce base AUC exactly.
GO for main-article novel algorithm claims requires a frozen source
selection policy and improvement on an untouched independent data
acquisition; V25 supplies neither. Otherwise manuscript remains
V24 measurement-validity paper with an exploratory V25 supplement.
