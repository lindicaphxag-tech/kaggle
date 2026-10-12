# GID invariance, no-free-lunch and priority check
2026-10-13 | BSPC sidecar; source-locked Lee study NOT a claimed success.

## What the operator guarantees, exactly
From source participant g and class c compute mean m_gc.
W= sum_{g,c,i in g,c} (x_i-m_gc)(x_i-m_gc)^T/(N-2G).
d_eq=(1/G)*sum_g(m_g1-m_g0).
GID(lambda) = [(1-lambda)W+lambda tr(W)/p I]^(-1) d_eq.

For each arbitrary independent participant translation x_i -> x_i+b_g,
the source means translate equally within BOTH classes, cancelling
in W and d_eq. Thus in an unchanged FIXED source feature coordinate
system, the GID direction is invariant for all feasible lambdas.
Source participant-class sample counts may be unbalanced.
The original global class-centred pooled covariance and its
globally pooled contrast need not have this invariance.
A previously SOURCE-FITTED Riemannian tangent reference is fixed
at evaluation time. No guarantee is claimed under physically
modifying raw EEG then RE-FITTING that nonlinear reference.

This is a classic within-group discriminant covariance construction,
NOT an unprecedented mathematical operator. Existing prior art:
Hu, Zhang, Chen, Chan, Domain Generalization via Multidomain
Discriminant Analysis, UAI/PMLR 2020 (v115:292-302),
https://proceedings.mlr.press/v115/hu20a.html
Other within-domain centring/class-balanced Fisher discriminant
methods may have materially overlapping ingredients; an
exhaustive novelty search is still required.

## A no-free-lunch obstruction for zero-target sign forecasts
Let w_a and w_b be nonparallel linear discriminants learned from
the EXACT SAME fixed sources. There exists a vector v satisfying
w_a.v>0 and w_b.v<0. (For nonparallel normals, use a linear
combination spanning both and choose a halfspace intersection.)
Consider two target worlds with the exact SAME source data.
In T1 positive examples are v and negatives 0; in T2 reverse
the target labels. Adding arbitrarily tiny nondegenerate noise
does not change the almost-sure limiting AUC.
In T1 AUC(w_a) approaches 1 and AUC(w_b) approaches 0,
whereas in T2 the rankings reverse. A source-only algorithm
receives identical source observations in both worlds,
and cannot certify the sign of its prospective target AUC gap
for ALL possible unobserved target distributions.
A positive signed forecast is possible only under additional,
testable source-to-target relationship assumptions.
If all candidate score directions rank all pairs identically
(positive collinearity), the result is trivial equivalence,
not a new transfer prediction problem.

## Source-only, falsifiable but conditional forecaster
The first Lee2019 session-1 SOURCE grouped CV compares GID vs
original grouped macro-AUC LDA on the original identical 27-lambda
path, training only on groups 1..12. Forecast POSITIVE only if
paired SOURCE macro delta >= +.005 and >=8 of 12 source held
groups improve; else FORECAST NEGATIVE. This heuristic is frozen
before reading ANY target IDs. Target A 13..24 and target B
25..36, no retune between blocks, are a test of this forecast.
Neither a two-block success on one acquisition nor a two-block
failure establishes a universal domain-shift law.
A nonparallel source classifier, robust real AUC gain against
strong classical baselines and repeated independent acquisition
would still be needed to promote this as an L8/L9 EEG algorithm.

## Relation to flagship paper
The focused BSPC main draft (PR #140) is the empirically verified
validation-estimand and three-source-acquisition predictive score
anatomy paper; it is NOT replaced by a new negative GID study.
If held Lee result is negative, keep it with V31/32 negatives.
The separate V26-V30 recording-content identity paper stays separate.
