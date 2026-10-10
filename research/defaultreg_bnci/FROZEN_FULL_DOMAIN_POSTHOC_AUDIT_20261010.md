# BNCI2014_001 full-domain shrinkage audit — explicitly post-hoc

Date: 2026-10-10.

This audit is frozen **after** the first BNCI2014_001 9-subject LOSO result was
opened. It therefore cannot serve as independent confirmation of a new method.
Its purpose is narrower: address the reviewer concern that the original
{0.2,0.4,0.6,0.8,0.95} search grid structurally excluded analytic
Ledoit-Wolf-like shrinkage values (~0.002--0.004 on the first run).

## Fixed audit

Dataset/task/splits are unchanged from the first BNCI2014_001 run:
- 9 outer leave-one-subject-out folds;
- left-hand vs right-hand motor imagery;
- 8--30 Hz;
- source-only tangent geometry inside every fold;
- target subject excluded from geometry, regularisation selection and model fitting.

Replace the truncated grid with the complete predeclared scalar path:
{0, .0005, .001, .002, .003, .005, .0075, .01, .015, .02, .03,
 .05, .075, .10, .15, .20, .30, .40, .50, .60, .70, .80,
 .90, .95, .98, .99, 1.0}.

For every outer fold report:
1. analytic Ledoit-Wolf and OAS shrinkage;
2. source-participant macro-AUC optimum over the full path;
3. source covariance-predictive-NLL optimum over the full path;
4. domain-boundary hits;
5. full criterion curves;
6. held-target AUC for LW, OAS, full-domain source-discriminability,
   full-domain covariance-NLL and logistic regression.

## Interpretation

This audit can answer whether the earlier BNCI result was merely forced away
from the low-lambda region by the truncated grid. It cannot rehabilitate
tail-signed/CFCTS as prospective algorithms and cannot be called a new
confirmation cohort because all nine target outcomes were already exposed.

The untouched confirmation hierarchy remains:
- EEGMMIDB S021--S040 full-domain block;
- EEGMMIDB S041--S060 representation x regularisation block;
- independent Cho2017 / Lee2019_MI cohorts.

If the full-domain source-discriminability optimum remains far from LW in
BNCI2014_001, it supports the *mechanism-development* claim that the objectives
can separate. If it moves into the LW region, the earlier mismatch magnitude
must be weakened accordingly.
