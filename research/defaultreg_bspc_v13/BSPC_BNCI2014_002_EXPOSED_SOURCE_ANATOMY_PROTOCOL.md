# Additional descriptive source anatomy on a higher-dimensional EEG montage
2026-10-13 | Strictly POST-OUTCOME exploratory, not a preregistered test.

BNCI2014_002 was already a target collection in the historical
DefaultReg EEG study. Here all 14 released IDs are reused
*only as grouped SOURCE-validation observations* for an EXACT
estimand decomposition. NO NEW independent generalisation or
unseen-person AUC claim is possible.

This third acquisition has a different right-hand-vs-feet motor imagery
task and native 15-channel, 120-tangent-coordinate feature set,
rather than the left-vs-right 3-channel/6-feature Cho and Dreyer families.

Keep original old-source_runner BNCI2014_002 preprocessing unchanged:
MOABB 1.7.2, native montage, 8–30 Hz, resample 160Hz, class labels
right_hand/feet with positive feet, all IDs 1..14.
Original grouped 5-fold source group split and the feasible ORIGINAL
27-point lambda path; refit original tangent reference and covariance
on every source fold. Do not access any alternative or target data.
Use exact three-factor:
P_location, P_contrast_score_aligned, P_contrast_score_orthogonal.
Report all 14 people and every feasible lambda plus per-dataset selected
lambda at historical CovScore, proper source-trained predictive NLL,
and original grouped source macro AUC. Independently verify complete
P_total=P_location+P_aligned+P_orthogonal and original
source macro and historical CovScore selected lambda parity.
A lower AUC-invariant fraction on this native montage must be
retained; no posthoc processing/hyperparameter retuning is allowed.
