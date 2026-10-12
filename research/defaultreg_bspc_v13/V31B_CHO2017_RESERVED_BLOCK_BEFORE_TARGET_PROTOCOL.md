# V31B reserved participant block: fixed Cho2017 source selection, no retuning
Committed 2026-10-13 AFTER inspecting the first Cho2017 target block (13..32),
but BEFORE reading target EEG or labels for the reserved IDs 33..52.

This is a **sequential exploratory replication within the same acquisition**,
not a separate-site external confirmation and NOT an independently registered
confirmatory trial. Optional follow-up is explicit.

## No new algorithmic degrees of freedom
Re-use the original V31 code and source group IDs 1..12, sensors
C3/Cz/C4, left/right imagery, 8-30 Hz, 160 Hz, and MOABB 1.7.2.
All five source-selected scalar shrinkage lambdas are precisely:
SCE rho1 = 1.0; source macro = 0.5;
proper source-mean Gaussian NLL = 0.1;
historical class-centred = 0.0; SCE rho0 = 0.3.
No radius, margin, geometry, number of channels, stage, source ID,
preprocessing option or class label is changed.

The runner MUST reproduce the ORIGINAL SHA256 frozen first-run
source-choice JSON **exactly** before reading ANY person in 33..52:
677f848de0e3bc3122892e5bec55b20dc7808ab04093bdaeb33f1bd0a923f796.
Otherwise it FAILS CLOSED, and there is no follow-up outcome.
It only switches the held-person list after this lock; it never
re-selects parameters on the reserved block.

## Outcomes
Evaluate all and exactly 20 reserved IDs 33..52, each class >=20,
no missing-subject substitution. Identical seven matched methods:
SCE rho1, original macro-grid, SCE rho0, correct proper NLL,
historical centred covariance, LW and OAS.
Primary descriptive contrast is macro participant AUC:
SCE rho1 minus source macro, report signed all-20 effects and exact
two-sided paired sign-flip p, >.02 harms, untrimmed all-person mean.

If successful, both disjoint released-ID target blocks can be shown
side by side, but there is only one shared source model and acquisition;
no claim of two independent studies or a prospective preregistration.
No revised method is permitted to use ANY 13..52 target outcomes.
The original V31 threshold p<.05 for the first cohort was not reached;
a favorable second p value cannot retroactively change that result.
