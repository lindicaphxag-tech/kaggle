# SemRepair — rigorous source-repair theory, trusted actuator envelopes, falsifying baselines
October 10, 2026. All new unverified CI outcomes must remain PENDING.

## Correct problem: change in physical action set, not merely better fault detection

For invertible 2x2 interface stages P and C and internal clip limit L, the plant sees
F(u) = C clip(Pu,[-L,L]^2). If an unrestricted upstream black-box compensator
can only change u, its attainable output set is exactly C([-L,L]^2).
In official Pusher-v5 a final [-2,2]^2 actuator clamp changes this set into
K = clip(C([-L,L]^2),[-2,2]^2).
Our physically strongest existing action-oracle projects the desired action
exactly onto K: truly instantaneous Euclidean-optimal, NOT an optimal
100-step original-task control policy. Source correction may change K.

## Signed patch with necessary and sufficient nominal semantic check

For signed replacement producer P', controller C' and source clip L', exact
implementation of identity over every u in D=[-R,R]^2 requires
C'P'=I and R*max_i sum_j |P'_ij| <= L', provided C' invertible and R>0.
The first is forced by the derivative at zero, the second by absence of a
saturated intermediate action that would eliminate local invertibility.
These are STANDARD linear algebra and polytope facts, not claimed inventions.

Before switching real on-disk adapter bytes, test-key Ed25519 2-of-3 source
activation checks old/new SHA256, current+target epoch, nonce, actor family,
deployment action radius, allowed component edits, and whole-command-box
mapping+clipping. Valid signatures on semantically invalid changes are REFUSED.

The latest completed 16 adversarial signing/semantic tests and 4 native
third-party frozen pretrained SAC task tests are GREEN:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38010872050
Actual faulty FILE-loaded Pusher policy mean reward -56.395976, object-to-goal
distance .245791m; actual signed/installed FILE-loaded action adapter
-26.379423 and .175368m, identical healthy reference (16 already-used seeds
for each of 3 arms, 48 episodes). This is controlled software integration,
NOT independent vendor sign-off, real robot deployment, or new efficacy data.
All signers are author-generated test identities.

## Robustness extension for bounded real coefficient drift

Suppose ACTUAL producer and controller matrices obey
|delta P_ij| <= dP_ij and |delta C_ij| <= dC_ij,
where NONNEGATIVE error intervals are asserted by an EXTERNAL credible
calibration authority, not estimated as trustworthy by these same taps.

For every u in [-R,R]^2, stage saturation is excluded if
R*max_i sum_j(|P_ij| + dP_ij) <= L.

The worst-case output infinity error is upper-bounded by
R*max_i sum_j[ |(CP-I)_ij| +
   sum_k( |C_ik|*dP_kj + dC_ik*|P_kj| + dC_ik*dP_kj ) ].
If this is <= epsilon as well, then all allowed coefficient disturbances
satisfy ||F_actual(u)-u||_infinity <= epsilon on the WHOLE specified action
box. The derivation is triangle inequality applied to
(C+delta C)(P+delta P)-I. This is a SUFFICIENT interval bound, not new math,
and does not cover unknown sources, untrusted calibration, time-varying
feedback, backlash or adversarial OS access. Fail closed when uncertain.

Code: research/semrepair/robust_source_envelope_certificate.py
Tests: research/semrepair/test_robust_source_envelope_certificate.py
CI: .github/workflows/semrepair-robust-source-uncertainty.yml
Its test includes 6000 bounded matrix/command combinations. No claim of
successful CI until the actual run passes.

## Complete external pretrained policy comparison

All actors genuinely published by Farama/Minari and frozen at immutable
checkpoint revisions: two SAC actors on OFFICIAL Reacher-v5 and two on
OFFICIAL Pusher-v5; two robot bodies, not four separate robot manufacturers.
The planted compensated clipping source fault is AUTHOR-ENGINEERED.

Strongest severity source minus exact instantaneous action black-box return:
Reacher expert +.482485, Reacher medium +.427307, Pusher expert +8.738653,
Pusher medium +5.544641. Actual Pusher object-to-goal mean distances reduced
relative to that oracle by .058898 m and .020738 m, respectively. Full
expert Pusher ORIGINAL reward/physics decomposition, per-seed paired analysis
and 10-contrast multiplicity checks: successful archive reanalysis
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009661288 .
Mild scenarios tied or sometimes slightly favored the blackbox.

## Falsification is more important than rewriting gains

A globally future-reward-optimal blackbox could complete pushing despite
the inability to exactly replicate ideal actuator torques. The prior
Euclidean-optimal per-step oracle therefore cannot establish universal task
superiority. To challenge this thesis directly, first-party code adds a
fully informed genuine original-reward Pusher planner and a three-step
receding horizon MPC which calls the FROZEN external SAC on its simulated
future Pusher 23D physical observations. It retains all losses and counts
every privileged native model rollout, future actor inference and executed
task effect.

The initial native model failed its strict true-reward identity test
(~1.36e-4 discrepancy); the failure is explicitly retained:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38010613448 .
The cause may involve the contact solver's DERIVED state buffers.
New MuJoCo 3.3.7-compatible clones use MjData.__copy__ backed by
native mj_copyData, not the restricted mjSTATE_INTEGRATION transfer.
The new one-step and three-step run results remain UNVERIFIED while queued,
so neither negative nor positive MPC performance may be cited.
No tolerance loosening from exact original-reward 1e-7 gate is permitted.

## Top accepted paper comparison and publication gates

ICML 2026 VLA-FixBench by Yan et al. actually studies 20 VLMs, LIBERO,
real robots and recovery across perception/planning/control:
https://proceedings.mlr.press/v306/yan26r.html .
CVPR 2026 LIBERO-Plus covers VLA progressive robustness generalization:
https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html .

The present source authority claim is DIFFERENT but much narrower.
Required for top venue equivalence: validated original-reward planning
baselines, independent non-Farama robot/policy, genuine external VLA or
natural fault cases, cryptographic manufacturer/telemetry trust roots and
independent end-to-end replication. No T-RO, RA-L, CCF or CAS acceptance
guarantee follows from these experiments. Do not transfer these findings
to C2P/GeoTrust/ClinGeom or any other of the six separate paper lines.
