# Repair-authority identifiability: exact finite certificate + prior-art boundary

**Source CI (Python 3.10/3.12/3.13):** [MuJoCo run #37993090103](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37993090103). Source: `minimal_trusted_evidence.py` + `test_minimal_trusted_evidence.py`. Actual MuJoCo dynamics, not hand-assigned sensor outcomes. **This is an author-created finite controlled model**, not a newly trained robot policy, third-party replication, or real system deployment.

## Scientific object

Let W be a finite set of implementation worlds, O:W -> Z the declared full physical sensor transcript, and a:W -> A the **authorized source patch** (not just the resultant physical trajectory). A sensor-only decision d:Z -> A with *total coverage and zero wrong authority* exists **iff** a is constant on every fiber O^{-1}(z). Necessity: a pair with equal observations but distinct valid patches requires contradictory decisions for one identical input. Sufficiency: choose the shared patch on each fiber.

If that condition fails, any zero-wrong-authorization decision rule must abstain on the conflicting observation, OR use additional independently trusted evidence. More active queries cannot break truly identical full-source observation transcripts when their menus are already contained in O. These statements depend on frozen finite worlds and observed channel menu, and are NOT general identifiability or distribution-free safety results.

For trusted provenance fields f_j and observed-alias conflict pairs (u,v) with a(u) != a(v), demand that each pair be separated by at least one selected field: f_j(u) != f_j(v). Selecting fields of least total cost is a weighted set-cover instance; the released finite solver independently checks *all* subsets of at most 18 fields and returns an explicit unseparated pair when impossible. For noisy observations, pairs must be **conservatively included whenever compatible within the declared noise model**; a single equality of rounded signatures is not a general stochastic equivalence proof.

## New native MuJoCo result

Existing genuine dynamics: the gear-fault and foreign-owned motor use exactly identical kinematics, controller inputs and actuator forces over THREE frozen forcing regimes; their necessary software patch permissions differ. The same physical world with two declared catalog build revisions illustrates worlds that differ as complete identities but share one permitted repair.

- Sensor-only correct total authorization: **impossible** on this constructed alias.
- One *trusted issuer namespace* field (assigned cost 1) distinguishes the incompatible repair permissions; exact minimum cost **1**.
- Distinguishing **all three build identities** needs the signed exact build revision (assigned cost 2); exact minimum cost **2**.
- A zero-cost but **untrusted self-report** of source namespace is explicitly rejected by the solver. Raw models, field values and all synthetic price units ship with the experiment source and JSON evidence.
- Separate 4-new-world x 3-wave MuJoCo study: sensor-only wrongly authorizes **3/12** out-of-catalog conditions; source-allowlist condition rejects **12/12**, only assuming the source identifier is genuinely trusted.
- Separate HMAC demonstration with an honest protected issuer (an explicit assumption), actual signed MuJoCo measurements: authorized 12/12 legitimate catalog traces, rejected 6/6 post-signing actuator-force swaps and 12/12 stale nonce replays. HMAC and nonce authentication are established cryptographic engineering, NOT claimed algorithmic innovations or a real hardware TEE.

## Rigorous comparison against PRIOR papers, not similarity by title

1. **Javdani et al., _Near Optimal Bayesian Active Learning for Decision Making_, AISTATS 2014:** https://proceedings.mlr.press/v33/javdani14.html . Already defines decision-region identification, necessary/sufficient certificates and Hyperedge Cutting. **We do not claim to invent deciding action class instead of complete world identity.**
2. **Cicalese et al., _Diagnosis determination: decision trees optimizing simultaneously worst and expected testing cost_, ICML 2014:** https://proceedings.mlr.press/v32/cicalese14.html . Already optimizes test cost for determining a discrete decision function; our set-cover toy is not a new complexity/approximation theorem.
3. **Active robot policy adaptation CoRL 2025:** https://2025.corl.org/program/main-conference . Main-conference standard is task-level benefit on real manipulation/adaptation rather than only deterministic physics source matches.
4. Existing internal exact active-identification oracle on the same known four-world MuJoCo family costs **4**, and the staged method costs **4**. **There is no active-diagnosis cost advantage.**

**Possible original robotics contribution after real evidence:** a *verified source-to-patch authorization contract*, with jointly measured physical residual, authenticated implementation identity, exact repair object and intervention-time safeguards, tested on a truly unseen independently implemented controller family, with a learned policy task-success endpoint. This is not secured by mathematical set cover alone. A manuscript claiming the 2014 decision-region theory as new would be technically incorrect.

## Hard go/no-go before T-RO or RA-L

- Unseen manufacturer/controller stack (not another author-generated MuJoCo knob), authentic patch provenance, same observation/risk budgets, strong HEC/decision-region and generic active identification comparators.
- Measured wrong repairs, correct repairs, abstentions, end-effector/physical recovery AND task-level policy success reported separately.
- Robust spoof/replay and compromised measurement-source controls; HMAC issuer trust boundaries cannot be hidden or counted as algorithmic superiority.
- Frozen experiment protocol and independent execution. No model-family overlap or post-outcome world tuning. As of this note, NO paper submission, NO external review, NO hardware result.
