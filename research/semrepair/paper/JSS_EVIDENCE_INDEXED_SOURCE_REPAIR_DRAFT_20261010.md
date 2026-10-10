# Intervention Surfaces Matter: Evidence-Indexed, Authority-Constrained Repair of Robot Action Software

**Journal of Systems and Software — targeted research-article draft, October 2026**

Status: **AUTHOR WORKING PAPER, NOT SUBMITTED OR ACCEPTED**. Results below are labeled as verified or awaiting CI. This article focuses on portable software engineering rather than claiming a new pretrained VLA method or real robot deployment.

## Abstract (research-article version, ~210 words)

Robot actions pass through software transforms whose semantics can depend on intermediate saturation, historical observations, and component ownership. Recovering a task after an incorrect command, however, does not establish that the incorrect software has been repaired, nor that the controller is authorized to modify it. We study source-repair decisions under two distinct constraints: actuator outputs unavailable to any upstream-only compensator, and latent software states that cannot be reconstructed from the compensator's current observations. We formulate action recovery as a minimax problem over physically reachable actions and observation-equivalent source worlds, and use full-command-domain contracts to admit or refuse source changes. A finite authorized patch catalog is searched for the least declared source-edit cost, subject to robust coefficient-uncertainty constraints and separately signed source and calibration evidence. We evaluate engineered clipped-interface faults with four independently published frozen SAC checkpoints across official Reacher-v5 and Pusher-v5 simulation tasks, and separately evaluate a historically anchored conversion error using unchanged, commit-pinned PyTorch code from LeRobot. In native Pusher experiments, a one-field, signed file-backed correction matches the original task outcomes of a three-field repair across 16 previously used initial states while reducing the number of changed configuration fields by two. Separately, a conditional two-world LeRobot source-interface experiment exhibits an average 0.2338 current-only maximum action error and zero error with the correct cached reference. A history-enabled black-box is explicitly permitted to match the source repair. These findings delimit source authority and information needs in software repair; they do not establish automatic correction of naturally occurring upstream bugs or real-world robotic safety.

## 1. Why this is a software engineering question

Prior robot fault recovery often intervenes at observations, desired controller output or learned action policies. Our object is a different intervention boundary: the **source-controlled transforms** between a pretrained policy and the physical command. The fix can belong to a source owner rather than the runtime policy maintainer. Two independently wrong transforms can cancel around nominal commands and create an unobservable failure under saturation; a correct temporal transform can also receive the wrong historical reference at its callsite, making actions history-dependent.

The proposed contribution is not the elementary algebraic fact of interval clipping, the Chebyshev center theorem, public-key signatures, exhaustive finite search or AST rewriting individually. The research proposition is that **a repair system must jointly constrain (i) what actions are physically deliverable; (ii) which hidden software worlds are observationally distinguishable; (iii) which source edits and evidence reads are authorized; and (iv) what complete program bytes are actually executed.**

The differentiation from generic monitoring is explicit: detecting a violated control invariant does not determine whether a permitted repair exists; and cryptographic signatures alone do not make the repair semantically correct.

## 2. Formal objects and contract

Let x denote hidden middleware state, h=H(x) its current observable information, u*(x) a desired physical control action and K(h) the set of physically deliverable actuated outputs under the current frozen middleware version. For the finite observational equivalence class X_h={x:H(x)=h}, any current-only action policy must produce a SINGLE y for the entire class. Its best achievable worst-world error under the infinity norm is

  rho(h) = min_{y in K(h)} max_{x in X_h} || y-u*(x) ||_infinity.

This is a **constrained Chebyshev center** problem over possible source worlds and the actual actuator image. The following statements are standard minimax/geometry observations formulated for the particular software authority context:

- If K(h) is the complete action space, rho is half the largest coordinate span of target actions over X_h. A source patch cannot be proved necessary if the black-box has a valid cached historical anchor which separates these worlds.
- If X_h has a single member, the obstacle becomes the distance between the desired command and K(h). This measures pointwise actuator reachability, NOT globally optimized future task return.
- If the world ambiguity is large AND target commands are outside K, both a history-information repair and a source-actuation change may be required.

### Algorithm 1: constrained two-dimensional information-reachability solver

For the double-clip source F(v)=clip(C clip(Pv,[-L,L]^2),[-B,B]^2), construct the exact finite union of convex polygons K by partitioning the original parallelogram into the nine combinations of lower/inside/upper actuator clipping regions. Avoid convex-hulling all projected polygons, which could invent unreachable points between disconnected components.

For candidate error bound r, all world targets define a single box

  D_r = product_j [ max_x(u*(x)_j-r), min_x(u*(x)_j+r) ].

The bound r is feasible if and only if at least one original K-polygon intersects D_r. Intersections use polygon half-plane clipping. Monotonic bisection yields upper/lower bounds around rho and a concrete reachable witness. The witness is conditional on trustworthy complete world targets and exact fixed control geometry. This computes a **numerical** action error floor, not a novel general LTL/SMT engine, proof for nonlinear contact trajectories, or executable inverse control sequence.

Code: research/semrepair/joint_information_reachability_operator.py
Adversarial tests: research/semrepair/test_joint_information_reachability_operator.py.
Green status: **awaiting actual dedicated GitHub CI outcome** when authored.

### Algorithm 2: source-authorized finite-catalog repair choice

An approved catalog offers candidate source edits c with declared positive or zero costs, producer/controller mappings P_c,C_c, internal per-channel bounds L_c, and a required edit scope. For n=2..32 channels and a policy input hyperrectangle with radii R_j, the candidate must satisfy robust source semantics under exogenous coefficient drift bounds dP,dC:

  sum_j (|P_ij| + dP_ij)*R_j <= L_i for all internal channels i,

  sum_j R_j [
    |(CP-I)_ij| +
    sum_k (|C_ik| dP_kj + dC_ik |P_kj| + dC_ik dP_kj)
  ] <= epsilon_i for all outputs i.

Candidates must additionally obey the externally signed component edit mask and each trusted hard actuator/software clip cap. Exhaustive selection minimizes declared cost over the **supplied finite approved catalog only**. A non-empty catalog can still yield REFUSE if all candidates fail; no supposed universal synthesis of all Python programs is claimed.

Code: research/semrepair/authority_aware_repair_synthesis.py.
Green 2..32 channel and authored 7D catalog CI: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38015187912 .

### Algorithm 3: real source-execution authorization

Patches to the authored actuator-adapter config require exact prior/replacement source hashes, actor/deployment/epoch/non-replay nonce binding, two distinct Ed25519 signers in a source-owner domain and two separate signers in a calibration domain. The calibration profile itself is pinned, binds to the exact proposed bytes, and contains an assumed error envelope plus a separately authorized maximum software clip. Only after semantic and robust verification does an atomic on-disk source replacement occur; the unchanged frozen SAC policy then executes commands through the loaded source config.

All signing identities are created by our tests; all calibration error intervals and signed hard maxima are first-party assumptions. This is not manufacturer approval, trusted sensors or a production filesystem adversary model. It demonstrates the algorithmic consequence of provenance assumptions, not their real-world establishment.

Successful ordinary source-activation CI:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38010872050 .
Successful dual-domain and coefficient uncertainty CI:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38014106998 .

## 3. Two real external software interfaces, two distinct semantic fault classes

**Case A — official Gymnasium robotics control/action middleware.** Frozen Farama/Minari SAC checkpoints have independently published weights and immutable model SHA256; Reacher and Pusher use genuine native MuJoCo physics and original task rewards. The reciprocal-shear plus intermediate clipping is expressly injected by the authors into the adapter. There is no claim that a Farama upstream issue contained this exact software bug. Because the source mappings compensate near zero, low-amplitude probes cannot certify the entire bounded control domain.

**Case B — Hugging Face LeRobot time-anchored relative actions.** The actual, unmodified publisher implementations of to_relative_actions and to_absolute_actions are loaded from commit b9cb121cb7d4e3e26ec5c906d08dda68d151ba52, with original Git blob SHA1 3405402904cf15ca18227b3a3fe006d7936f9d53 and verified original source byte SHA256 1c99f2a66649579cf6536d1970f71172b0b50bcb8decd226a9aa8969c0b0f171. We create a FIRST-PARTY caller error that replaces the original chunk-start reference with the current state; both broken and corrected calls execute the original PyTorch conversion function. This is **not** a discovered LeRobot upstream bug. The question is whether a current-only repair is identifiable without history.

Successful 64 pairs of indistinguishable historical worlds, each with two alternatives, 8 temporally dependent chunk steps: current-only minimax max absolute action error averaged 0.233774; the true correct-anchor caller has exactly zero numerical error. A blackbox that retained correct original chunk-start state CAN tie the source correction; strict source repair superiority is therefore false under equal historical evidence. Actual baseline:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38016101114 .

Additional AST source patch work transforms exactly one positional source argument in an AUTHOR-INJECTED callsite, while executing the pinned actual vendor tensor conversion kernel. No production code or upstream vendor source is modified. Original-case CI **awaiting verification**.

## 4. Methods and actual executed experiments

### 4.1 Pusher minimal edit

On 16 previously inspected seeds under the same frozen SAC expert, four 100-step official simulator arms were evaluated. The artifact genuinely signs and installs a source config; it compares the original broken source, official health reference, a one-field clip-only source repair, and the original three-field matrix+clip atomic rewrite.

| Source variant | Authored source fields changed | Original official mean return | Mean true object-target distance (m) |
|---|---:|---:|---:|
| Broken adapter | 0 | -56.395976 | 0.245791 |
| Healthy external actor | 0 | -26.379423 | 0.175368 |
| Signed one-field clip-only | 1 | -26.379423 | 0.175368 |
| Signed three-field atomic | 3 | -26.379423 | 0.175368 |

**Observed source edit cardinality reduction: 3 to 1 (66.7%) at equal observed original reward and task geometry.** This is an *authored* interface where the intermediate clip may be expanded to 9.05 because a 9.1 software max was **assumed**. If the true hardware/software licensing cap is below 9.05, this patch is NOT admissible and a more invasive edit or refusal is needed. Do not call it universally safer or less risky just because fewer fields changed. Completed 64 native-task raw records:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38015338180 .

### 4.2 Prior task-level external policy evidence

Two frozen SAC expert/medium checkpoints on each official Reacher-v5 and Pusher-v5 task were evaluated under the same planned five fault severities and six same-seed arms. A strong blackbox with exact fault parameters globally minimizes **instantaneous actuator deviation** over the actual doubly-clipped reachable physical output set. The source-restored policy outperforms that named comparator under severe planted faults; weak faults tie or sometimes slightly favor the comparator.

Pusher expert severe fault: original mean cumulative reward gain +8.738653 and mean true physical object-goal distance improvement 0.058898m versus exact instantaneous action projection; Pusher medium +5.544641 and 0.020738m. Original archive with independent 32-seed two-endpoint audit:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009661288 .

Stronger blackboxes granted exact native dynamics, true physical state, planted interface identity and complete original reward. On 16 **previously inspected** Pusher expert starts, greedy one-step and SAC-in-the-loop three-step MPC return -47.696714 and -47.480364, compared with -26.379423 for the healthy/source-repaired reference; native rewards match simulator execution to 1e-7 and numerical parity at ~1e-16 for selected MPC actions. These are finite candidate, short-horizon baselines, not globally reward-optimal arbitrary blackbox policies. CI:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38013779442
and
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38013970154 .

### 4.3 Source temporal information/fault interaction

A conceptual two-world pair using identical current observations and different historical anchors can be represented through the minimum physical control error operator rho(h). Source clip repair changes K(h); history read changes the world equivalence class. Full numerical six-mode ablation has code and tests, but must remain labeled **pending CI** until confirmed. This model experiment is not an end-to-end combined LeRobot+Pusher robot benchmark: it reuses established two-stage clip geometry with author-constructed world targets.

## 5. Editorial-level baseline and ablation requirements

To avoid an easy self-generated benchmark:
- A1: no repair vs unconditional atomic matrix+clip rewrite vs minimal licensed repair vs best privileged physically feasible action-only compensation.
- A2: signature-only vs signature+nominal semantics vs separate calibration-signature+drift-envelope vs hard-max cap and fail-closed constraints.
- A3: current observation only vs equal-information history-enabled blackbox vs source-anchor read/repair vs strict missing-trust abstention on the pinned LeRobot code.
- A4: full-action-domain certificate vs single numeric sample, one random point, or held-out numeric probe, including adversarial clipping boundary cases.
- A5: seven-source case in which clip-only permitted then physically HARD-forbidden; measures correct refusal, not numeric robot score improvement.

Evidence denominator always equals distinct task seeds or constructed world pairs, not individual 100-step time samples or 960 statistically independent robot bodies.

## 6. Most relevant *actual accepted* literature and positioning

**Nenchev et al., Journal of Systems and Software (2025), "Compositional code-level safety verification for automated driving controllers."** DOI 10.1016/j.jss.2025.112499. This is a much stronger independent CODE-LEVEL verification comparison: contracts plus bounded model checking, scalability to neural-network-based controllers and seven controller types, reported runtime in minutes. We currently do not match its reusable production verification or independent code case count. Our differentiated problem is licensed source repair selection and observation-bound recovery, not general software safety verification.

**Päßler et al., JSS (2025), "Analysing Self-Adaptive Systems as Software Product Lines."** DOI 10.1016/j.jss.2024.112324. A software product line/probabilistic formal model with an underwater robotic case study. Demonstrates that a software METHOD with a robot case can fit JSS without hardware, but only because the software-level contribution and model-checking framework are the main results.

**Luckcuck et al., Robotics and Autonomous Systems (accepted July 2026), "A compositional approach to verifying modular robotic systems."** DOI 10.1016/j.robot.2026.105648. This RAS paper contains an implemented first-order-logic ROS Contract Language, a reusable Vanda parser/monitor generator and a separate modeled robotic case study. It ALREADY solves important compositional contract verification issues. Merely adding signatures and an elementary interval inequality without verified code/source patches is not competitive with its formal breadth. Our candidate difference is repair intervention permission and information-physical joint obstruction; RAS is therefore a less precise lead journal for the PRESENT software claim.

JSS official aims require contributions to software engineering independently of a single application domain, and permit formal proofs, simulation, empirical work and negative results:
https://shop.elsevier.com/journals/journal-of-systems-and-software/0164-1212 .

RAS official scope includes theoretical/computational robotics but expects robotics contribution:
https://shop.elsevier.com/journals/robotics-and-autonomous-systems/0921-8890 .

## 7. Strongest remaining reject reasons and exact improvements needed

1. There are currently TWO external published SOFTWARE KERNELS/ENVIRONMENTS, but **not two independently documented genuine vendor bugs**. Both claimed fault-injection source cases are author-created. Need actual upstream before/after released implementations where the defect and fix are externally corroborated, or explicit JSS fault-injection-methodology framing with multi-project code mutation coverage.
2. The AST source repair handles one predeclared argument substitution. It is not currently a sound general source-program synthesizer, and general nD finite catalog optimization is exact ONLY relative to a manually supplied candidate set.
3. The two-world history ambiguity model assumes the entire possible historical-anchor set is given. In actual robots, learning this set and authenticating it would be an important separate problem. The source-read flag is not independent proof of valid issuer credentials.
4. Real-world signer root keys and calibration intervals are author-generated. Strong cryptographic tests are meaningful SOFTWARE TESTS, not evidence of manufacturer trust.
5. Shared publisher and same base family for the four external SAC actor checkpoints; no pretrained VLA, heterogeneous software maintenance team or real robot source failure evaluation.
6. CPU time, program analysis wall-clock cost, false authorization, timeout, certificate precision and AST-patch search coverage need measured cross-version controls, ideally on two independently maintained software repositories.
7. Novelty is a research direction, not a demonstrated top-accepted-paper contribution until independent code history and stronger generality are shown. No SCI-2/CCF-B acceptance probability is justified.

**Editorial decision:** concentrate on JSS *regular research paper* (software-source semantics and authorized proof-carrying patch choice). RAS is alternative if real robotics control recovery over independent non-Farama simulator families becomes the primary contribution. EAAI/Neurocomputing are poor fits absent a genuinely new learned repair model. No physical device is required by the research design, but broad independent reproducible evidence is.

Do not submit a duplicated version of the same findings simultaneously to JSS and RAS; freeze original source hashes, open scripts and truthful negative outcomes.
