# SemRepair — physically equivalent but semantically WRONG repairs (2026-10-10)

**Result type:** new, author-operated **native MuJoCo 3.3.7 physics** counterfactual against a previously constructed four-world scalar-actuator family. This is NOT a new robot platform, third-party reproduction, learned-policy success, repair discovery on hidden implementations, or proof of field safety.

## Exact executed evidence

- Protocol: `semrepair-native-wrong-interface-counterfactual-v1`
- Original script: `research/semrepair/mujoco_wrong_repair_counterfactual.py`
- GitHub Actions: [frozen PR workflow #37973635217](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37973635217), **SUCCESS**
- Commit examined by that job: `1279559600eab89f594ac695d0f58983434e20ad`
- Python 3.12, `mujoco==3.3.7`
- Raw 72-cell JSON: [GitHub Actions artifact 11637842420](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37973635217/artifacts/11637842420), retention until approximately 2026-11-08; copy it to a permanent, content-addressed author-controlled archive if research authors later elect to release it
- JSON SHA-256: `490c04920fc223313295b7adb045dc21d48afd0883aa2e0463dd285053b2d8a8`
- Stored artifact ZIP digest as reported by GitHub: `sha256:fedd40551d1cf9f1a0fd35dbd2e88bb8cea7edec1e8f94eccc9fd99aaf1d3fb4` (ZIP digest is **not** the JSON digest)
- Fixed command-set SHA-256: `c0ccfac3c810ac097d9afc6ab602d6452d3ffea16eded6645c10dd52dfff2630`
- Seed: `20261010`. 24 deterministic distinct command sequences × 3 fixed faulty worlds = 72 **paired fault-sequence cells**, not 72 independent fault classes or robots.

## Observed values

| Fixed physical metric | Verified output |
|---|---:|
| Original faulty trajectory, *minimum* sup-norm deviation from reference | 0.04360371743231671 |
| Largest authorized (correct-interface) repair sup-norm deviation | 0.0 |
| Largest unauthorized (wrong-interface) compensating repair deviation | 0.0 |
| Wrong-interface patches whose numeric end trajectory matches the reference | 72 / 72 |
| Correct interface and wrong interface share the same semantic patch authority | **No**, by construction |

`max_error()` takes an elementwise L-infinity metric over actual scalar MuJoCo qpos/qvel traces; these values are **not millimetres**, success rates, or calibrated physical harm.

### Why this is nontrivial but narrow

For each fixed faulty world, restore reference execution in two ways:
1. **Correct**: change the declared defective software producer or controller transmission field.
2. **Wrong**: change a different interface whose multiplicative effect compensates the original mismatch.

Both corrections produce the **same nominal motion**, but only one has the declared patch/implementation authority. The wrong fix introduces an additional latent semantic inconsistency. Therefore task-trajectory similarity alone is an inadequate label for repair authorization in this *controlled construction*. This supports the identifiability motivation without claiming a policy beats a fair active-diagnosis baseline.

### Publication red flags

- All 72 cells share **three** known parameterized worlds from **one** synthetic model family, which were designed to possess these scale symmetries. They are repeated excitations, not independent scientific replications of the central structural equivalence.
- Exact zero reflects the fixed algebraic compensation and the chosen trace measurement. It does not establish numerical equality for all robotics APIs or production controllers.
- The experiment does **not** show correct source localization by a method with source hidden from the algorithm. The worlds' semantic labels are provided to the experiment harness.
- The result does not demonstrate active probe efficiency, cross-framework repair acceptance, independently chosen counterexamples, clinical effectiveness, or new upstream maintainer adoption.
- Do NOT use a 72/72 binomial CI as a cross-robot success probability. The independent treatment levels are the hand-constructed world family and experiment generator, not the individual source-trajectory cells.

**Next falsifier:** hold out previously unseen robotic interface implementations and choose one concrete repair with the source hidden from the active diagnostician, while competing against fair information-gain and identify-all baselines with matched sensing/action costs. Incorrect patch authorization and real executed task closure must be separate endpoints.
