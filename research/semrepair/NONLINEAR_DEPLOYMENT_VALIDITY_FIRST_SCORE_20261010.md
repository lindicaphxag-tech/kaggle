# SemRepair: model-validity boundary under nonlinear interface aliases

**Status (2026-10-10):** first-party native MuJoCo falsification, deterministic linear-noise interval checks, and theoretical finite-query obstruction. Public PR remains Draft. No real VLA, new manufacturer family, externally authenticated source or autonomous hardware repair was tested.

## Actual reproducible evidence

- Source + tests: research/semrepair/deployment_validity_shield.py and test_deployment_validity_shield.py
- Public successful pinned MuJoCo 3.3.7 CI: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38003118008
- The workflow executes all **16** tests on Python 3.12, including the native simulator experiment and full raw JSON archive.
- Historical solver uses 2 linearly independent numeric input/tap reads and 1 held-out read to infer two linear ABI maps. We constructed two coordinate-wise cubic action mappings whose perturbations are identically zero at all 3 historical probes, while becoming nonzero on execution-relevant additional commands.
- Across **12 deliberately designed and dependent** nonlinear parameter cases (2 axes x 6 nonzero strengths), the historical 3-probe linear null was never falsified (0/12 detection). With the identical two training taps, old holdout and four extra explicit deployment probes (7 total), 12/12 were falsified.
- A **7-tap generic fixed probe baseline**, supplied the same numeric measurements and cost, ALSO detected 12/12. The new probing layout has **NO empirical fixed-budget algorithmic advantage over this baseline** on the constructed family. Seven-vs-three is an information-cost difference, not a claim of better inference at equal budget.

## Sound conditional statements (versus unsafe claims)

1. **Negative identifiability:** for ANY finite set of selected input commands, a polynomial term can be constructed that vanishes at every observed x coordinate yet differs elsewhere. A successful finite test cannot certify arbitrary nonlinear ABI correctness. The stronger 7-probe counterexample was independently tested. Elementary polynomial interpolation is prior art.
2. **Noise-aware compatibility:** when each producer and controller numeric tap coordinate has a declared deterministic error <=epsilon and the true ABI is linear, fitting an input-to-tap map on two independent commands gives an exact per-validation-coordinate residual tolerance epsilon * (1 + ||U_inverse u||_1). Reject values beyond this conservative necessary test. This is NOT a sufficient global linearity proof.
3. **Bounded-noise repair classification:** the new interval-based method bounds EVERY feasible producer-to-identity matrix deviation and controller-output-minus-producer input effect under bounded tap noise, refuses rank-ambiguous interfaces and undecidable near-threshold classes, and returns only an untrusted repair candidate when classification is sound under the stated linearity model. No signer or execution authority is supplied.
4. **Conditional full envelope:** for externally established L-Lipschitz residuals on a specific square deployment domain, a complete Cartesian epsilon-net with max-norm covering radius rho gives max residual <= max_grid residual + L*rho. The implementation checks mesh completeness and refuses to call any globally valid result without the EXTERNAL L bound. This is standard covering theory; the code cannot certify the supplied L or authenticate its issuer.

## What the tests do NOT demonstrate

- 12 crafted nonlinearity parameters are not independently sampled robot systems or manufacturers.
- A solver intentionally challenged with nonlinear polynomials rooted at its original probes cannot alone support state-of-the-art detection rates.
- A 7-probe generic baseline ties the new method. No equal-budget active-probing win, no optimized selection algorithm or query-cost novelty is claimed.
- Every tap can be spoofed if its physical origin is untrusted. Existing provenance group tests remain conditional on honest independent issuer roots. A hash of asserted bytes does not authenticate a runtime force sensor.
- The original 2DOF native MuJoCo closed-loop results concern a hand-coded PD policy; no frozen pretrained VLA and no truly unseen controller family has yet been run.
- A finite numerical check plus external unverified Lipschitz claim is not end-to-end software/robot safety or protection against correlation in device compromises.

## Defensible contribution and nearest accepted peer

The general research challenge is **source-bound repair authority conditional on observation support, evidence integrity and true deploy-time action range**, not a new polynomial/linear algebra theorem. ICML 2026 Yan et al. VLA-FixBench (https://proceedings.mlr.press/v306/yan26r.html) evaluates 20 VLMs over faults and robot recovery in LIBERO and real platforms; our present hardware and pretrained-policy evidence remains weaker. The strongest next experiment must be freeze-then-test on an external policy/controller/plant with matched access, risk and probe budget, and report task success, unsafe acceptance, conservative refusals, and independent source-family uncertainty.

**Current publication gate:** method correctness strengthened, key old interpretation falsified; RAS/RA-L/T-RO or CCF-level ready status NOT established.


## Second falsification, recorded after the first 12 cases (not hidden test)

The first extra four probes were axis-aligned and the same-budget generic comparator matched 12/12 on deliberately constructed single-axis cubic faults. A **new, separate cross-axis interaction** was then constructed: producer-x perturbation proportional to x*y*(x-0.31)*(y+0.27). It vanishes at the original training/holdout points AND at every extra axis-only probe, but differs at paired nonzero x/y inputs.

In native MuJoCo, across **six dependent coefficients**, three historic probes detected 0/6; seven axis-aligned probes detected 0/6; the seven-probe fixed generic baseline detected 6/6; and a seven-probe diagonal corner layout detected 6/6. This is a demonstration that the first proposed probe selection was incomplete, **not** out-of-distribution manufacturer transfer or unique new active learning. The corner method is tied with generic at equal budget on this designed family. New public successful CI with **17 tests** and raw data: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38003533144 .

Hence combined constructed challenges include 12 axis-cubic and 6 cross-axis parameter conditions. They were authored sequentially as adversarial mechanisms and must not be pooled into a preregistered independent test set or turned into a 100% universal detection claim. A truly frozen external controller family and independent probe selection design remain mandatory.
