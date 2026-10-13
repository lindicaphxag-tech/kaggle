# P4 / Numerical Algorithms — rank-event sweep and verified numerical boundaries (PARTIAL PROTOTYPE)
Date: 2026-10-13 | The exact rational event aggregation prototype has been executed; root isolation and real-input guarantees remain open.

## Evidence hierarchy
- For fixed source-fitted SPD S, class contrast d and held source pair v, score difference N(lambda)/prod_i q_i(lambda), positive denominator on feasible interval. deg N<=p-1. Elementary rational polynomial identity; not original root-isolation mathematics.
- Real Weibo 3ch p6 source rank coverage: old termwise 94.121%, Bernstein 94.267% (full interval): +.146 pp; previously examined source folds, NOT new target AUC.
- Original rational 960 fixture: centered Horner and Bernstein 924/960, centered faster, 36 actual crossing intervals: **no exclusive Bernstein edge there**.
- NEW corrected 3-family, p=2,3,6,9,12, 4320-rational-case exploratory sweep: centered Horner 4212, Bernstein 4232; +20/4320 = 0.463 pp, with 88 genuine sign crossings. Tiny but nonzero selected synthetic advantage, NOT floating EEG end-to-end guarantee. First sweep had generator scalar-multiple defect and was corrected post inspection.
- Executed exact rational incremental event-map parity: dimensions p=3,6; constructed full-degree p=6 numerators; 4–32 pair crossing fixtures; ties, endpoint roots and even-multiplicity touching events. Identical complete event outputs to full pair re-evaluation using the SAME SymPy exact root routine. Local median p=3/32 pairs/64 events 0.450 versus 0.037 s; p=6/16 pairs/32 events 0.380 versus 0.056 s. In no-event controls the incremental form is slightly SLOWER (p=3: 0.461 versus 0.482 s; p=6: 3.523 versus 3.547 s). Machine-dependent and constructed; independent CI: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38354399661.
- A new journal novelty argument cannot be based on standard Bernstein coefficients or merely recovering from a weak Horner baseline.

## Doctoral-level problem
Compute and VERIFY the entire scalar-shrinkage held-source ROC-AUC event map (all rank changes and tie values) efficiently and with a controllable numerical error budget, exploiting shared spectral poles and low-degree rational numerators. Certified constant-sign intervals are only a subroutine; the principal product should be exact or reliably enclosed event locations and an accurate continuous AUC-optimum set.

## Proposed algorithm: pole-shared rank event sweep (event aggregation parity verified on rational fixtures; remaining parts unverified)
1. For each source-trained SPD eigensystem, factor q_i(lambda)=s_i+(nu-s_i)lambda. Positive feasible interval must be proved; reject singular bounds.
2. Precompute p leave-one-factor numerator basis polynomials B_i(lambda)=prod_{j!=i} q_j(lambda), so for each held positive-negative pair N_pair(lambda)=sum_i a_{i,pair} B_i(lambda). Share B_i across ALL pairs to reduce redundant polynomial construction.
3. For every nonzero N_pair, isolate ALL roots in feasible interval by an independently verified exact-rational Sturm/Descartes method for rational inputs, or outward-rounded interval arithmetic + input-error enclosure for floating EEG. Multiplicities, exact ties, endpoint equality and near-coincident roots need explicit treatment.
4. Sort isolated root EVENT BUNDLES; sweep pairwise sign-order credits between events and evaluate tie credit 1/2 EXACTLY at an event. Compress changes in person macro-AUC, return maximal open cells and exact event-optimal points; no point-only dense sampling may be called a proof.
5. Use midpoint-Horner as cheap first-stage certifier; use Bernstein only for unresolved intervals where it adds signs at lower budget; escalate root isolation when crossing exists. With verified bounds, unknown intervals remain unknown and cannot be mislabeled safe.
6. Derive and test complexity: shared basis O(p^2) precomputation (implementation dependent), pair root-isolation cost and O(R log R) event sorting; compare from actual profiler, no assumption a priori of asymptotic speedup.

## Strong baselines and tests
- Centershift Horner (exact rational and outward-rounded), Bernstein, adaptive subdivision, Sturm exact root isolation (not merely unshifted Horner), numerical eigendecomposition with high precision, brute 27-grid and dense lambda sweep only as *approximation* baselines.
- Exact rational stress: multi-spectrum 3+ families, generated distinct coefficient shapes, repeated/near endpoint roots, stationary ties, 10x condition-number variations. Include adversarial inputs selected independently from algorithm design fixtures.
- Real EEG: >=3 different native montages (p6, p>6) and repeated source folds; compare complete source AUC event map, continuous-grid argmax discrepancies and runtime/certification failures. Must not equate source ranking quality with target AUC.
- Verified end-to-end claim only with bounded feature/covariance estimation error, SPD margins, outward rounding and separate third-party arithmetic checker; otherwise label exact for rationalised surrogate inputs.
- Benchmark quality: accuracy of complete root/event map, zero UNSOUND certificates, % unresolved, wall-clock at matched precision and memory, plus Pareto frontier. Show optimality value and computational use beyond 0.463 pp selected synthetics.

## Conditional acceptance-level gates
(A) correct full-event ROC AUC at roots with ties;
(B) materially faster / tighter than sound state-of-art root isolation at equivalent precision and representation, on independent instances;
(C) documented failure modes, degeneracies and actual implementation complexity;
(D) real high-dimensional source datasets and input-error guarantees.
If B or D fail, do NOT claim exceptional Numerical Algorithms novelty. Consider a specialist technical note only with appropriately narrow claims. Target journal alignment first, acceptance rankings after a defensible matched-paper audit.