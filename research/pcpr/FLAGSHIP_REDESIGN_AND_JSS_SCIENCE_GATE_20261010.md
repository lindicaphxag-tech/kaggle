# PCPR vNext — Flagship redesign, provenance-led evidence, and JSS scientific gate (2026-10-10)

> **Status: substantial research-tool engineering, NOT a JSS full-paper empirical breakthrough; SCIENCE HOLD.** Maintainer approval, journal acceptance, independent benchmark accuracy, and superiority to prior work are not claimed.

## Why the current argument still fails a strict JSS regular review

- The pre-diff frozen v7 run on **12 authentic PRs** yielded **0 source-grounded differentiating witnesses and 12 abstentions**; its outcome-locked selection and result remain immutable in `lindicaphxag-tech/cs`.
- In **six new, pre-inspection, merged upstream PRs**, the first generic source-forward-read detector yielded **0/6 PRs with source-anchored candidates**, **0/6 native witnesses**. Exact run: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38027317715.
- **After inspecting those six**, four native PyTorch operation-pattern probes matched **6/6**, but these were **POST-HOC DEVELOPER TESTS**, not held-out generalization, patch correctness detection or independent ground truth. Run: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38027485614. The six include a feature PR; matching a PyTorch mechanism is NOT a verified defect.
- Using the **frozen v0.2 algorithm** (commit `92aeab877d341e8e923f451e2e45757c4b6e7516`) on **eight differently owned, not-yet-inspected upstream PRs** produced **0/8 local semantic probes**, **0 external package validations** and **0 patch-correctness decisions**. Exact report: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38027541722. This is an intentionally hard transfer falsifier, not something to conceal.
- **Real Git-pinned source** `deepchem/deepchem#5190`: original source classes plus unchanged optimizer and owning graph; 14 independent **field** bounded native registration witnesses and 13 field abstentions across only two classes; one synthesized partial fix rejected. These are correlated field-level measurements, not 14 independent projects or adjudicated incorrect patches. Proof of **field-register operator effect only**, not end-to-end DeepChem.fit. Reproducible `run_real_deepchem_source.py`.
- **Real source-tree native Python-package replay** `kornia/kornia#5446`: verified before `457fa2ec3182483528251241fc312fc49768d44f` and after `59a5de3c8ae87d3e9e020028510312451a138127`, with exact original Git blobs. Native Kornia `Quaternion` exposes buffer registration and outer-module dtype conversion difference, while explicit `nn.Parameter` remains a parameter in both. **3/3 predefined package-level effect checks PASS** on GitHub CI: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38027699749. This is a *previously inspected* PR and measures actual package behavior, NOT patch correctness or independent new-case generalization.
- A second Kornia exact-source-tree So2 pre/post replay for PR #5005 has been implemented; its result must be checked against the run receipt before claiming a pass.
- Synthetic regression suite: **36/36 PASS**, normal and `python -O`; validates code contracts, not APCA scientific effectiveness.

## Actual methodological opening — Source-Independent Effect Closure

A defensible new research target is *not* to add another string-matching PR classifier. It is to construct an **authentic, source-independent, executable path** of obligations and evidence that handles compound fixes without using the patched program's own assertions as a truth source.

A candidate procedure (not implemented end-to-end yet):

1. **Revision-attested provenance DAG**: fetch Git exact base/head, validate Git tree content identities and unchanged source (line-level source digest, call/dataflow, alias, parameter/buffer owner, state, shape, gradient effects). Do not assume a file hash alone establishes unchanged status.
2. **Cross-scope typed effect closure**: connect changed fields/expressions to unchanged consumers, model ownership, optimizer update, checkpoint `state_dict`, serialisation, device/dtype casts and runtime control flow. Unknown dynamic dispatch or unsupported APIs produce a typed uncertainty, not a fake proven edge.
3. **Independent obligations**: classify observed BEFORE source uses, public library API semantics, existing pre-patch tests and human issue specifications separately. A read or call generates a *hypothesis*, not automatic proof of desired intent. Newly written patch keywords may suggest test contexts but cannot define correct behavior.
4. **Executable counterfactuals**: generate cases that compare equal pre/post inputs under actual pinned package: parameter vs buffer vs plain tensor, detached gradient, missing serialization key, shape broadcast, stale runtime state, alias, accelerator context. Test both negative and preserving controls. Unsupported environment, e.g., CUDA-only, remains untested.
5. **Non-promotable evidence certificates**: report one of observed violation, observed local support, or abstention, preserving exact obligation, source provenance, context, execution layer, budget and unmeasured effects. No finite local test declares an entire PR correct.
6. **Budget-aware witness selection**: choose high-information independent obligations under actual CPU/GPU/LLM and reviewer costs, then test against a matched-budget random/coverage baseline; prior author-directed set cover had 11/12 vs random expected 10.67/12, so cost superiority cannot be assumed.

## Distinguishing contribution from existing 2025–2026 work

- ChangeGuard, Mokav: differential test generation; demonstrate independent semantic obligations and transparent authorization *beyond* merely exposing changed behavior.
- SWE-Proof (2026): faithful formal specifications are a significant pre-existing research result; our method must establish the incremental source/requirement provenance and effect-verification contribution rather than imply specification-fidelity discovery is new.
- 2026 ESEM LLM-as-Judge 326-patch study: correctness assessment has evidence leakage and false positives. Any claimed gains require sanitized inputs, real ground truth and matched budget.

## JSS regular empirical gates (must ALL pass before 'ready')

1. **Method generality**: automatically produce source-independent obligations on a new, prospectively locked **100+ real PRs** across independent repositories. Inspect neither patch diffs nor gold annotations when selecting/tuning final method. Freeze code SHA, input URLs, baseline budgets and metric code before running.
2. **Truth**: two independent expert reviewers assess *obligation truth* and correctness of naturally occurring wrong, partial and valid fixes. Human disagreements adjudicated and explicitly retained.
3. **Strong baselines**: compare with repo-native tests, differential/LLM-guided tools (ChangeGuard/Mokav on applicable Python tasks), and a high-quality LLM patch judge using sanitized evidence. Do not import published benchmark numbers as if they were results on this cohort.
4. **Critical endpoint**: incorrect **whole-patch** authorization at a *predeclared positive selective coverage* (e.g., coverage >= 30%), plus full risk-coverage AUC, positive witness recall, missed defects, abstention, native pass rate and costs. An always-abstain strategy is a baseline with 0 utility, not perfect accuracy.
5. **Independent generalization**: per-repository clustered intervals and held-out codebase types; report exact zero-coverage failures and the actual denominator. 100 PRs in two repos are **not** 100 independent environments.
6. **Reproduction**: accepted licenses, public runnable environment, exact full-source dependency versions, source hashes, raw ledgers, optional GPU conditions, at least one external independent reproduction.
7. **Novelty assessment**: empirical improvement at equal costs AND a mechanism showing why non-self-grounding effect closure works; mere runtime probe success or more unit tests is insufficient.

## Recommended manuscript positioning

- **JSS regular full research**: **HOLD** until above scientific gates and decisive held-out positive evidence.
- **JSS NITP**: a sober, falsifiable *research idea* article with current negative results is possible after editorial eligibility, literature and manuscript version checks, not guaranteed.
- No manuscript will claim `6/6` as generalization or `0/8` as high patch-correctness precision.

## Reproduction artifacts

- Main PR: https://github.com/lindicaphxag-tech/kaggle/pull/84
- Original source lock for six: `research/pcpr/VNEXT_PRE_INSPECTION_COHORT_LOCK_20261010.md`.
- Independent eight: `research/pcpr/V02_UNSEEN_EIGHT_PR_LOCK_20261010.md`.
- Exact v0.1 & v0.2 first-score JSON and isolated counterfactuals are GitHub Actions artifacts, not paper statistics.
- Kornia actual upstream package case: `research/pcpr/run_native_kornia_5446.py`; report SHA-backed.
- Kornia So2 second actual package case: `research/pcpr/run_native_kornia_5005.py`, status pending independent CI check.
