# JAX #41227 and #41228 — independently executed regression evidence

## 2026-10-11 latest current-PR-head status (supersedes older snapshot below)

- **[JAX #41227](https://github.com/jax-ml/jax/pull/41227)**: current one-commit head `5ca3f43d82c7bf81351074816781c9e1d755f6ed`; its Git tree matches the previously exact-source-tested `1d8cac8...` content (`870699f350e72c87dafbb36a5ae89ff4bc8769bf`). Its 20 existing/new Student-t test cases and true before/after numerical witness remain grounded on that byte-identical code tree.
- **[JAX #41228](https://github.com/jax-ml/jax/pull/41228)**: source-exact tested **`2af46d5f0bf14c458c8ffa3ff5bd8fc395d613f7`**, with the Poisson-local numerical correction and near-float32-maximum finite log-PMF regression. [Four green exact-source CI jobs #38081399284](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38081399284): **54 public Poisson distribution cases passed; 43 Poisson/sampling-related cases passed; 20 Student-t cases passed**. Focused tests overlap and are not counted again. 
- [New exact-source Poisson red/green #38081402392](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38081402392): 2/2 completed; original JAX source `93e877c...` variance/λ **1.4613796483** versus pinned repaired source `2af46d5...` **0.9996126592** with three seeds and 300,000 draws per revision at λ=10⁸.
- **Rebase caveat:** after that CI, #41228 moved to `fbd0d2256286b85961b46caa64564d369a9830ed` on a newer JAX mainline. The four relevant PR file Git blobs are **exactly unchanged** relative to tested `2af46d5...`: `jax/_src/random/core.py` blob `ffe5ca968973fcf63981fdd648dc53dcb63eb42f`, `jax/_src/scipy/stats/poisson.py` blob `9d519197342499fb8757e585644ce55d74f5e7ef`, `tests/random_lax_test.py` blob `26b32f59933b3b27981c0d8a2d1769349790a256`, `tests/scipy_stats_test.py` blob `8e3e91617fe95c02355dc5991b75e9771095ef71`. **This proves the Poisson contribution file content is identical, not that every dependency in the rebased full JAX tree has been re-executed.**
- Official upstream JAX reviews remain binding. Independent CI, immutable commit receipts, verified file blobs and reproducible data **are not maintainer approvals or merged status**.

---

Public reproducibility note for maintainer review, 2026-10-11 (UTC+9).
The writer is the external contributor; this is **not an upstream review approval, official JAX CI, or a merged change**.

## Exact source identity and upstream code scope

| Official upstream PR | Current single-commit HEAD | CI-executed equivalent head | Proven identical Git tree |
|---|---|---|---|
| [jax-ml/jax#41227](https://github.com/jax-ml/jax/pull/41227) stable Student-t logpdf | `5ca3f43d82c7bf81351074816781c9e1d755f6ed` | `1d8cac844d810025b6189617508f8043631e3de6` | `870699f350e72c87dafbb36a5ae89ff4bc8769bf` |
| [jax-ml/jax#41228](https://github.com/jax-ml/jax/pull/41228) Poisson transformed rejection | `3ef5a045d0492f2f6e5f4f1d8e7e01f4216ebaa9` | `c5e895b766b791d95139dfbc0de30e69095616e7` | `0ebb4d21231f0d52135ab594aaa3ef75bbc27020` |

Git tree identities are exact values returned by GitHub's git-commit endpoints. Source files are byte-identical despite the later one-commit squash; the logs retain their genuine executed (pre-squash) commit IDs.

## CI summary — all jobs completed successfully

[**Four-job exact-source independent Actions #38077439200**](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38077439200)

- #41227 current-equivalent tree: Student-t `tests/scipy_stats_test.py -k TLogPdf`, **20 passed**, including extreme valid scales. Focused `-k TLogPdfLargeDf`: **1 passed**, overlapping the broader selection.
- #41228 current-equivalent tree: Poisson `tests/scipy_stats_test.py -k Poisson`, **54 passed**; RNG/Poisson/stirling-related `tests/random_lax_test.py`, **42 passed**; focused random/public checks **10 + 5 passed**, overlapping broader results.
- All four jobs used Ubuntu Python 3.12, installed the actual JAX source checkout as editable and ran the repository's own test suite. Four jobs, no algorithm-test failures.
- Artifacts: [Student-t summary](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38077439200/artifacts/11678532259) and [Poisson summary](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38077439200/artifacts/11679201781).

## Before → after reproducible defect witnesses

**#41227 — Student-t float32 large-df cancellation**

- [**Source-exact red/green Actions #38077445868**](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38077445868), both jobs success.
- Inputs: `x=0`, `df=[30,1000,100000,10000000]`, `float32`, eager and JIT; reference `scipy.stats.t.logpdf` evaluated in float64.
- Original JAX base `656227de179fc76b43197ad2086f31a2db6c1637`: **max absolute logpdf error = 0.9189385582**.
- Patched reviewed source tree: **max absolute logpdf error = 4.89076255e-7**.
- Raw JSONs: [base](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38077445868/artifacts/11679272517), [patch](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38077445868/artifacts/11679016821).

**#41228 — Poisson variance inflation at lambda=1e8**

- [**Source-exact red/green Actions #38077442588**](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38077442588), both jobs success.
- Each run: `lambda=1e8`, `float32`, 3 fixed PRNG seeds × 100,000 samples each (300,000 samples per source), CPU, Python 3.12. This directly exercises upstream [bug #41202](https://github.com/jax-ml/jax/issues/41202).
- Original JAX base `93e877c99dc05a1fe9409a9789c25e05bb187982`: **variance / lambda = 1.4613796483**, mean / lambda = 0.9999996050.
- Patched reviewed source tree: **variance / lambda = 0.9996126592**, mean / lambda = 1.0000001471.
- Raw JSONs: [base](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38077442588/artifacts/11678692145), [patch](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38077442588/artifacts/11678846948).

## Fully executable replication

- [Student-t harness](./jax_student_t_pr41227_redgreen.py) + [workflow](../.github/workflows/upstream-jax-student-t-redgreen.yml).
- [Poisson harness](./jax_poisson_pr41228_redgreen.py) + [workflow](../.github/workflows/upstream-jax-poisson-redgreen.yml).
- [Existing + new official test selections](../.github/workflows/upstream-jax-flagship-review.yml).
- Workflows lock exact upstream SHAs, install those sources and record the genuine commit identity and package versions. A later maintainer change requires fresh testing or verified byte-identical tree identity.

## Review / interpretation boundaries

For #41227, the current diff implements reviewer suggestions to use `mul(df, half)`, parameterized floating dtype and the vectorized official `_CheckAgainstNumpy` / `_CompileAndCheck` test pattern. The later fix avoids `scale**2 * df` intermediate overflow/underflow. For #41228, the public SciPy-compatible Poisson log-PMF shares one stabilized helper with the rejection sampler; Stirling asymptotics are now local to Poisson, not the Binomial-shared helper. These statements are based on read source diffs, not substitute maintainer approval.

The regression probes cover focused parameters, and the Poisson samples are pseudo-random trials, **not independent machines**. Do not extrapolate to universal numerical accuracy or full platform support. Both upstream PRs currently require explicit JAX maintainer re-review. No upstream comment or review request was submitted by this workflow.
