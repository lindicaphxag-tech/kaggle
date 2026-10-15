# P3 Lee2019 planned target B pre-target numerical provenance FAILED — FORENSIC AUDIT EXECUTED
- CI forensic check success: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38312802226
- Machine-readable immutable receipt: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38312802226/artifacts/11774744407
- Original first SOURCE12 manifest:
  `27efc2565d5bc2bc22344b67ee994e42b985c16c34cabf6cdf55136270fa18f3`
- Halted B run recalculated SOURCE12 manifest:
  `0b2869570c6ba4c466a46e9a7d9ea335c195cb9a64292e2b6de9cdcad167e7ff`
- The 12/12 per-person source-covariance byte SHA differ.
  All source-only scientific choices and fields EXCEPT original byte hashes
  and `nuisance_cov_between_fraction` match exactly; last metric differed
  by 2.220446049250313e-16.
- This supports a possible numerical-pipeline reproducibility issue,
  but WITHOUT original binary arrays we cannot certify that all
  array differences are merely floating-point noise.
- Target B was NOT READ during halted run and remains NOT evaluated.
  No relaxation of SHA lock, retuning or invented target B claim.
- P3 still HOLD: independent common-policy benchmark and distinct
  enough outcomes beyond P1 needed before submission.
