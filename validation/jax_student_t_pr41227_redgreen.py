#!/usr/bin/env python3
"""Source-exact Student-t logpdf pre-fix/post-fix numerical error comparison."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import scipy.stats
import jax
import jax.numpy as jnp
import jax.scipy.stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expect", choices=("broken_baseline", "corrected"), required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()
    df64 = np.array([30.0, 1e3, 1e5, 1e7], np.float64)
    df32 = df64.astype(np.float32)
    xs = np.zeros(4, np.float32)
    scipy_ref = scipy.stats.t.logpdf(np.zeros(4, np.float64), df64)
    fn = jax.scipy.stats.t.logpdf
    eager = np.asarray(fn(jnp.asarray(xs), jnp.asarray(df32)), np.float64)
    compiled = np.asarray(jax.jit(fn)(jnp.asarray(xs), jnp.asarray(df32)), np.float64)
    err_eager = np.abs(eager - scipy_ref)
    err_jit = np.abs(compiled - scipy_ref)
    maxerr = float(max(err_eager.max(), err_jit.max()))
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    # The thresholds intentionally distinguish the original catastrophic
    # lgamma difference cancellation from the stable beta-identity form.
    good = maxerr > .05 if args.expect == "broken_baseline" else maxerr < 5e-4
    report = {
        "upstream": "jax-ml/jax",
        "pr": 41227,
        "source_commit": sha,
        "jax_version": jax.__version__,
        "jax_import_path": str(Path(jax.__file__).resolve()),
        "backend": jax.default_backend(),
        "dtype": "float32",
        "inputs_df": df64.tolist(),
        "inputs_x": xs.tolist(),
        "scipy_float64_reference": scipy_ref.tolist(),
        "jax_eager": eager.tolist(),
        "jax_jit": compiled.tolist(),
        "absolute_errors_eager": err_eager.tolist(),
        "absolute_errors_jit": err_jit.tolist(),
        "max_abs_error": maxerr,
        "expected_state": args.expect,
        "condition_passed": bool(good),
        "criterion": "baseline max-error>0.05; corrected max-error<0.0005",
        "limits": "4 targeted large-DF points, CPU and one float32 dtype; not universal accuracy proof",
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not good:
        raise SystemExit("Source-exact red/green expectation not satisfied")


if __name__ == "__main__":
    main()
