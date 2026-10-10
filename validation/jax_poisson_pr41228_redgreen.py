#!/usr/bin/env python3
"""Reproduce jax-ml/jax#41202, compare original regression vs upstream PR #41228.

Run from an *exact* JAX source checkout. Never mutate the source under test.
Scientific unit: one pseudo-random draw; the multi-seed summary is descriptive,
NOT an independent test across 3 machine environments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", choices=("broken_baseline", "corrected"), required=True)
    parser.add_argument("--n-per-seed", type=int, default=100_000)
    parser.add_argument("--rate", type=float, default=1e8)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    if args.n_per_seed < 10000:
        raise ValueError("n-per-seed must be >=10000 for a useful variance witness")
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=pathlib.Path.cwd()
    ).strip()
    rate = float(np.float32(args.rate))
    seeds = (0, 1, 2)
    values = []
    summaries = []
    for seed in seeds:
        sample = np.asarray(
            jax.random.poisson(
                jax.random.key(seed), rate,
                shape=(args.n_per_seed,), dtype=jnp.int32
            ),
            dtype=np.float64,
        )
        values.append(sample)
        summaries.append({
            "seed": seed,
            "mean_to_rate": float(sample.mean() / rate),
            "variance_to_rate": float(sample.var(ddof=0) / rate),
            "n": int(sample.size),
        })

    all_samples = np.concatenate(values)
    mean_ratio = float(all_samples.mean() / rate)
    variance_ratio = float(all_samples.var(ddof=0) / rate)
    # With n=300,000, Poisson sample-variance relative SE ≈ sqrt(2/n)=0.0026;
    # 0.08 leaves a much wider guard band and is not a CI.
    if args.expected == "broken_baseline":
        accepted = variance_ratio > 1.15
    else:
        accepted = (
            abs(variance_ratio - 1.0) < 0.08
            and abs(mean_ratio - 1.0) < 0.001
        )
    payload = {
        "source_repository": "jax-ml/jax",
        "source_commit": sha,
        "jax_version": jax.__version__,
        "jax_import_path": str(pathlib.Path(jax.__file__).resolve()),
        "jax_backend": jax.default_backend(),
        "numpy_version": np.__version__,
        "rate_float32": rate,
        "expected_state": args.expected,
        "seeds": summaries,
        "n_total": int(all_samples.size),
        "pooled_mean_to_rate": mean_ratio,
        "pooled_variance_to_rate": variance_ratio,
        "red_green_condition_satisfied": bool(accepted),
        "criterion": "broken variance_ratio>1.15; corrected abs(variance_ratio-1)<0.08 and abs(mean_ratio-1)<0.001",
        "interpretation": "source-exact descriptive stochastic regression; not production-SLA guarantee",
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    output = pathlib.Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not accepted:
        raise SystemExit(
            "red-green criterion failed: observed variance ratio " + str(variance_ratio)
        )


if __name__ == "__main__":
    main()
