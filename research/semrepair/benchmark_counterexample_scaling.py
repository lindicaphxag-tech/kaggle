"""CPU-only reproducible finite catalog counterexample cover scaling benchmark.

Timing is MACHINE SPECIFIC and uses SYNTHETIC random incidence, not actual
vendor source mutations. Never extrapolate to arbitrary Python/robot source.
Limits intentionally <=18 wrong hypotheses; costs and candidate catalogs
are external assumptions, and 2^n worst-case search remains exponential.
"""
from __future__ import annotations
import json
import random
import statistics
import time

from .minimum_counterexample_probe_cover import (
    Probe,exact_min_cost_counterexamples
)


def benchmark_scaling():
    rows=[]
    for n in (5,8,12,16,18):
        rng=random.Random(20261010+n)
        ids=tuple(f"wrong_{i}" for i in range(n))
        bank=[]
        for i in range(n*2):
            covered=tuple(x for x in ids if rng.random()<.32)
            bank.append(Probe(f"probe_{i:02d}",
                              rng.choice([1,2,3]),covered))
        bank.append(Probe("guaranteed_complete",n+3,ids))
        times=[]
        certificates=[]
        for _ in range(3):
            t=time.perf_counter()
            proof=exact_min_cost_counterexamples(ids,bank)
            times.append((time.perf_counter()-t)*1000.)
            certificates.append(proof)
        if not all(x["decision"]=="MINIMUM_COST_COUNTEREXAMPLE_SUITE_CERTIFIED"
                   for x in certificates):
            raise AssertionError("synthetic full-coverage bank failed optimization")
        if len({x["provable_minimum_cost"] for x in certificates})!=1:
            raise AssertionError("exact optimizer is nondeterministic")
        rows.append({
            "incorrect_candidate_count":n,
            "synthetic_probe_bank_size":len(bank),
            "exact_minimum_declared_cost":proof["provable_minimum_cost"],
            "selected_probe_count":len(proof["selected_probes"]),
            "reachable_bitmask_DP_states":proof["dp_reachable_states"],
            "bitmask_DP_transitions":proof["dp_transitions"],
            "median_wall_milliseconds_three_runs":round(statistics.median(times),3),
            "complete_witness_checked":proof["certificate_complete_against_catalog"],
        })
    return {
        "protocol":"finite-source-program-test-cover-machine-specific-CPU-scale-v1",
        "single_machine_walltime_is_not_reproducible_across_processors":True,
        "original_upstream_vendor_source_evaluations_in_this_benchmark":0,
        "synthetic_independent_incidence_random_seed_rule":"20261010 + n",
        "scope":"finite exactly optimizable wrong-patch set at most 18 candidates, not industrial APR scalability",
        "rows":rows,
    }


if __name__=="__main__":
    print(json.dumps(benchmark_scaling(),indent=2,sort_keys=True))
