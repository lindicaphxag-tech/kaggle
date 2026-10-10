"""PCPR robust adaptive experiment selection on a finite semantic-hypothesis graph.

A hypothesis predicts deterministic finite outcomes for each permitted native
probe, and has a reviewer decision label. The planner selects an experiment
adaptively, minimizing WORST-CASE acquisition cost until all surviving
hypotheses agree on the label. No execution yields a whole-patch proof.
This is a bounded planning operator; finite-hypothesis completeness is not
established from real software source.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence
import math

@dataclass(frozen=True)
class Experiment:
    name: str
    cost: int
    outcomes: tuple[str, ...]
    evidence_tier: str="native_operator"

@dataclass(frozen=True)
class Plan:
    upper_cost: int | None
    next_experiment: str | None
    hypothesis_count: int
    terminal_label: str | None

def _groups(hypotheses: tuple[int,...], experiment: Experiment):
    groups={}
    for h in hypotheses: groups.setdefault(experiment.outcomes[h],[]).append(h)
    return {k:tuple(v) for k,v in groups.items()}

def minimax_plan(labels:Sequence[str], experiments:Sequence[Experiment], max_cost:int|None=None):
    """Globally optimal deterministic minimax strategy for a fixed finite family.

    Unknown/unresolved equivalence classes refuse with cost=None. This is
    predictive finite-model cost, NOT observed PR success or a correctness
    certificate; no future information is used in planning.
    """
    labels=tuple(labels);exp=tuple(experiments)
    n=len(labels)
    if not n: raise ValueError("No hypotheses")
    if len({e.name for e in exp})!=len(exp):raise ValueError("Duplicate experiment")
    if any(e.cost<=0 or len(e.outcomes)!=n for e in exp):
        raise ValueError("Invalid nonpositive cost or prediction length")
    by_name={e.name:e for e in exp}
    @lru_cache(None)
    def solve(hs):
        ls={labels[h] for h in hs}
        if len(ls)==1:return Plan(0,None,len(hs),next(iter(ls)))
        solutions=[]
        for e in exp:
            gr=_groups(hs,e)
            if len(gr)<=1:continue
            parts=[solve(z) for z in gr.values()]
            if any(p.upper_cost is None for p in parts):continue
            cost=e.cost+max(p.upper_cost for p in parts)
            if max_cost is not None and cost>max_cost:continue
            solutions.append((cost,e.name))
        if not solutions:return Plan(None,None,len(hs),None)
        best_cost,best_name=min(solutions,key=lambda x:(x[0],x[1]))
        return Plan(best_cost,best_name,len(hs),None)
    return solve,by_name

def follow(labels, experiments, hypothesis:int):
    """Replay one latent-hypothesis trace against the fixed policy."""
    solve,by_name=minimax_plan(labels,experiments)
    hs=tuple(range(len(labels)));spent=0;trace=[]
    while True:
        p=solve(hs)
        if p.next_experiment is None:
            return {"cost":spent,"terminal_label":p.terminal_label,
                    "refused":p.upper_cost is None,"trace":trace,"remaining":len(hs)}
        e=by_name[p.next_experiment]
        answer=e.outcomes[hypothesis]
        trace.append({"experiment":e.name,"observed":answer,"cost":e.cost,
                      "hypotheses_before":len(hs)})
        spent+=e.cost
        hs=_groups(hs,e)[answer]

def fixed_order_cost(labels, experiments, hypothesis, permutation=None):
    """Fair nonadaptive test order, early-stopping on decision-label agreement."""
    exp=list(experiments)
    if permutation is not None:exp=[exp[i] for i in permutation]
    hs=tuple(range(len(labels)));spent=0
    for e in exp:
        if len({labels[h] for h in hs})==1:
            return {"cost":spent,"refused":False}
        gr=_groups(hs,e)
        if len(gr)<=1:continue
        hs=gr[e.outcomes[hypothesis]];spent+=e.cost
    return {"cost":spent,"refused":len({labels[h] for h in hs})>1}

def compare(labels,experiments):
    """No p-values here: a deterministic comparison, no independent PRs."""
    p=[follow(labels,experiments,i) for i in range(len(labels))]
    b=[fixed_order_cost(labels,experiments,i) for i in range(len(labels))]
    return {"n_hypotheses":len(labels),
            "minimax_worst_cost":max(x["cost"] for x in p),
            "fixed_order_worst_cost":max(x["cost"] for x in b),
            "minimax_mean_cost":sum(x["cost"] for x in p)/len(p),
            "fixed_order_mean_cost":sum(x["cost"] for x in b)/len(b),
            "minimax_refusals":sum(x["refused"] for x in p),
            "fixed_order_refusals":sum(x["refused"] for x in b)}

def generate_finite_world(seed:int,n:int=10,m:int=7):
    """Deterministic synthetic correlated effects; no result-dependent tuning."""
    import random
    rng=random.Random(seed)
    labels=tuple(["conforming"]*(n//2)+["incomplete"]*(n-n//2))
    exp=[]
    for j in range(m):
        # Give each test a plausible cheap/expensive native cost.
        outcomes=tuple(str(rng.randrange(2+(j%3==0))) for _ in range(n))
        exp.append(Experiment(f"probe_{j}",1+(j*3)%6,outcomes))
    # Final sound discriminant of label, costly.
    exp.append(Experiment("manual_adjudication",15,labels,"human_full_source"))
    return labels,exp

if __name__=="__main__":
    import json
    results=[compare(*generate_finite_world(k)) for k in range(48)]
    d={"status":"SYNTHETIC_FINITE_HYPOTHESIS_SIMULATION_ONLY",
       "n_worlds":len(results),
       "mean_minimax_worst_cost":sum(x["minimax_worst_cost"] for x in results)/len(results),
       "mean_fixed_order_worst_cost":sum(x["fixed_order_worst_cost"] for x in results)/len(results),
       "worlds_minimax_strictly_cheaper":sum(x["minimax_worst_cost"]<x["fixed_order_worst_cost"] for x in results),
       "worlds_minimax_tied":sum(x["minimax_worst_cost"]==x["fixed_order_worst_cost"] for x in results),
       "worlds_minimax_costlier":sum(x["minimax_worst_cost"]>x["fixed_order_worst_cost"] for x in results),
       "independent_real_repos":0,"claims_of_full_correctness":0}
    with open("pcpr_minimax_synthetic.json","w") as f:json.dump({"summary":d,"rows":results},f,indent=2)
    print(json.dumps(d,indent=2))
