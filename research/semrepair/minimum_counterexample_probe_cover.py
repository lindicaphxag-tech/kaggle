"""Exact minimum-cost distinguishing tests for a FINITE source-patch grammar.

Source-backed baseline: symmetric action bounds admit several incorrect
ManiSkill controller rotation patches. This tool finds the least-cost tests
that reject every known incorrect training-plausible patch, with a dynamic
programming proof and explicit refusal if the probe budget is insufficient.

This is standard finite weighted SET COVER, not new mathematics or a proof
of correctness for programs outside the enumerated candidate catalog.
Worst-case O(|bank| * 2^|wrong_candidates|); deliberate cap at 18 hypotheses.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class Probe:
    name:str
    integer_cost:int
    excludes:tuple[str,...]


def exact_min_cost_counterexamples(
    wrong_candidates:Sequence[str],probes:Sequence[Probe],
    *,max_budget:int|None=None,max_candidates:int=18
)->dict:
    ids=tuple(wrong_candidates)
    if not ids or len(ids)>max_candidates or len(set(ids))!=len(ids):
        raise ValueError("nonempty unique bounded incorrect-candidate set required")
    if not probes or len(probes)>128 or len({p.name for p in probes})!=len(probes):
        raise ValueError("nonempty unique bounded probe bank required")
    if max_budget is not None and (type(max_budget) is not int or max_budget<0):
        raise ValueError("nonnegative integer probe budget required")
    idmap={name:i for i,name in enumerate(ids)}
    full=(1<<len(ids))-1
    bank=[]
    for p in probes:
        if not isinstance(p,Probe) or not p.name or \
           type(p.integer_cost) is not int or p.integer_cost<=0:
            raise ValueError("positive integer-priced probes required")
        if len(set(p.excludes))!=len(p.excludes) or \
           any(item not in idmap for item in p.excludes):
            raise ValueError("unknown or duplicate excluded hypothesis")
        mask=sum(1<<idmap[x] for x in p.excludes)
        bank.append((p.name,p.integer_cost,mask))
    dp:dict[int,tuple[int,tuple[str,...]]]={0:(0,())}
    transitions=0
    for name,cost,mask in sorted(bank):
        for covered,(price,chosen) in list(dp.items()):
            transitions+=1
            nxt=covered|mask
            entry=(price+cost,tuple(sorted((*chosen,name))))
            old=dp.get(nxt)
            if old is None or (entry[0],len(entry[1]),entry[1])< \
               (old[0],len(old[1]),old[1]):
                dp[nxt]=entry
    best=dp.get(full)
    if best is None:
        return {
            "decision":"REFUSE_NO_COMPLETE_DISTINGUISHING_PROBE_SET",
            "selected_probes":None,"provable_minimum_cost":None,
            "candidate_count":len(ids),"bank_size":len(bank),
            "dp_reachable_states":len(dp),"dp_transitions":transitions,
            "certificate_complete_against_catalog":False,
        }
    feasible=max_budget is None or best[0]<=max_budget
    chosen=best[1] if feasible else None
    byname={p.name:set(p.excludes) for p in probes}
    verified=chosen is not None and \
       set().union(*(byname[n] for n in chosen))==set(ids)
    if feasible and not verified:
        raise AssertionError("cover verifier disagrees with DP")
    return {
        "decision":("MINIMUM_COST_COUNTEREXAMPLE_SUITE_CERTIFIED"
                    if feasible else "REFUSE_INSUFFICIENT_PROBE_BUDGET"),
        "selected_probes":list(chosen) if chosen is not None else None,
        "provable_minimum_cost":best[0],
        "candidate_count":len(ids),"bank_size":len(bank),
        "dp_reachable_states":len(dp),"dp_transitions":transitions,
        "certificate_complete_against_catalog":bool(verified),
        "scope":"only finite declared source patch candidates and assumed test oracle",
        "all_possible_programs_certified":False,
    }


def baseline_maniskill_grammar_probe_suite():
    """Standalone independent ANALYTIC test; real vendor code tested separately."""
    wrong=("negate_lower","absolute_lower","hardcode_training_upper",
           "half_span","average_positive_gains")
    bank=[]
    for i,(lo,hi) in enumerate(((-.1,.1),(-.08,.16),(-.15,.03),
                                (-.1,.1),(-.06,.18),(-.125,.21))):
        outputs={
            "negate_lower":-lo,"absolute_lower":abs(lo),
            "hardcode_training_upper":.1,
            "half_span":(hi-lo)/2,
            "average_positive_gains":(hi+abs(lo))/2,
        }
        eliminated=tuple(k for k in wrong if abs(outputs[k]-hi)>1e-9)
        bank.append(Probe(f"cfg_{i}",1 if i==1 else 2,eliminated))
    result=exact_min_cost_counterexamples(wrong,bank)
    result["ground_truth_source"]="declared positive rot_upper controller contract"
    result["source_code_executed_in_this_stdlib_only_test"]=False
    result["probes"]=[{"name":p.name,"cost":p.integer_cost,
                       "excluded":list(p.excludes)} for p in bank]
    return result


if __name__=="__main__":
    import json
    print(json.dumps(baseline_maniskill_grammar_probe_suite(),
                     indent=2,sort_keys=True))
