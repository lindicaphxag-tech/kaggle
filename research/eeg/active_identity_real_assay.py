"""Real EEG closed-loop simulation of minimax external electrode provenance queries.

The test-time oracle knows the *injected permutation* and reveals physical
identity only when queried. No class label or true mapping is accessible during
query planning. Every candidate identity mapping uses the SAME frozen spatial
classifier. No real electrode hardware faults or pretrained EEG model here.
"""
import argparse
import json
from itertools import product, permutations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from .filterbank_logeuclidean_real import (
    load_trials, filter_bank_log_covariance, SPLITS, CS,
)
from .active_provenance_queries import DecisionTableQueryPlanner

PAIRS=(("C1","C3"),("C2","C4"))
SEED=314159

def allowed_assignments(n, groups):
    result=[]
    for settings in product(*(permutations(g) for g in groups)):
        row_to_identity=list(range(n))
        for group,perm in zip(groups,settings):
            for row,identity in zip(group,perm):
                row_to_identity[row]=identity
        result.append(tuple(row_to_identity))
    return tuple(result)

def evaluate_queries(X_test, y_test, channel_names, frozen_model, *, seed=SEED):
    x=np.asarray(X_test,dtype=np.float64)
    y=np.asarray(y_test,dtype=int)
    by_name={str(ch).upper():i for i,ch in enumerate(channel_names)}
    if any(a not in by_name or b not in by_name for a,b in PAIRS):
        raise ValueError("Missing predeclared physical electrodes")
    groups=tuple((by_name[a],by_name[b]) for a,b in PAIRS)
    maps=allowed_assignments(len(channel_names),groups)
    rng=np.random.default_rng(seed)
    observed=x.copy()
    actual_maps=[]
    for k in range(len(x)):
        p=list(range(len(channel_names)))
        for group in groups:
            if bool(rng.integers(0,2)):
                p[group[0]],p[group[1]]=p[group[1]],p[group[0]]
        observed[k]=x[k,p,:]
        actual_maps.append(tuple(p))
    # Query selection must not access these actual_maps, only logits of
    # all supported identity hypotheses and trusted answers to chosen queries.
    logits=[]
    for mapping in maps:
        recovered=observed[:,np.argsort(mapping),:]
        logits.append(frozen_model.decision_function(
            filter_bank_log_covariance(recovered)))
    logits=np.stack(logits,axis=1)
    oracle=frozen_model.decision_function(filter_bank_log_covariance(x))
    naive=frozen_model.decision_function(filter_bank_log_covariance(observed))
    rows=[]
    for i in range(len(y)):
        planner=DecisionTableQueryPlanner(maps,logits[i],groups)
        initial=planner.propose()
        completed,asked=planner.interact(actual_maps[i])
        correct=int(oracle[i]>0)
        if not completed.authorized or completed.label!=correct:
            raise RuntimeError("Accepted identity decision differs from known original")
        # Strong fair random comparator: at each iteration randomize ONLY
        # among still-informative physical identity queries. A row whose
        # identity is inferred from prior answers cannot incur query cost.
        seen=[]
        random_done=planner.propose()
        while not random_done.authorized:
            consistent=[p for p in planner.mappings
                        if all(p[row]==identity for row,identity in seen)]
            informative=[row for row in planner.rows
                         if len({p[row] for p in consistent})>1]
            if not informative:
                raise RuntimeError("Ambiguous prediction with no informative ID query")
            row=int(rng.choice(informative))
            seen.append((row,actual_maps[i][row]))
            random_done=planner.propose(seen)
        if not random_done.authorized or random_done.label!=correct:
            raise RuntimeError("Random oracle query failed to resolve true mapping")
        rows.append({
            "task_label":int(y[i]),
            "known_identity_prediction":correct,
            "naive_corrupted_prediction":int(naive[i]>0),
            "active_prediction":int(completed.label),
            "initial_authorized":initial.authorized,
            "minimax_worst_case_query_budget":initial.worst_case_queries,
            "active_queries":len(asked),
            "random_queries":len(seen),
            "revealed_row_indices":[int(q[0]) for q in asked],
        })
    original=np.array([r["known_identity_prediction"] for r in rows])
    restored=np.array([r["active_prediction"] for r in rows])
    naive_predictions=np.array([r["naive_corrupted_prediction"] for r in rows])
    return {
        "n_test":len(y),
        "initial_coverage":float(np.mean([r["initial_authorized"] for r in rows])),
        "mean_active_queries":float(np.mean([r["active_queries"] for r in rows])),
        "mean_random_queries":float(np.mean([r["random_queries"] for r in rows])),
        "max_active_queries":int(max(r["active_queries"] for r in rows)),
        "invariance_violations":int(np.sum(original!=restored)),
        "known_identity_bacc":float(balanced_accuracy_score(y,original)),
        "naive_corrupted_bacc":float(balanced_accuracy_score(y,naive_predictions)),
        "post_query_bacc":float(balanced_accuracy_score(y,restored)),
        "per_trial":rows,
        "scope":"Trust-assisted simulated identity queries on existing EEG, not real hardware verification",
        "limitations":["Prediction invariance is not true task correctness",
                       "Only two predeclared channel pairs are uncertain",
                       "A generic finite-set minmax decision tree is not a new theorem",
                       "No pretrained NeuroRVQ or EEGCLIP checkpoint used"],
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",choices=SPLITS,default="mini")
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.data_dir.mkdir(parents=True,exist_ok=True)
    split=SPLITS[args.cohort]
    X,y,subject,names,hashes=load_trials(
        split["train"]+split["val"]+split["test"],args.data_dir)
    train=np.isin(subject,split["train"])
    valid=np.isin(subject,split["val"])
    test=np.isin(subject,split["test"])
    tfeat=filter_bank_log_covariance(X[train])
    vfeat=filter_bank_log_covariance(X[valid])
    best=None
    for C in CS:
        m=make_pipeline(StandardScaler(),LogisticRegression(
            C=C,max_iter=1000,random_state=SEED))
        m.fit(tfeat,y[train])
        acc=float(balanced_accuracy_score(y[valid],m.predict(vfeat)))
        if best is None or (acc,-C)>(best[0],-best[1]):
            best=(acc,C)
    model=make_pipeline(StandardScaler(),LogisticRegression(
        C=best[1],max_iter=1000,random_state=SEED))
    model.fit(tfeat,y[train])
    result=evaluate_queries(X[test],y[test],names,model)
    result.update({"cohort":args.cohort,
                   "selected_C_using_validation_only":best[1],
                   "validation_bacc":best[0],
                   "split_subjects":{k:list(v) for k,v in split.items()},
                   "EDF_sha256":hashes})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print("ACTIVE_EEG_TRUSTED_QUERY_REAL",json.dumps({k:result[k] for k in (
        "n_test","initial_coverage","mean_active_queries","mean_random_queries",
        "invariance_violations","known_identity_bacc","naive_corrupted_bacc",
        "post_query_bacc")}))

if __name__=="__main__":
    main()
