#!/usr/bin/env python3
"""P3 real EEG shared-source selector comparison, BNCI2014_004.

Discovery result, not independently registered or replicating 4 old
method-specific negative experiments. 1-8 source -> subject9 reserved
is frozen before any subject9 signal access. 9-fold complete LOPO
is a SEPARATELY flagged exposed exploratory comparison.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/"defaultreg_proper_predictive"))
import original_frozen_runner as old
from run_proper_predictive_posthoc import select_proper_source_nll

SEED=20261013
METHODS=("source_macro","proper_predictive_nll","historical_class_centred","LedoitWolf","OAS","precommitted_lambda_0p1")
def jsonwrite(x,p):
    data=(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def load_people(ids,ds,paradigm):
    c=[];ys=[];groups=[];receipts=[]
    for sid in ids:
        X,labels,meta=paradigm.get_data(dataset=ds,subjects=[sid])
        labels=np.asarray(labels)
        X=np.asarray(X,dtype=np.float64)
        if X.ndim!=3 or X.shape[1]!=3 or not np.isfinite(X).all():
            raise RuntimeError(f"Unexpected BNCI2014_004 3-channel EEG schema for {sid}: {X.shape}")
        y=(labels=="right_hand").astype(np.int64)
        if set(labels.tolist())!={"left_hand","right_hand"} or set(y.tolist())!={0,1}:
            raise RuntimeError(f"BNCI2014_004 bad class labels in {sid}")
        c.append(old.covariances(X));ys.append(y)
        groups.extend([sid]*len(y))
        receipts.append({"subject":sid,"n":len(y),"n_positive":int(y.sum()),"channels":int(X.shape[1])})
        print("P3_BNCI004_RELEASE_INPUT",sid,len(y),flush=True)
    return np.concatenate(c),np.concatenate(ys),np.asarray(groups),receipts

def selectors(C,y,g):
    path=old.source_path(C,y,g)
    allowed=path["feasible_full_lambdas"]
    proper,source_nll=select_proper_source_nll(C,y,g,allowed)
    return {"source_macro":float(path["lambda_source_macro"]),
            "proper_predictive_nll":float(proper),
            "historical_class_centred":float(path["lambda_covariance_full"]),
            "precommitted_lambda_0p1":0.1,
            "source_groups":sorted(set(map(int,g))),
            "source_grid_sha256":hashlib.sha256(json.dumps(
                 path["inner_fold_receipts"],sort_keys=True).encode()).hexdigest(),
            "n_feasible_lambdas":len(allowed),
            "correct_source_train_mean_predictive_is_distinct":True,
            "all_target_outcomes_held_from_selector":True}

def score(Ctr,ytr,Cte,chosen):
    inv=old.reference(Ctr);Xtr=old.tangent(Ctr,inv);Xte=old.tangent(Cte,inv)
    state=old.spectral_state(Xtr,ytr)
    ret={k:old.scalar_scores(state,Xte,chosen[k]) for k in (
        "source_macro","proper_predictive_nll","historical_class_centred","precommitted_lambda_0p1")}
    for k in ("LedoitWolf","OAS"):
        ret[k]=old.analytic_scores(Xtr,ytr,Xte,"LW" if k=="LedoitWolf" else "OAS")[0]
    return ret

def auc_record(scores,y):
    return {k:float(roc_auc_score(y,v)) for k,v in scores.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args();out=args.out
    ds=BNCI2014_004()
    p=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
                   fmin=8,fmax=30,resample=160,baseline=None,
                   reject_by_annotation=False)
    # Source phase: the reserved final person 9 is NEVER read here.
    C8,y8,g8,rec=load_people(list(range(1,9)),ds,p)
    selected9=selectors(C8,y8,g8)
    chosen_file=out/"BEFORE_SUBJECT9_SIGNAL_SOURCE_ONLY_CHOICES.json"
    lock=jsonwrite({"status":"PRE_RESERVED_HELD_9_SOURCE_LOCK",
        "dataset":"BNCI2014_004","source_ids":list(range(1,9)),"held_id":9,
        "fmin":8,"fmax":30,"resample":160,"seed":SEED,
        "source_processing_receipts":rec,"selection":selected9,
        "no_subject9_EEG_loaded_before_lock":True},chosen_file)
    print("P3_RESERVED_ID9_CHOICES_SHA_BEFORE_EEG_READ",lock,flush=True)
    # Only NOW is final held subject9's EEG opened.
    C9,y9,g9,receipt9=load_people([9],ds,p)
    scores9=auc_record(score(C8,y8,C9,selected9),y9)
    strict={"held_id":9,"auc":scores9,"source_lock_sha256":lock,
            "n_target_trials":len(y9),
            "source_only_Z0_at_time_of_selection":True,
            "independent_targets_in_this_locked_experiment":1}
    print("P3_RESERVED_HELD_ID9_RESULT",json.dumps(strict,sort_keys=True),flush=True)
    # Supplementary LOPO: now released ALL trial contents have been read.
    # Not an independent confirmation and NOT globally blinded.
    C=np.concatenate([C8,C9]);y=np.concatenate([y8,y9]);g=np.concatenate([g8,g9])
    selected=[];receipt=[]
    for held in range(1,10):
        tr=g!=held
        z=selected9 if held==9 else selectors(C[tr],y[tr],g[tr])
        selected.append({"held_id":held,"source_selected_parameters":z})
    second_lock=jsonwrite({"dataset":"BNCI2014_004",
        "methods":list(METHODS),"design":"post-reserved-9 exposed retrospective 9-fold LOPO",
        "source_choices":selected,"overlapping_source_training_sets":True,
        "not_9_independent_acquisitions":True,
        "source_choices_not_from_held_labels_per_fold":True},out/"EXPLORATORY_9FOLD_CHOICES_BEFORE_AUC.json")
    for item in selected:
        held=item["held_id"]
        if held==9: vals=scores9
        else:
            tr=g!=held; va=~tr
            vals=auc_record(score(C[tr],y[tr],C[va],item["source_selected_parameters"]),y[va])
        receipt.append({"held_id":held,"n_held":int(np.sum(g==held)), "AUC":vals})
        print("P3_SHARED_9_FOLD_RESULT",held,vals,flush=True)
    differences={}
    for m in METHODS:
        vals=np.asarray([r["AUC"][m] for r in receipt])
        against=np.asarray([r["AUC"]["source_macro"] for r in receipt])
        differences[m]={"macro_AUC":float(vals.mean()),
           "vs_source_macro_mean_delta":float(np.mean(vals-against)),
           "n_material_harm_vs_source_macro":int(np.sum(vals<against-0.02)),
           "n_material_benefit_vs_source_macro":int(np.sum(vals>against+0.02))}
    final={"status":"EXECUTED_BNCI2014_004_SOURCE_ONLY_COMMON_CANDIDATE_MATRIX",
      "reserved_subject9_source_sha":lock,"exploratory_all9_source_lock_sha":second_lock,
      "one_held_subject_primary_not_a_confirmatory_cohort":strict,
      "all9_held_people_EXPLORATORY_not_independent":receipt,
      "summary_descriptive_no_causal_inference":differences,
      "information_budget":"all policies selected from corresponding eight source IDs only",
      "no_new_general_universal_negative_theorem":True,
      "manuscript_gate":"do not call four old negative methods jointly replicated",
      "no_clinical_participant_identity_access":True}
    digest=jsonwrite(final,out/"P3_ACTUAL_BNCI004_SHARED_MATRIX_RESULTS.json")
    print("P3_NEW_REAL_EEG_COMMON_MATRIX_DONE",json.dumps(differences,sort_keys=True),flush=True)
    print("P3_FULL_RESULT_SHA",digest,flush=True)
if __name__=="__main__":main()
