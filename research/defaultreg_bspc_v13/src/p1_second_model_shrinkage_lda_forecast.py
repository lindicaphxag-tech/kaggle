#!/usr/bin/env python3
"""Independent common-covariance LDA score implementation for P1.

BNCI2014_001 was exposed in prior project work. The one held target is
unread until a source-only lock in this execution, but this is NOT a
prospective or external untouched confirmation. Tangent features reuse
the common EEG preprocessing; scoring/NLL are implemented independently.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'defaultreg_proper_predictive'))
import original_frozen_runner as old

GRID=(0.,.01,.05,.1,.25,.5,.75,1.)

def fit(X,y,lam):
    if set(np.unique(y))!={0,1}:raise ValueError('both classes required')
    means=np.stack([X[y==k].mean(0) for k in (0,1)])
    residual=X-means[y]
    pooled=residual.T@residual/len(X)
    scale=np.trace(pooled)/X.shape[1]
    sigma=(1-lam)*pooled+lam*scale*np.eye(X.shape[1])
    sign,logdet=np.linalg.slogdet(sigma)
    if sign<=0:raise ValueError('non-PD source covariance')
    w=np.linalg.solve(sigma,means[1]-means[0])
    return means,sigma,w,logdet

def measure(model,X,y):
    means,sigma,w,logdet=model
    score=(X-(means[0]+means[1])/2)@w
    residual=X-means[y]
    quadratic=np.sum(residual*np.linalg.solve(sigma,residual.T).T,axis=1)
    return float(roc_auc_score(y,score)),float(.5*(logdet+quadratic.mean()))

def source_validation(C,y,g):
    ids=sorted(set(map(int,g)));folds=old._held_groups(ids)
    result={float(l):[] for l in GRID}
    for held in folds:
        v=np.isin(g,held);t=~v
        ref=old.reference(C[t]);Xt=old.tangent(C[t],ref);Xv=old.tangent(C[v],ref)
        for lam in GRID:
            model=fit(Xt,y[t],lam)
            for sid in held:
                mask=g[v]==sid
                auc,nll=measure(model,Xv[mask],y[v][mask])
                result[lam].append((int(sid),auc,nll))
    if any(sorted(s for s,_,_ in v)!=ids for v in result.values()):
        raise AssertionError('one source-fold result per participant required')
    return result,ids

def choice(validation,ids):
    candidates=list(GRID)
    def mean(lam,index,train_ids):
        return float(np.mean([row[index] for row in validation[lam] if row[0] in train_ids]))
    macro=max(candidates,key=lambda l:(mean(l,1,ids),l))
    proper=min(candidates,key=lambda l:(mean(l,2,ids),l))
    oof=[]
    for held in ids:
        train=[i for i in ids if i!=held]
        a=max(candidates,key=lambda l:(mean(l,1,train),l))
        b=min(candidates,key=lambda l:(mean(l,2,train),l))
        oof.append(mean(a,1,[held])-mean(b,1,[held]))
    return {'source_macro':macro,'source_predictive_nll':proper,
            'forecast_macro_minus_predictive_auc':float(np.mean(oof)),
            'n_source_participants':len(ids),'choice_crossfit_by_source_person':True}

def load(ds,paradigm,ids):
    c=[];ys=[];groups=[];receipts=[]
    for sid in ids:
        X,labels,_=paradigm.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X,dtype=float);labels=np.asarray(labels)
        if X.ndim!=3 or X.shape[1]!=3 or set(labels.tolist())!={'left_hand','right_hand'}:
            raise ValueError('BNCI2014_001 expected three-channel schema')
        yy=(labels=='right_hand').astype(int)
        c.append(old.covariances(X));ys.append(yy);groups.extend([sid]*len(yy))
        receipts.append({'id':sid,'trials':len(yy),'right':int(yy.sum())})
        print('P1_LDA_RELEASE_LOADED',sid,len(yy),flush=True)
    return np.concatenate(c),np.concatenate(ys),np.asarray(groups),receipts

def write(path,obj):
    data=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def one_target(C,y,g,target,selected):
    tr=g!=target;te=~tr
    ref=old.reference(C[tr]);Xt=old.tangent(C[tr],ref);Xv=old.tangent(C[te],ref)
    auc={k:measure(fit(Xt,y[tr],selected[k]),Xv,y[te])[0]
         for k in ('source_macro','source_predictive_nll')}
    observed=auc['source_macro']-auc['source_predictive_nll']
    forecast=selected['forecast_macro_minus_predictive_auc']
    return {'target_id':target,'n_target_trials':int(te.sum()),'auc':auc,
            'observed_macro_minus_predictive_auc':observed,
            'source_forecast_macro_minus_predictive_auc':forecast,
            'direction_correct_excluding_ties':bool(forecast*observed>0),
            'absolute_forecast_error_auc':abs(forecast-observed)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();ds=BNCI2014_001()
    paradigm=MotorImagery(n_classes=2,events=['left_hand','right_hand'],
                          channels=['C3','Cz','C4'],fmin=8,fmax=30,resample=160,
                          baseline=None,reject_by_annotation=False)
    C,y,g,rec=load(ds,paradigm,list(range(1,9)))
    source,ids=source_validation(C,y,g);selected=choice(source,ids)
    lock=write(args.out/'P1_LDA_SOURCE_ONLY_BEFORE_ID9.json',
               {'dataset':'BNCI2014_001','source_ids':ids,'held_id':9,'grid':GRID,
                'model':'independent pooled within-class shrinkage LDA',
                'selection':selected,'source_receipts':rec,
                'historically_exposed_acquisition':True,
                'held9_signal_unread_in_this_execution':True})
    print('P1_LDA_BEFORE_ID9_LOCK_SHA',lock,flush=True)
    Ct,yt,gt,_=load(ds,paradigm,[9])
    C=np.concatenate([C,Ct]);y=np.concatenate([y,yt]);g=np.concatenate([g,gt])
    rows=[one_target(C,y,g,9,selected)]
    for target in range(1,9):
        tr=g!=target
        validation,source_ids=source_validation(C[tr],y[tr],g[tr])
        rows.append(one_target(C,y,g,target,choice(validation,source_ids)))
    rows.sort(key=lambda row:row['target_id'])
    nonzero=[r for r in rows if abs(r['observed_macro_minus_predictive_auc'])>1e-12 and
             abs(r['source_forecast_macro_minus_predictive_auc'])>1e-12]
    summary={'direction_correct':sum(r['direction_correct_excluding_ties'] for r in nonzero),
             'direction_evaluable':len(nonzero),
             'mean_absolute_forecast_error_auc':float(np.mean([r['absolute_forecast_error_auc'] for r in rows])),
             'macro_minus_predictive_auc_mean':float(np.mean([r['observed_macro_minus_predictive_auc'] for r in rows]))}
    result={'status':'RETROSPECTIVE_SECOND_SCORER_EXPLORATORY_SOURCE_FORECAST',
            'source_lock_before_id9_sha256':lock,'rows':rows,'summary':summary,
            'held9_source_lock_before_signal_true_only_within_this_run':True,
            'all_subject_outcomes_previously_exposed_projectwide':True,
            'not_an_independent_confirmatory_model_or_acquisition':True}
    sha=write(args.out/'P1_LDA_SECOND_SCORER_RESULTS.json',result)
    print('P1_LDA_SECOND_SCORER_SUMMARY',json.dumps(summary,sort_keys=True),flush=True)
    print('P1_LDA_SECOND_SCORER_SHA',sha,flush=True)
if __name__=='__main__':main()
