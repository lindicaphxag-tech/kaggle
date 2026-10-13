#!/usr/bin/env python3
"""Retrospective common-source comparison of four previously distinct policies.

BNCI2014_004 ID9 has been exposed by earlier six-policy work; this is
explicitly a harmonized reanalysis, not a new blinded external validation.
All selection uses the same IDs 1..8 and identical outer source folds.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research'/'defaultreg_proper_predictive'))
import original_frozen_runner as old
from source_contrast_envelope_v31 import source_envelope,select_source_envelope
from noise_debiased_subject_operator_v32 import group_cv,fit as fit_aniso,score as score_aniso
from source_group_invariant_lda import source_group_cv,fit_group_state,predict as predict_group
from certified_continuous_selector_v23 import certified_source_auc_search,source_person_auc_at_lambda


def load(p,ds,ids):
    c=[];y=[];g=[];receipt=[]
    for sid in ids:
        X,labels,_=p.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X);labels=np.asarray(labels)
        if X.ndim!=3 or X.shape[1]!=3 or set(labels.tolist())!={'left_hand','right_hand'}:
            raise RuntimeError('data schema mismatch')
        yy=(labels=='right_hand').astype(int)
        c.append(old.covariances(X));y.append(yy);g.extend([sid]*len(yy))
        receipt.append({'id':sid,'trials':len(yy),'right':int(yy.sum())})
    return np.concatenate(c),np.concatenate(y),np.asarray(g),receipt


def source_models(C,y,g):
    models=[]
    for held in old._held_groups(sorted(set(map(int,g)))):
        va=np.isin(g,held);tr=~va
        inv=old.reference(C[tr]);state=old.spectral_state(old.tangent(C[tr],inv),y[tr])
        Xt=old.tangent(C[va],inv)
        for sid in held:
            mask=g[va]==sid
            models.append((state,Xt[mask],y[va][mask]))
    return models


def select(C,y,g):
    path=old.source_path(C,y,g)
    grid=[float(x) for x in path['feasible_full_lambdas']]
    baseline=float(path['lambda_source_macro'])
    models=source_models(C,y,g)
    expected=path['criteria_by_lambda'][str(baseline)]['source_macro_auc']
    observed=np.mean([source_person_auc_at_lambda(m,baseline) for m in models])
    if abs(expected-observed)>1e-10: raise AssertionError('source-fold comparator mismatch')
    cont=certified_source_auc_search(models,low=0.,high=1.,tolerance=.02,
                                     max_splits=5,segments=1,initial_lambdas=grid)
    inv=old.reference(C);features=old.tangent(C,inv)
    sce=select_source_envelope(source_envelope(features,y,g),grid,radius=1.)
    aniso=group_cv(C,y,g,grid,(0.,.5,2.),beta=.25)
    group=source_group_cv(C,y,g,grid)
    if abs(aniso['baseline']['lambda']-baseline)>1e-12 or abs(group['decisions']['original']['lambda']-baseline)>1e-12:
        raise AssertionError('source macro baseline not exactly reproduced')
    return {'source_macro':baseline,
            'continuous':float(cont['selected_lambda_source_macro']),
            'continuous_source_certificate_met':cont['source_optimality_certificate_met'],
            'continuous_source_gap_upper':cont['residual_source_optimality_gap_upper_bound'],
            'SCE':float(sce['selected_lambda']),
            'anisotropic':aniso['risk_best'],
            'group_centered':group['decisions']['gid'],
            'source_macro_auc':float(expected),
            'source_group_centered_forecast_positive':group['decisions']['gid']['source_only_forecast_positive']}


def evaluate(C,y,g,Cte,yte,choices):
    inv=old.reference(C);X=old.tangent(C,inv);Z=old.tangent(Cte,inv)
    st=old.spectral_state(X,y)
    iso=fit_aniso(X,y,g)
    gid=fit_group_state(X,y,g)
    scores={'source_macro':old.scalar_scores(st,Z,choices['source_macro']),
            'continuous':old.scalar_scores(st,Z,choices['continuous']),
            'SCE':old.scalar_scores(st,Z,choices['SCE']),
            'anisotropic':score_aniso(iso,Z,choices['anisotropic']['lambda'],choices['anisotropic']['alpha']),
            'group_centered':predict_group(gid,Z,choices['group_centered']['lambda'],'gid')}
    assert np.allclose(old.scalar_scores(st,Z,choices['source_macro']),
                       predict_group(gid,Z,choices['source_macro'],'original'),rtol=1e-8,atol=1e-8)
    return {k:float(roc_auc_score(yte,v)) for k,v in scores.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();ds=BNCI2014_004()
    p=MotorImagery(n_classes=2,events=['left_hand','right_hand'],fmin=8,fmax=30,
                   resample=160,baseline=None,reject_by_annotation=False)
    C,y,g,rec=load(p,ds,list(range(1,9)))
    chosen=select(C,y,g)
    lock={'status':'RETROSPECTIVE_SOURCE_ONLY_REANALYSIS_TARGET_ALREADY_EXPOSED',
          'source_ids':list(range(1,9)),'held_id':9,'method_hyperparameters':
          {'SCE_radius':1.,'anisotropic_alphas':[0.,.5,2.],
           'anisotropic_beta':.25,'continuous_tolerance':.02,'continuous_max_splits':5},
          'source_receipt':rec,'chosen':chosen,'no_subject9_read_in_this_process_yet':True}
    args.out.mkdir(parents=True,exist_ok=True)
    frozen=args.out/'P3_FOUR_OPERATORS_SOURCE_LOCK.json'
    data=(json.dumps(lock,sort_keys=True,indent=2)+'\n').encode();frozen.write_bytes(data)
    digest=hashlib.sha256(data).hexdigest()
    print('P3_FOUR_OPERATORS_SOURCE_LOCK_SHA',digest,flush=True)
    Cte,yte,_,target_rec=load(p,ds,[9]);auc=evaluate(C,y,g,Cte,yte,chosen)
    result={'status':'EXECUTED_RETROSPECTIVE_FOUR_OPERATOR_COMMON_SOURCE_BNCI004',
            'cannot_call_id9_independently_blinded':True,
            'one_held_person_only':True,'source_lock_sha256':digest,
            'target_receipt':target_rec,'target_auc':auc,
            'paired_delta_vs_source_macro':{k:v-auc['source_macro'] for k,v in auc.items()},
            'no_population_superiority_or_null_established':True}
    (args.out/'P3_FOUR_OPERATORS_RESULTS.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('P3_FOUR_OPERATORS_COMMON_SOURCE_RESULT',json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__':main()
