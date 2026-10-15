#!/usr/bin/env python3
"""Fair tuned-CAR benchmark on six CHB-MIT real digital EDF prefixes.
Development: chb01-02. Evaluation: chb03-06. All data positives / negatives are
CONTROLLED MODIFICATIONS, not natural clinical duplicate recordings.

Fitted-CAR uses source-only development selection over a PREDECLARED threshold grid.
The held sources are excluded from choice in this workflow run, but were previously
exposed project-wide and therefore this is not an independent prospective trial.
"""
import json,os,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import p2_external_chbmit_shared_clock as io
import p2_provguard_v4_calibrated as raw
import p2_provguard_v5_artifact_residual as v5

OUT=Path(os.environ.get('P2_OUTPUT','p2_car_tuned'));OUT.mkdir(parents=True,exist_ok=True)
REL=[f'chb{i:02d}/chb{i:02d}_01.edf' for i in range(1,7)]
POS=['exact','relabel_permute','global_lag','crop','gain','drop','noise','combined']
NEG=['different_subject','same_file_different_time','one_shared_channel',
     'async_same_source','repeated_constant_artifact','shared_global_artifact']
TRAIN_TH=(.90,.93,.95,.97,.985)
VERIFY_TH=(.90,.95,.97)
RULES=[raw.Policy(train_threshold=a,verify_threshold=b) for a in TRAIN_TH for b in VERIFY_TH]
inputs={x:io.read_one(x) for x in REL}
def car(z):
    k=list(z)
    A=np.stack([np.asarray(z[x],float) for x in k])
    A=A-A.mean(axis=0,keepdims=True)
    return {x:a for x,a in zip(k,A)}
cases=[]
for si,rel in enumerate(REL):
    src,same,meta=inputs[rel]
    other=inputs[REL[(si+1)%len(REL)]][0]
    for ci,kind in enumerate(POS+NEG):
        if kind=='shared_global_artifact':
            time=np.arange(768);wave=np.sin(time*.051)+np.sin(time*.113)
            amp=100*max(np.std(np.stack(list(src.values()))),np.std(np.stack(list(other.values()))))
            a={k:v+amp*wave for k,v in src.items()}
            b={k:v+amp*wave for k,v in other.items()}
        else:
            a=src
            b=io.transform(src,same if kind=='same_file_different_time' else other,kind,io.SEED+si*31+ci)
        cases.append(dict(group='development' if si<2 else 'previously_exposed_held',
            original_record=rel,kind=kind,label=int(kind in POS),
            car_src=car(a),car_tgt=car(b),raw_src=a,raw_tgt=b))
rows=[]
for case in cases:
    preds=[int(raw.match(case['car_src'],case['car_tgt'],policy)['status']=='NUMERIC_WITNESS') for policy in RULES]
    original=int(raw.match(case['raw_src'],case['raw_tgt'])['status']=='NUMERIC_WITNESS')
    guarded=int(v5.match(case['raw_src'],case['raw_tgt'])['status']=='DUAL_WITNESS')
    rows.append(dict(group=case['group'],original_record=case['original_record'],kind=case['kind'],
                     label=case['label'],CAR_grid=preds,old_original=original,v5_residual=guarded))
    if len(rows)%14==0:print('P2_TUNED_CAR_SOURCE',case['original_record'],flush=True)
dev=[z for z in rows if z['group']=='development'];held=[z for z in rows if z['group']!='development']
# Prespecified development objective: first zero false positives, then maximum sensitivity.
# If none achieves zero false positives, minimize FP, then maximize sensitivity.
def stat(rows,key):
    pos=[r for r in rows if r['label']];neg=[r for r in rows if not r['label']]
    return {'tp':sum(r[key] for r in pos),'pos':len(pos),
            'fp':sum(r[key] for r in neg),'neg':len(neg)}
devscores=[]
for i,p in enumerate(RULES):
    tp=sum(z['CAR_grid'][i] for z in dev if z['label'])
    fp=sum(z['CAR_grid'][i] for z in dev if not z['label'])
    devscores.append((fp,-tp,-p.train_threshold,-p.verify_threshold,i))
_,_,_,_,chosen=sorted(devscores)[0]
for r in rows:r['tuned_CAR']=r['CAR_grid'][chosen]
out={'dataset':'Six real CHB-MIT EDF prefixes, 84 controlled variants','n_original_sources':6,
     'selection':'thresholds selected on chb01..chb02 only in this execution, held chb03..06 previously exposed projectwide',
     'development_groups':['chb01','chb02'],'held_groups':['chb03','chb04','chb05','chb06'],
     'candidate_policies':[{'fit':p.train_threshold,'verify':p.verify_threshold} for p in RULES],
     'chosen_CAR_policy':{'fit':RULES[chosen].train_threshold,'verify':RULES[chosen].verify_threshold},
     'summary':{'development':{m:stat(dev,m) for m in ('old_original','tuned_CAR','v5_residual')},
                'held':{m:stat(held,m) for m in ('old_original','tuned_CAR','v5_residual')}},
     'cases':[ {k:v for k,v in z.items() if k!='CAR_grid'} for z in rows ],
     'limitations':['method known from full-project prior held evaluations', 'positive and negative examples are experimentally constructed',
      'small 2-record development calibration; no natural clinical duplicate ground truth',
      'CAR grid comparison is exploratory, not multiplicity corrected across model families',
      'later publication-level comparisons should independently benchmark DTW, robust fingerprints, rereferencing variants']}
(OUT/'P2_TUNED_CAR_84_CONTROLLED_CASES.json').write_text(json.dumps(out,indent=2)+'\n')
print('P2_TUNED_CAR_RESULT',json.dumps({'chosen':out['chosen_CAR_policy'],'summary':out['summary']},sort_keys=True),flush=True)
