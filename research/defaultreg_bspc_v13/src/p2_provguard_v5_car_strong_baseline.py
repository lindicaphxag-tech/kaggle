#!/usr/bin/env python3
"""Fair CAR+old baseline vs V5 common-mode residual, CHBMIT six actual EDF prefixes.

Matched subject/group controlled transformations. All methods receive same arrays.
CAR subtracts contemporaneous within-record channel mean, no leakage from target labels.
Shared-global-artifact constructed negatives, NOT clinical noise. No new superiority claims.
"""
from pathlib import Path
import json,os,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import p2_external_chbmit_shared_clock as source_reader
import p2_provguard_v4_calibrated as old
import p2_provguard_v5_artifact_residual as new
OUT=Path(os.environ.get('P2_OUTPUT','p2_v5_car_baseline'));OUT.mkdir(parents=True,exist_ok=True)
files=[f'chb{i:02d}/chb{i:02d}_01.edf' for i in range(1,7)]
records={p:source_reader.read_one(p) for p in files}
pos=['exact','relabel_permute','global_lag','crop','gain','drop','noise','combined']
neg=['different_subject','same_file_different_time','one_shared_channel','async_same_source','repeated_constant_artifact','shared_global_artifact']
def car(d):
    names=list(d);lengths={len(d[k]) for k in names}
    if len(lengths)!=1:raise ValueError('CAR needs same length among channels')
    M=np.stack([np.asarray(d[k],float) for k in names])
    M=M-M.mean(axis=0,keepdims=True)
    return {k:v for k,v in zip(names,M)}
rows=[]
for subject,path in enumerate(files):
    src,same,_=records[path]
    other=records[files[(subject+1)%len(files)]][0]
    for idx,kind in enumerate(pos+neg):
        if kind=='shared_global_artifact':
            q=np.arange(768);artifact=np.sin(q*.051)+np.sin(q*.113)
            amplitude=100*max(np.std(np.stack(list(src.values()))),np.std(np.stack(list(other.values()))))
            a={k:v+amplitude*artifact for k,v in src.items()}
            b={k:v+amplitude*artifact for k,v in other.items()}
        else:
            a=src
            b=source_reader.transform(src,same if kind=='same_file_different_time' else other,kind,source_reader.SEED+subject*31+idx)
        ans={
            'raw_shared_clock':old.match(a,b)['status']=='NUMERIC_WITNESS',
            'car_then_shared_clock':old.match(car(a),car(b))['status']=='NUMERIC_WITNESS',
            'dual_residual':new.match(a,b)['status']=='DUAL_WITNESS'}
        rows.append({'source':path,'condition':kind,'label':int(kind in pos),**{k:int(v) for k,v in ans.items()}})
    print('CAR_BASELINE_SOURCE_DONE',path,flush=True)
summary={}
for method in ('raw_shared_clock','car_then_shared_clock','dual_residual'):
    summary[method]={'positive_hits':sum(x[method] for x in rows if x['label']),
       'positive_total':sum(x['label'] for x in rows),
       'negative_false_alarms':sum(x[method] for x in rows if not x['label']),
       'negative_total':sum(not x['label'] for x in rows),
       'shared_artifact_false_alarms':sum(x[method] for x in rows if x['condition']=='shared_global_artifact')}
obj={'scope':'CHB-MIT real digital EDF prefixes, 6 sources x 14 controlled conditions; no natural duplicate labels',
     'summary':summary,'cases':rows,
     'limitations':['CAR is a strong informed nuisance-removal baseline; V5 superiority must be established against it',
        '6 original source units not 84 independent observations','any-channel content reuse endpoint has different labels']}
(OUT/'P2_V5_CAR_STRONG_BASELINE_84.json').write_text(json.dumps(obj,indent=2)+'\n')
print('P2_CAR_STRONG_BASELINE_RESULT',json.dumps(summary),flush=True)
