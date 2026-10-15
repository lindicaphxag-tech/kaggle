#!/usr/bin/env python3
"""Twelve-source fixed-CAR versus selected-subset residual research extension.
Original CHB-MIT EDF waveforms, controlled transformations; NOT natural
duplicates, new dataset, or prospectively never-touched held-out subjects.
CAR thresholds were frozen on CHB01-02 in run 38677222169.
"""
import os,json,sys,hashlib
from pathlib import Path
from urllib.request import urlopen
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import p2_external_chbmit_shared_clock as io
import p2_provguard_v4_calibrated as baseline
import p2_provguard_v5_artifact_residual as guard
OUT=Path(os.environ.get('P2_OUTPUT','p2_v8_grouped'));OUT.mkdir(parents=True,exist_ok=True)
with urlopen('https://physionet.org/files/chbmit/1.0.0/RECORDS',timeout=75) as r:manifest=r.read()
available=[line.strip() for line in manifest.decode().splitlines() if line.strip().endswith('.edf')]
def first(case):
    candidates=sorted(x for x in available if x.startswith(f'chb{case:02d}/'))
    if not candidates:raise RuntimeError('missing public EDF case')
    return candidates[0]
files=[first(i) for i in range(9,21)]
originals={x:io.read_one(x) for x in files}
policy=baseline.Policy(train_threshold=.93,verify_threshold=.90)
pos=['exact','relabel_permute','global_lag','crop','gain','drop','noise','combined']
neg=['different_subject','same_file_different_time','one_shared_channel','async_same_source',
     'repeated_constant_artifact','shared_global_artifact']
def car(d):
    names=list(d);X=np.stack([np.asarray(d[k],dtype=float) for k in names])
    X-=X.mean(axis=0,keepdims=True)
    return {k:v for k,v in zip(names,X)}
rows=[]
for i,rel in enumerate(files):
    src,same,_=originals[rel]
    other=originals[files[(i+1)%len(files)]][0]
    for j,kind in enumerate(pos+neg):
        if kind=='shared_global_artifact':
            t=np.arange(768);w=np.sin(t*.051)+np.sin(t*.113)
            amp=100*max(np.std(np.stack(list(src.values()))),
                        np.std(np.stack(list(other.values()))))
            a={k:v+amp*w for k,v in src.items()}
            b={k:v+amp*w for k,v in other.items()}
        else:
            a=src;b=io.transform(src,same if kind=='same_file_different_time' else other,
                                 kind,io.SEED+i*31+j)
        old=baseline.match(a,b)['status']=='NUMERIC_WITNESS'
        tuned=baseline.match(car(a),car(b),policy)['status']=='NUMERIC_WITNESS'
        proposed=guard.match(a,b)['status']=='DUAL_WITNESS'
        rows.append({'record':rel,'kind':kind,'positive':kind in pos,
                     'raw':int(old),'tuned_car':int(tuned),'residual':int(proposed)})
    print('V8_RECORD_DONE',rel,flush=True)
summary={}
for key in ['raw','tuned_car','residual']:
    pp=[r for r in rows if r['positive']];nn=[r for r in rows if not r['positive']]
    summary[key]={'tp':sum(x[key] for x in pp),'pos':len(pp),
                  'fp':sum(x[key] for x in nn),'neg':len(nn)}
grouped=[]
for rel in files:
    group=[r for r in rows if r['record']==rel]
    grouped.append({'record':rel,'car_positive':sum(x['tuned_car'] for x in group if x['positive']),
                    'residual_positive':sum(x['residual'] for x in group if x['positive']),
                    'car_negative_fp':sum(x['tuned_car'] for x in group if not x['positive']),
                    'residual_negative_fp':sum(x['residual'] for x in group if not x['positive'])})
result={'status':'CONTROLLED_RESEARCH','manifest_sha256':hashlib.sha256(manifest).hexdigest(),
 'source_digest':{x:originals[x][2]['sha256_prefix'] for x in files},
 'frozen_development':'chb01-02 CAR threshold fit=.93 verify=.90 run 38677222169',
 'n_original_records':len(files),'n_dependent_constructed_cases':len(rows),
 'results':summary,'source_grouped':grouped,'cases':rows,
 'limitations':['same CHB-MIT corpus with previously inspected case IDs 09-20',
  'only controlled digital transformations rather than natural duplicates',
  '12 independent original EDFs are inferential units; constructs are correlated',
  'no published optimally reference-adjusted CAR baseline in this workflow']}
(OUT/'P2_V8_12_SOURCE_FIXED_CAR_CASES.json').write_text(json.dumps(result,indent=2)+'\n')
print('P2_V8_GROUPED_12_SOURCE_RESULT',json.dumps(summary,sort_keys=True),flush=True)
