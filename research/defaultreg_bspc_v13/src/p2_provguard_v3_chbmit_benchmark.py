#!/usr/bin/env python3
"""Frozen real-EDF CHB-MIT prefix controlled-transform benchmark. Never redistribute EEG.
Evaluates previous baseline AND V3 using identical public EDF controlled pairs.
This is NOT natural clinical duplicate discovery.
"""
import os,sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import p2_external_chbmit_shared_clock as old
import p2_provguard_v3_frozen as v3
OUT=Path(os.environ.get('P2_OUTPUT','p2_v3_chbmit_results'))
OUT.mkdir(parents=True,exist_ok=True)
SOURCES=['chb01/chb01_01.edf','chb02/chb02_01.edf','chb03/chb03_01.edf','chb04/chb04_01.edf','chb05/chb05_01.edf','chb06/chb06_01.edf']
positives=['exact','relabel_permute','global_lag','crop','gain','drop','noise','combined']
negatives=['different_subject','same_file_different_time','one_shared_channel','async_same_source','repeated_constant_artifact']
p=v3.Policy(max_lag=100,min_channels=4,train_threshold=.985,verify_threshold=.97,
            min_overlap=230,verify_fraction=.25,guard_samples=32)
datasets={}
for rel in SOURCES:
    datasets[rel]=old.read_one(rel)
rows=[]
for i,rel in enumerate(SOURCES):
    s,neg_same,meta=datasets[rel]
    diff=datasets[SOURCES[(i+1)%len(SOURCES)]][0]
    for j,kind in enumerate(positives+negatives):
        alt=neg_same if kind=='same_file_different_time' else diff
        target=old.transform(s,alt,kind,old.SEED+i*31+j)
        old_ans=old.three_matchers(s,target)
        cert=v3.match(s,target,p)
        accepted=cert['status']=='NUMERIC_WITNESS'
        verified=v3.verify(s,target,cert,p) if accepted else False
        if accepted and not verified:raise RuntimeError('invalid V3 numerical witness')
        rows.append({'source':rel,'class':'controlled_positive' if kind in positives else 'controlled_negative',
                     'kind':kind,'shared_clock_previous':old_ans['common_clock'],
                     'independent_lag_hungarian':old_ans['independent_lag_assignment'],
                     'v3_witness':int(accepted),'v3_verified':int(verified),
                     'v3_lag':cert.get('lag'),'v3_channels':cert.get('matched_channels',0)})
    print('BENCHMARK_PROGRESS',rel,flush=True)
summary={}
for key in ['shared_clock_previous','independent_lag_hungarian','v3_witness']:
    pos=[r for r in rows if r['class']=='controlled_positive']
    neg=[r for r in rows if r['class']=='controlled_negative']
    summary[key]={'positive_detected':sum(r[key] for r in pos),'positive_total':len(pos),
                  'negative_false_alarms':sum(r[key] for r in neg),'negative_total':len(neg)}
out={'status':'FINISHED','source':'PhysioNet CHB-MIT 1.0.0 public EDF digital prefixes',
     'endpoint':'synchronous multichannel numerical content reuse',
     'policy':p.__dict__,'sources':[{'relpath':k,'prefix_sha256':d[2]['sha256_prefix']} for k,d in datasets.items()],
     'case_count':len(rows),'summary':summary,'cases':rows,
     'limitations':['Only controlled variants on original public EEG','NOT independent natural provenance labels',
                    'No published DTW or robust synchronizer reproduction in this workflow',
                    'No hyperparameter tuning on held-out sources','Significance uses recording as unit not variants']}
(OUT/'P2_CHBMIT_V3_CASES.json').write_text(json.dumps(out,indent=2)+'\n')
print('P2_V3_BENCHMARK_RESULT',json.dumps(summary,sort_keys=True),flush=True)
