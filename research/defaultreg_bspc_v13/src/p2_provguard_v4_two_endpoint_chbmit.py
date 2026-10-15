#!/usr/bin/env python3
"""Two distinct ProvGuard V4 endpoints on six original CHB-MIT EDF digital prefixes.
REAL source waveforms; experimentally transformed targets; NOT natural duplicates.
"""
import sys,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import p2_external_chbmit_shared_clock as old
import p2_provguard_v4_calibrated as v4
OUTPUT=Path(os.environ.get('P2_OUTPUT','p2_v4_two_endpoints'));OUTPUT.mkdir(parents=True,exist_ok=True)
files=['chb01/chb01_01.edf','chb02/chb02_01.edf','chb03/chb03_01.edf',
       'chb04/chb04_01.edf','chb05/chb05_01.edf','chb06/chb06_01.edf']
data={rel:old.read_one(rel) for rel in files}
pos=['exact','relabel_permute','global_lag','crop','gain','drop','noise','combined']
neg=['different_subject','same_file_different_time','one_shared_channel',
     'async_same_source','repeated_constant_artifact']
p=v4.Policy()
rows=[]
for i,rel in enumerate(files):
    s,same,_=data[rel]; diff=data[files[(i+1)%len(files)]][0]
    for j,kind in enumerate(pos+neg):
        other=same if kind=='same_file_different_time' else diff
        target=old.transform(s,other,kind,old.SEED+i*31+j)
        coherent=v4.match(s,target,p)
        content=v4.any_content_witness(s,target,p)
        w_coh=coherent['status']=='NUMERIC_WITNESS'
        w_any=content['status']=='NUMERIC_WITNESS'
        if w_coh and not v4.verify(s,target,coherent,p):raise ValueError('invalid coherent witness')
        if w_any and not v4.verify_any_content(s,target,content,p):raise ValueError('invalid single-channel witness')
        ground_coh=kind in pos
        ground_any=kind in pos or kind in ['one_shared_channel','async_same_source']
        rows.append({'record':rel,'kind':kind,
          'ground_truth_coherent':ground_coh,'ground_truth_any_content':ground_any,
          'predicted_coherent':w_coh,'predicted_any_content':w_any,
          'coherent_lag':coherent.get('lag'),'any_content_lag':content.get('lag')})
    print('DONE',rel,flush=True)
summary={}
for endpoint in ('coherent','any_content'):
    y='ground_truth_'+endpoint;t='predicted_'+endpoint
    summary[endpoint]={
      'hits':sum(r[y] and r[t] for r in rows),
      'positives':sum(r[y] for r in rows),
      'false_alarms':sum(not r[y] and r[t] for r in rows),
      'negatives':sum(not r[y] for r in rows)}
result={'status':'FINISHED','dataset':'CHB-MIT 1.0.0 real digital EDF prefixes, controlled variants',
       'thresholds':p.__dict__,'data_prefix_sha256':{x:data[x][2]['sha256_prefix'] for x in files},
       'warning':'Labels differ by endpoint. No physical patient identity proof, no natural duplicate ground truth.',
       'summary':summary,'cases':rows}
(OUTPUT/'P2_V4_TWO_ENDPOINTS.json').write_text(json.dumps(result,indent=2)+'\n')
print('TWO_ENDPOINT_RESULTS',json.dumps(summary),flush=True)
