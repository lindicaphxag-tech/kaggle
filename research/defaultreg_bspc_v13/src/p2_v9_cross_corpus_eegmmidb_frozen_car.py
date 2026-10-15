#!/usr/bin/env python3
"""Cross-corpus physically independent public EDF test: PhysioNet EEGMMIDB 1.0.0.
Labels/tasks are ignored: this experiment concerns DIGITAL SIGNAL CONTENT, not MI accuracy.
Uses six source EDFs S001..S006 R04, each real EEG digital waveform plus controlled variants.
The method/policies are frozen from CHB-MIT work prior to these per-file measurements.
This is cross-release evidence, not natural duplicate ground-truth or subject-blind new task.
"""
import os,json,hashlib,sys
from pathlib import Path
from urllib.request import Request,urlopen
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import p2_external_chbmit_shared_clock as common
import p2_provguard_v4_calibrated as base
import p2_provguard_v5_artifact_residual as proposed
OUT=Path(os.environ.get('P2_OUTPUT','p2_v9_eegmmidb'));OUT.mkdir(parents=True,exist_ok=True)
RELS=[f'S{i:03d}/S{i:03d}R04.edf' for i in range(1,7)]
BASE='https://physionet.org/files/eegmmidb/1.0.0/'
# Only first digital EDF prefix; no whole-file digest or redistribution.
def read_edf(rel):
    req=Request(BASE+rel,headers={'Range':'bytes=0-2999999','User-Agent':'research-repro/1.0'})
    with urlopen(req,timeout=115) as res: buf=res.read(3000000)
    if len(buf)<150000:raise RuntimeError('EDF prefix too short '+rel)
    def fld(b):return b.decode('ascii','replace').strip()
    ns=int(fld(buf[252:256]));header=int(fld(buf[184:192]))
    if not 8<=ns<=128 or header!=256+ns*256:raise ValueError('Invalid EDF header '+rel)
    names=[fld(buf[256+i*16:256+(i+1)*16]) for i in range(ns)]
    smpbase=256+216*ns
    samples=np.array([int(fld(buf[smpbase+i*8:smpbase+(i+1)*8])) for i in range(ns)])
    recbytes=int(samples.sum())*2
    if header+60*recbytes>len(buf):raise RuntimeError('EDF Range too short '+rel)
    eeg=[(i,n) for i,n in enumerate(names) if n and
        not any(k in n.upper() for k in ('ANNOT','STATUS','ECG','EOG','EMG','STIM'))]
    eeg=eeg[:8]
    if len(eeg)!=8 or len({int(samples[i]) for i,n in eeg})!=1:
        raise ValueError('Unexpected EEG channel format '+rel)
    N=int(samples[eeg[0][0]])
    if N<128:raise RuntimeError('Unexpected short sampling '+rel)
    # Same amount of observed 768 digital samples for both releases.
    offsets=np.r_[0,samples.cumsum()]
    def window(first):
        result={n:[] for i,n in eeg}
        nrec=int(np.ceil(768/N))
        for idx in range(first,first+nrec):
            buf0=np.frombuffer(buf,dtype='<i2',count=int(samples.sum()),offset=header+idx*recbytes)
            for i,name in eeg:
                result[name].append(buf0[int(offsets[i]):int(offsets[i+1])].astype(float))
        return {n:np.concatenate(v)[:768] for n,v in result.items()}
    src=window(4);nonoverlap=window(42)
    return src,nonoverlap,{'url':BASE+rel,'sha256_prefix':hashlib.sha256(buf).hexdigest(),
     'eeg_channel_labels':[x[1] for x in eeg],'edf_samples_per_record':N,'prefix_only':True}
def car(d):
    names=list(d);x=np.stack([d[k] for k in names])
    x-=x.mean(axis=0,keepdims=True)
    return {k:v for k,v in zip(names,x)}
pos=['exact','relabel_permute','global_lag','crop','gain','drop','noise','combined']
neg=['different_subject','same_file_different_time','one_shared_channel',
     'async_same_source','repeated_constant_artifact','shared_global_artifact']
records={}
for r in RELS:
    records[r]=read_edf(r)
    print('EEGMMIDB_SOURCE_LOADED',r,flush=True)
frozen_car=base.Policy(train_threshold=.93,verify_threshold=.90)
rows=[]
for i,rel in enumerate(RELS):
    src,nonoverlap,meta=records[rel]
    different=records[RELS[(i+1)%len(RELS)]][0]
    for j,kind in enumerate(pos+neg):
        if kind=='shared_global_artifact':
            q=np.arange(768);wave=np.sin(q*.051)+np.sin(q*.113)
            amplitude=100*max(np.std(np.stack(list(src.values()))),np.std(np.stack(list(different.values()))))
            a={k:v+amplitude*wave for k,v in src.items()}
            b={k:v+amplitude*wave for k,v in different.items()}
        else:
            a=src
            b=common.transform(src,nonoverlap if kind=='same_file_different_time' else different,
                               kind,common.SEED+31*i+j)
        row={'source':rel,'kind':kind,'positive':kind in pos,
         'raw':int(base.match(a,b)['status']=='NUMERIC_WITNESS'),
         'frozen_tuned_car':int(base.match(car(a),car(b),frozen_car)['status']=='NUMERIC_WITNESS'),
         'residual':int(proposed.match(a,b)['status']=='DUAL_WITNESS')}
        rows.append(row)
    print('V9_SOURCE_EVALUATED',rel,flush=True)
summary={}
for method in ('raw','frozen_tuned_car','residual'):
    pp=[r for r in rows if r['positive']];nn=[r for r in rows if not r['positive']]
    summary[method]={'TP':sum(r[method] for r in pp),'positive_total':len(pp),
                     'FP':sum(r[method] for r in nn),'negative_total':len(nn)}
source_stats=[]
for rel in RELS:
    a=[r for r in rows if r['source']==rel]
    source_stats.append({'source':rel,'car_TP':sum(r['frozen_tuned_car'] for r in a if r['positive']),
       'residual_TP':sum(r['residual'] for r in a if r['positive']),
       'car_FP':sum(r['frozen_tuned_car'] for r in a if not r['positive']),
       'residual_FP':sum(r['residual'] for r in a if not r['positive'])})
doc={'status':'EXECUTED_REAL_EXTERNAL_EEG_WAVEFORMS_CONTROLLED_VARIANTS',
 'corpus':'PhysioNet eegmmidb/1.0.0; S001..S006 R04; unmodified digital EDF prefixes',
 'contrasts':'CHB-MIT dev-frozen CAR vs raw shared clock vs residual verification',
 'n_original_records':len(RELS),'n_controlled_cases':len(rows),
 'frozen_thresholds':{'CAR_train':.93,'CAR_verify':.90,'residual_v5_train':.985,'residual_v5_verify':.97},
 'source_manifest':{r:records[r][2] for r in RELS},
 'summary':summary,'per_record':source_stats,'cases':rows,
 'limitations':['R04 EEGMMIDB task labels never read or evaluated',
   'data collection different from CHB-MIT but digital variants are synthetic',
   'source IDs project-wide prior exposure unknown, not prospectively untouched',
   '6 source EDFs not 84 independent observations',
   'not a patient or clinical acquisition identity classifier',
   'no direct published SOTA fingerprint or DTW benchmark in this experiment']}
(OUT/'P2_V9_EEGMMIDB_CROSS_CORPUS_84.json').write_text(json.dumps(doc,indent=2)+'\n')
print('P2_V9_EEGMMIDB_SUMMARY',json.dumps(summary,sort_keys=True),flush=True)
