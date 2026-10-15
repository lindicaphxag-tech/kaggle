#!/usr/bin/env python3
"""EEGMMIDB source-specific tuned CAR vs frozen V5 residual.
Dev EDF S001..S002; held EDF S009..S014. All are original digital signals
with CONTROLLED perturbation labels; not natural duplicate detection.
This is a post-discovery exploratory study, not pre-registered SOTA.
"""
import os,sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import p2_v9_cross_corpus_eegmmidb_frozen_car as reader
import p2_external_chbmit_shared_clock as transforms
import p2_provguard_v4_calibrated as base
import p2_provguard_v5_artifact_residual as residual
OUT=Path(os.environ.get('P2_OUTPUT','p2_v10_tuned_eegmmidb'));OUT.mkdir(exist_ok=True,parents=True)
DEV=[f'S{i:03d}/S{i:03d}R04.edf' for i in [1,2]]
HELD=[f'S{i:03d}/S{i:03d}R04.edf' for i in range(9,15)]
POS=['exact','relabel_permute','global_lag','crop','gain','drop','noise','combined']
NEG=['different_subject','same_file_different_time','one_shared_channel',
     'async_same_source','repeated_constant_artifact','shared_global_artifact']
TRAIN=(.80,.85,.90,.93,.95,.97,.985)
VERIFY=(.80,.85,.90,.93,.95,.97)
GRID=[base.Policy(train_threshold=a,verify_threshold=b) for a in TRAIN for b in VERIFY]
def make_case(src,other,same,kind,seed):
    if kind=='shared_global_artifact':
        q=np.arange(768);artifact=np.sin(q*.051)+np.sin(q*.113)
        amp=100*max(np.std(np.stack(list(src.values()))),
                   np.std(np.stack(list(other.values()))))
        return ({k:v+amp*artifact for k,v in src.items()},
                {k:v+amp*artifact for k,v in other.items()})
    return src,transforms.transform(src,same if kind=='same_file_different_time' else other,kind,seed)
def run():
    allfiles=DEV+HELD
    originals={}
    for path in allfiles:
        originals[path]=reader.read_edf(path)
        print('EEGMMIDB_ORIGINAL_LOADED',path,flush=True)
    devcases=[]
    for idx,rel in enumerate(DEV):
        src,same,_=originals[rel];other=originals[DEV[(idx+1)%len(DEV)]][0]
        for j,kind in enumerate(POS+NEG):
            a,b=make_case(src,other,same,kind,transforms.SEED+idx*31+j)
            devcases.append((rel,kind,int(kind in POS),reader.car(a),reader.car(b)))
    scored=[]
    for ci,p in enumerate(GRID):
        fp=tp=0
        for rel,kind,label,a,b in devcases:
            hit=int(base.match(a,b,p)['status']=='NUMERIC_WITNESS')
            tp+=label*hit;fp+=(1-label)*hit
        scored.append((fp,-tp,-p.train_threshold,-p.verify_threshold,ci))
        if (ci+1)%7==0:print('EEGMMIDB_CAR_GRID',ci+1,'of',len(GRID),flush=True)
    fp,neg_tp,fit_neg,verify_neg,ci=min(scored)
    chosen=GRID[ci]
    print('EEGMMIDB_CAR_POLICY_FROZEN',json.dumps({'fit':chosen.train_threshold,
        'verify':chosen.verify_threshold,'dev_true_positives':-neg_tp,'dev_false_positives':fp}),flush=True)
    cases=[]
    for idx,rel in enumerate(HELD):
        src,same,_=originals[rel]
        other=originals[HELD[(idx+1)%len(HELD)]][0]
        for j,kind in enumerate(POS+NEG):
            a,b=make_case(src,other,same,kind,transforms.SEED+(idx+100)*31+j)
            car=int(base.match(reader.car(a),reader.car(b),chosen)['status']=='NUMERIC_WITNESS')
            v5=int(residual.match(a,b)['status']=='DUAL_WITNESS')
            cases.append({'source':rel,'case':kind,'positive':int(kind in POS),
                          'CAR_source_tuned':car,'V5_CHB_frozen':v5})
        print('EEGMMIDB_V10_HELD_ORIGINAL_DONE',rel,flush=True)
    summary={}
    for method in ('CAR_source_tuned','V5_CHB_frozen'):
        pp=[x for x in cases if x['positive']];nn=[x for x in cases if not x['positive']]
        summary[method]={'TP':sum(x[method] for x in pp),'pos':len(pp),
                         'FP':sum(x[method] for x in nn),'neg':len(nn)}
    source_rows=[]
    for rel in HELD:
        group=[x for x in cases if x['source']==rel]
        source_rows.append({'source':rel,
           'CAR_TP':sum(x['CAR_source_tuned'] for x in group if x['positive']),
           'V5_TP':sum(x['V5_CHB_frozen'] for x in group if x['positive']),
           'CAR_FP':sum(x['CAR_source_tuned'] for x in group if not x['positive']),
           'V5_FP':sum(x['V5_CHB_frozen'] for x in group if not x['positive'])})
    output={'status':'EXECUTED_IF_CI_SUCCESS','dataset':'PhysioNet EEGMMIDB 1.0.0',
        'development_EDF':DEV,'evaluation_EDF':HELD,
        'CAR_search_grid_fit':TRAIN,'CAR_search_grid_verify':VERIFY,
        'CAR_frozen_choice':{'fit':chosen.train_threshold,'verify':chosen.verify_threshold},
        'selection_objective':'min dev false alarms, then max dev TPs, deterministic tie break',
        'V5_policy':'fixed default CHB-MIT thresholds and channel-match-first residual check',
        'n_original_held_sources':len(HELD),'n_controlled_held_cases':len(cases),
        'EDF_prefix_digests':{p:originals[p][2]['sha256_prefix'] for p in allfiles},
        'summary':summary,'per_source':source_rows,'cases':cases,
        'limitations':['EDA and examples selected after prior EEGMMIDB tests',
          'digital real EEG waveforms; synthetic controlled duplicates and artifacts',
          'no natural clinical duplicate adjudication',
          'published robust synchronizers and fingerprints absent',
          'inference must use original EDF not transformed cases']}
    (OUT/'P2_EEGMMIDB_SOURCE_TUNED_CAR_FINAL.json').write_text(json.dumps(output,indent=2)+'\n')
    print('P2_EEGMMIDB_SOURCE_TUNED_COMPARISON',json.dumps(summary,sort_keys=True),flush=True)
if __name__=='__main__':run()
