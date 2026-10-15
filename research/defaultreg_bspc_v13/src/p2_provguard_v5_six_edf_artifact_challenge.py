#!/usr/bin/env python3
"""Six real CHB-MIT digital EDF source prefixes, controlled transformations only.
New hard negative: two different public EEG sources share an injected additive
global artifact, while their channel-specific waveforms remain unrelated.
That is negative for coherent PHYSIOLOGICAL segment provenance but positive
for the broad numerical reuse of the injected artifact.
"""
import os,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import p2_external_chbmit_shared_clock as reader
import p2_provguard_v4_calibrated as baseline
import p2_provguard_v5_artifact_residual as guard
OUT=Path(os.environ.get('P2_OUTPUT','p2_v5_artifact'));OUT.mkdir(parents=True,exist_ok=True)
records=[f'chb{i:02d}/chb{i:02d}_01.edf' for i in range(1,7)]
data={x:reader.read_one(x) for x in records}
positive=['exact','relabel_permute','global_lag','crop','gain','drop','noise','combined']
negative=['different_subject','same_file_different_time','one_shared_channel','async_same_source','repeated_constant_artifact','shared_global_artifact']
rows=[]
for idx,rel in enumerate(records):
    src,same,meta=data[rel]
    other=data[records[(idx+1)%len(records)]][0]
    for j,kind in enumerate(positive+negative):
        if kind=='shared_global_artifact':
            # Both represent different source identities but same nuisance waveform.
            w=np.sin(np.arange(768)*.051)+np.sin(np.arange(768)*.113)
            # A fixed synthetic artifact common to both recordings.
            amp=100*max(np.std(np.concatenate(list(src.values()))),
                        np.std(np.concatenate(list(other.values()))))
            s={k:v+amp*w for k,v in src.items()}
            t={k:v+amp*w for k,v in other.items()}
        else:
            alt=same if kind=='same_file_different_time' else other
            t=reader.transform(src,alt,kind,reader.SEED+idx*31+j);s=src
        old=baseline.match(s,t)
        new=guard.match(s,t)
        if new['status']=='DUAL_WITNESS' and not guard.verify(s,t,new):
            raise RuntimeError('new witness fails independent verifier')
        rows.append({'source':rel,'kind':kind,'is_coherent_positive':kind in positive,
                     'old':int(old['status']=='NUMERIC_WITNESS'),
                     'v5_dual':int(new['status']=='DUAL_WITNESS'),
                     'v5_reason':new.get('reason'),
                     'min_residual':min(new.get('residual_scores',[0]))})
    print('P2_V5_ARTIFACT_PROGRESS',rel,flush=True)
stats={}
for method in ('old','v5_dual'):
    stats[method]={'positive_hits':sum(r[method] for r in rows if r['is_coherent_positive']),
                   'positive_total':sum(r['is_coherent_positive'] for r in rows),
                   'false_alarm_all':sum(r[method] for r in rows if not r['is_coherent_positive']),
                   'negative_total':sum(not r['is_coherent_positive'] for r in rows),
                   'false_alarm_shared_artifact':sum(r[method] for r in rows if r['kind']=='shared_global_artifact')}
report={'status':'FINISHED','dataset':'CHB-MIT original real EDF digital prefixes','results':stats,'cases':rows,
        'important':['All labels controlled: 6 independent source files, variants correlated within source',
        'Global artifact injected is a constructed numeric reuse, not an observed clinical artifact',
        'Residual guard may hurt legitimate shared only-common-mode signal',
        'Strong published multi-device synchronizers/DTW not included in this run']}
(OUT/'P2_V5_REAL_WAVEFORM_ARTIFACT_CHALLENGE.json').write_text(json.dumps(report,indent=2)+'\n')
print('P2_V5_ARTIFACT_SUMMARY',json.dumps(stats),flush=True)
