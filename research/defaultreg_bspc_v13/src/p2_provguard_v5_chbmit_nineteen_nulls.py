#!/usr/bin/env python3
"""19-distinct-public-person nominal null calibration, single prespecified query.
Data are REAL CHB-MIT EDF digital prefixes but positive is CONTROLLED transformation.
If fewer than 19 calibrators remain, 0.05 rank p significance is not possible.
Reject repeated known subject CHB21, which is related to CHB01; no patient identities inferred.
"""
import json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import p2_external_chbmit_shared_clock as reader
import p2_provguard_v5_continuous_search as model
OUTPUT=Path(os.environ.get('P2_OUTPUT','p2_v5_source_level'));OUTPUT.mkdir(parents=True,exist_ok=True)
source_rel='chb01/chb01_01.edf'
source,_,meta=reader.read_one(source_rel)
nulls=[];manifest=[]
for i in range(2,21):
    rel=f'chb{i:02d}/chb{i:02d}_01.edf'
    x,unused,m=reader.read_one(rel)
    nulls.append(x)
    manifest.append({'record':rel,'sha256_edf_prefix':m['sha256_prefix']})
    print('CALIBRATOR_LOADED',rel,flush=True)
if len(nulls)!=19:raise RuntimeError('insufficient independent null case indices')
results={}
for kind,endpoint in [('combined','clock'),('async_same_source','any')]:
    # Challenge generation is fixed independently of any calibration outcome.
    positive=reader.transform(source,nulls[0],kind,20261015)
    output=model.calibrate({'chb01':source},positive,nulls,endpoint=endpoint)
    results[kind+'_'+endpoint]=output
    print('CALIBRATED',kind,endpoint,json.dumps({'score':output['score'],
         'p_value':output['p_value'],'max_null':max(output['null_scores']),
         'n_null':output['n_null']}),flush=True)
output={'dataset':'CHB-MIT 1.0.0 EDF DIGITAL PREFIX; source chb01; null chb02..20',
        'endpoint_definitions':'clock=coherent multichannel; any=one-channel reuse',
        'null_subjects_nominal_distinct':19,'source_sha256_edf_prefix':meta['sha256_prefix'],
        'manifest':manifest,'results':results,
        'predeclared_primary':'combined_clock; other query exploratory and not independently multiplicity-controlled',
        'limitations':['controlled positive not natural duplicate','source/validation transformations known',
                       'corpus max for one source only','calibrators presumed distinct CHB subjects; no independent medical verification',
                       'same dataset acquisition environment; external independence not tested',
                       'under exchangeability and no calibration leakage rank p is conditionally valid',
                       'p-value resolution 1/20 and no general population FPR estimate']}
(OUTPUT/'P2_V5_NINETEEN_RECORD_NULL_CALIBRATION.json').write_text(json.dumps(output,indent=2)+'\n')
