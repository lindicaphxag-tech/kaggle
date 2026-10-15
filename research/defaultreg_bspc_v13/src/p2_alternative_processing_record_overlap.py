#!/usr/bin/env python3
"""P2 alternative-processing exact recording-content overlap robustness.

Real Liu2024 public EEG, no target AUC or model training.
This is NOT a second dataset, independent patient confirmation,
or a preregistered untouched target result.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from moabb.datasets import Liu2024
from moabb.paradigms import MotorImagery

sys.path.insert(0,str(Path(__file__).resolve().parent))
from record_identity_groups_v26 import assign_recording_groups

def scan(fmin,fmax,resample):
    ds=Liu2024()
    p=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
                   fmin=fmin,fmax=fmax,resample=resample,
                   baseline=None,reject_by_annotation=False)
    records=[]
    shapes=set()
    for sid in range(1,51):
        X,labels,_=p.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(np.asarray(labels)=="right_hand",dtype=np.uint8)
        if len(X)!=40 or X.shape[1]!=29 or int(y.sum())!=20 or not np.isfinite(X).all():
            raise RuntimeError("unexpected fixed released EEG structure")
        shapes.add(tuple(X.shape))
        records.append({"subject":sid,"signal":X,"labels":y})
        print("P2_ALTERNATE_EEG_PROCESSING_READ",sid,flush=True)
    if len(shapes)!=1:raise AssertionError("dimension mismatch")
    ident,groupof=assign_recording_groups(records)
    return {"fmin":fmin,"fmax":fmax,"resample":resample,
       "shape_trials_channels_samples":list(next(iter(shapes))),
       "full_record_content_components":[g for g in ident["exact_record_groups"] if len(g)>1],
       "unique_processed_record_count":ident["independent_processed_record_count"],
       "retrospective_global_signal_discovery":True,
       "does_not_establish_patient_identity":True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    result={"status":"P2_PUBLIC_LIU2024_ALTERNATE_PIPELINE_CONTENT_AUDIT",
            "baseline_previous_pipeline":"8-30 Hz 160 Hz (already reported)",
            "new_independent_processing_setting":"6-35 Hz 128 Hz",
            "is_not_second_software_reader":True,
            "not_independent_external_dataset":True,
            "reanalysis":scan(6,35,128)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(result,indent=2,sort_keys=True)+"\n"
    a.out.write_text(body)
    print("P2_NEW_PROCESSING_RESULT",json.dumps(result["reanalysis"],sort_keys=True))
    print("RECEIPT_SHA",hashlib.sha256(body.encode()).hexdigest())
if __name__=="__main__":main()
