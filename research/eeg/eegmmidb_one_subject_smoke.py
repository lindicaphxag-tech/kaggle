"""Read three original EEGMMIDB motor-imagery EDF recordings for ONE subject.

This smoke verifies data ingestion and electrode labels only. It is NOT a
subject-disjoint classification experiment and produces no task accuracy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from .eegmmidb_real_pilot import load_real_data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.data_dir.mkdir(parents=True,exist_ok=True)
    X,y,s,names,coords,files=load_real_data((1,),args.data_dir)
    if len(files)!=3 or X.ndim!=3 or X.shape[1:]!=(64,2):
        raise RuntimeError("Expected three real 64-channel spectral EDF sources")
    if set(np.unique(y))!={0,1} or set(np.unique(s))!={1}:
        raise RuntimeError("Missing both imagery classes or unexpected subject")
    if len(set(names))!=64 or not np.isfinite(X).all() or not np.isfinite(coords).all():
        raise RuntimeError("Invalid source electrode identity contract")
    record={
        "dataset":"PhysioNet EEGMMIDB v1.0.0", "subject_id":1,
        "runs":[4,8,12], "real_edf_source_count":len(files),
        "real_trial_count":len(y), "labels":{str(k):int((y==k).sum()) for k in (0,1)},
        "channels":list(names),"spectral_feature_shape":list(X.shape),
        "raw_edf_sha256":files,
        "status":"ingestion_smoke_only_no_task_accuracy_or_originality_claim",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False))
    print("REAL_EEG_EDF_SMOKE",json.dumps({
        "n_trials":len(y),"class_counts":record["labels"],
        "channels":len(names),"sources":len(files),
    }))


if __name__=="__main__":
    main()
