"""All-50 acute-stroke public EEG recording-identity forensic audit.

Never display raw patient EEG. Only identifies exact duplicate source
recordings under the original fixed MOABB v1.7.2 protocol and separately
compares the bytes of the public EDF files. Source recording duplicates
are NOT legitimate independent outer held-person observations.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from moabb.datasets import Liu2024
from moabb.paradigms import MotorImagery


def digest_array(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda:file.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    ds=Liu2024()
    paradigm=MotorImagery(
        n_classes=2,events=["left_hand","right_hand"],
        fmin=8,fmax=30,resample=160,baseline=None,
        reject_by_annotation=False,
    )
    receipt=[]
    for sid in range(1,51):
        X,y,meta=paradigm.get_data(dataset=ds,subjects=[sid])
        X=np.ascontiguousarray(X,dtype=np.float64)
        y=np.asarray(y)
        if X.shape[0]!=40 or X.shape[1]!=29 or len(y)!=40:
            raise ValueError("wrong original 50-patient source schema")
        if np.min(np.bincount((y=="right_hand").astype(int),minlength=2))!=20:
            raise ValueError("original 20/20 class balance violation")
        edf=ds.data_path(sid)[0]
        row={
            "subject":sid,
            "preprocessed_EEG_sha256":digest_array(X),
            "labels_sha256":digest_array(np.asarray((y=="right_hand"),np.uint8)),
            "released_EDF_file_sha256":digest_file(edf),
            "shape":list(X.shape)
        }
        receipt.append(row)
        print("LIU50_RECORD_FINGERPRINT_DONE",sid,flush=True)
    def group_by(key):
        d=defaultdict(list)
        for r in receipt:
            d[r[key]].append(r["subject"])
        return sorted((sorted(v) for v in d.values() if len(v)>1),
                      key=lambda x:(x[0],len(x)))
    pre_groups=group_by("preprocessed_EEG_sha256")
    label_map={r["subject"]:r["labels_sha256"] for r in receipt}
    pre_same_label=all(len({label_map[s] for s in g})==1 for g in pre_groups)
    edf_groups=group_by("released_EDF_file_sha256")
    identity={}
    for g in pre_groups:
        for s in g:identity[s]=g[0]
    unique=[s for s in range(1,51) if s not in identity or identity[s]==s]
    out={
        "status":"COMPLETE_50_PUBLIC_LIU2024_PREPROCESSED_RECORD_IDENTITY_AUDIT",
        "subjects_checked":50,
        "original_trial_count":2000,
        "all_recordings_mapped_to_original_subject_ids":True,
        "exact_duplicate_preprocessed_EEG_groups":pre_groups,
        "all_exact_signal_duplicates_also_same_event_labels":pre_same_label,
        "exact_duplicate_released_EDF_file_groups":edf_groups,
        "unique_EEG_records_count":len(unique),
        "identity_based_representative_subject_ids":unique,
        "identities_from_waveforms_only_not_prediction_performance":True,
        "not_confirmation_of_patient_identity_or_source_cause":True,
        "not_new_target_AUC_test":True,
        "receipts":receipt,
    }
    p=Path("/tmp/defaultreg-v25-liu-all50/LIU2024_ALL50_PUBLIC_DATA_FINGERPRINTS.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(out,sort_keys=True,indent=2)+"\n"
    p.write_text(body)
    print(json.dumps({
        "status":out["status"],
        "exact_duplicate_preprocessed_EEG_groups":pre_groups,
        "exact_duplicate_released_EDF_file_groups":edf_groups,
        "unique_EEG_records_count":len(unique),
        "identity_group_labels_concordant":pre_same_label,
        "report_sha256":hashlib.sha256(body.encode()).hexdigest()
    },indent=2))


if __name__=="__main__":
    main()
