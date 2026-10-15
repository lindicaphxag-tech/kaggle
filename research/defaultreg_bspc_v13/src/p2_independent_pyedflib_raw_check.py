#!/usr/bin/env python3
"""Independent raw EDF numerical-reader cross-check for P2.

pyEDFlib is used rather than the previous MNE EDF read path.
Do not infer that equal digital signals imply the same patient.
No signal samples or file paths are exported in the artifact.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from moabb.datasets import Liu2024
import pyedflib

def raw_signature(subject):
    ds=Liu2024()
    path=Path(ds.data_path(subject)[0])
    if not path.exists(): raise FileNotFoundError("dataset release file absent")
    with pyedflib.EdfReader(str(path)) as edf:
        count=edf.signals_in_file
        labels=edf.getSignalLabels()
        srate=[float(edf.getSampleFrequency(i)) for i in range(count)]
        nsamp=[int(edf.getNSamples()[i]) for i in range(count)]
        arrays=[edf.readSignal(i,digital=False) for i in range(count)]
    if len(arrays)!=count or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("invalid EDF channel parsing")
    return {"n":count,"labels":labels,"sample_rates":srate,"samples":nsamp,"arrays":arrays,
            "released_file_binary_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def compare_pair(a,b):
    A=raw_signature(a);B=raw_signature(b)
    if A["n"]!=B["n"] or A["labels"]!=B["labels"]:
        raise RuntimeError("raw released EDF channel schema mismatch")
    count=A["n"]
    eq=[]
    for i in range(count):
        same=(A["sample_rates"][i]==B["sample_rates"][i] and
              A["samples"][i]==B["samples"][i] and
              np.array_equal(A["arrays"][i],B["arrays"][i]))
        eq.append(bool(same))
    return {"released_ids":[a,b],"n_channels":count,
       "reader":"pyedflib EDF physical samples (not MNE)",
       "n_exact_equal_raw_channels":sum(eq),
       "n_not_equal":count-sum(eq),
       "unequal_channel_indices":[i for i,v in enumerate(eq) if not v],
       "unnamed_or_unequal_labels":[str(A["labels"][i]) for i,v in enumerate(eq) if not v],
       "released_binary_file_sha256_different":A["released_file_binary_sha256"]!=B["released_file_binary_sha256"],
       "does_not_establish_patient_identity":True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    out={"status":"INDEPENDENT_PYEDFLIB_RAW_SIGNALS_EDF_AUDIT",
      "dataset":"public MOABB Liu2024","not_human_identity_evidence":True,
      "not_a_new_target_classifier_validation":True,
      "pairs":[compare_pair(4,5),compare_pair(47,49)]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    # Conservative: do NOT write a success assertion for the previously
    # reported 32-of-33 MNE outcome before independent verification.
    print(json.dumps({k:v for k,v in out.items() if k!="pairs"},sort_keys=True))
    for x in out["pairs"]:
        print("EDF_INDEPENDENT_PAIR_COUNTS",json.dumps(x,sort_keys=True))
if __name__=="__main__":main()
