"""Conventional filter-bank log-Euclidean EEG covariance baseline.

Real PhysioNet EEGMMIDB EDF; fixed subject-disjoint train/validation/test.
No new method or pretrained NeuroRVQ is claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

BANDS=((8.,13.),(13.,30.))
RUNS=(4,8,12)
SPLITS={
    "mini":{"train":(1,2),"val":(3,),"test":(4,)},
    "pilot":{"train":tuple(range(1,7)),
             "val":tuple(range(7,10)),"test":tuple(range(10,13))},
}
CS=(.0001,.001,.01,.1,1.)

def fingerprint(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def load_trials(ids,data_dir):
    xs,ys,groups=[],[],[]
    hashes={}; names=None
    montage=mne.channels.make_standard_montage("standard_1005")
    for sid in ids:
        paths=mne.datasets.eegbci.load_data(int(sid),list(RUNS),path=str(data_dir),
                                            update_path=False,verbose=False)
        if len(paths)!=3:
            raise RuntimeError("Expected three run-specific EDF files")
        for file in paths:
            hashes[str(Path(file).resolve().relative_to(data_dir.resolve()))]=fingerprint(file)
            raw=mne.io.read_raw_edf(file,preload=True,verbose=False)
            mne.datasets.eegbci.standardize(raw)
            raw.set_montage(montage,on_missing="raise",verbose=False)
            if names is None:
                names=tuple(raw.ch_names)
            if len(names)!=64 or set(names)!=set(raw.ch_names):
                raise ValueError("Unexpected electrode names")
            raw.reorder_channels(names)
            raw.set_eeg_reference("average",projection=False,verbose=False)
            ev,_=mne.events_from_annotations(raw,event_id={"T1":1,"T2":2},verbose=False)
            ep=mne.Epochs(raw,ev,event_id={"T1":1,"T2":2},
                          tmin=.5,tmax=3.5,baseline=None,preload=True,verbose=False)
            x=ep.get_data(units="V")
            if x.ndim!=3 or x.shape[1]!=64 or x.shape[-1]<400:
                raise ValueError("Unexpected time-series epoch")
            xs.append(x)
            ys.append((ep.events[:,-1]==2).astype(int))
            groups.extend([int(sid)]*len(x))
            raw.close()
    return np.concatenate(xs),np.concatenate(ys),np.asarray(groups),names,hashes

def filter_bank_log_covariance(trials,sfreq=160.,shrink=.08):
    x=np.asarray(trials,dtype=np.float64)
    if x.ndim!=3 or not len(x) or x.shape[1]<2 or x.shape[2]<64:
        raise ValueError("Expected nonempty trial-channel-time arrays")
    if not np.isfinite(x).all() or not 0<shrink<1:
        raise ValueError("Nonfinite array or illegal covariance shrinkage")
    n,c,t=x.shape
    idx=np.triu_indices(c)
    factor=np.where(idx[0]==idx[1],1.,2.**.5)
    feature=[]
    for lo,hi in BANDS:
        sos=butter(4,(lo,hi),btype="bandpass",fs=sfreq,output="sos")
        z=sosfiltfilt(sos,x,axis=-1)
        z-=z.mean(axis=-1,keepdims=True)
        cov=z@np.swapaxes(z,-2,-1)/(t-1)
        trace=np.trace(cov,axis1=-2,axis2=-1)
        if np.any(trace<=0) or not np.isfinite(trace).all():
            raise ValueError("Nonpositive power")
        cov*= (c/trace)[:,None,None]
        cov*=1-shrink
        cov[:,np.arange(c),np.arange(c)]+=shrink
        eig,v=np.linalg.eigh(cov)
        if np.any(eig<=0):
            raise RuntimeError("Non-SPD covariance")
        logcov=(v*np.log(eig)[:,None,:])@np.swapaxes(v,-2,-1)
        feature.append(logcov[:,idx[0],idx[1]]*factor[None,:])
    return np.concatenate(feature,axis=1)

def evaluate(feats,y,subjects,cohort="mini"):
    split=SPLITS[cohort]
    if set(subjects)!=set(split["train"]+split["val"]+split["test"]):
        raise ValueError("Unregistered or missing subjects")
    masks={k:np.isin(subjects,v) for k,v in split.items()}
    if not all(m.any() for m in masks.values()) or set(y[masks["train"]])!={0,1}:
        raise ValueError("Invalid subject split or training classes")
    valid=[]
    for C in CS:
        model=make_pipeline(StandardScaler(),LogisticRegression(
            C=C,max_iter=1000,random_state=314159))
        model.fit(feats[masks["train"]],y[masks["train"]])
        bacc=float(balanced_accuracy_score(
            y[masks["val"]],model.predict(feats[masks["val"]])))
        valid.append((bacc,C))
    chosen=max(valid,key=lambda z:(z[0],-z[1]))[1]
    model=make_pipeline(StandardScaler(),LogisticRegression(
        C=chosen,max_iter=1000,random_state=314159))
    model.fit(feats[masks["train"]],y[masks["train"]])
    ix=masks["test"]
    pred=model.predict(feats[ix])
    return {"subject_splits":{k:list(v) for k,v in split.items()},
            "selected_C_validation_only":chosen,
            "validation_grid":[{"C":c,"bacc":acc} for acc,c in valid],
            "test_bacc":float(balanced_accuracy_score(y[ix],pred)),
            "n_test":int(ix.sum()),"test_labels":y[ix].tolist(),
            "test_predictions":pred.tolist(),"test_subjects":subjects[ix].tolist()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",choices=SPLITS,default="mini")
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    a.data_dir.mkdir(parents=True,exist_ok=True)
    sp=SPLITS[a.cohort]
    X,y,subjects,channels,files=load_trials(
        sp["train"]+sp["val"]+sp["test"],a.data_dir)
    features=filter_bank_log_covariance(X)
    record=evaluate(features,y,subjects,a.cohort)
    record.update({
        "name":"Classical filter-bank log-Euclidean covariance baseline",
        "n_channels":len(channels),"n_trials":len(y),
        "features_per_trial":features.shape[-1],
        "EDF_sha256":files,
        "versions":{"mne":mne.__version__,"numpy":np.__version__,
                    "scipy":scipy.__version__,"sklearn":sklearn.__version__},
        "limitations":["Not a novel EEG method","No pretrained foundation model",
                       "Exploratory; prior pilot used overlapping subjects",
                       "Only validation subject labels select hyperparameter"],
    })
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(record,indent=2,allow_nan=False))
    print("REAL_EEG_SPATIAL_BASELINE",json.dumps({
        "test_bacc":record["test_bacc"],"n_test":record["n_test"],
        "C":record["selected_C_validation_only"]}))

if __name__=="__main__":
    main()
