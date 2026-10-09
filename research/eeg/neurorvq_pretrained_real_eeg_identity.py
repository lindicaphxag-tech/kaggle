"""Frozen, real-pretrained Braindecode NeuroRVQ electrode-binding diagnostic.

This is a real EEG *representation drift* measurement, NOT test accuracy,
a new training algorithm, a reliable device-fault detector, or a clinical claim.

Pinned original public EEGMMIDB source S001/R04 and release model:
- library source commit a619f4b38333de6060811ffb8a5a4b1104551ff9
- checkpoint braindecode/neurorvq-eeg-pretrained (HF Hub model revision
  intentionally external; record the exact resolved SHA in run evidence).
License CC BY-NC 4.0, noncommercial research only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import mne

CHANNELS=("C3","C4","Cz","F3")
SOURCE_RUN=4
MODEL_REPO="braindecode/neurorvq-eeg-pretrained"
LIBRARY_SHA="a619f4b38333de6060811ffb8a5a4b1104551ff9"


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def extract_real_windows(cache):
    paths=mne.datasets.eegbci.load_data(1,[SOURCE_RUN],path=str(cache),
                                        update_path=False,verbose=False)
    if len(paths)!=1:
        raise RuntimeError("Exactly one real original EDF required")
    raw=mne.io.read_raw_edf(paths[0],preload=True,verbose=False)
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                    on_missing="raise",verbose=False)
    if not set(CHANNELS).issubset(set(raw.ch_names)):
        raise ValueError("Required four channel physical identities missing")
    raw.pick(list(CHANNELS))
    raw.reorder_channels(list(CHANNELS))
    raw.set_eeg_reference("average",projection=False,verbose=False)
    raw.filter(0.5,45.0,verbose=False)
    raw.resample(200.0,verbose=False)
    events,_=mne.events_from_annotations(raw,event_id={"T1":1,"T2":2},verbose=False)
    ep=mne.Epochs(raw,events,event_id={"T1":1,"T2":2},tmin=0.0,tmax=2.995,
                  baseline=None,preload=True,verbose=False)
    x=ep.get_data(units="uV")[:6].astype("float32")
    if x.shape != (6,4,600):
        raise RuntimeError(f"Bad raw EEG window geometry {x.shape}")
    x=np.clip(x,-500,500)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite raw EEG")
    return x,sha256(paths[0]),str(paths[0]),int(len(ep))


def feature_array(model,signal):
    with torch.inference_mode():
        z=model(torch.from_numpy(np.array(signal,copy=True)),return_features=True)
    if not isinstance(z,dict) or "features" not in z:
        raise ValueError("Model did not expose pretrained encoder features")
    out=z["features"].detach().cpu().numpy()
    if not np.isfinite(out).all():
        raise ValueError("Invalid frozen encoder features")
    return out


def assess(ref,wrong,restored,eps=1e-5):
    ref=np.asarray(ref,dtype=float)
    wrong=np.asarray(wrong,dtype=float)
    restored=np.asarray(restored,dtype=float)
    if ref.shape!=wrong.shape or ref.shape!=restored.shape or ref.ndim!=2:
        raise ValueError("Expected matched 2D frozen feature vectors")
    error=np.max(np.abs(ref-restored))
    if error>eps:
        raise AssertionError(f"Exact electrode reorder failed to recover representation ({error})")
    norm=np.maximum(np.linalg.norm(ref,axis=1),1e-12)
    drift=np.linalg.norm(wrong-ref,axis=1)/norm
    return {"n_trials":int(len(ref)),
            "relative_feature_l2_drift":drift.tolist(),
            "median_relative_drift":float(np.median(drift)),
            "max_relative_drift":float(np.max(drift)),
            "restored_max_abs_error":float(error),
            "feature_dim":int(ref.shape[1]),
            "wrong_order_nonidentical_count":int(np.sum(drift>1e-8))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True)
    import braindecode
    from braindecode.models import NeuroRVQ
    from huggingface_hub import HfApi
    info=HfApi().model_info(MODEL_REPO)
    repo_revision=str(info.sha)
    torch.manual_seed(314159)
    model=NeuroRVQ.from_pretrained(
        MODEL_REPO,n_outputs=2,n_chans=4,n_times=600,sfreq=200,
        channel_names=[name.lower() for name in CHANNELS],
    )
    model.eval()
    x,edf_hash,path,n_total=extract_real_windows(args.cache)
    permutation=np.array([1,0,2,3])
    original=feature_array(model,x)
    wrong=feature_array(model,x[:,permutation,:])
    restored=feature_array(model,x[:,permutation,:][:,np.argsort(permutation),:])
    result=assess(original,wrong,restored)
    result.update({
        "raw_edf_sha256":edf_hash,
        "original_edf_name":Path(path).name,
        "raw_edf_run":SOURCE_RUN,
        "all_source_epochs":n_total,
        "channels":CHANNELS,
        "wrong_permutation":permutation.tolist(),
        "model":MODEL_REPO,
        "hf_checkpoint_repo_revision":repo_revision,
        "library_source_revision":LIBRARY_SHA,
        "braindecode_version":getattr(braindecode,"__version__",None),
        "torch_version":torch.__version__,
        "mne_version":mne.__version__,
        "preprocess":"CAR, filter 0.5-45 Hz, resample 200 Hz, uV, clip +-500, epoch 3 sec",
        "limits":[
            "Frozen PRETRAINED encoder, but randomly initialized task head NOT scored",
            "One source subject, only six epochs, no downstream task performance",
            "Known channel correction is deterministic data bookkeeping",
            "Unknown true electrode identity cannot be certified without provenance",
            "No clinical, device, safety or SOTA claim",
        ]})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print("REAL_PRETRAINED_NEURORVQ_ID_DRIFT",json.dumps({
        "n_trials":result["n_trials"],
        "feature_dim":result["feature_dim"],
        "median_relative_drift":result["median_relative_drift"],
        "max_relative_drift":result["max_relative_drift"],
        "restored_max_abs_error":result["restored_max_abs_error"],
        "checkpoint_revision":repo_revision,
    }))

if __name__=="__main__":
    main()
