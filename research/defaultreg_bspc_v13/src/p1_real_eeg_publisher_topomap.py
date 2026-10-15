#!/usr/bin/env python3
"""Real BNCI2014_001 class-conditional scalp bandpower; NOT estimand weights."""
import json,hashlib,os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
CH=["Fz","FC3","FC1","FCz","FC2","FC4","C5","C3","C1","Cz","C2","C4","C6","CP3","CP1","CPz","CP2","CP4","P1","Pz","P2","POz"]
OUT=os.environ.get("FIGURE_OUT","/tmp/default-eeg-real-topomap")
os.makedirs(OUT,exist_ok=True)
pp=MotorImagery(n_classes=2,events=["left_hand","right_hand"],fmin=8,fmax=30,resample=128,baseline=None,reject_by_annotation=False)
X,y,_=pp.get_data(dataset=BNCI2014_001(),subjects=[1])
X=np.asarray(X,dtype=np.float64);y=np.asarray(y)
assert X.ndim==3 and X.shape[1]==len(CH) and set(y)=={"left_hand","right_hand"} and np.isfinite(X).all()
info=mne.create_info(CH,128,"eeg")
info.set_montage(mne.channels.make_standard_montage("standard_1020"),on_missing="raise")
f,P=welch(X,fs=128,nperseg=min(256,X.shape[-1]),axis=-1)
fig,axs=plt.subplots(2,3,figsize=(11,6),layout="constrained")
for row,(name,low,high) in enumerate([("Mu",8,12),("Beta",13,30)]):
    ix=(f>=low)&(f<=high)
    z=np.log10(np.maximum(np.trapezoid(P[:,:,ix],f[ix],axis=-1),1e-30))
    a,b=[z[y==event].mean(axis=0) for event in ("left_hand","right_hand")]
    bound=(min(a.min(),b.min()),max(a.max(),b.max()))
    for col,v in enumerate((a,b,b-a)):
        vlim=(-np.max(np.abs(v)),np.max(np.abs(v))) if col==2 else bound
        im,_=mne.viz.plot_topomap(v,info,axes=axs[row,col],show=False,cmap="RdBu_r" if col==2 else "viridis",vlim=vlim,contours=4,sensors=True)
        axs[row,col].set_title(f"{name} | "+["Left","Right","Right minus left"][col])
        fig.colorbar(im,ax=axs[row,col],fraction=.046,pad=.02)
fig.savefig(os.path.join(OUT,"P1_actual_BNCI_subject1_bandpower_topomap.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(OUT,"P1_actual_BNCI_subject1_bandpower_topomap.png"),bbox_inches="tight",dpi=300)
audit={"status":"REAL_EEG_CONTEXT_ONLY","subject":1,"dataset":"BNCI2014_001","n_trials":len(y),"shape":list(X.shape),"sha256_float64_epochs":hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),"not_decoding_weights":True,"not_unseen_subject":True,"no_erd_baseline":True}
open(os.path.join(OUT,"provenance.json"),"w").write(json.dumps(audit,indent=2)+"\n")
print(json.dumps(audit))
