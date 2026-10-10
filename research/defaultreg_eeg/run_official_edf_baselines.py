"""Public-data-only DEFAULT REGULARISATION official EDF feasibility baseline.

Pre-registered train subjects S001..S008, heldout S009+S010; runs R04/R08/R12.
Published PhysioNet EEGMMIDB v1.0.0 (10.13026/C28G6P), ODC-By v1.
NO target-person training or target-batch moments; NO abandoned T1/T2 events.
ONE band 8-30 Hz, 21 sensors, source-fitted SPD-log tangent coordinates.

Baseline arms: LedoitWolf LDA, OAS LDA, source-person LOPO macro-AUC
tuned ridge-LDA, LogisticRegression. All use the same source-only final
geometry, original untouched heldout target trials, and no target labels
before scoring. This is a 2-target feasibility pilot, NOT multi-cohort
replication, historic manuscript matching, clinical result, or novel method.

This standalone runner contains no private manuscript or proprietary code.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time,urllib.request
from pathlib import Path

import numpy as np
import mne
from sklearn.covariance import LedoitWolf,OAS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CHANS=("FC5","FC3","FC1","FCZ","FC2","FC4","FC6",
       "C5","C3","C1","CZ","C2","C4","C6",
       "CP5","CP3","CP1","CPZ","CP2","CP4","CP6")
RUNS=(4,8,12)
SOURCE=tuple(range(1,9))
HELDOUT=(9,10)
GRID=(0.2,0.4,0.6,0.8,0.95)
URL="https://physionet.org/files/eegmmidb/1.0.0"

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def load_or_fetch(root,subject,run):
 target=root/f"S{subject:03d}"/f"S{subject:03d}R{run:02d}.edf"
 target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  url=f"{URL}/S{subject:03d}/S{subject:03d}R{run:02d}.edf"
  part=target.with_suffix(".edf.incomplete")
  if part.exists():part.unlink()
  req=urllib.request.Request(url,headers={"User-Agent":"DefaultReg-source-only-reproduction/1.0"})
  with urllib.request.urlopen(req,timeout=80) as response,open(part,"wb") as output:
   while True:
    chunk=response.read(1<<20)
    if not chunk:break
    output.write(chunk)
  if part.stat().st_size<300_000:raise ValueError("EDF suspiciously small "+str(part))
  part.rename(target)
 return target

def extract_original_trials(path,subject,run):
 raw=mne.io.read_raw_edf(str(path),preload=True,verbose="ERROR")
 names={x:x.replace(".","").replace("-","").upper() for x in raw.ch_names}
 if len(set(names.values()))!=len(names):raise ValueError("Duplicate normalized channels")
 raw.rename_channels(names)
 if not set(CHANS).issubset(raw.ch_names):
  raise ValueError("Missing required 21 EEG channels in "+str(path))
 raw.pick(list(CHANS))
 events,_=mne.events_from_annotations(raw,event_id={"T1":1,"T2":2},verbose=False)
 if not len(events):raise ValueError("No task events")
 raw.filter(8,30,method="iir",verbose="ERROR")
 ep=mne.Epochs(raw,events,event_id={"T1":1,"T2":2},
                tmin=0.5,tmax=3.5,baseline=None,
                reject_by_annotation=False,proj=False,
                preload=True,verbose="ERROR")
 if not np.array_equal(ep.events,events):
  raise ValueError("Lost original task cue before quality selection")
 data=ep.get_data(copy=True)
 if not np.isfinite(data).all():raise ValueError("Nonfinite original EDF data")
 data=data-data.mean(axis=-1,keepdims=True)
 cov=data@data.transpose(0,2,1)/data.shape[-1]
 scale=np.trace(cov,axis1=1,axis2=2)/len(CHANS)
 cov+=np.eye(len(CHANS))[None,:,:]*np.maximum(scale[:,None,None]*1e-6,1e-20)
 y=(ep.events[:,2]==2).astype(np.int64)
 if len(np.unique(y))<2:raise ValueError("Missing one task class")
 return cov,y,[{"subject":subject,"run":run,"event_sample":int(e[0]),
                "cue":int(e[2])} for e in ep.events]

def reference(covs):
 center=np.mean(covs,axis=0)
 val,vec=np.linalg.eigh(center)
 if np.any(val<=0):raise ValueError("Source reference not SPD")
 return (vec*(val**(-0.5)))@vec.T

def tangent(covs,source_inv_sqrt):
 z=np.einsum("ab,nbc,cd->nad",source_inv_sqrt,covs,source_inv_sqrt)
 eig,rot=np.linalg.eigh(z)
 if np.any(eig<=0):raise ValueError("Trial not SPD in source geometry")
 mat=(rot*np.log(eig)[:,None,:])@rot.transpose(0,2,1)
 p=mat.shape[1]
 idx=np.triu_indices(p)
 result=mat[:,idx[0],idx[1]]
 result[:,idx[0]!=idx[1]]*=np.sqrt(2)
 return result

def within_residual(X,y):
 c0=X[y==0];c1=X[y==1]
 if len(c0)<2 or len(c1)<2:raise ValueError("Both classes need >=2 source trials")
 mean0=c0.mean(0);mean1=c1.mean(0)
 resid=np.concatenate((c0-mean0,c1-mean1),axis=0)
 d=mean1-mean0
 midpoint=(mean1+mean0)/2
 return resid,d,midpoint

def ridge_score(train_X,train_y,test_X,lambda_value):
 resid,d,midpoint=within_residual(train_X,train_y)
 S=(resid.T@resid)/(len(train_X)-2)
 trace=max(np.trace(S)/len(d),np.finfo(float).eps)
 reg=(1-lambda_value)*S+lambda_value*trace*np.eye(len(d))
 w=np.linalg.solve(reg,d)
 return (test_X-midpoint)@w

def fit_analytic(train_X,train_y,test_X,estimator):
 resid,d,midpoint=within_residual(train_X,train_y)
 fitted=estimator.fit(resid)
 reg=fitted.covariance_
 w=np.linalg.solve(reg+np.eye(len(d))*max(float(np.trace(reg)),1.)*1e-10,d)
 return (test_X-midpoint)@w, float(fitted.shrinkage_)

def subject_lopo_lambda(covs,y,groups,tail_fraction=0.25):
 """Same source-subject validation folds for four scalar selection criteria.
 No target information enters any criterion. This is a development comparison,
 not evidence that tail risk is novel or superior.
 """
 rows={lam:{"auc":[],"signed":[],"squared":[]} for lam in GRID}
 for group in sorted(set(groups)):
  tr=groups!=group;val=~tr
  if len(set(y[val]))!=2:raise ValueError("Missing class for source participant")
  inv=reference(covs[tr])   # INNER fold source-only geometry, no validation subject
  Xtr=tangent(covs[tr],inv)
  Xva=tangent(covs[val],inv)
  resid_val,d_val,_=within_residual(Xva,y[val])
  S_val=(resid_val.T@resid_val)/(len(Xva)-2)
  for lam in GRID:
   prediction=ridge_score(Xtr,y[tr],Xva,lam)
   auc=float(roc_auc_score(y[val],prediction))
   resid,d_train,mid=within_residual(Xtr,y[tr])
   S_train=(resid.T@resid)/(len(Xtr)-2)
   trace=max(np.trace(S_train)/len(d_train),np.finfo(float).eps)
   reg=(1-lam)*S_train+lam*trace*np.eye(len(d_train))
   w=np.linalg.solve(reg,d_train)
   den=float(w@S_val@w)
   floor=np.finfo(float).eps*max(1.0,float(w@w))
   if not np.isfinite(den) or den<=floor:raise ValueError("Degenerate LOPO validation variance")
   signed=float(w@d_val)/np.sqrt(den)
   rows[lam]["auc"].append(auc)
   rows[lam]["signed"].append(signed)
   rows[lam]["squared"].append(signed*signed)
 n_groups=len(set(groups));k=max(1,int(np.ceil(tail_fraction*n_groups)))
 criterion={}
 for lam in GRID:
  signed=np.asarray(rows[lam]["signed"])
  criterion[lam]={
   "source_macro_auc":float(np.mean(rows[lam]["auc"])),
   "signed_mean":float(np.mean(signed)),
   "squared_margin":float(np.mean(rows[lam]["squared"])),
   "tail_signed":float(np.mean(np.sort(signed)[:k])),
   "negative_fraction":float(np.mean(signed<0))}
 selected={
   "source_macro_auc":float(max(GRID,key=lambda lam:(criterion[lam]["source_macro_auc"],lam))),
   "signed_mean":float(max(GRID,key=lambda lam:(criterion[lam]["signed_mean"],lam))),
   "squared_margin":float(max(GRID,key=lambda lam:(criterion[lam]["squared_margin"],lam))),
   "tail_signed":float(max(GRID,key=lambda lam:(criterion[lam]["tail_signed"],lam)))}
 return selected,{str(lam):criterion[lam] for lam in GRID}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--edf-root",default="edf_original")
 ap.add_argument("--out",default="defaultreg_public_edf_feasibility")
 args=ap.parse_args()
 t0=time.monotonic()
 root=Path(args.edf_root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 records=[];covs=[];labels=[];groups=[];ids=[];files=[]
 for subject in SOURCE+HELDOUT:
  for run in RUNS:
   file=load_or_fetch(root,subject,run)
   original_sha=sha(file)
   cov,y,trialids=extract_original_trials(file,subject,run)
   files.append({"subject":subject,"run":run,"name":str(file.name),
                 "sha256":original_sha,"edf_bytes":file.stat().st_size,
                 "all_original_cues":len(trialids),"selected_and_scored":len(y)})
   covs.append(cov);labels.append(y);groups.extend([subject]*len(y))
   ids.extend(trialids)
 matrix=np.concatenate(covs)
 y=np.concatenate(labels);g=np.asarray(groups)
 tr=np.isin(g,SOURCE);te=np.isin(g,HELDOUT)
 if np.intersect1d(np.unique(g[tr]),np.unique(g[te])).size:
  raise ValueError("Target subject contaminated source training")
 selected,cvscores=subject_lopo_lambda(matrix[tr],y[tr],g[tr])
 fit_inv=reference(matrix[tr]) # SOURCE ONLY: never target covariance
 Xtrain=tangent(matrix[tr],fit_inv)
 Xheld=tangent(matrix[te],fit_inv) # Apply frozen reference one trial at a time
 scores={}
 lw,lwval=fit_analytic(Xtrain,y[tr],Xheld,LedoitWolf(store_precision=False))
 oa,oaval=fit_analytic(Xtrain,y[tr],Xheld,OAS(store_precision=False))
 scores["LedoitWolf_LDA"]=lw
 scores["OAS_LDA"]=oa
 scores["nested_source_macro_auc_LDA"]=ridge_score(Xtrain,y[tr],Xheld,selected["source_macro_auc"])
 scores["source_signed_mean_LDA"]=ridge_score(Xtrain,y[tr],Xheld,selected["signed_mean"])
 scores["source_squared_margin_LDA"]=ridge_score(Xtrain,y[tr],Xheld,selected["squared_margin"])
 scores["source_tail_signed_LDA"]=ridge_score(Xtrain,y[tr],Xheld,selected["tail_signed"])
 logistic=LogisticRegression(max_iter=2000,solver="lbfgs",C=1.0)
 logistic.fit(Xtrain,y[tr])
 scores["LogisticRegression"]=logistic.decision_function(Xheld)
 if len({len(x) for x in scores.values()})!=1:raise ValueError("Unequal trial counts")
 result={
  "status":"REAL_ORIGINAL_EDF_IF_DOWNLOADED_SHA_VERIFIED",
  "warning":"2-target single-band feasibility only; not significant gain, external replication, historical protocol reproduction or TNSRE submission.",
  "official_dataset":"PhysioNet EEGMMIDB 1.0.0 DOI 10.13026/C28G6P",
  "license":"Open Data Commons Attribution v1.0",
  "target_data_used_for_training_or_hyperparam_selection":False,
  "source_participants":list(SOURCE),"heldout_participants":list(HELDOUT),
  "runs":list(RUNS),"features":"8-30Hz, 21 channels, source-trained arithmetic-SPD-whitened log tangent, 231D",
  "inner_geometry_refit_on_source_training_groups_only":True,
  "no_task_cues_dropped":True,
  "train_original_task_trials":int(tr.sum()),
  "target_original_task_trials":int(te.sum()),
  "source_only_validation_criteria_by_lambda":cvscores,
  "source_selected_lambda_by_method":selected,
  "LedoitWolf_shrinkage":lwval,"OAS_shrinkage":oaval,
  "per_target_subject":{},
  "source_files":files,
  "wall_seconds":round(time.monotonic()-t0,2)}
 for subject in HELDOUT:
  m=g[te]==subject
  result["per_target_subject"][str(subject)]={
   "target_trials":int(m.sum()),
   "task_label_counts":{"T1":int(np.sum(y[te][m]==0)),
                        "T2":int(np.sum(y[te][m]==1))},
   "auc":{arm:round(float(roc_auc_score(y[te][m],score[m])),8)
          for arm,score in scores.items()}}
 (out/"full_pilot_json.json").write_text(json.dumps(result,indent=2,sort_keys=True))
 np.savez_compressed(out/"all_original_trial_covariances.npz",
                      covs=matrix,y=y,subject=g,
                      run=np.asarray([x["run"] for x in ids]),
                      event_sample=np.asarray([x["event_sample"] for x in ids]),
                      cue=np.asarray([x["cue"] for x in ids]))
 result["cov_npz_sha256"]=sha(out/"all_original_trial_covariances.npz")
 (out/"full_pilot_json.json").write_text(json.dumps(result,indent=2,sort_keys=True))
 print(json.dumps(result,indent=2,sort_keys=True))
 return result
if __name__=="__main__":main()
