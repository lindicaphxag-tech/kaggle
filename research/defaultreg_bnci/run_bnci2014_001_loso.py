"""Default Regularisation — preregistered BNCI2014_001 full LOSO.

Public external-dataset evidence runner. All target-subject outcomes are
strictly held out from geometry/model/regularisation selection. Left/right
motor-imagery trials only. One wide 8–30 Hz representation. Primary new
criterion (frozen before target scores): lower-tail signed source transfer.

This is an external validation *stratum*, not a clinical claim or SOTA proof.
"""
from __future__ import annotations
import argparse,json,math,platform,sys,time
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf,OAS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

SUBJECTS=tuple(range(1,10))
GRID=(0.2,0.4,0.6,0.8,0.95)
TAIL=0.25
METHODS=(
 "LedoitWolf_LDA","OAS_LDA","nested_source_macro_auc_LDA",
 "source_signed_mean_LDA","source_squared_margin_LDA",
 "source_tail_signed_LDA","LogisticRegression")

def covariances(X):
 X=np.asarray(X,dtype=np.float64)
 if X.ndim!=3 or not np.isfinite(X).all():raise ValueError("Expected finite [trial,channel,time] EEG")
 X=X-X.mean(axis=-1,keepdims=True)
 c=X@X.transpose(0,2,1)/X.shape[-1]
 scale=np.trace(c,axis1=1,axis2=2)/c.shape[1]
 c+=np.eye(c.shape[1])[None,:,:]*np.maximum(scale[:,None,None]*1e-6,1e-20)
 return c

def reference(covs):
 m=np.mean(covs,axis=0)
 eig,U=np.linalg.eigh((m+m.T)/2)
 if not np.isfinite(eig).all() or np.any(eig<=0):raise ValueError("Source arithmetic reference not SPD")
 return (U*(eig**-0.5))@U.T

def tangent(covs,inv):
 z=np.einsum("ab,nbc,cd->nad",inv,covs,inv)
 z=(z+z.transpose(0,2,1))/2
 e,U=np.linalg.eigh(z)
 if not np.isfinite(e).all() or np.any(e<=0):raise ValueError("Whitened trial covariance not SPD")
 logm=(U*np.log(e)[:,None,:])@U.transpose(0,2,1)
 p=logm.shape[1];i,j=np.triu_indices(p)
 out=logm[:,i,j]
 out[:,i!=j]*=np.sqrt(2.)
 return out

def within(X,y):
 a=X[y==0];b=X[y==1]
 if len(a)<2 or len(b)<2:raise ValueError("Every training/validation block needs both classes")
 ma=a.mean(0);mb=b.mean(0)
 Z=np.concatenate([a-ma,b-mb],axis=0)
 S=(Z.T@Z)/(len(Z)-2)
 return (S+S.T)/2,mb-ma,(ma+mb)/2

def ridge(trainX,trainy,testX,lam):
 S,d,mid=within(trainX,trainy)
 nu=max(float(np.trace(S))/S.shape[0],np.finfo(float).eps)
 R=(1-lam)*S+lam*nu*np.eye(S.shape[0])
 w=np.linalg.solve(R,d)
 return (testX-mid)@w

def analytic(trainX,trainy,testX,cls):
 S,d,mid=within(trainX,trainy)
 # fit estimator to centered within-class residuals so mean shift is not covariance.
 a=trainX[trainy==0];b=trainX[trainy==1]
 resid=np.concatenate([a-a.mean(0),b-b.mean(0)],axis=0)
 est=cls(store_precision=False).fit(resid)
 R=(est.covariance_+est.covariance_.T)/2
 R+=np.eye(R.shape[0])*max(float(np.trace(R)),1.0)*1e-12
 w=np.linalg.solve(R,d)
 return (testX-mid)@w,float(est.shrinkage_)

def source_criteria(covs,y,groups):
 rows={lam:{"auc":[],"signed":[]} for lam in GRID}
 unique=sorted(set(map(int,groups)))
 for held in unique:
  tr=groups!=held;va=~tr
  if len(np.unique(y[va]))!=2:raise ValueError("Source validation subject lacks one class")
  inv=reference(covs[tr])
  Xtr=tangent(covs[tr],inv);Xva=tangent(covs[va],inv)
  Sva,dva,_=within(Xva,y[va])
  for lam in GRID:
   pred=ridge(Xtr,y[tr],Xva,lam)
   rows[lam]["auc"].append(float(roc_auc_score(y[va],pred)))
   Str,dtr,_=within(Xtr,y[tr])
   nu=max(float(np.trace(Str))/Str.shape[0],np.finfo(float).eps)
   R=(1-lam)*Str+lam*nu*np.eye(Str.shape[0])
   w=np.linalg.solve(R,dtr)
   den=float(w@Sva@w)
   if not np.isfinite(den) or den<=np.finfo(float).eps*max(1.,float(w@w)):
    raise ValueError("Degenerate held-source discriminant variance")
   rows[lam]["signed"].append(float(w@dva)/np.sqrt(den))
 k=max(1,int(math.ceil(TAIL*len(unique))))
 crit={}
 for lam in GRID:
  signed=np.asarray(rows[lam]["signed"])
  crit[lam]={
   "source_macro_auc":float(np.mean(rows[lam]["auc"])),
   "signed_mean":float(np.mean(signed)),
   "squared_margin":float(np.mean(signed**2)),
   "tail_signed":float(np.mean(np.sort(signed)[:k])),
   "negative_fraction":float(np.mean(signed<0)),
   "per_source_auc":list(map(float,rows[lam]["auc"])),
   "per_source_signed_margin":list(map(float,rows[lam]["signed"]))}
 selected={
  "source_macro_auc":float(max(GRID,key=lambda x:(crit[x]["source_macro_auc"],x))),
  "signed_mean":float(max(GRID,key=lambda x:(crit[x]["signed_mean"],x))),
  "squared_margin":float(max(GRID,key=lambda x:(crit[x]["squared_margin"],x))),
  "tail_signed":float(max(GRID,key=lambda x:(crit[x]["tail_signed"],x)))}
 return selected,{str(k):v for k,v in crit.items()}

def score_fold(covs,y,groups,target):
 train=groups!=target;test=~train
 if target not in set(groups):raise ValueError("Unknown target")
 if target in set(groups[train]):raise AssertionError("Target leakage")
 selected,crit=source_criteria(covs[train],y[train],groups[train])
 inv=reference(covs[train])
 Xtr=tangent(covs[train],inv);Xte=tangent(covs[test],inv)
 scores={}
 lw,lwlam=analytic(Xtr,y[train],Xte,LedoitWolf)
 oa,oalam=analytic(Xtr,y[train],Xte,OAS)
 scores["LedoitWolf_LDA"]=lw;scores["OAS_LDA"]=oa
 scores["nested_source_macro_auc_LDA"]=ridge(Xtr,y[train],Xte,selected["source_macro_auc"])
 scores["source_signed_mean_LDA"]=ridge(Xtr,y[train],Xte,selected["signed_mean"])
 scores["source_squared_margin_LDA"]=ridge(Xtr,y[train],Xte,selected["squared_margin"])
 scores["source_tail_signed_LDA"]=ridge(Xtr,y[train],Xte,selected["tail_signed"])
 lr=LogisticRegression(C=1.0,solver="lbfgs",max_iter=3000)
 lr.fit(Xtr,y[train]);scores["LogisticRegression"]=lr.decision_function(Xte)
 return {
  "target_subject":int(target),
  "source_subjects":[int(x) for x in sorted(set(groups[train]))],
  "target_trials":int(test.sum()),
  "target_label_counts":{"left_hand":int(np.sum(y[test]==0)),"right_hand":int(np.sum(y[test]==1))},
  "auc":{k:float(roc_auc_score(y[test],v)) for k,v in scores.items()},
  "selected_lambda_by_source_only_criterion":selected,
  "source_criteria_by_lambda":crit,
  "LedoitWolf_shrinkage":lwlam,"OAS_shrinkage":oalam,
  "target_data_used_for_selection":False}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",default="defaultreg_bnci2014_001_loso")
 args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 t=time.monotonic()
 dataset=BNCI2014_001(artifact_handling="ignore")
 paradigm=MotorImagery(events=["left_hand","right_hand"],n_classes=2,
                       fmin=8,fmax=30,tmin=0.0,tmax=4.0,
                       baseline=None,reject_by_annotation=False)
 X,labels,metadata=paradigm.get_data(dataset=dataset,subjects=list(SUBJECTS))
 if X.ndim!=3:raise ValueError("Unexpected MOABB output shape "+str(X.shape))
 labels=np.asarray(labels)
 if set(labels)!={"left_hand","right_hand"}:raise ValueError("Unexpected labels "+str(set(labels)))
 y=(labels=="right_hand").astype(np.int64)
 groups=np.asarray(metadata["subject"],dtype=int)
 if set(groups)!=set(SUBJECTS):raise ValueError("Missing BNCI subjects "+str(set(groups)))
 counts={str(s):int(np.sum(groups==s)) for s in SUBJECTS}
 if len(set(counts.values()))!=1:raise ValueError("Unequal trial denominator across subjects "+str(counts))
 covs=covariances(X)
 folds=[score_fold(covs,y,groups,target) for target in SUBJECTS]
 auc_matrix={m:[f["auc"][m] for f in folds] for m in METHODS}
 primary=np.asarray(auc_matrix["source_tail_signed_LDA"])
 nested=np.asarray(auc_matrix["nested_source_macro_auc_LDA"])
 paired=primary-nested
 summary={
  "status":"REAL_BNCI2014_001_FULL_LOSO_COMPLETE",
  "dataset":"BNCI2014_001 / BCI Competition IV 2a",
  "subjects":list(SUBJECTS),"n_subjects":len(SUBJECTS),
  "task":"left_hand_vs_right_hand","band_hz":[8,30],"epoch_seconds":[0,4],
  "artifact_handling":"ignore","reject_by_annotation":False,
  "target_batch_statistics_used":False,
  "target_labels_used_before_scoring":False,
  "trial_count_by_subject":counts,
  "feature_dimension":int(covs.shape[1]*(covs.shape[1]+1)//2),
  "primary_new_method":"source_tail_signed_LDA",
  "primary_baseline":"nested_source_macro_auc_LDA",
  "mean_auc_by_method":{m:float(np.mean(v)) for m,v in auc_matrix.items()},
  "primary_minus_nested_mean_auc":float(np.mean(paired)),
  "primary_minus_nested_median_auc":float(np.median(paired)),
  "primary_minus_nested_min_auc":float(np.min(paired)),
  "primary_harmed_by_more_than_0_02":int(np.sum(paired < -0.02)),
  "primary_improved_subjects":int(np.sum(paired>0)),
  "primary_tied_subjects":int(np.sum(paired==0)),
  "folds":folds,
  "environment":{"python":sys.version,"platform":platform.platform()},
  "wall_seconds":round(time.monotonic()-t,2),
  "interpretation":"External real-dataset stratum. N=9 subjects. No cross-paper SOTA inference."}
 (out/"bnci2014_001_loso.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
 print(json.dumps(summary,indent=2,sort_keys=True))
 return summary

if __name__=="__main__":main()
