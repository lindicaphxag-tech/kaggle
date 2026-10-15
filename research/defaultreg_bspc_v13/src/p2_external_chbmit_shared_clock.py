#!/usr/bin/env python3
"""External CHB-MIT 1.0.0 real EDF digital EEG, frozen shared-clock test.

Two distinct subjects, one file each; public EDF *prefixes* obtained using HTTP Range.
Clinical seizure annotations and subject identity are never used.
The goal is matching a coherent multi-sensor DIGITAL RECORDING, not any single-sensor reuse.
"""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
from urllib.request import Request,urlopen
import numpy as np
from scipy.signal import correlate,correlation_lags
from scipy.optimize import linear_sum_assignment
OUT=Path(os.environ.get("P2_OUTPUT","p2_chbmit_external"));OUT.mkdir(parents=True,exist_ok=True)
TH=.985;K=4;L=100;MINLEN=230;SEED=20261015
FILES=["chb01/chb01_01.edf","chb02/chb02_01.edf"]
def field(a):
    return a.decode("ascii","replace").strip()
def read_one(rel):
    url="https://physionet.org/files/chbmit/1.0.0/"+rel
    nbytes=2400000
    req=Request(url,headers={"Range":f"bytes=0-{nbytes-1}","User-Agent":"research-repro/1.0"})
    with urlopen(req,timeout=110) as resp:
        status=resp.status
        buf=resp.read(nbytes) # never redistribute raw EEG
        content_range=resp.headers.get("Content-Range","")
        etag=resp.headers.get("ETag","")
    if len(buf)<200000:raise RuntimeError(f"unexpected EDF prefix {rel}: {len(buf)}")
    ns=int(field(buf[252:256]))
    header_bytes=int(field(buf[184:192]))
    assert 4<=ns<=128 and header_bytes==256+ns*256,(ns,header_bytes)
    dur=float(field(buf[244:252]))
    start=256+216*ns
    samples=[int(field(buf[start+i*8:start+(i+1)*8])) for i in range(ns)]
    labels=[field(buf[256+i*16:256+(i+1)*16]) for i in range(ns)]
    record_size=2*sum(samples)
    assert (header_bytes+54*record_size)<=len(buf),("not enough prefix",len(buf),record_size)
    assert dur>0 and len(set(samples))<8
    channels=[]
    for i,label in enumerate(labels):
        if "EEG" in label.upper() or "-" in label:
            if "ECG" not in label.upper() and "VNS" not in label.upper() and label not in [q[1] for q in channels]:
                channels.append((i,label))
        if len(channels)>=8:break
    assert len(channels)==8,(rel,labels)
    def epoch(first_record):
        dig={name:[] for _,name in channels}
        for r in range(first_record,first_record+3):
            pos=header_bytes+r*record_size
            record=np.frombuffer(buf,dtype="<i2",count=sum(samples),offset=pos)
            acc=np.r_[0,np.cumsum(samples)]
            for ix,name in channels:
                z=record[acc[ix]:acc[ix+1]].astype(np.float64)
                assert len(z)==256,(rel,ix,len(z))
                dig[name].append(z)
        return {name:np.concatenate(v) for name,v in dig.items()}
    # First epoch 4--7s, independent same-file negative epoch 41--44s.
    return epoch(4),epoch(41),{"url":url,"prefix_bytes":len(buf),
           "sha256_prefix":hashlib.sha256(buf).hexdigest(),
           "sha256_edf_header":hashlib.sha256(buf[:header_bytes]).hexdigest(),
           "http_status":status,"content_range":content_range,"etag":etag,
           "edf_labels":labels,"chosen_eeg_labels":[n for _,n in channels],
           "record_duration":dur,"samples_per_record":samples,
           "note":"prefix only; full-file SHA256 NOT verified"}
def corr_grid(target,source):
    ns=len(source);nt=len(target)
    lag=correlation_lags(nt,ns,"full")
    a=np.maximum(0,-lag);b=np.maximum(0,lag)
    n=np.minimum(ns-a,nt-b)
    x=np.r_[0,np.cumsum(source)];x2=np.r_[0,np.cumsum(source*source)]
    y=np.r_[0,np.cumsum(target)];y2=np.r_[0,np.cumsum(target*target)]
    mx=x[a+n]-x[a];my=y[b+n]-y[b]
    vx=np.maximum(x2[a+n]-x2[a]-mx*mx/n,0)
    vy=np.maximum(y2[b+n]-y2[b]-my*my/n,0)
    dot=correlate(target,source,mode="full",method="fft")-mx*my/n
    denom=np.sqrt(vx*vy)
    valid=(np.abs(lag)<=L)&(n>=MINLEN)
    p=lag[valid];a=np.divide(dot[valid],denom[valid],
            out=np.zeros(np.sum(valid)),where=denom[valid]>1e-13)
    return p,np.abs(a)
def three_matchers(src,tgt):
    ss=list(src);tt=list(tgt)
    grid=np.zeros((2*L+1,len(ss),len(tt)),dtype=np.float32)
    for i,k in enumerate(ss):
        for j,v in enumerate(tt):
            l,c=corr_grid(tgt[v],src[k]);grid[l+L,i,j]=c
    shared=False
    for idx in range(2*L+1):
        m=np.where(grid[idx]>=TH,grid[idx],0)
        if (m.max(axis=1)>=TH).sum()<K:continue
        ii,jj=linear_sum_assignment(-m)
        if np.count_nonzero(m[ii,jj]>=TH)>=K:
            shared=True;break
    independent=grid.max(axis=0)
    independent=np.where(independent>=TH,independent,0)
    i,j=linear_sum_assignment(-independent)
    independent_ans=np.count_nonzero(independent[i,j]>=TH)>=K
    named_zero=sum(k in tgt and grid[L,ix,tt.index(k)]>=TH
                   for ix,k in enumerate(ss))>=K
    return {"common_clock":int(shared),"independent_lag_assignment":int(independent_ans),
            "named_zero_lag":int(named_zero)}
def transform(src,other,kind,seed):
    rng=np.random.default_rng(seed)
    items=list(src.items())
    if kind=="exact":return {k:v.copy() for k,v in items}
    if kind=="relabel_permute":return {"R"+str(i):v.copy() for i,(_,v) in enumerate(items[::-1])}
    if kind=="global_lag":return {k:np.r_[rng.normal(0,np.std(v),31),v[:-31]] for k,v in items}
    if kind=="crop":return {k:v[61:702].copy() for k,v in items}
    if kind=="gain":return {k:v*(.8+.035*i)+v.mean()*.1 for i,(k,v) in enumerate(items)}
    if kind=="drop":return dict(items[:5])
    if kind=="noise":return {k:v+rng.normal(0,np.std(v)*.015,len(v)) for k,v in items}
    if kind=="combined":
        return {"X"+str(i):v[43:701]*(.7+.03*i)+rng.normal(0,np.std(v)*.005,658)
                for i,(_,v) in enumerate(items[::-1][:6])}
    if kind=="different_subject":return {k:v.copy() for k,v in zip(src,list(other.values()))}
    if kind=="same_file_different_time":return {k:v.copy() for k,v in zip(src,list(other.values()))}
    if kind=="one_shared_channel":
        t={k:v.copy() for k,v in zip(src,list(other.values()))}
        k=items[0][0];t[k]=src[k].copy();return t
    if kind=="async_same_source":
        delays=[7,23,39,55,71,87,13,29]
        return {"Z"+str(i):np.r_[rng.normal(0,np.std(v),delays[i]),v[:-delays[i]]]
                for i,(_,v) in enumerate(items)}
    if kind=="repeated_constant_artifact":
        t={k:v.copy() for k,v in zip(src,list(other.values()))}
        wave=np.sin(np.arange(768)*2*np.pi/31)
        return {k:v+.15*np.std(v)*wave for k,v in t.items()}
    raise KeyError(kind)
def main():
    inputs={}
    for rel in FILES:
        z,remote,meta=read_one(rel);inputs[rel]=(z,remote,meta)
    cases=[]
    positives=["exact","relabel_permute","global_lag","crop","gain","drop","noise","combined"]
    negatives=["different_subject","same_file_different_time","one_shared_channel","async_same_source","repeated_constant_artifact"]
    for p,rel in enumerate(FILES):
        src,same_file,meta=inputs[rel]
        another=inputs[FILES[1-p]][0]
        for ix,kind in enumerate(positives+negatives):
            neg=same_file if kind=="same_file_different_time" else another
            target=transform(src,neg,kind,SEED+p*31+ix)
            preds=three_matchers(src,target)
            cases.append({"source":rel,"kind":kind,"clock_coherent_positive":int(ix<len(positives)),"pred":preds})
    methods=list(cases[0]["pred"])
    summary={}
    for name in methods:
        summary[name]={"positive_hits":sum(z["clock_coherent_positive"] and z["pred"][name] for z in cases),
            "positive_total":sum(z["clock_coherent_positive"] for z in cases),
            "false_positive":sum((not z["clock_coherent_positive"]) and z["pred"][name] for z in cases),
            "negative_total":sum(not z["clock_coherent_positive"] for z in cases)}
    output={"status":"REAL_PUBLISHED_CHBMIT_DIGITAL_PREFIX_CONTROLLED_TRANSFORMS",
        "dataset":"CHB-MIT 1.0.0","dataset_doi":"10.13026/C2K01R",
        "source_corpus_independent_of_eegmmidb":True,
        "endpoints":"same coherent multichannel recording clock; any independent channel reuse is different endpoint",
        "pre_frozen":{"threshold_abs_correlation":TH,"minimum_channels":K,"max_abs_shift":L,"minimum_overlap":MINLEN},
        "limitations":["2 source persons, all positives experimentally transformed",
                       "counterfactual async negative is positive for any-channel-content reuse",
                       "no independent natural clinical duplicate ground truth",
                       "downloaded EDF prefixes not full-file hashes",
                       "common-clock constraint is a known generic signal-synchronization concept"],
        "EDF_prefix_provenance":{rel:inputs[rel][2] for rel in FILES},
        "summary":summary,"cases":cases}
    (OUT/"CHBMIT_EXTERNAL_COHERENCE_20261015.json").write_text(json.dumps(output,indent=2)+"\n")
    print("CHBMIT_EXTERNAL_REPRODUCIBLE",json.dumps(summary))
if __name__=="__main__":
    main()
