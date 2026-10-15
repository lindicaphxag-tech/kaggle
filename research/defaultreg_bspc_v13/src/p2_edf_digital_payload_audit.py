#!/usr/bin/env python3
"""Independent EDF 16-bit raw digital payload audit: no MNE or pyEDFlib parser.
Invalid physical calibration header fields do not affect digital content
comparison. It does NOT prove physical values or patient identity.
"""
from pathlib import Path
import json,argparse
from moabb.datasets import Liu2024

def number(b):
    return int(b.decode("ascii","replace").strip())
def header(path):
    path=Path(path)
    with path.open("rb") as f:
        h=f.read(256)
        if len(h)!=256:raise ValueError("short EDF header")
        n=number(h[252:256]);hb=number(h[184:192])
        records=number(h[236:244])
        if not 0<n<4096 or hb!=256+256*n:raise ValueError("EDF header schema")
        sh=f.read(n*256)
    lengths=[16,80,8,8,8,8,8,80,8,32]
    fields=[];off=0
    for width in lengths:
        fields.append([sh[off+i*width:off+(i+1)*width] for i in range(n)])
        off+=width*n
    if off!=len(sh):raise ValueError("EDF per-signal header truncated")
    names=[s.decode("ascii","replace").strip() for s in fields[0]]
    count=[number(s) for s in fields[8]]
    if min(count)<1:raise ValueError("invalid samples per channel")
    bpr=2*sum(count);remaining=path.stat().st_size-hb
    if remaining<0 or remaining%bpr:raise ValueError("EDF unaligned payload")
    actual=remaining//bpr
    if records not in (-1,actual):raise ValueError("inconsistent record count")
    return {"n":n,"names":names,"samples":count,"records":actual,"header":hb}

def compare(path_a,path_b):
    A,B=header(path_a),header(path_b)
    if any(A[k]!=B[k] for k in ("n","names","samples","records")):
        raise ValueError("incompatible EDF digital structures")
    eq=[True]*A["n"]
    with open(path_a,"rb") as fa,open(path_b,"rb") as fb:
        fa.seek(A["header"]);fb.seek(B["header"])
        for _ in range(A["records"]):
            for j,n in enumerate(A["samples"]):
                a=fa.read(2*n);b=fb.read(2*n)
                if len(a)!=2*n or len(b)!=2*n:raise IOError("EDF premature EOF")
                if a!=b:eq[j]=False
    return {
      "n_raw_channels":A["n"],
      "n_identical_digital_channels":sum(eq),
      "different_digital_channel_indices":[i for i,v in enumerate(eq) if not v],
      "different_channel_names":[A["names"][i] for i,v in enumerate(eq) if not v],
      "comparison":"byte-for-byte raw EDF digital int16 sample payload",
      "not_physical_amplitude_or_patient_identity_confirmation":True,
      "reader_independent_of_mne_and_pyedflib":True}

def main():
    pa=argparse.ArgumentParser();pa.add_argument("--out",type=Path,required=True)
    o=pa.parse_args();ds=Liu2024()
    pairs=[]
    for a,b in ((4,5),(47,49)):
        q=compare(ds.data_path(a)[0],ds.data_path(b)[0])
        q["released_ids"]=[a,b];pairs.append(q)
        print("EDF_DIGITAL_PAIR_AUDITED",a,b,q["n_identical_digital_channels"],flush=True)
    result={"status":"RAW_EDF_DIGITAL_PAYLOAD_INDEPENDENT_CHECK",
            "limitations":"Malformed physical calibration header may prevent EDFlib reader; digital payload compared only",
            "public_data_not_identifiable_patient_proof":True,"pairs":pairs}
    o.out.parent.mkdir(parents=True,exist_ok=True)
    o.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
