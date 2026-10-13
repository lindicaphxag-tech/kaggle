#!/usr/bin/env python3
"""Real public NEMAR v1.0.3 EDF verification, without raw file artifacts.
Recompute full-file sha256 from publicly hosted bytes; compare as-stored
per-channel EDF digital samples for 04/05, 11/15, 47/49.
This is the same Liu2024 study, NOT independent patient acquisitions.
"""
import argparse,hashlib,json,struct,time,urllib.request
from pathlib import Path

IDS=(4,5,11,15,47,49)
COMMIT="11e3eeb4b439d5f8ee1b5e3b8e6032824bc2de83"
ROOT="https://data.nemar.org/nm000158/v1.0.3"
MANIFEST=("https://raw.githubusercontent.com/nemarDatasets/nm000158/"
          +COMMIT+"/sourcedata/sourcedata_provenance.json")

def field(b,i,j):
    return b[i:j].decode("ascii",errors="strict").strip()

def number(x):
    val=int(x)
    if val<=0:raise ValueError("nonpositive EDF record count")
    return val

def parse_edf(blob):
    if len(blob)<256:raise ValueError("truncated header")
    hd=number(field(blob,184,192))
    records=number(field(blob,236,244))
    count=number(field(blob,252,256))
    if hd<256+256*count or hd>len(blob):
        raise ValueError("EDF header length")
    # Standard EDF per-channel header blocks:
    # label16 transducer80 physical_unit8 physical_min8
    # physical_max8 digital_min8 digital_max8 prefilter80
    # sample_count8 reserved32.
    off=256+(16+80+8+8+8+8+8+80)*count
    sizes=[number(field(blob,off+8*i,off+8*(i+1)))
           for i in range(count)]
    if len(blob)!=hd+2*records*sum(sizes):
        raise ValueError("EDF size / sample layout mismatch")
    streams=[hashlib.sha256() for _ in range(count)]
    pos=hd
    for _ in range(records):
        for i,k in enumerate(sizes):
            chunk=blob[pos:pos+2*k]
            if len(chunk)!=2*k:raise AssertionError("EDF record underflow")
            streams[i].update(chunk);pos+=2*k
    if pos!=len(blob):raise AssertionError("EDF record overflow")
    return {"channels":count,"records":records,"samples_per_record":sizes,
            "channel_digest":[h.hexdigest() for h in streams]}

def compare(x,y):
    if (x["channels"]!=y["channels"] or x["records"]!=y["records"]
        or x["samples_per_record"]!=y["samples_per_record"]):
        return {"comparable":False}
    aa=x["channel_digest"];bb=y["channel_digest"]
    return {"comparable":True,
            "same_index_exact_digital_channel_count":
                sum(a==b for a,b in zip(aa,bb)),
            "permutation_invariant_exact_channel_count":
                sum(min(aa.count(k),bb.count(k)) for k in set(aa)),
            "total_channels":len(aa)}

def fetch(url,limit):
    last=None
    for retry in range(3):
        try:
            req=urllib.request.Request(url,headers={
                "User-Agent":"public-EEG-numerical-content-audit/1.0"})
            with urllib.request.urlopen(req,timeout=120) as r:
                raw=r.read(limit+1)
            if len(raw)>limit:raise ValueError("oversized public asset")
            return raw
        except Exception as e:
            last=e;time.sleep(retry+1)
    raise RuntimeError("Verified public release cannot be fetched") from last

def selftest():
    def make(chans):
        n=len(chans);recs=2;per=3
        h=("0       "+" "*80+" "*80+"01.01.01"+"01.01.01"
           +str(256+256*n).ljust(8)+" "*44
           +str(recs).ljust(8)+"1       "+str(n).ljust(4))
        assert len(h)==256
        def fmt(width,seq):
            return "".join(str(x).ljust(width)[:width] for x in seq)
        meta=(fmt(16,[f"ch{i}" for i in range(n)])
              +fmt(80,[""]*n)+fmt(8,["uV"]*n)
              +fmt(8,[-50]*n)+fmt(8,[50]*n)
              +fmt(8,[-32768]*n)+fmt(8,[32767]*n)
              +fmt(80,[""]*n)+fmt(8,[per]*n)+fmt(32,[""]*n))
        assert len(meta)==256*n
        signal=b"".join(struct.pack("<"+"h"*per,*chans[j][i*per:(i+1)*per])
                        for i in range(recs) for j in range(n))
        return (h+meta).encode("ascii")+signal
    x=make([[1,2,3,4,5,6],[10,20,30,40,50,60]])
    y=make([[1,2,3,4,5,6],[10,20,30,40,50,61]])
    z=make([[10,20,30,40,50,60],[1,2,3,4,5,6]])
    assert compare(parse_edf(x),parse_edf(y))[
        "same_index_exact_digital_channel_count"]==1
    assert compare(parse_edf(x),parse_edf(z))[
        "permutation_invariant_exact_channel_count"]==2
    try:parse_edf(x[:-2])
    except ValueError:pass
    else:raise AssertionError("truncated EDF accepted")
    return {"status":"SYNTHETIC_EDF_PARSER_PASS","no_real_data_read":True}

def live():
    manifest=json.loads(fetch(MANIFEST,150000))
    rows=manifest["files"]
    ids={int(z["subject"]):z for z in rows}
    if len(rows)!=50 or len(ids)!=50 or manifest["moabb_version_used"]!="1.5.0":
        raise ValueError("unexpected versioned public manifest")
    stats={};digests={}
    for k in IDS:
        row=ids[k]
        path=f'{ROOT}/sourcedata/{row["file"]}'
        data=fetch(path,int(row["bytes"])+4096)
        sha=hashlib.sha256(data).hexdigest()
        if sha!=row["sha256"]:
            raise RuntimeError("independent whole-file digest mismatch")
        digests[k]=parse_edf(data)
        stats[str(k)]={"verified_whole_file_sha256":True,
                       "bytes":len(data),
                       "record_count":digests[k]["records"],
                       "digital_channel_count":digests[k]["channels"]}
    pairs=[]
    for a,b in ((4,5),(11,15),(47,49)):
        pairs.append({"public_ids":[a,b],
                      "different_whole_file_sha256":ids[a]["sha256"]!=ids[b]["sha256"],
                      **compare(digests[a],digests[b])})
    return {"status":"PUBLIC_NEMAR_EDF_BYTES_DIGITALLY_AUDITED",
            "manifest_commit":COMMIT,"independent_rehash_of_6_full_EDF_files":True,
            "public_file_results":stats,"pair_results":pairs,
            "source_dataset_is_Liu2024_not_independent_acquisition":True,
            "MAT_to_EDF_export_operator_unverified":True,
            "clinical_patient_identity_unverified":True,
            "no_EEG_waveforms_in_report":True}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--out",type=Path)
    arg=p.parse_args()
    if arg.self_test:z=selftest()
    else:
        if not arg.out:raise SystemExit("--out required for real public audit")
        z=live()
    s=json.dumps(z,sort_keys=True,indent=2)+"\n"
    print(s)
    if arg.out:
        arg.out.parent.mkdir(parents=True,exist_ok=True)
        arg.out.write_text(s)
