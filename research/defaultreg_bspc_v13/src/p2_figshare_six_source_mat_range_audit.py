#!/usr/bin/env python3
"""P2 Figshare v5 source MAT provenance: range-limited, public EEG signals only.

Archives: published Liu2024 Figshare article 21679035 version 5, source-data
file 38516555. Read exactly six public MAT members (IDs 4,5,11,15,47,49),
never the full ~1.875 GB ZIP. Reject HTTP 200 for Range requests. Strict
ZIP central-directory, local-header, payload length and CRC-32 validation.
Keep EEG arrays in temporary runner storage; artifacts export ONLY aggregate
signal comparisons, no raw EEG, clinical header or patient metadata.

This is retrospective analysis of public release products, NOT independent
clinical patient/session identification, a causal leakage estimate, or an
analysis of independent prospective acquisitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import tempfile
import zipfile
import zlib
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from p2_figshare_sourcedata_remote_zip_index import FILE_ID, URL

IDS=(4,5,11,15,47,49)
PAIRS=((4,5),(47,49),(11,15))
MAX_MEMBER_UNCOMPRESSED=500_000_000
MAX_TOTAL_REMOTE_BYTES=1_100_000_000
CHUNK_BYTES=2*1024*1024

class StrictRange:
    def __init__(self,url,local_bytes=None):
        self.url=url
        self.local_bytes=local_bytes
        self.total_received=0
        self.remote_total=None
    def get(self,start,end):
        if start>=0 and (end is None or end<start):
            raise ValueError("empty/invalid forward range")
        if start<0 and end is not None:
            raise ValueError("suffix HTTP range cannot have explicit end")
        expected=(end-start+1) if start>=0 else -start
        if self.local_bytes is not None and start<0:
            expected=min(expected,len(self.local_bytes))
        if expected>CHUNK_BYTES+131072:
            raise ValueError("requested range too large")
        self.total_received+=expected
        if self.total_received>MAX_TOTAL_REMOTE_BYTES:
            raise RuntimeError("strict cumulative network range budget exceeded")
        if self.local_bytes is not None:
            body=self.local_bytes[start:end+1] if start>=0 else self.local_bytes[start:]
            if len(body)!=expected:raise IOError("short synthetic local range")
            return body,len(self.local_bytes)
        span=f"bytes={start}-{end}" if start>=0 else f"bytes={start}"
        request=Request(self.url,headers={
            "Range":span,"Accept-Encoding":"identity",
            "User-Agent":"P2-public-six-source-mat-range-comparison/1.0"
        })
        with urlopen(request,timeout=120) as rsp:
            if rsp.status!=206:
                raise RuntimeError(f"Range ignored (HTTP {rsp.status}); refuse full archive")
            hdr=rsp.headers.get("Content-Range","")
            m=re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)",hdr.strip())
            if not m:raise ValueError("Content-Range missing or invalid")
            actual_start,actual_end,total=map(int,m.groups())
            want_start=start if start>=0 else total+start
            want_end=end if start>=0 else total-1
            if (actual_start,actual_end)!=(want_start,want_end):
                raise ValueError("HTTP Content-Range did not match requested bytes")
            data=rsp.read(expected+1)
            if len(data)!=expected:raise IOError("incomplete or oversized range body")
            if self.remote_total is not None and total!=self.remote_total:
                raise ValueError("remote archive size changed during checks")
            self.remote_total=total
            return data,total

def directory(reader):
    tail,total=reader.get(-131072,None)
    marker=bytes.fromhex("504b0506")
    idx=tail.rfind(marker)
    if idx<0 or idx+22>len(tail):raise ValueError("ZIP central directory end missing")
    end=struct.unpack_from("<I4H2IH",tail,idx)
    _,disk,cdisk,on_disk,count,length,offset,comment=end
    if disk or cdisk or on_disk!=count or offset==0xFFFFFFFF or length>32_000_000:
        raise ValueError("unsupported ZIP topology")
    raw,_=reader.get(offset,offset+length-1)
    result=[]
    pos=0
    for _ in range(count):
        if raw[pos:pos+4]!=bytes.fromhex("504b0102"):
            raise ValueError("invalid central header")
        flags,method=struct.unpack_from("<HH",raw,pos+8)
        crc,comp,uncomp=struct.unpack_from("<III",raw,pos+16)
        nlen,xlen,clen=struct.unpack_from("<HHH",raw,pos+28)
        loff=struct.unpack_from("<I",raw,pos+42)[0]
        name_bytes=raw[pos+46:pos+46+nlen]
        name=name_bytes.decode("utf-8" if flags&0x800 else "cp437")
        if ".." in Path(name).parts:raise ValueError("unsafe archive member path")
        result.append({"name":name,"flags":flags,"method":method,
                       "crc":crc,"compressed":comp,"uncompressed":uncomp,
                       "local_offset":loff})
        pos+=46+nlen+xlen+clen
    if len(result)!=count or pos!=len(raw):
        raise ValueError("central directory entry accounting mismatch")
    return result,total

def write_verified_member(reader,member,destination):
    if member["flags"]&1 or member["method"] not in (0,8):
        raise ValueError("encrypted or unsupported ZIP method")
    if member["compressed"]==0xFFFFFFFF or member["uncompressed"]==0xFFFFFFFF:
        raise ValueError("ZIP64 per-member unsupported")
    if member["uncompressed"]>MAX_MEMBER_UNCOMPRESSED:
        raise ValueError("source MAT member too large; fail closed")
    off=member["local_offset"]
    hdr,_=reader.get(off,off+29)
    if hdr[:4]!=bytes.fromhex("504b0304"):raise ValueError("invalid local header")
    flag,method=struct.unpack_from("<HH",hdr,6)
    name_length,extra_length=struct.unpack_from("<HH",hdr,26)
    if method!=member["method"] or flag!=member["flags"]:
        raise ValueError("local-central method/flag drift")
    name_data,_=reader.get(off+30,off+30+name_length-1)
    expected_name=member["name"].encode("utf-8" if flag&0x800 else "cp437")
    if name_data!=expected_name:
        raise ValueError("local-central member name drift")
    begin=off+30+name_length+extra_length
    end=begin+member["compressed"]
    decomp=zlib.decompressobj(-15) if method==8 else None
    crc=0;sha=hashlib.sha256();n=0
    with Path(destination).open("wb") as fout:
        cursor=begin
        while cursor<end:
            stop=min(end,cursor+CHUNK_BYTES)
            chunk,_=reader.get(cursor,stop-1)
            cursor=stop
            decoded=decomp.decompress(chunk) if decomp else chunk
            n+=len(decoded)
            if n>MAX_MEMBER_UNCOMPRESSED:raise ValueError("unexpected inflate size")
            sha.update(decoded);crc=zlib.crc32(decoded,crc);fout.write(decoded)
        if decomp:
            final=decomp.flush()
            n+=len(final)
            sha.update(final);crc=zlib.crc32(final,crc);fout.write(final)
            if not decomp.eof or decomp.unused_data:
                raise ValueError("invalid deflate stream")
    if n!=member["uncompressed"] or (crc&0xFFFFFFFF)!=member["crc"]:
        raise ValueError("source MAT uncompressed bytes / CRC mismatch")
    return {"sha256":sha.hexdigest(),"uncompressed_bytes":n,
            "zip_crc32_matches":True,"zip_compressed_bytes":member["compressed"]}

def candidate_arrays(file):
    """No patient IDs/metadata or source waveforms are returned."""
    import scipy.io
    arrays=[]
    def add(x,depth=0):
        if depth>5 or len(arrays)>100: return
        if isinstance(x,np.ndarray):
            if x.dtype.names:
                for key in x.dtype.names:
                    add(x[key],depth+1)
            elif x.dtype==object:
                for item in x.flat[:40]:add(item,depth+1)
            elif x.dtype.kind in ("i","u","f") and x.size>=2048 and x.ndim>=2:
                if np.all(np.isfinite(x)):
                    arrays.append(np.asarray(x))
        elif hasattr(x,"_fieldnames"):
            for key in x._fieldnames:
                add(getattr(x,key),depth+1)
    backend="scipy.io.loadmat"
    try:
        o=scipy.io.loadmat(file,struct_as_record=False,squeeze_me=False,
                          simplify_cells=False)
        for key,val in o.items():
            if key.startswith("__"):continue
            add(val)
    except (NotImplementedError,ValueError) as ex:
        # v7.3 MAT stores HDF5; scan public numeric datasets, not metadata.
        import h5py
        backend="h5py-MAT-v7.3"
        with h5py.File(file,"r") as f:
            def walk(name,node):
                if isinstance(node,h5py.Dataset) and len(node.shape)>=2 and np.prod(node.shape)>=2048:
                    if node.dtype.kind in "iuf" and node.size<=80_000_000:
                        arr=node[()]
                        if np.isfinite(arr).all():arrays.append(arr)
            f.visititems(walk)
    arrays.sort(key=lambda a:(-a.size,tuple(a.shape)))
    # Avoid duplicating candidates of identical content in one MAT struct.
    rows=[]; unique=set()
    for x in arrays:
        if x.nbytes>500_000_000:continue
        little=np.ascontiguousarray(x)
        # Do not export content-hashes: these are only used internally.
        signature=hashlib.sha256(little.tobytes()).hexdigest()
        key=(little.shape,signature)
        if key in unique:continue
        unique.add(key)
        rows.append((little,signature))
    return backend,rows[:12]

def numeric_compare(a,b):
    """Same shape/dtype, normalized correlation without exporting waveforms."""
    x=np.asarray(a);y=np.asarray(b)
    out={"same_shape":x.shape==y.shape,
         "same_dtype":str(x.dtype)==str(y.dtype),
         "n_elements_a":int(x.size),"n_elements_b":int(y.size)}
    if x.shape!=y.shape:
        return out
    out["exact_numeric_equality"]=bool(np.array_equal(x,y))
    # Strictly the same scalar coordinate ordering.
    if x.size>=2 and np.issubdtype(x.dtype,np.number) and np.issubdtype(y.dtype,np.number):
        # Stride over long vectors to bound temporary memory; deterministic and transparent.
        z=np.asarray(x,dtype=np.float64).reshape(-1)
        w=np.asarray(y,dtype=np.float64).reshape(-1)
        if not(np.isfinite(z).all() and np.isfinite(w).all()):
            out["pearson"]=None
        else:
            z=z-z.mean();w=w-w.mean()
            denom=np.linalg.norm(z)*np.linalg.norm(w)
            out["pearson"]=float(np.clip(np.dot(z,w)/denom,-1,1)) if denom>0 else None
    return out

def run(reader):
    entries,total=directory(reader)
    select={}
    for i in IDS:
        wanted=f"sourcedata/sub-{i:02d}/sub-{i:02d}_task-motor-imagery_eeg.mat"
        matches=[e for e in entries if e["name"]==wanted]
        if len(matches)!=1:raise ValueError(f"missing/ambiguous public source MAT for ID {i}")
        select[i]=matches[0]
    meta={"status":"EXECUTED_SOURCE_MAT_SIX_PUBLIC_FILES_RANGE_AUDIT",
          "official_figshare_file_id":FILE_ID,
          "full_archive_bytes":total,
          "full_archive_not_downloaded":True,
          "request_range_206_required":reader.local_bytes is None,
          "only_six_selected_source_MAT_members_read":True,
          "patient_identity_or_session_not_inferred":True,
          "no_eeg_waveform_content_or_patient_metadata_exported":True,
          "data_seen_before_current_audit_not_prospective":True,
          "per_file":[],
          "pair_comparisons":[]}
    with tempfile.TemporaryDirectory() as t:
        saved={}
        for i in IDS:
            path=Path(t)/f"public-id-{i}.mat"
            proof=write_verified_member(reader,select[i],path)
            saved[i]=path
            meta["per_file"].append({"released_public_id":i,
                                     "source_mat_filename":select[i]["name"],
                                     **proof})
            print("P2_SOURCE_MAT_RANGE_EXTRACTED",i,proof["uncompressed_bytes"],flush=True)
        for ia,ib in PAIRS:
            back_a,arr_a=candidate_arrays(saved[ia])
            back_b,arr_b=candidate_arrays(saved[ib])
            scores=[]
            for ai,(a,_) in enumerate(arr_a[:8]):
                for bi,(b,_) in enumerate(arr_b[:8]):
                    if a.size<2048 or b.size<2048:continue
                    v=numeric_compare(a,b)
                    if v["same_shape"]:
                        scores.append({"source_numeric_candidate_a":ai,
                                       "source_numeric_candidate_b":bi,
                                       "shape":list(a.shape),"dtype_a":str(a.dtype),
                                       "dtype_b":str(b.dtype),**v})
            scores.sort(key=lambda z:(not z.get("exact_numeric_equality",False),
                                     -(z.get("pearson") if z.get("pearson") is not None else -2),
                                     -z["n_elements_a"]))
            result={"released_public_ids":[ia,ib],
                    "original_mat_containers_byte_identical":
                     next(x["sha256"] for x in meta["per_file"] if x["released_public_id"]==ia)==
                     next(x["sha256"] for x in meta["per_file"] if x["released_public_id"]==ib),
                    "candidate_numeric_arrays_a":len(arr_a),
                    "candidate_numeric_arrays_b":len(arr_b),
                    "mat_loader_a":back_a,"mat_loader_b":back_b,
                    "same_shape_numeric_candidate_comparisons":len(scores),
                    "best_matching_numeric_array":scores[0] if scores else None,
                    "all_reported_numeric_candidates_are_unverified_EEG_semantics":True}
            meta["pair_comparisons"].append(result)
            print("P2_SOURCE_MAT_PAIR_ANALYZED",ia,ib,"matched-shapes",len(scores),
                  "best",result["best_matching_numeric_array"],flush=True)
    meta["total_http_range_body_bytes"]=reader.total_received
    meta["no_source_mat_to_processed_EDF_alignment_proven"]=True
    return meta

def self_test():
    import scipy.io
    with tempfile.TemporaryDirectory() as t:
        t=Path(t)
        members={}
        original=np.arange(9000,dtype=np.float64).reshape(3,3000)
        for i in IDS:
            path=t/f"{i}.mat"
            data=original if i in (4,5) else original+i
            scipy.io.savemat(path,{"eeg":data})
            members[i]=path.read_bytes()
        zp=t/"archive.zip"
        with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
            for i,payload in members.items():
                z.writestr(f"sourcedata/sub-{i:02d}/sub-{i:02d}_task-motor-imagery_eeg.mat",payload)
        reader=StrictRange(URL,local_bytes=zp.read_bytes())
        result=run(reader)
        assert len(result["pair_comparisons"])==3
        assert result["pair_comparisons"][0]["best_matching_numeric_array"]["exact_numeric_equality"]
        assert result["pair_comparisons"][1]["best_matching_numeric_array"]["exact_numeric_equality"] is False
        assert result["total_http_range_body_bytes"]<len(zp.read_bytes())*2
        print("P2_SOURCE_MAT_RANGE_ONLY_SYNTHETIC_EXTRACTION_AND_COMPARE_PASS",flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if args.out is None:parser.error("--out is required for real public source MAT scan")
    o=run(StrictRange(URL))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+"\n"
    args.out.write_text(body,encoding="utf-8")
    print("P2_SOURCE_MAT_SIXPAIR_AUDIT_SHA256",hashlib.sha256(body.encode()).hexdigest(),flush=True)

if __name__=="__main__":
    main()
