#!/usr/bin/env python3
"""Fetch only the ZIP central directory of the official Figshare v5 sourcedata.zip.

Must use HTTP byte ranges (206 Partial Content) and refuses a full-body
200 response: no inadvertent ~1.875 GB source download. This inventories
public FILE NAMES ONLY; does NOT open MAT data or clinical metadata.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
import struct
import tempfile
import zipfile
from pathlib import Path

FILE_ID=38516555
URL=f"https://ndownloader.figshare.com/files/{FILE_ID}"

def decode_zip_central(tail: bytes,read_range):
    marker=bytes.fromhex("504b0506")
    where=tail.rfind(marker)
    if where<0 or where+22>len(tail):raise ValueError("ZIP end of central directory missing")
    _,disk,central_disk,n_disk,n_entries,nbytes,offset,comment=struct.unpack_from("<I4H2IH",tail,where)
    if disk!=0 or central_disk!=0 or n_disk!=n_entries:
        raise ValueError("unsupported multi-disk ZIP")
    if nbytes<=0 or nbytes>32_000_000 or offset==0xFFFFFFFF:
        raise ValueError("unsupported ZIP64/oversized central directory; do not download")
    central=read_range(offset,offset+nbytes-1)
    if len(central)!=nbytes:raise IOError("truncated ZIP central directory range")
    entries=[];pos=0
    for i in range(n_entries):
        if central[pos:pos+4]!=bytes.fromhex("504b0102"):
            raise ValueError("ZIP central directory signature mismatch")
        size_compressed,size_uncompressed=struct.unpack_from("<II",central,pos+20)
        l_name,l_extra,l_comment=struct.unpack_from("<HHH",central,pos+28)
        local_offset=struct.unpack_from("<I",central,pos+42)[0]
        raw=central[pos+46:pos+46+l_name]
        try:name=raw.decode("utf-8")
        except UnicodeDecodeError:name=raw.decode("cp437")
        entries.append({"name":name,"compressed_bytes":size_compressed,"uncompressed_bytes":size_uncompressed,
                        "local_header_offset":local_offset})
        pos+=46+l_name+l_extra+l_comment
        if pos>len(central):raise ValueError("invalid ZIP central directory record size")
    if len(entries)!=n_entries:raise AssertionError("ZIP entry count drift")
    return entries

def public_index():
    from urllib.request import Request,urlopen
    endpoint=URL
    def request_range(first,last):
        ranges=f"bytes={first}-{last}" if first>=0 else f"bytes={first}"
        req=Request(endpoint,headers={"Range":ranges,"User-Agent":"P2-ZIP-directory-range-audit/1.0"})
        with urlopen(req,timeout=90) as response:
            status=response.status
            if status!=206:
                raise RuntimeError(f"HTTP {status}: server did not respect Range; refusing full archive download")
            typ=response.headers.get("Content-Range","")
            if not typ.startswith("bytes "):
                raise RuntimeError("Missing Content-Range in response")
            body=response.read(131073 if first<0 else (last-first+2))
            if first>=0 and len(body)!=last-first+1:
                raise RuntimeError("Incomplete requested central-directory bytes")
            return body,typ
    tail,source_range=request_range(-131072,None)
    match=re.search(r"/([0-9]+)$",source_range)
    if not match:raise ValueError("no content-size in HTTP Content-Range")
    total=int(match.group(1))
    def reader(l,r):
        z,_=request_range(l,r)
        return z
    entries=decode_zip_central(tail,reader)
    candidates=[e for e in entries if e["name"].lower().endswith(".mat")]
    targets=[]
    for subject in (4,5,11,15,47,49):
        regex=re.compile(r"(?:^|[-_/ ])(?:sub(?:ject)?[-_/ ]?0*|s)"+str(subject)+r"(?:[-_/. ]|$)",re.I)
        matches=[e["name"] for e in candidates if regex.search(e["name"])]
        targets.append({"public_released_id":subject,"mat_member_count_candidate":len(matches),
                        "candidate_public_mat_members":matches[:10]})
    return {
        "source_archive_file_id":FILE_ID,
        "remote_zip_size_bytes":total,
        "http_range_206_required_and_observed":True,
        "full_source_archive_not_downloaded":True,
        "zip_entry_count":len(entries),
        "mat_member_count":len(candidates),
        "edf_member_count":sum(e["name"].lower().endswith(".edf") for e in entries),
        "zip_entry_name_examples":[e["name"] for e in entries[:12]],
        "candidate_original_mat_paths":targets,
        "source_zip_index_only_not_original_mat_signal_comparison":True,
        "no_patient_demographic_or_waveform_data_read":True,
        "source_api":"Figshare v5 file 38516555 via range-only central directory"
    }

def self_test():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"test.zip"
        with zipfile.ZipFile(p,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("sourcedata/sub-04/eeg/sub-04_eeg.mat",b"not-real-eeg")
            z.writestr("sourcedata/sub-05/eeg/sub-05_eeg.mat",b"also-not-real")
        data=p.read_bytes()
        index=decode_zip_central(data[-131072:],lambda a,b:data[a:b+1])
        assert len(index)==2 and index[0]["name"].endswith("sub-04_eeg.mat")
        assert index[1]["uncompressed_bytes"]==len(b"also-not-real")
    print("P2_FIGSHARE_REMOTE_ZIP_CENTRAL_DIRECTORY_SYNTHETIC_TEST_PASS")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if args.out is None:parser.error("--out required")
    x=public_index()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(x,ensure_ascii=True,indent=2,sort_keys=True)+"\n"
    args.out.write_text(body,encoding="utf-8")
    print("P2_REMOTE_SOURCE_ZIP_METADATA",{
        "entry_count":x["zip_entry_count"],
        "mat_count":x["mat_member_count"],
        "matched_ID_count":sum(bool(v["mat_member_count_candidate"]) for v in x["candidate_original_mat_paths"])
    },flush=True)
    print("P2_SOURCE_ZIP_INDEX_SHA256",hashlib.sha256(body.encode()).hexdigest())

if __name__=="__main__":
    main()
