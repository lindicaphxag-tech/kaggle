#!/usr/bin/env python3
"""Static audit of AUTHOR-PUBLISHED Liu2024 Figshare v5 processing code ZIP.

This program reads only Figshare metadata and a declared code archive;
it DOES NOT execute any downloaded MATLAB/Python and DOES NOT fetch the
raw source MAT, EDF or patient records. It checks whether available
code can explain the MAT->EDF conversion, but it never infers patient
identity from a file name. Fail closed if source code is missing.
"""
import argparse,hashlib,io,json,re,urllib.request,urllib.parse,zipfile
from pathlib import Path

ARTICLE=21679035
VERSION=5
BASE=f"https://api.figshare.com/v2/articles/{ARTICLE}/versions/{VERSION}"
MAX_CODE_BYTES=60_000_000
MAX_MEMBER_BYTES=2_000_000
EXTS={".m",".py",".txt",".md",".json",".csv",".mlx"}
KEYWORDS=re.compile(
    r"(readmatrix|load\(|save\(|edf|mat\b|filt|fir|eeglab|pop_|baseline|"
    r"sub[-_]?0*[0-9]|sourcedata|export|resampl|event|label|duplicate|copyfile|"
    r"movefile|fullfile|for\s+\w+\s*=|eeg_)",
    re.IGNORECASE)

def safe_fetch(url,limit):
    u=urllib.parse.urlparse(url)
    if u.scheme!="https" or u.hostname not in {
        "api.figshare.com","ndownloader.figshare.com",
        "api.figshare.com","figshare.com"}:
        raise ValueError("Host not allowlisted")
    req=urllib.request.Request(url,headers={"User-Agent":"reproducible-provenance-static-code-audit"})
    with urllib.request.urlopen(req,timeout=90) as r:
        data=r.read(limit+1)
    if len(data)>limit:raise RuntimeError("bounded retrieval exceeded")
    return data

def inspect_archive(binary):
    zipinfo=[]
    findings=[]
    with zipfile.ZipFile(io.BytesIO(binary)) as z:
        for info in z.infolist():
            path=Path(info.filename)
            if info.is_dir():continue
            if (".." in path.parts or path.is_absolute() or
                info.file_size>MAX_MEMBER_BYTES):
                zipinfo.append({"name":info.filename,"size":info.file_size,
                                "status":"SKIPPED_SAFETY_BOUND"})
                continue
            compressed=info.compress_size
            if info.file_size>0 and compressed>0 and info.file_size/compressed>1000:
                zipinfo.append({"name":info.filename,"size":info.file_size,
                                "status":"SKIPPED_COMPRESSION_RATIO"})
                continue
            row={"name":info.filename,"size":info.file_size,"crc32":f"{info.CRC:08x}"}
            if path.suffix.lower() in EXTS:
                raw=z.read(info)
                if len(raw)!=info.file_size:raise AssertionError("ZIP length mismatch")
                row["sha256"]=hashlib.sha256(raw).hexdigest()
                if path.suffix.lower() in {".m",".py",".txt",".md"}:
                    text=raw.decode("utf-8",errors="replace")
                    matched=[]
                    for n,line in enumerate(text.splitlines(),1):
                        if KEYWORDS.search(line):
                            # Static snippets are CODE not instructions
                            matched.append({"line":n,"text":line[:220]})
                    findings.append({"name":info.filename,
                                     "sha256":row["sha256"],
                                     "line_count":len(text.splitlines()),
                                     "relevant_snippets":matched[:120],
                                     "matching_line_count":len(matched)})
            zipinfo.append(row)
    return zipinfo,findings

def main(out):
    meta=json.loads(safe_fetch(BASE,1_500_000))
    files=meta.get("files",[])
    if not files:raise RuntimeError("Figshare v5 file metadata absent: no inference")
    listing=[{"id":f.get("id"),"name":f.get("name"),
              "size":f.get("size"),"supplied_md5":f.get("supplied_md5")}
             for f in files]
    codes=[f for f in files
           if re.search(r"(^|[/\\])(code|scripts?)\.zip$",f.get("name",""),re.I)]
    receipt={"source":"Figshare DOI 10.6084/m9.figshare.21679035.v5",
             "article_id":ARTICLE,"version":VERSION,
             "source_metadata_url":BASE,
             "manifest_files":listing,
             "available_code_files_count":len(codes),
             "source_raw_MAT_not_downloaded":True,
             "EDF_not_downloaded":True,
             "no_patient_information_read":True,
             "downloaded_scripts_NEVER_executed":True,
             "cannot_infer_provenance_or_patient_identity_from_filename":True}
    if not codes:
        receipt["status"]="AUTHOR_CODE_ZIP_NOT_FOUND_IN_PINNED_FIGSHARE_METADATA"
    elif len(codes)>1:
        receipt["status"]="AMBIGUOUS_MULTIPLE_CODE_ARCHIVES"
    else:
        f=codes[0]
        if f.get("size") is None or int(f["size"])>MAX_CODE_BYTES:
            receipt["status"]="AUTHOR_CODE_ARCHIVE_EXCEEDS_SAFE_BYTE_LIMIT"
        else:
            binary=safe_fetch(f["download_url"],int(f["size"])+4096)
            if len(binary)!=int(f["size"]):raise ValueError("Figshare code zip length mismatch")
            receipt["archive_sha256"]=hashlib.sha256(binary).hexdigest()
            receipt["archive_size_bytes"]=len(binary)
            members,findings=inspect_archive(binary)
            receipt.update(status="AUTHOR_CODE_ZIP_STATIC_INSPECTED",
                           code_zip_member_inventory=members,
                           candidate_processing_scripts=findings,
                           found_matlab_sources=sum(m["name"].endswith(".m") for m in members),
                           direct_MAT_to_EDF_mapping_still_requires_operator_replay=True)
    payload=json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(payload)
    print("P2_AUTHOR_CODE_ARCHIVE_AUDIT",json.dumps(
      {"status":receipt["status"],
       "code_files":receipt["available_code_files_count"],
       "scripts":len(receipt.get("candidate_processing_scripts",[])),
       "file_inventory_count":len(listing)},sort_keys=True),flush=True)
    if receipt["status"]!="AUTHOR_CODE_ZIP_STATIC_INSPECTED":
        raise RuntimeError("No fully static-inspected pinned published code archive: see JSON")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();main(a.out)
