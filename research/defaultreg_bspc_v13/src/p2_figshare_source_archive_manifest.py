#!/usr/bin/env python3
"""Publisher public Figshare v5 archive manifest: NO patient or signal data read.

Establishes which public downloadable artifacts are the processed EDF
archive and the independently released original-source MAT archive. This
is only a manifest audit; it does not assert the MAT EEG arrays match
or differ, nor does it infer that two clinical patients are identical.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

ARTICLE=21679035
URLS=[
    f"https://api.figshare.com/v2/articles/{ARTICLE}/versions/5",
    f"https://api.figshare.com/v2/articles/{ARTICLE}"
]

def public_manifest():
    errors=[]
    for url in URLS:
        try:
            request=Request(url,headers={"Accept":"application/json","User-Agent":"P2-public-manifest-audit/1.0"})
            with urlopen(request,timeout=35) as response:
                info=json.load(response)
            if not isinstance(info.get("files"),list):
                raise ValueError("Figshare response lacks files array")
            files=[]
            for f in info["files"]:
                name=str(f.get("name",""))
                if any(x in name.lower() for x in ("patient","clinical","demo","person")):
                    # Do not enumerate subject metadata filenames in results.
                    continue
                files.append({
                    "file_id":int(f["id"]),
                    "file_name":name,
                    "size_bytes":int(f.get("size",0)),
                    "remote_md5":str(f.get("computed_md5") or f.get("supplied_md5") or "")[:32],
                    "does_not_describe_internal_archive_contents":True,
                })
            return {
                "source_api":url,
                "figshare_article_id":ARTICLE,
                "reported_version":info.get("version"),
                "publisher_doi":info.get("doi"),
                "archive_files":files,
                "contains_filename_sourcedata_zip":any("sourcedata" in v["file_name"].lower() for v in files),
                "contains_filename_edffile_zip":any("edffile" in v["file_name"].lower() for v in files),
                "known_MOABB_EDF_archive_public_file_id":38516654,
                "public_metadata_only":True,
                "no_signals_downloaded_or_patient_identity_checked":True,
                "no_claim_MAT_correspondence_verified":True,
            }
        except Exception as exc:
            errors.append(type(exc).__name__)
    raise RuntimeError("Figshare public archive metadata unavailable via configured URLs: "+", ".join(errors))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    x=public_manifest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(x,indent=2,sort_keys=True,ensure_ascii=True)+"\n"
    args.out.write_text(body,encoding="utf-8")
    print("P2_PUBLIC_SOURCE_ARCHIVE_MANIFEST",[(v["file_name"],v["size_bytes"],v["file_id"]) for v in x["archive_files"]],flush=True)
    print("P2_FIGSHARE_METADATA_SHA256",hashlib.sha256(body.encode()).hexdigest())

if __name__=="__main__":
    main()
