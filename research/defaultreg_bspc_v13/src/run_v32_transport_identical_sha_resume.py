#!/usr/bin/env python3
"""Transport-only resume of V32 source-locked first-run after OSF read timeout.

All scientific code and all original choices remain bitwise unchanged.
The original 12-source SHA is reproduced BEFORE any held target I/O.
This is NOT a new independent first-run or a new scientific method.
"""
from pathlib import Path
import argparse,sys,time
import requests
import pooch
import pooch.core
import pooch.downloaders

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_v32_dreyer_strict_source_only as original

SOURCE_LOCK="9bfc60e1da10575315917aa6662532198a892dfa99f045341bb71adde58a04b9"

# Restrict network transport change to OSF file endpoint.
legacy_choose=pooch.core.choose_downloader
def reliable_downloader(url,progressbar=False):
    if "files.de-1.osf.io" in url:
        return pooch.HTTPDownloader(progressbar=progressbar,timeout=180,verify=False)
    return legacy_choose(url,progressbar=progressbar)
pooch.core.choose_downloader=reliable_downloader

legacy_person=original.person
def retry_person(paradigm,dataset,sid):
    for attempt in range(4):
        try:
            return legacy_person(paradigm,dataset,sid)
        except (requests.exceptions.Timeout,ConnectionError,OSError) as exc:
            if attempt==3:
                raise
            print("V32_TRANSPORT_ONLY_RETRY",sid,attempt+1,type(exc).__name__,flush=True)
            time.sleep(min(3,attempt+1))
    raise AssertionError("unreachable")
original.person=retry_person

legacy_save=original.save
def verify_before_any_target_io(obj,path):
    digest=legacy_save(obj,path)
    if Path(path).name==original.SOURCE_FILE:
        if digest!=SOURCE_LOCK:
            raise AssertionError("V32 original source choices and source EEG "
                                 "receipts did NOT reproduce; deny held target I/O")
        print("V32_IDENTICAL_FIRST_RUN_SOURCE_SHA_REPRODUCED",digest,flush=True)
    return digest
original.save=verify_before_any_target_io

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    original.main(p.parse_args().out)
