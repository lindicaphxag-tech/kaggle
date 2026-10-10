"""Fail closed if retired, stale or chronology-invalid Default Regularisation claims re-enter paper_vnext."""
from __future__ import annotations
import argparse,re
from pathlib import Path

# These are manuscript-prose claims that must not re-enter .tex files.
BANNED_TEX_TERMS=(
    "all seven protocols",
    "five of five cohorts",
    "five of five",
    "4--19",
    "four to nineteen",
    "four-to-nineteen",
    "98.3\\%",
    "0.8300 pooled",
    "validation-free",
    "consumes no validation set",
    "every evaluated protocol",
    "statistically equivalent",
    "formally equivalent",
    "Kaggle PR #97",
)

# These indicate a stale evidence state even inside status/claim-control files.
STALE_ALL_FILE_TERMS=(
    "ONE TRUE PRE-DATA COHORT HAS POINT-ESTIMATE SUPPORT",
    "MULTI-COHORT AND UNCERTAINTY GATES REMAIN OPEN",
    "frontier remains a later claim with one PRE-DATA cohort completed",
)

BANNED_REGEX=(
    (
        r"cho2017.{0,180}lee2019.{0,180}(?:untouched|prospective).{0,120}(?:validity[- ]frontier|lexicographic)",
        "Cho/Lee were exposed before later frontier/lexicographic claims and cannot be called untouched confirmation for those claims.",
    ),
    (
        r"(?:fctr|cfcts).{0,120}(?:is|was|remains)?\s*(?:robust|superior|safer|state[- ]of[- ]the[- ]art)",
        "FCTR/CFCTS are demoted or negative controls under current evidence.",
    ),
    (
        r"s021.{0,120}(?:prospectively|prospective).{0,120}(?:confirm|validat).{0,120}(?:validity[- ]frontier|lexicographic)",
        "S021-S040 predates the later frontier/lexicographic constructs.",
    ),
    (
        r"bnci2015\\?_001.{0,120}(?:pending|awaiting|not yet completed)",
        "BNCI2015_001 v5 TRUE PRE-DATA execution is complete.",
    ),
    (
        r"(?:bnci2014\\?_002|bnci2015\\?_001).{0,160}(?:holm[- ]confirmed|holm[- ]significant|significant after holm)",
        "Neither TRUE PRE-DATA external H2/H3 family is individually Holm-significant.",
    ),
)

REQUIRED={
 "00_abstract_evidence_locked.tex":(
   "complete scalar path",
   "neither target labels nor distribution statistics",
   "The benefit is not uniform",
   "do not propose",
   "26/26",
 ),
 "01_intro.tex":(
   "operator geometry",
   "S021--S040",
   "negative algorithm results",
   "BNCI2014\\_002",
   "BNCI2015\\_001",
   "material-fragility",
 ),
 "02_results_current.tex":(
   "S021--S040",
   "harmed by more than",
   "deterministic score-regeneration",
   "26/26",
   "delta_{0.02}",
   "historical hidden-filter-bank narrative",
 ),
 "03_discussion_current.tex":(
   "26/26",
   "0.916",
   "0.175",
   "FCTR",
   "Z0",
   "TNSRE",
 ),
 "80_methods.tex":(
   "Operator geometry of objective mismatch",
   "Regularisation Validity Frontier",
   "material-fragility",
   "score-family",
   "BNCI2015\\_001",
   "20,000",
 ),
}

REQUIRED_MD={
 "TITLE_CLAIM_AND_VENUE_LOCK.md":(
   "Regularisation Validity in Cross-Subject EEG",
   "TWO TRUE PRE-DATA COHORTS WITH UNCERTAINTY-COMPLETE FRONTIERS",
   "NO NEW SELECTOR IS PROMOTED",
 ),
 "STATUS.md":(
   "SCIENCE CORE: COMPLETE",
   "SUBMISSION-PACKAGE HOLD",
   "V5_MATERIAL_FRAGILITY_FRONTIER_RECEIPT_20261010.md",
 ),
}

def norm(text:str)->str:
    return re.sub(r"\s+"," ",text.lower()).strip()

def audit(root:Path):
    violations=[];missing=[]
    tex_files=list(sorted(root.glob("*.tex")))
    control_files=[]
    for name in REQUIRED_MD:
        p=root/name
        if p.exists():control_files.append(p)

    for p in tex_files:
        text=p.read_text()
        lower=text.lower()
        compact=norm(text)
        for term in BANNED_TEX_TERMS:
            if term.lower() in lower:
                violations.append((p.name,term,"BANNED_TEX_TERM"))
        for term in STALE_ALL_FILE_TERMS:
            if norm(term) in compact:
                violations.append((p.name,term,"STALE_EVIDENCE_STATE"))
        for pattern,reason in BANNED_REGEX:
            if re.search(pattern,compact,flags=re.I):
                violations.append((p.name,pattern,reason))

    for p in control_files:
        compact=norm(p.read_text())
        for term in STALE_ALL_FILE_TERMS:
            if norm(term) in compact:
                violations.append((p.name,term,"STALE_EVIDENCE_STATE"))

    for name,terms in REQUIRED.items():
        p=root/name
        if not p.exists():
            missing.append((name,"FILE_MISSING"))
            continue
        compact=norm(p.read_text())
        for term in terms:
            if norm(term) not in compact:
                missing.append((name,term))

    for name,terms in REQUIRED_MD.items():
        p=root/name
        if not p.exists():
            missing.append((name,"FILE_MISSING"))
            continue
        compact=norm(p.read_text())
        for term in terms:
            if norm(term) not in compact:
                missing.append((name,term))

    if violations or missing:
        raise SystemExit("VNEXT_CLAIM_GUARD_FAIL "+repr({
            "banned":violations,"missing":missing
        }))
    print("VNEXT_CLAIM_GUARD_PASS tex_files="+str(len(tex_files))+
          " control_files="+str(len(control_files)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    a=ap.parse_args()
    audit(a.root)

if __name__=="__main__":
    main()
