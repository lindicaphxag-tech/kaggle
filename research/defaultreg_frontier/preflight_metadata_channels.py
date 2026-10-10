"""Metadata-only preflight for future Validity-Frontier cohorts.

NO task epochs, labels, features, covariance, classifier or AUC are created.
The only purpose is to freeze an outcome-independent sensor montage based on
raw channel availability before a separate scientific runner is authored.
"""
from __future__ import annotations
import argparse,json,platform,sys
from pathlib import Path
import moabb
from moabb.datasets import Shin2017A,GrosseWentrup2009

CANDIDATE=("FC5","FC3","FC1","FC2","FC4","FC6",
           "C5","C3","C1","Cz","C2","C4","C6",
           "CP5","CP3","CP1","CPz","CP2","CP4","CP6")

def norm(x):
    return x.replace(".","").replace("-","").replace(" ","").upper()

def build(name):
    if name=="Shin2017A":
        d=Shin2017A(accept=True,motor_imagery=True,mental_arithmetic=False)
        expected=tuple(range(1,30))
    elif name=="GrosseWentrup2009":
        d=GrosseWentrup2009()
        expected=tuple(range(1,11))
    else:
        raise ValueError(name)
    subjects=tuple(map(int,d.subject_list))
    if subjects!=expected:
        raise ValueError(f"unexpected subject roster {subjects}")
    # Raw metadata only. We intentionally do not invoke MotorImagery,
    # events_from_annotations, epochs, labels, features or scores.
    raw_tree=d.get_data(subjects=[subjects[0]])
    sessions=raw_tree[subjects[0]]
    raw_objects=[]
    run_keys={}
    for session,runs in sessions.items():
        run_keys[str(session)]=sorted(map(str,runs.keys()))
        raw_objects.extend(runs.values())
    if not raw_objects:
        raise ValueError("no raw runs")
    first=raw_objects[0]
    raw_names=list(first.ch_names)
    normalized={norm(x):x for x in raw_names}
    selected=[ch for ch in CANDIDATE if norm(ch) in normalized]
    actual=[normalized[norm(ch)] for ch in selected]
    if len(selected)<7:
        raise ValueError(f"fewer than seven frozen candidate channels survive: {selected}")
    sfreqs=sorted({float(r.info["sfreq"]) for r in raw_objects})
    result={
      "status":"METADATA_ONLY_CHANNEL_PREFLIGHT_NO_OUTCOME",
      "dataset":name,
      "moabb_version":moabb.__version__,
      "subject_roster":list(subjects),
      "n_subjects":len(subjects),
      "inspected_subject":subjects[0],
      "session_keys":sorted(map(str,sessions.keys())),
      "run_keys":run_keys,
      "sampling_frequencies":sfreqs,
      "raw_channel_count":len(raw_names),
      "raw_channel_names":raw_names,
      "frozen_candidate_family":list(CANDIDATE),
      "selected_channel_labels_canonical":selected,
      "selected_raw_channel_names":actual,
      "selected_channel_count":len(selected),
      "tangent_dimension":len(selected)*(len(selected)+1)//2,
      "task_events_accessed":False,
      "class_labels_accessed":False,
      "epochs_created":False,
      "features_created":False,
      "covariances_created":False,
      "classifier_fit":False,
      "auc_computed":False,
      "environment":{"python":sys.version,"platform":platform.platform()}}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True,choices=["Shin2017A","GrosseWentrup2009"])
    p.add_argument("--out",required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    r=build(a.dataset)
    q=out/(a.dataset+"_metadata_preflight.json")
    q.write_text(json.dumps(r,indent=2,sort_keys=True))
    print(json.dumps(r,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
