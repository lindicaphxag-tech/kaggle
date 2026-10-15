#!/usr/bin/env python3
"""Weibo2014 replication of four operators on one fixed source set.

Source 1..8 is locked before reading target 9 or 10 in this process.
Weibo2014 outcomes were exposed in earlier project work: this is a
retrospective second-acquisition test, NOT prospective confirmation.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from moabb.datasets import Weibo2014
from moabb.paradigms import MotorImagery
from p3_bnci004_four_operator_same_source import select,evaluate
from p3_real_weibo_common_six_selector_matrix import load_people

def write(path,obj):
    data=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();ds=Weibo2014()
    paradigm=MotorImagery(n_classes=2,events=['left_hand','right_hand'],
                          channels=['C3','Cz','C4'],fmin=8,fmax=30,
                          resample=160,baseline=None,reject_by_annotation=False)
    source=list(range(1,9));targets=[9,10]
    C,y,g,receipt=load_people(source,ds,paradigm)
    choices=select(C,y,g)
    lock={'status':'RETROSPECTIVE_SECOND_ACQUISITION_SOURCE_ONLY_LOCK',
          'source_ids':source,'target_ids':targets,'dataset':'Weibo2014',
          'methods':['source_macro','continuous','SCE','anisotropic','group_centered'],
          'source_receipt':receipt,'source_choices':choices,
          'processing':'C3/Cz/C4 8-30Hz 160Hz right vs left',
          'already_exposed_target_outcomes_in_wider_research':True,
          'target_EEG_unread_in_this_run_until_lock':True}
    sha=write(args.out/'P3_WEIBO_FOUR_OPERATORS_SOURCE_LOCK.json',lock)
    print('P3_WEIBO_FOUR_OPERATORS_SOURCE_LOCK_SHA',sha,flush=True)
    rows=[]
    for target in targets:
        Ct,yt,_,rec=load_people([target],ds,paradigm)
        auc=evaluate(C,y,g,Ct,yt,choices)
        rows.append({'target_id':target,'n_target':int(len(yt)),
                     'target_receipt':rec,'auc':auc,
                     'delta_vs_source_macro':{k:float(v-auc['source_macro']) for k,v in auc.items()}})
        print('P3_WEIBO_FOUR_OPERATORS_HELD_RESULT',target,json.dumps(rows[-1],sort_keys=True),flush=True)
    methods=sorted(rows[0]['auc'])
    result={'status':'EXECUTED_RETROSPECTIVE_SECOND_ACQUISITION_COMMON_SOURCE_FOUR_OPERATORS',
            'source_lock_sha256':sha,'rows':rows,
            'mean_delta_vs_source_macro':{k:float(np.mean([r['delta_vs_source_macro'][k] for r in rows])) for k in methods},
            'all_target_outcomes_previously_exposed_to_project':True,
            'two_subjects_one_acquisition_not_two_independent_acquisitions':True,
            'not_a_prospective_null_or_superiority_test':True}
    digest=write(args.out/'P3_WEIBO_FOUR_OPERATORS_TWO_TARGET_RESULTS.json',result)
    print('P3_WEIBO_FOUR_OPERATORS_RESULT_SHA',digest,flush=True)
if __name__=='__main__':main()
