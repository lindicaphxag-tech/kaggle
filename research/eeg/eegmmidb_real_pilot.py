"""PhysioNet EEGMMIDB left/right imagery pilot with subject-disjoint electrode tests.

This is a frozen *classical feature baseline*, not EEG-CLIP/NeuroRVQ inference,
not proof of a novel model, and not a source of clinical claims.

Usage:
    python -m research.eeg.eegmmidb_real_pilot --cohort pilot --data-dir ./mne_data --out pilot.json
    python -m research.eeg.eegmmidb_real_pilot --cohort full --data-dir ./mne_data --out full.json

All 64 EEG channels, task runs 4/8/12, T1 = left fist imagery, T2 = right.
Public source: https://physionet.org/content/eegmmidb/1.0.0/
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import mne
import numpy as np
import scipy
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn

SEED = 314159
RUNS = (4, 8, 12)
SPLITS = {
    'pilot': {'train': tuple(range(1, 7)), 'val': tuple(range(7, 10)), 'test': tuple(range(10, 13))},
    'full': {'train': tuple(range(1, 17)), 'val': tuple(range(17, 21)), 'test': tuple(range(21, 25))},
}
BANDS = ((8., 13.), (13., 30.))


def fingerprint(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def spectral_features(epochs, sfreq):
    """(trial,channel,time) -> (trial,channel,2), strict 8-13/13-30 Hz log PSD."""
    epochs = np.asarray(epochs, dtype='float64')
    if epochs.ndim != 3 or epochs.shape[-1] < 256 or not np.isfinite(epochs).all():
        raise ValueError('Bad epoch tensor or insufficient time samples')
    freqs, psd = welch(epochs, fs=sfreq, nperseg=256, axis=-1)
    result=[]
    for low, high in BANDS:
        ix=(freqs >= low) & (freqs < high)
        if not ix.any():
            raise ValueError('Band has no spectral bins')
        result.append(np.log(np.maximum(np.mean(psd[..., ix], axis=-1), 1e-20)))
    return np.stack(result, axis=-1)


def aligned_electrodes(channels):
    channels = tuple(channels)
    if len(channels) != 64 or len(set(channels)) != 64:
        raise ValueError('Expected exactly 64 uniquely named EEG electrodes')
    montage=mne.channels.make_standard_montage('standard_1005')
    positions=montage.get_positions()['ch_pos']
    if any(name not in positions for name in channels):
        raise ValueError('Electrode location unknown: no synthetic location fallback')
    xyz=np.asarray([positions[name] for name in channels],dtype='float64')
    if not np.isfinite(xyz).all():
        raise ValueError('Invalid montage coordinates')
    return xyz


def stress_conditions(test_features, train_features, names, xyz, seed=SEED):
    """Conditions receive the same held-out features; never fit to test labels.

    IDW here interpolates *log spectral features*, NOT real EEG voltages and NOT
    equivalent to MNE's spherical-spline raw-signal interpolation.
    """
    xt=np.asarray(test_features, dtype=float)
    tr=np.asarray(train_features,dtype=float)
    n=len(names)
    if xt.ndim!=3 or tr.ndim!=3 or xt.shape[1:]!=(n,2) or tr.shape[1:]!=(n,2):
        raise ValueError('Expected [trial, electrode, 2 frequency bands]')
    if len(set(names))!=n or xyz.shape!=(n,3) or n < 5:
        raise ValueError('Invalid named montage')
    rng=np.random.default_rng(seed)
    perm=rng.permutation(n)
    inv=np.argsort(perm)
    shifted=xt[:,perm,:]
    restored=shifted[:,inv,:]
    if not np.array_equal(xt,restored):
        raise AssertionError('Known-identity permutation must be exactly reversible')
    trial_mean=tr.mean(axis=0)
    result={'original':xt, 'shape_only_permutation':shifted,
            'identity_reindexed':restored}
    missing_record={}
    for pct in (10,25,40):
        nmissing=max(1,int(round(n*pct/100)))
        absent=np.sort(rng.choice(n,size=nmissing,replace=False))
        present=np.setdiff1d(np.arange(n),absent)
        filled=xt.copy()
        filled[:,absent,:]=trial_mean[absent,:]
        result[f'feature_mean_fill_{pct}']=filled
        interp=xt.copy()
        for j in absent:
            dd=np.linalg.norm(xyz[present]-xyz[j],axis=1)
            nn=present[np.argsort(dd)[:4]]
            dd4=np.maximum(np.linalg.norm(xyz[nn]-xyz[j],axis=1),1e-8)
            ww=(1/dd4**2);ww=ww/ww.sum()
            interp[:,j,:]=np.einsum('c,tcb->tb',ww,xt[:,nn,:])
        result[f'feature_idw_{pct}']=interp
        missing_record[str(pct)]=[names[int(i)] for i in absent]
    return result, {'permutation_indices':perm.tolist(),
                    'missing_electrodes_by_percentage':missing_record,
                    'unknown_or_duplicated_identity':'ABSTAIN, not auto-corrected'}


def evaluate_holdout(X,y,subject,channels,xyz,cohort='pilot'):
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y,dtype=int)
    subject=np.asarray(subject,dtype=int)
    split=SPLITS[cohort]
    train=np.isin(subject,split['train'])
    val=np.isin(subject,split['val'])
    test=np.isin(subject,split['test'])
    if not all(m.any() for m in (train,val,test)):
        raise ValueError('Empty subject split: abort')
    if len(set(split['train'])&set(split['val'])) or len(set(split['train'])&set(split['test'])) or len(set(split['val'])&set(split['test'])):
        raise ValueError('Subject leakage')
    if set(subject)!=set(split['train']+split['val']+split['test']):
        raise ValueError('Data does not contain the exact preregistered cohort')
    if set(y[train])!={0,1}:
        raise ValueError('Both classes required for training')
    if X.shape!=(len(y),len(channels),2):
        raise ValueError('Bad feature shape')
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000,random_state=SEED))
    model.fit(X[train].reshape(train.sum(),-1),y[train])
    conditions, perturbations=stress_conditions(X[test],X[train],channels,xyz)
    normal=model.predict(conditions['original'].reshape(test.sum(),-1))
    recovered=model.predict(conditions['identity_reindexed'].reshape(test.sum(),-1))
    if not np.array_equal(normal,recovered):
        raise AssertionError('Identity reindex did not recover identical predictions')
    names=sorted(conditions)
    summary={}
    for name in names:
        preds=model.predict(conditions[name].reshape(test.sum(),-1))
        summary[name]={
            'overall_bacc':float(balanced_accuracy_score(y[test],preds)),
            'per_subject':[
                {'subject':int(s),'n_trials':int(np.sum(subject[test]==s)),
                 'bacc':float(balanced_accuracy_score(y[test][subject[test]==s],preds[subject[test]==s]))}
                for s in split['test']],
            'per_trial_prediction':preds.astype(int).tolist(),
        }
    pred_val=model.predict(X[val].reshape(val.sum(),-1))
    return {
        'claim_scope':'Real task feature-baseline pilot, NOT a trained foundation model or novel method benefit',
        'cohort':cohort,
        'split_subjects':{k:list(map(int,v)) for k,v in split.items()},
        'n_trials':{'train':int(train.sum()),'val':int(val.sum()),'test':int(test.sum())},
        'validation_bacc':float(balanced_accuracy_score(y[val],pred_val)),
        'test_labels':y[test].astype(int).tolist(),
        'test_subject_ids':subject[test].astype(int).tolist(),
        'perturbations':perturbations,
        'condition_metrics':summary,
        'model':'StandardScaler + LogisticRegression(C=1), fixed log PSD 8-13/13-30',
        'limits':['No pretrained EEGCLIP or NeuroRVQ weights','Not MNE spherical interpolation',
                  'No independent external replication','No clinical conclusions',
                  'Unknown identity: reject/abstain; not recoverable without provenance'],
    }


def load_real_data(subject_ids,data_dir):
    all_x=[];all_y=[];all_s=[];files={};reference=None;xyz=None
    for sid in subject_ids:
        filepaths=mne.datasets.eegbci.load_data(int(sid),list(RUNS),path=str(data_dir),update_path=False,verbose=False)
        if len(filepaths)!=3:
            raise RuntimeError(f'Expected exactly 3 EDF recordings for subject {sid}')
        for p in filepaths:
            files[str(Path(p).resolve().relative_to(data_dir.resolve()))]=fingerprint(p)
            raw=mne.io.read_raw_edf(p,preload=True,verbose=False)
            mne.datasets.eegbci.standardize(raw)
            raw.set_montage(mne.channels.make_standard_montage('standard_1005'),on_missing='raise',verbose=False)
            if reference is None:
                reference=tuple(raw.ch_names)
                xyz=aligned_electrodes(reference)
            if set(raw.ch_names)!=set(reference):
                raise ValueError(f'Unexpected channel set in subject {sid}')
            raw.reorder_channels(reference)
            raw.set_eeg_reference('average',projection=False,verbose=False)
            event_ids={'T1':1,'T2':2}
            events,_=mne.events_from_annotations(raw,event_id=event_ids,verbose=False)
            epochs=mne.Epochs(raw,events,event_id=event_ids,tmin=0.5,tmax=3.5,
                              baseline=None,preload=True,verbose=False)
            if len(epochs)==0:
                raise RuntimeError(f'No motor imagery epochs found in {p}')
            x=spectral_features(epochs.get_data(units='V'),raw.info['sfreq'])
            y=(epochs.events[:,-1]==2).astype(int)
            all_x.append(x);all_y.append(y);all_s.extend([int(sid)]*len(y))
            del raw,epochs
    return np.concatenate(all_x),np.concatenate(all_y),np.asarray(all_s),reference,xyz,files


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cohort',choices=list(SPLITS),default='pilot')
    ap.add_argument('--data-dir',type=Path,default=Path('./mne_data'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.data_dir.mkdir(parents=True,exist_ok=True)
    cohort=SPLITS[args.cohort]
    subjects=cohort['train']+cohort['val']+cohort['test']
    X,y,s,channels,xyz,files=load_real_data(subjects,args.data_dir)
    result=evaluate_holdout(X,y,s,channels,xyz,args.cohort)
    result['data_source']='PhysioNet EEGMMIDB v1.0.0, task runs 4/8/12, T1/T2 only'
    result['file_sha256']=files
    result['software_versions']={'python':platform.python_version(),'mne':mne.__version__,
                                 'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('ELECTRODE_PILOT_RESULT',json.dumps({
        'cohort':args.cohort,'n_test':result['n_trials']['test'],
        'metrics':{k:round(v['overall_bacc'],4) for k,v in result['condition_metrics'].items()},
        'out':str(args.out)}))


if __name__=='__main__':main()
