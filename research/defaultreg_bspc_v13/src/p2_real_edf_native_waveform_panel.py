#!/usr/bin/env python3
"""Four real public EDFs: cryptographic checksum gated channel waveforms.
Public release digital samples, NOT patient identity or raw acquisition."""
import json,hashlib,urllib.request,struct,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT='https://data.nemar.org/nm000158/v1.0.3/sourcedata/'
MAN='https://raw.githubusercontent.com/nemarDatasets/nm000158/11e3eeb4b439d5f8ee1b5e3b8e6032824bc2de83/sourcedata/sourcedata_provenance.json'
OUT=os.environ.get('FIGURE_OUT','/tmp/p2-waveforms')
os.makedirs(OUT,exist_ok=True)
def get(url):
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DefaultEEG-public-repro/1.0'}),timeout=150) as q:
    return q.read()
def field(b,i,j):return b[i:j].decode('ascii').strip()
def parse(b):
  hd=int(field(b,184,192));N=int(field(b,236,244));count=int(field(b,252,256))
  dur=float(field(b,244,252))
  labels=[field(b,256+16*i,256+16*(i+1)) for i in range(count)]
  offset=256+(16+80+8+8+8+8+8+80)*count
  countper=[int(field(b,offset+8*i,offset+8*(i+1))) for i in range(count)]
  if hd+2*N*sum(countper)!=len(b):raise ValueError('unexpected EDF layout')
  def record(k,name):
    j=labels.index(name)
    pos=hd+2*k*sum(countper)+2*sum(countper[:j])
    return np.frombuffer(b,dtype='<i2',count=countper[j],offset=pos).astype('int32')
  return {'n_records':N,'labels':labels,'duration_s':dur,'record':record}
manifest=json.loads(get(MAN))
rows={int(r['subject']):r for r in manifest['files']}
ids=[4,5,47,49];dec={}
for i in ids:
  r=rows[i];data=get(ROOT+r['file'])
  if hashlib.sha256(data).hexdigest()!=r['sha256']:raise RuntimeError(f'EDF SHA mismatch {i}')
  dec[i]=parse(data)
assert all(dec[i]['n_records']==320 for i in ids)
# A pre-reported overlap segment: 47/49 has >=28/29 matching digital channels 128..306.
# Pair 4/5 is identical across all 29 named time-varying EEG streams.
cases=[(4,5,80),(47,49,150)]
fig,axs=plt.subplots(2,3,figsize=(11,5.6),layout='constrained')
counts={}
for row,(i,j,record_idx) in enumerate(cases):
  common=[]
  for ch in ['C3','C4','P4','Fz','CP3','P3']:
    x=dec[i]['record'](record_idx,ch);y=dec[j]['record'](record_idx,ch)
    if x.shape==y.shape and np.array_equal(x,y):common.append(ch)
  if row==0 and len(common)<3:raise AssertionError('4/5 should have at least 3 matches')
  if row==1 and len(common)<2:raise AssertionError('47/49 overlap segment should have matches')
  differing=[c for c in ['P4','P3','C3','C4','Fz','CP3'] if c not in common]
  chans=['C3','C4','P4'] if row==0 else [common[0],common[1],differing[0] if differing else common[2]]
  counts[f'{i}/{j}']={'record':record_idx,'channels':chans,'same':[]}
  for col,ch in enumerate(chans):
    ax=axs[row,col]
    x=dec[i]['record'](record_idx,ch);y=dec[j]['record'](record_idx,ch)
    if x.shape!=y.shape:raise ValueError('incomparable samples per record')
    xx=np.linspace(0,dec[i]['duration_s'],len(x),endpoint=False)
    same=bool(np.array_equal(x,y));counts[f'{i}/{j}']['same'].append(same)
    ax.plot(xx,x,color='#276b93',lw=1.15,label=f'EDF ID {i}')
    ax.plot(xx,y,color='#b34b47',lw=.8,alpha=.83,ls='--',label=f'EDF ID {j}')
    ax.set_title(f'({chr(97+row*3+col)}) IDs {i}/{j} • {ch} • '+('EXACT' if same else 'DIFF'),fontsize=9)
    ax.set_xlabel(f'Native EDF record {record_idx}, time (s)')
    ax.set_ylabel('Native digital amplitude (int16)')
    ax.spines[['top','right']].set_visible(False)
    if col==0:ax.legend(frameon=False,fontsize=8)
fig.savefig(os.path.join(OUT,'P2_real_native_EDF_waveform_array.pdf'),bbox_inches='tight')
fig.savefig(os.path.join(OUT,'P2_real_native_EDF_waveform_array.png'),bbox_inches='tight',dpi=300)
report={'status':'REAL_PUBLIC_EDF_WAVEFORMS_SHA_VERIFIED','IDs':ids,
 'public_release':'NEMAR nm000158 v1.0.3','commit':'11e3eeb4b439d5f8ee1b5e3b8e6032824bc2de83',
 'cases':counts,'patient_identity_not_assessed':True,'not_original_raw_MAT':True,
 'no_preprocessing_transform_inference':True}
open(os.path.join(OUT,'provenance.json'),'w').write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,sort_keys=True))
