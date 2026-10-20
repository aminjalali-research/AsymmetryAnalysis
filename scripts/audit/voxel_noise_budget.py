"""Noise budget of the voxel-wise AI normative model (FM_40_59 band): control SD of AI
in GM / hippocampus / temporal cortex -> asymmetry (%) needed to reach |zAI|=4; floor
fraction; and the same SD after 6-mm Gaussian smoothing of perfusion (20 controls)."""
import numpy as np, nibabel as nib, json, sys
from scipy import ndimage
from pathlib import Path
B=Path('results_zscore'); G='FM_40_59'
sd=nib.load(str(B/'asymmetry/groups'/G/'sd_asymmetry.nii.gz')).get_fdata(dtype=np.float32)
meanai=nib.load(str(B/'asymmetry/groups'/G/'mean_asymmetry.nii.gz')).get_fdata(dtype=np.float32)
bm=nib.load(str(B/'asymmetry/groups'/G/'brain_mask.nii.gz')).get_fdata().astype(bool)
parc=nib.load(str(B/'groups'/G/'consensus_parcellation.nii.gz')).get_fdata().astype(np.int32)
meta=json.load(open(B/'asymmetry/groups'/G/'metadata.json')); print("metadata:",{k:v for k,v in meta.items() if k!='subject_ids'})
ctx=((parc>=1000)&(parc<=1035))|((parc>=2000)&(parc<=2035)); sub=np.isin(parc,[10,49,17,53,18,54])
gm=(ctx|sub)&bm
hip=np.isin(parc,[17,53])&bm; amy=np.isin(parc,[18,54])&bm
temp=np.isin(parc,[1006,1007,1009,1015,1016,1030,1033,1034,2006,2007,2009,2015,2016,2030,2033,2034])&bm
def rep(name,m):
    v=sd[m]; q=np.percentile(v,[10,25,50,75,90]); print(f"{name:14} n={m.sum():9,} SD%AI p10/25/50/75/90 = {q.round(1)}  at_floor(5%)={100*(v<=5.001).mean():.1f}%  |AI| for z=4 (median) = {4*q[2]:.0f}%  |mean AI| median={np.median(np.abs(meanai[m])):.1f}%")
for n,m in [('brain',bm),('GM(37ROI)',gm),('hippocampus',hip),('amygdala',amy),('temporal ctx',temp)]: rep(n,m)
# smoothing experiment: 20 controls, Welford on AI from unsmoothed vs 6-mm smoothed perfusion
ids=meta['subject_ids'][:20]; sig=6/2.3548/0.8
acc={k:[np.zeros(sd.shape,np.float64),np.zeros(sd.shape,np.float64),np.zeros(sd.shape,np.int32)] for k in ('raw','sm6')}
def ai_of(p):
    L=p.astype(np.float64); R=np.flip(L,0); v=(L>0)&(R>0); ai=np.zeros_like(L); ai[v]=100*(L[v]-R[v])/((L[v]+R[v])/2); return ai,v
for i,s in enumerate(ids):
    p=nib.load(f'symreg/sym_perf/{s}_perf_sym.nii.gz').get_fdata(dtype=np.float32)
    psm=ndimage.gaussian_filter(np.where(p>0,p,0).astype(np.float32),sig)
    # renormalise smoothing by mask to limit edge dilution
    w=ndimage.gaussian_filter((p>0).astype(np.float32),sig); psm=np.where(w>0.5,psm/np.maximum(w,1e-6),0)
    for k,arr in (('raw',p),('sm6',psm)):
        ai,v=ai_of(arr); S,SS,C=acc[k]; S[v]+=ai[v]; SS[v]+=ai[v]**2; C[v]+=1
    print(f"  {i+1}/{len(ids)} {s}",flush=True)
for k in ('raw','sm6'):
    S,SS,C=acc[k]; ok=C>=10; m=np.where(ok,S/np.maximum(C,1),0); var=np.where(ok,SS/np.maximum(C,1)-m**2,0); s_=np.sqrt(np.maximum(var,0))
    for n,msk in [('GM(37ROI)',gm),('hippocampus',hip),('temporal ctx',temp)]:
        v=s_[msk&ok]; print(f"{k:4} {n:13} SD%AI median={np.median(v):.1f} p25={np.percentile(v,25):.1f} p75={np.percentile(v,75):.1f} -> |AI| for z=4: {4*np.median(v):.0f}%")
