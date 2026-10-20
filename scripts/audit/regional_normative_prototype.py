"""Prototype: regional normative asymmetry. For each of the 37 ROI pairs, LI = (L-R)/(L+R) of
mean CBF (symmetric space, band consensus parcellation). Controls (60/band) give mean/SD of LI
per ROI; patients get z_LI. Reports control SD per ROI (noise budget at ROI level) and a simple
mesial-temporal rule (sign of hippocampus+amygdala+temporal z) vs MDT side. Read-only, ~1 GB."""
import numpy as np, nibabel as nib, json, pandas as pd
from pathlib import Path
import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from ez_ground_truth import EZ_GROUND_TRUTH as GT
B=Path('results_zscore'); SYM=Path('symreg/sym_perf')
PAIRS=[(17,53,'hippocampus'),(18,54,'amygdala'),(10,49,'thalamus')]+[(1000+i,2000+i,f'ctx{i}') for i in range(1,36) if i!=4]
TEMPORAL={'hippocampus','amygdala','ctx6','ctx7','ctx9','ctx15','ctx16','ctx30','ctx33','ctx34'}
def roi_means(perf,parc):
    out={}
    for l,r,n in PAIRS:
        for lab,side in ((l,'L'),(r,'R')):
            m=(parc==lab)&(perf>0)&(perf<150); out[(n,side)]=perf[m].mean() if m.sum()>50 else np.nan
    return out
def li_of(d): return {n:(d[(n,'L')]-d[(n,'R')])/(d[(n,'L')]+d[(n,'R')]) for _,_,n in PAIRS}
bands=json.load(open('results_zscore/groups/FM_20_39/metadata.json')) and ['FM_20_39','FM_40_59','FM_60_79']
ctrl={}; parcs={}
for b in bands:
    meta=json.load(open(B/'groups'/b/'metadata.json')); parc=nib.load(str(B/'groups'/b/'consensus_parcellation.nii.gz')).get_fdata().astype(np.int32); parcs[b]=parc
    rows=[]
    for s in meta['subject_ids']:
        p=nib.load(str(SYM/f'{s}_perf_sym.nii.gz')).get_fdata(dtype=np.float32); rows.append(li_of(roi_means(p,parc)))
    ctrl[b]=pd.DataFrame(rows); print(b,'controls',len(rows),flush=True)
pd.concat(ctrl,names=['band']).to_csv('ez_analysis_mdt/audit_control_roi_li.csv')
print("\nControl LI SD (x100 = % units of (L-R)/(L+R); AI%≈2x) per band:")
sdtab=pd.DataFrame({b:100*ctrl[b].std() for b in bands}); print(sdtab.loc[['hippocampus','amygdala','thalamus','ctx6','ctx7','ctx9','ctx15','ctx16','ctx30','ctx33','ctx28','ctx24']].round(1).to_string())
print("median SD over 37 ROIs:",sdtab.median().round(1).to_dict())
print("control mean LI x100 hippocampus:",{b:round(100*ctrl[b]['hippocampus'].mean(),2) for b in bands})
# patients
pb={}
for b in bands:
    for s in json.load(open(B/'asymmetry/groups'/b/'metadata.json')).get('subject_ids',[]): pass
import glob
res=[]
for f in sorted(glob.glob('results_zscore/patients/P0*/P0*_vs_FM_*_summary.json')):
    d=json.load(open(f)); pid,b=d['patient_id'],d['group']
    p=nib.load(str(SYM/f'{pid}_perf_sym.nii.gz')).get_fdata(dtype=np.float32); li=li_of(roi_means(p,parcs[b]))
    z={n:(li[n]-ctrl[b][n].mean())/ctrl[b][n].std() for n in li}
    mes=np.nanmean([z['hippocampus'],z['amygdala']]); tmp=np.nanmean([z[n] for n in TEMPORAL]); allm=np.nanmean(list(z.values()))
    ez=GT[pid]['ez']; pred=lambda v: 'R' if v>0 else 'L'   # positive LI = L>R = right hypoperfused -> EZ right
    res.append(dict(pid=pid,band=b,mdt=ez,z_hip=round(z['hippocampus'],2),z_amy=round(z['amygdala'],2),z_mesial=round(mes,2),z_temporal=round(tmp,2),z_all=round(allm,2),
                    pred_mesial=pred(mes),pred_temporal=pred(tmp),n_roi_abs_z_gt2=int(np.nansum(np.abs(list(z.values()))>2))))
R=pd.DataFrame(res); print("\n",R.to_string(index=False))
u=R[R.mdt.isin(['L','R'])]
for c in ('pred_mesial','pred_temporal'):
    agree=(u[c]==u.mdt); bal=0.5*(agree[u.mdt=='L'].mean()+agree[u.mdt=='R'].mean())
    print(f"{c}: agree {agree.sum()}/{len(u)}  balanced acc {bal:.2f}  (L {agree[u.mdt=='L'].sum()}/{(u.mdt=='L').sum()}, R {agree[u.mdt=='R'].sum()}/{(u.mdt=='R').sum()})")
print("controls LOO: fraction of controls with |z_mesial|>2 (false-positive proxy):")
for b in bands:
    c=ctrl[b][['hippocampus','amygdala']]; zs=[]
    for i in range(len(c)):
        o=c.drop(i); zz=((c.iloc[i]-o.mean())/o.std()).mean(); zs.append(zz)
    zs=np.array(zs); print(f"  {b}: {np.mean(np.abs(zs)>2):.2f}  (|z|>1.64: {np.mean(np.abs(zs)>1.64):.2f})")
