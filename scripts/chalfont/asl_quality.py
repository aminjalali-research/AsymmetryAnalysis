#!/usr/bin/env python3
"""Objective ASL quality metrics for every registered Chalfont subject.

Computed in symmetric-template space with the fixed Desikan-Killiany atlas:
  gm_cbf      mean TC-CBF in cortical + subcortical GM (atlas labels)
  wm_cbf      mean TC-CBF in cerebral white matter (labels 2, 41)
  gm_wm_ratio gm_cbf / wm_cbf  (GM:WM ~ 2-3 expected for pCASL at 3 T)
  frac_sat    fraction of brain voxels at the int16 ceiling (>= 32000) or <= -32000
  frac_neg    fraction of brain voxels <= 0
  lr_gm_ratio mean GM CBF left / right (gross hemispheric failure)
Rule (pre-registered here, before any lateralization result is seen):
  FAIL if gm_wm_ratio < 1.2 or gm_cbf < 15 or gm_cbf > 100 or frac_sat > 0.01
       or lr_gm_ratio outside [0.8, 1.25]
"""
import sys, numpy as np, nibabel as nib, pandas as pd
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]
EXT = BASE / "external_chalfont"
atlas = nib.load(str(EXT / "atlas/consensus_parcellation_sym.nii.gz")).get_fdata().astype(np.int32)
gm = np.isin(atlas, list(range(1001, 1036)) + list(range(2001, 2036)) + [10, 17, 18, 49, 53, 54])
wm = np.isin(atlas, [2, 41])
_aff = nib.load(str(EXT / "atlas/consensus_parcellation_sym.nii.gz")).affine
_xw = _aff[0, 0] * np.arange(atlas.shape[0]) + _aff[0, 3]       # world x per i index (RAS: x<0 = left)
left = np.zeros_like(gm); left[_xw < 0, :, :] = True              # fixed 2026-10-20 (audit finding 7)
brain = atlas > 0
rows = []
variant = sys.argv[1] if len(sys.argv) > 1 else "perf"
for p in sorted((EXT / "sym_perf").glob(f"*_{variant}_sym.nii.gz")):
    sid = p.name.split(f"_{variant}_sym")[0]
    d = nib.load(str(p)).get_fdata(dtype=np.float32)
    g, w = d[gm], d[wm]
    gl, gr = d[gm & left], d[gm & ~left]
    r = dict(id=sid, gm_cbf=float(g.mean()), wm_cbf=float(w.mean()),
             gm_wm_ratio=float(g.mean() / max(w.mean(), 1e-3)),
             gm_sd=float(g.std()),
             frac_sat=float(np.mean(np.abs(d[brain]) >= 32000)),
             frac_neg=float(np.mean(d[brain] <= 0)),
             lr_gm_ratio=float(gl.mean() / max(gr.mean(), 1e-3)))
    r["fail"] = (r["gm_wm_ratio"] < 1.2 or r["gm_cbf"] < 15 or r["gm_cbf"] > 100
                 or r["frac_sat"] > 0.01 or not (0.8 <= r["lr_gm_ratio"] <= 1.25))
    rows.append(r); print(f"{sid} gm={r['gm_cbf']:.1f} wm={r['wm_cbf']:.1f} ratio={r['gm_wm_ratio']:.2f} lr={r['lr_gm_ratio']:.2f} sat={r['frac_sat']:.4f} {'FAIL' if r['fail'] else ''}", flush=True)
df = pd.DataFrame(rows)
out = EXT / "qc" / f"asl_quality_{variant}.csv"; df.to_csv(out, index=False)
print("\nwrote", out); print(df.describe().T[["mean", "min", "50%", "max"]].round(2)); print("FAIL:", df.loc[df.fail, "id"].tolist())
