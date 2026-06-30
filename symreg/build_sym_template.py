#!/usr/bin/env python3
"""Build a left-right symmetric T1 template on the native HCP-MNI 0.8mm grid.

The template is constructed as T_sym = (T_avg + flip0(T_avg)) / 2 where flip0 is
numpy's np.flip(axis=0) -- the EXACT operation used by 02_compute_zai.py's
compute_asymmetry_map. By construction T_sym is invariant under flip0, so any
subject nonlinearly registered onto T_sym becomes flip0-symmetric, which makes
the mirror-AI compare homologous voxels.
"""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib

BASE = Path(__file__).resolve().parent.parent
sample = (BASE / "symreg/template_sample.txt").read_text().split()

acc = None
ref = None
n = 0
for d in sample:
    p = Path(d) / "MNINonLinear" / "T1w_restore_brain.nii.gz"
    img = nib.load(str(p))
    data = img.get_fdata(dtype=np.float32)
    # Normalize each brain to its own robust 98th pct so intensity scales match
    scale = np.percentile(data[data > 0], 98)
    data = data / scale
    if acc is None:
        acc = np.zeros_like(data, dtype=np.float64)
        ref = img
    acc += data
    n += 1
    print(f"  [{n}/{len(sample)}] {Path(d).name}  scale98={scale:.1f}", flush=True)

avg = acc / n
sym = (avg + np.flip(avg, axis=0)) / 2.0

# Save both the plain average (diagnostic) and the symmetric template (target)
out = BASE / "symreg/templates"
nib.save(nib.Nifti1Image(avg.astype(np.float32), ref.affine, ref.header),
         str(out / "T1_avg.nii.gz"))
nib.save(nib.Nifti1Image(sym.astype(np.float32), ref.affine, ref.header),
         str(out / "T1_sym.nii.gz"))

# Residual asymmetry of the plain average vs the symmetric template (sanity)
m = avg > 0.05
flipavg = np.flip(avg, axis=0)
valid = (avg + flipavg) / 2 > 0.05
ai_avg = 100 * (avg[valid] - flipavg[valid]) / ((avg[valid] + flipavg[valid]) / 2)
print(f"\nPlain average T1 residual mean|AI| = {np.mean(np.abs(ai_avg)):.1f}% "
      f"(this is the systematic template asymmetry we remove)")
print(f"Template voxels (>0.05): {m.sum():,}")
print(f"Saved T1_avg.nii.gz and T1_sym.nii.gz to {out}")
