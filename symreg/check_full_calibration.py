#!/usr/bin/env python3
"""Post-pipeline GM-specific calibration check (run after run_sym_pipeline.sh).

For each patient, reports %|zAI|>=1.96 within the gray-matter consensus mask of
its age-matched band, using the freshly-built symmetric-space group stats. This
is the manuscript headline calibration number. Compares against the archived
original-space zAI maps when present.
"""
from pathlib import Path
import numpy as np
import nibabel as nib

BASE = Path(__file__).resolve().parent.parent
GROUPS = BASE / "results_zscore/groups"
ZAI = BASE / "results_zscore/asymmetry/patients"
ARCH = BASE / "_archived_origspace_zai_20260622/asymmetry/patients"

CORT = set(range(1001, 1036)) | set(range(2001, 2036))
SUBC = {10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60}
GM = CORT | SUBC

PAT_BAND = {'FM_20_39': ['P013','P015','P016','P017','P019','P020','P026','P028','P029','P030'],
            'FM_40_59': ['P018','P021','P022','P023','P024','P027'],
            'FM_60_79': ['P025']}


def gm_mask(band):
    parc = nib.load(str(GROUPS/band/"consensus_parcellation.nii.gz")).get_fdata().astype(np.int32)
    m = np.zeros(parc.shape, bool)
    for lab in GM:
        m |= (parc == lab)
    return m


def pct(zpath, mask):
    if not zpath.exists():
        return None
    z = nib.load(str(zpath)).get_fdata(dtype=np.float32)
    m = mask & (z != 0)
    n = int(m.sum())
    return (100.0 * np.sum(np.abs(z[m]) >= 1.96) / n, n) if n else (float('nan'), 0)


print(f"{'patient':8} {'band':9} {'NEW %|z|>=1.96 (GM)':>22} {'OLD %':>8} {'GMvox':>9}")
new_all = []
for band, pats in PAT_BAND.items():
    mask = gm_mask(band)
    for p in pats:
        newp = pct(ZAI/p/f"{p}_asymmetry_zscore.nii.gz", mask)
        # original-space archived map used a different (per-band) GM mask, but
        # the magnitude is comparable; report with the same sym GM mask if shapes match
        oldz = ARCH/p/f"{p}_asymmetry_zscore.nii.gz"
        oldp = pct(oldz, mask) if oldz.exists() else None
        if newp:
            new_all.append(newp[0])
        ns = f"{newp[0]:.1f}%" if newp else "MISSING"
        os_ = f"{oldp[0]:.1f}%" if oldp else "n/a"
        nv = newp[1] if newp else 0
        print(f"{p:8} {band:9} {ns:>22} {os_:>8} {nv:>9,}")
print(f"\nNEW cohort mean %|zAI|>=1.96 in GM = {np.mean(new_all):.1f}% (n={len(new_all)})")
print("Target: ~5-15% for a calibrated normative asymmetry model.")
