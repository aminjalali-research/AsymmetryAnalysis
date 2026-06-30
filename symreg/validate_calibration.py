#!/usr/bin/env python3
"""Validate the symmetric-registration zAI fix on the FM_20_39 subset.

Calibration metric: with the control AI normative model built from the
registered controls, what fraction of GM voxels exceed |z|>=1.96 for
(a) the controls themselves (leave-one-out -> should be ~5%), and
(b) the patients. Over-dispersion = controls flag far more than 5%.

Compares the NEW (symmetric-registered) perfusion against the OLD
(un-registered, original MNI) perfusion using the SAME 20 controls.
"""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib

BASE = Path(__file__).resolve().parent.parent
SYM = BASE / "symreg/sym_perf"
SD_FLOOR = 5.0

CTRL = [l.split('\t')[2].split('/')[-1] for l in
        (BASE/'symreg/jobs_val.txt').read_text().splitlines()
        if l.split('\t')[2].split('/')[-1].startswith('HCA')]
PAT = [l.split('\t')[2].split('/')[-1] for l in
       (BASE/'symreg/jobs_val.txt').read_text().splitlines()
       if l.split('\t')[2].split('/')[-1].startswith('P')]


def ai_map(perf):
    left = perf.astype(np.float64)
    right = np.flip(left, axis=0)
    mean_lr = (left + right) / 2.0
    valid = mean_lr > 1.0
    ai = np.zeros_like(left)
    ai[valid] = 100.0 * (left[valid] - right[valid]) / mean_lr[valid]
    return ai, valid


def build_stats(ai_list, valid_list):
    """Welford mean/SD across subjects, + bilateral pooling + 5% floor."""
    shape = ai_list[0].shape
    mean = np.zeros(shape); M2 = np.zeros(shape); cnt = np.zeros(shape, int)
    for ai, v in zip(ai_list, valid_list):
        cnt[v] += 1
        d = np.where(v, ai - mean, 0.0)
        mean[v] += d[v] / cnt[v]
        d2 = np.where(v, ai - mean, 0.0)
        M2[v] += d[v] * d2[v]
    sd = np.zeros(shape); ok = cnt >= 2
    sd[ok] = np.sqrt(M2[ok] / (cnt[ok] - 1))
    sd = np.maximum(sd, np.flip(sd, axis=0))   # bilateral pooling
    sd = np.maximum(sd, SD_FLOOR)               # floor
    return mean, sd, cnt


def pct_exceed(ai, valid, mean, sd, mask):
    z = np.zeros_like(ai)
    m = valid & (sd > 0) & mask
    z[m] = np.clip((ai[m] - mean[m]) / sd[m], -20, 20)
    n = int(m.sum())
    return 100.0 * np.sum(np.abs(z[m]) >= 1.96) / n if n else float('nan'), n


def run(which, perf_path_fn):
    print(f"\n{'='*60}\n  {which}\n{'='*60}")
    ai_c, v_c = [], []
    for cid in CTRL:
        p = perf_path_fn(cid, 'ctrl')
        if not p.exists():
            print(f"  skip missing {cid}"); continue
        ai, v = ai_map(nib.load(str(p)).get_fdata(dtype=np.float32))
        ai_c.append(ai); v_c.append(v)
    mean, sd, cnt = build_stats(ai_c, v_c)
    # GM-ish mask: voxels present in >=75% of controls
    mask = (cnt / len(ai_c)) >= 0.75
    print(f"  controls={len(ai_c)}  mask voxels={int(mask.sum()):,}  "
          f"median SD in mask={np.median(sd[mask]):.1f}%")
    # Control self-exceedance (leave-one-out approx: rebuild stats w/o i)
    cex = []
    for i, (ai, v) in enumerate(zip(ai_c, v_c)):
        loo_ai = ai_c[:i] + ai_c[i+1:]
        loo_v = v_c[:i] + v_c[i+1:]
        m_l, s_l, _ = build_stats(loo_ai, loo_v)
        pe, _ = pct_exceed(ai, v, m_l, s_l, mask)
        cex.append(pe)
    print(f"  CONTROL LOO %|z|>=1.96: mean={np.mean(cex):.1f}%  "
          f"range=[{np.min(cex):.1f},{np.max(cex):.1f}]  (target ~5%)")
    # Patients
    pex = []
    for pid in PAT:
        p = perf_path_fn(pid, 'pat')
        if not p.exists():
            print(f"  skip missing patient {pid}"); continue
        ai, v = ai_map(nib.load(str(p)).get_fdata(dtype=np.float32))
        pe, n = pct_exceed(ai, v, mean, sd, mask)
        pex.append((pid, pe))
        print(f"    {pid}: %|z|>=1.96 = {pe:.1f}%")
    print(f"  PATIENT mean %|z|>=1.96 = {np.mean([x[1] for x in pex]):.1f}%")
    return cex, pex


def new_path(cid, kind):
    return SYM / f"{cid}_perf_sym.nii.gz"

def old_path(cid, kind):
    if kind == 'ctrl':
        return BASE / f"DatasetControls/FM_20_39/{cid}/{cid}_MNISpace_perfusion_calib_upsampled.nii.gz"
    return BASE / f"Dataset/{cid}/{cid}_perfusion_calib_resampled_to_T1w.nii.gz"


if __name__ == "__main__":
    run("OLD (un-registered, original MNI)", old_path)
    run("NEW (symmetric-registered)", new_path)
