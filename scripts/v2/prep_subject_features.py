#!/usr/bin/env python3
"""v2 pipeline, step 1: per-subject features for the enhanced asymmetry analysis.

Why (audit 2026-10-20, docs/audit/METHODS_AUDIT_2026-10-20.md): the voxel-wise AI at
0.8 mm has a control SD of ~55-70 % in gray matter, so realistic 10-20 % asymmetries are
undetectable. The enhanced analysis works on (a) region means and (b) heavily smoothed
maps on a coarse, mirror-symmetric grid, both derived here once per subject.

Per subject (symmetric-template space, 0.8 mm):
  * ROI table: mean CBF per label (physiological voxels 0 < CBF < 150) using the subject's
    own symmetric parcellation (in-house) or the fixed template atlas (chalfont), plus
    the gray-matter mean for global normalisation.
  * Smoothed map: Gaussian FWHM (default 8 mm) applied to CBF within the coverage mask
    (mask-normalised), then subsampled every `sub` voxels with axis-0 offset chosen so
    that flipping the subsampled grid still maps every voxel onto its mirror voxel
    (full grid 227 wide: indices 2 + 3k mirror onto 224 - 3k).
  * The parcellation subsampled on the same grid (nearest) for scope masks.

Outputs: v2_features/<dataset>/<ID>.npz  (perf float32, parc int16, meta) and
         v2_features/<dataset>/<ID>_roi.json
Usage: python scripts/v2/prep_subject_features.py --dataset inhouse|chalfont [--jobs 4]
"""
import argparse, json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, nibabel as nib
from scipy import ndimage

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
import dataset_config  # noqa: E402

CORTEX = list(range(1001, 1036)) + list(range(2001, 2036))
SUBCORT = [10, 49, 17, 53, 18, 54]
LABELS = SUBCORT + CORTEX


def sym_slices(shape, sub):
    nx = shape[0]
    c = (nx - 1) / 2.0
    assert float(c).is_integer(), "axis 0 must have odd length for an exact mirror grid"
    c = int(c)
    k = c // sub
    start = c - k * sub
    return (slice(start, nx, sub), slice(0, shape[1], sub), slice(0, shape[2], sub))


def process(args):
    sid, perf_path, parc_path, out_dir, fwhm, sub = args
    out_npz = out_dir / f"{sid}.npz"; out_json = out_dir / f"{sid}_roi.json"
    if out_npz.exists() and out_json.exists():
        return sid, "exists"
    try:
        img = nib.load(str(perf_path)); perf = img.get_fdata(dtype=np.float32)
        parc = np.asarray(nib.load(str(parc_path)).dataobj).astype(np.int32)
        if parc.shape != perf.shape:
            return sid, f"shape mismatch {parc.shape} vs {perf.shape}"
        # --- ROI table ---------------------------------------------------
        phys = (perf > 0) & (perf < 150)
        lab = np.where(phys, parc, 0).ravel()
        sums = np.bincount(lab, weights=perf.ravel(), minlength=2100)
        cnts = np.bincount(lab, minlength=2100)
        roi = {}
        for L in LABELS:
            roi[str(L)] = {"mean": float(sums[L] / cnts[L]) if cnts[L] > 0 else None, "n": int(cnts[L])}
        gm_lab = np.isin(parc, LABELS) & phys
        roi["gm_mean"] = float(perf[gm_lab].mean()) if gm_lab.any() else None
        roi["gm_n"] = int(gm_lab.sum())
        roi["brain_mean"] = float(perf[phys].mean()); roi["brain_n"] = int(phys.sum())
        with open(out_json, "w") as f:
            json.dump(roi, f)
        # --- smoothed, subsampled map -------------------------------------
        m = (perf > 0).astype(np.float32)
        sigma = fwhm / 2.3548 / 0.8
        num = ndimage.gaussian_filter(perf * m, sigma)
        den = ndimage.gaussian_filter(m, sigma)
        sm = np.where(den > 0.5, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
        sl = sym_slices(perf.shape, sub)
        np.savez_compressed(out_npz, perf=sm[sl], parc=parc[sl].astype(np.int16),
                            fwhm=fwhm, sub=sub, offset=sl[0].start, shape=np.array(perf.shape))
        return sid, "ok"
    except Exception as e:  # noqa: BLE001
        return sid, f"ERROR {e!r}"


def main():
    ap = argparse.ArgumentParser(); dataset_config.add_dataset_args(ap)
    ap.add_argument("--fwhm", type=float, default=8.0); ap.add_argument("--sub", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    cfg = dataset_config.configure(a.dataset, perf_variant=getattr(a, "perf_variant", "tc"),
                                   band_policy=getattr(a, "band_policy", None))
    out_dir = BASE / "v2_features" / cfg.name; out_dir.mkdir(parents=True, exist_ok=True)
    ids = a.ids or sorted({*cfg.patient_ids(), *[c for g in cfg.available_groups() for c in cfg.control_ids(g)]})
    jobs = []
    for sid in ids:
        pp = cfg.perf_path(sid)
        if not pp.exists():
            print(f"  skip {sid}: no perfusion"); continue
        jobs.append((sid, pp, cfg.subject_parc_path(sid), out_dir, a.fwhm, a.sub))
    print(f"dataset={cfg.name} subjects={len(jobs)} fwhm={a.fwhm} sub={a.sub} jobs={a.jobs} -> {out_dir}", flush=True)
    with Pool(a.jobs) as pool:
        for i, (sid, st) in enumerate(pool.imap_unordered(process, jobs), 1):
            print(f"  [{i}/{len(jobs)}] {sid}: {st}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
