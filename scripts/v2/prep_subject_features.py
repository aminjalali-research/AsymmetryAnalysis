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


# --- partial-volume maps (2026-11-04, improvement list "partial-volume correction") -------------------
# GIF NeuroMorph segmentation volumes (verified on C001): 1 CSF, 2 cortical GM, 3 WM, 4 deep GM, 5 other brain,
# 0 and 6-8 non-brain.  The GM and WM probabilities are warped with the subject's own T1 -> T1_sym warp straight
# onto the mirror-symmetric subsampled grid, smoothed with the same FWHM as the CBF, and stored in the npz as
# `gm` / `wm`; analyze_v2.py --pvc then forms CBF_gm = (CBF - WM * CBF_wm) / GM where GM >= 0.3.
GIF_GM_VOLS = (2, 4)
GIF_WM_VOL = 3
GIF_BRAIN_VOLS = (1, 2, 3, 4, 5)


def ref_subgrid(feat_dir, template, sub):
    """Reference image (zeros) of the subsampled mirror-symmetric grid, for applywarp --ref."""
    out = feat_dir / "_ref_subgrid.nii.gz"
    if out.exists():
        return out
    t = nib.load(str(template)); A = t.affine.copy(); shape = t.shape[:3]
    sl = sym_slices(shape, sub)
    new = np.eye(4); new[:3, :3] = A[:3, :3] * sub; new[:3, 3] = (A @ np.array([sl[0].start, 0.0, 0.0, 1.0]))[:3]
    nshape = tuple(len(range(*s.indices(n))) for s, n in zip(sl, shape))
    nib.save(nib.Nifti1Image(np.zeros(nshape, np.float32), new), str(out))
    return out


def add_tissue(args):
    sid, seg_path, warp_path, ref_path, out_npz, fwhm = args
    import os, subprocess
    try:
        if not out_npz.exists():
            return sid, "no npz"
        z = dict(np.load(out_npz))
        if "gm" in z:
            return sid, "exists"
        tmp = out_npz.parent / f"_{sid}_seg_sub.nii.gz"
        subprocess.run(["applywarp", f"--in={seg_path}", f"--ref={ref_path}", f"--warp={warp_path}", f"--out={tmp}",
                        "--interp=trilinear", "--datatype=float"], check=True, capture_output=True,
                       env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"})
        S = nib.load(str(tmp)).get_fdata(dtype=np.float32); tmp.unlink()
        if S.shape[:3] != z["perf"].shape:
            return sid, f"shape mismatch {S.shape[:3]} vs {z['perf'].shape}"
        gm = S[..., list(GIF_GM_VOLS)].sum(-1); wm = S[..., GIF_WM_VOL]
        brain = (S[..., list(GIF_BRAIN_VOLS)].sum(-1) > 0.5).astype(np.float32)
        sigma = fwhm / 2.3548 / (0.8 * float(z["sub"]))
        den = ndimage.gaussian_filter(brain, sigma)
        def sm(x):
            num = ndimage.gaussian_filter(x * brain, sigma)
            return np.where(den > 0.5, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
        z["gm"] = sm(gm); z["wm"] = sm(wm); z["tissue_fwhm"] = fwhm
        np.savez_compressed(out_npz, **z)
        return sid, "ok"
    except Exception as e:  # noqa: BLE001
        return sid, f"ERROR {e!r}"


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
    ap.add_argument("--feat-dir", default=None, help="output directory (default v2_features/<dataset>)")
    ap.add_argument("--tissue", action="store_true",
                    help="chalfont only: add GIF gray/white-matter probability maps (`gm`, `wm`) to existing npz files")
    a = ap.parse_args()
    cfg = dataset_config.configure(a.dataset, perf_variant=getattr(a, "perf_variant", "tc"),
                                   band_policy=getattr(a, "band_policy", None))
    out_dir = Path(a.feat_dir) if a.feat_dir else BASE / "v2_features" / cfg.name; out_dir.mkdir(parents=True, exist_ok=True)
    ids = a.ids or sorted({*cfg.patient_ids(), *[c for g in cfg.available_groups() for c in cfg.control_ids(g)]})
    if a.tissue:
        if cfg.is_inhouse:
            sys.exit("--tissue needs the GIF segmentation of the external cohort; the in-house cohort has no tissue maps staged")
        ref = ref_subgrid(out_dir, BASE / "symreg/templates/T1_sym.nii.gz", a.sub)
        staged = BASE / "external_chalfont/staged"; symp = BASE / "external_chalfont/sym_perf"
        jobs = [(sid, staged / sid / "gif_seg.nii.gz", symp / f"{sid}_warp.nii.gz", ref, out_dir / f"{sid}.npz", a.fwhm)
                for sid in ids if (staged / sid / "gif_seg.nii.gz").exists() and (symp / f"{sid}_warp.nii.gz").exists()]
        print(f"tissue maps: {len(jobs)} subjects -> {out_dir}", flush=True)
        with Pool(a.jobs) as pool:
            for i, (sid, st) in enumerate(pool.imap_unordered(add_tissue, jobs), 1):
                print(f"  [{i}/{len(jobs)}] {sid}: {st}", flush=True)
        return
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
