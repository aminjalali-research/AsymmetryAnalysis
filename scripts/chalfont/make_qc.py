#!/usr/bin/env python3
"""Registration QC for Chalfont subjects in symmetric template space.

Per subject: external_chalfont/qc/<ID>.png with three rows
  1. T1_sym (2 mm) with contours of the warped T1 (red) -> T1->template fit
  2. T1_sym with the warped transit-corrected CBF overlay          -> ASL->T1->template chain
  3. native T1_brain with the TC-CBF resampled by asl2t1.mat (spline, inside the T1 brain + 2 mm) -> ASL->T1 fit
and one row in external_chalfont/qc/registration_qc.csv:
  asl_t1_method        bbr (normmi refined by boundary-based registration) or normmi
  asl_t1_gm_corr       ASL->T1 fit: correlation of the CBF with the GIF cortex indicator inside the T1 brain
                       (within the ASL slab); 0.40 median after BBR, < 0.20 flagged (low_asl_t1_fit)
  asl_t1_vent_ratio    median CBF in the lateral ventricles (GIF 50-53) / median CBF in the cortex; ~0.2 is
                       what a perfectly aligned map gives at 4-mm slices, 0.35 the cohort median
  ncc_t1_template      normalized cross-correlation warped T1 (2 mm) vs template, template mask
  mean_abs_ai_t1       mean |100 (x - flip0 x)/mean| of the warped T1 at 2 mm (flip-symmetry sanity)
  mean_abs_ai_t1_tissue  same, restricted to voxels whose mirror pair are both > 30 % of p98
  mean_tccbf_pos       mean TC-CBF over template-mask voxels with CBF > 0 (ml/100g/min)
  mean_tccbf_mask      mean TC-CBF over all template-mask voxels (zeros outside the slab included)
  coverage             fraction of template-mask voxels with CBF > 0
  coverage_left/right  the same per hemisphere (world x < 0 is left on this RAS grid)
  coverage_lr_ratio    min/max of the two
  frac_negative        fraction of template-mask voxels with CBF < 0 (noise)
  flag                 '' or ';'-joined reasons (low_ncc, low_coverage, lr_asymmetric, high_t1_ai, low_cbf)

Usage: make_qc.py [--ids ID ...] [--no-png]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chalfont_io as cio  # noqa: E402

# Reference values from the in-house pipeline (P013 warped T1 vs template, same
# functions): NCC 0.71, mean |AI| 14.2 % (tissue-restricted 13.1 %); C001 pilot:
# NCC 0.82, tissue |AI| 14.6 %.
FLAG_NCC = 0.70
FLAG_COVERAGE = 0.50
FLAG_LR_RATIO = 0.90
FLAG_T1_AI = 20.0
FLAG_CBF = 20.0   # ml/100g/min; mean TC-CBF over covered template-mask voxels below this is suspicious


def world_to_index(aff: np.ndarray, xyz) -> tuple:
    v = np.linalg.inv(aff) @ np.array([*xyz, 1.0])
    return tuple(int(round(c)) for c in v[:3])


VENTRICLES = (50, 51, 52, 53)      # GIF v3: inferior lateral and lateral ventricles, both sides
FLAG_ASL_T1 = 0.20                 # asl_t1_gm_corr below this: the ASL -> T1 fit is suspect


def asl_in_t1(sid: str, what: str = "tcflow", interp: str = "spline") -> np.ndarray | None:
    """An ASL-space image resampled into the T1 with the pipeline's asl2t1.mat (None without the matrix)."""
    st = cio.STAGED / sid
    mat = cio.SYM_PERF / f"{sid}_asl2t1.mat"
    if not mat.exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "in_t1.nii.gz")
        subprocess.run(["flirt", "-in", str(st / f"{what}.nii.gz"), "-ref", str(st / "T1_brain.nii.gz"),
                        "-applyxfm", "-init", str(mat), "-out", out, "-interp", interp],
                       check=True, env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return nib.load(out).get_fdata(dtype=np.float32)


def asl_t1_fit(sid: str) -> dict:
    """How well the CBF sits in the T1 (the ASL -> T1 step on its own): correlation with the cortex indicator and the
    ventricle / cortex CBF ratio, inside the T1 brain and the ASL slab. Also returns the masked CBF for the panel."""
    from scipy import ndimage
    st = cio.STAGED / sid
    cbf = asl_in_t1(sid, "tcflow", "spline")
    ref = asl_in_t1(sid, "asl_ref", "trilinear")
    out = dict(asl_t1_method="", asl_t1_gm_corr="", asl_t1_vent_ratio="", cbf_masked=None)
    if cbf is None:
        return out
    init = cio.SYM_PERF / f"{sid}_asl2t1_init.mat"
    final = cio.SYM_PERF / f"{sid}_asl2t1.mat"
    if init.exists():
        out["asl_t1_method"] = "normmi" if np.allclose(np.loadtxt(init), np.loadtxt(final)) else "bbr"
    else:
        out["asl_t1_method"] = "normmi_wholehead"       # matrices from before 2026-11-18
    mask = nib.load(str(st / "brain_mask.nii.gz")).get_fdata() > 0
    out["cbf_masked"] = cbf * ndimage.binary_dilation(mask, iterations=2)
    parc_p = st / "gif_parc.nii.gz"
    if not parc_p.exists():
        return out
    parc = np.asanyarray(nib.load(str(parc_p)).dataobj)
    fov = ndimage.binary_erosion(ref > 0, iterations=3)
    cortex, vent = (parc >= 100) & fov, np.isin(parc, VENTRICLES) & fov
    inb = mask & fov & (cbf != 0)
    mc = float(np.median(cbf[cortex & (cbf > 0)])) if (cortex & (cbf > 0)).any() else 0.0
    if inb.sum() > 100:
        out["asl_t1_gm_corr"] = round(float(np.corrcoef(cbf[inb], (parc[inb] >= 100).astype(float))[0, 1]), 4)
    if mc > 0 and (vent & (cbf > 0)).any():
        out["asl_t1_vent_ratio"] = round(float(np.median(cbf[vent & (cbf > 0)]) / mc), 4)
    return out


def panel_slices(vol: np.ndarray, aff: np.ndarray, cuts_world) -> list:
    i, j, k = world_to_index(aff, cuts_world)
    i = int(np.clip(i, 0, vol.shape[0] - 1)); j = int(np.clip(j, 0, vol.shape[1] - 1)); k = int(np.clip(k, 0, vol.shape[2] - 1))
    return [vol[:, :, k].T, vol[:, j, :].T, vol[i, :, :].T]   # axial, coronal, sagittal


def qc_subject(sid: str, group: str, tmpl2: nib.Nifti1Image, make_png: bool = True) -> dict:
    sp = cio.SYM_PERF
    t1w_img = nib.load(str(sp / f"{sid}_t1_sym_2mm.nii.gz"))
    t1w = t1w_img.get_fdata(dtype=np.float32)
    tdat = tmpl2.get_fdata(dtype=np.float32)
    aff = tmpl2.affine
    mask = cio.template_mask(tdat)
    perf_full = nib.load(str(sp / f"{sid}_perf_sym.nii.gz"))
    perf = cio.resample_to_sym_2mm(perf_full, order=1).get_fdata(dtype=np.float32)

    a, b = t1w[mask], tdat[mask]
    ncc = float(np.corrcoef(a, b)[0, 1])
    ai = cio.flip_ai_stats(t1w, mask)
    # Tissue-restricted variant (both mirrored voxels above 30% of the 98th percentile):
    # insensitive to zero/near-zero voxels at the brain-mask edge and in sulci.
    p98 = np.percentile(t1w[t1w > 0], 98) if (t1w > 0).any() else 1.0
    ai_t = cio.flip_ai_stats(t1w, mask & (t1w > 0.3 * p98) & (np.flip(t1w, axis=0) > 0.3 * p98))
    pos = (perf > 0) & mask
    xw = aff[0, 0] * np.arange(perf.shape[0]) + aff[0, 3]
    left = (xw < 0)[:, None, None] & mask
    right = (xw > 0)[:, None, None] & mask
    cov = pos.sum() / mask.sum()
    cov_l = (pos & left).sum() / left.sum()
    cov_r = (pos & right).sum() / right.sum()
    # Flip-consistency (2026-11-04, improvement list): Dice of the warped T1 tissue mask, and of the
    # perfusion coverage, with their own left-right mirror. A mis-registered or asymmetric subject
    # scores low; the exclusion rule (median - 3 MAD over all subjects) is applied in main().
    tis = mask & (t1w > 0.3 * p98)
    dice_t1 = 2.0 * (tis & np.flip(tis, axis=0)).sum() / max(1, 2 * tis.sum())
    dice_perf = 2.0 * (pos & np.flip(pos, axis=0)).sum() / max(1, 2 * pos.sum())
    row = dict(
        id=sid, group=group,
        ncc_t1_template=round(ncc, 4),
        mirror_dice_t1=round(float(dice_t1), 4), mirror_dice_perf=round(float(dice_perf), 4),
        mean_abs_ai_t1=round(ai["mean_abs_ai"], 2), median_abs_ai_t1=round(ai["median_abs_ai"], 2),
        mean_abs_ai_t1_tissue=round(ai_t["mean_abs_ai"], 2),
        mean_tccbf_pos=round(float(perf[pos].mean()) if pos.any() else 0.0, 2),
        mean_tccbf_mask=round(float(perf[mask].mean()), 2),
        coverage=round(float(cov), 4), coverage_left=round(float(cov_l), 4), coverage_right=round(float(cov_r), 4),
        coverage_lr_ratio=round(float(min(cov_l, cov_r) / max(cov_l, cov_r)) if max(cov_l, cov_r) > 0 else 0.0, 4),
        frac_negative=round(float(((perf < 0) & mask).sum() / mask.sum()), 4),
    )
    fit = asl_t1_fit(sid)
    row.update({k: fit[k] for k in ("asl_t1_method", "asl_t1_gm_corr", "asl_t1_vent_ratio")})
    flags = []
    if row["asl_t1_gm_corr"] != "" and row["asl_t1_gm_corr"] < FLAG_ASL_T1: flags.append("low_asl_t1_fit")
    if row["ncc_t1_template"] < FLAG_NCC: flags.append("low_ncc")
    if row["coverage"] < FLAG_COVERAGE: flags.append("low_coverage")
    if row["coverage_lr_ratio"] < FLAG_LR_RATIO: flags.append("lr_asymmetric")
    if row["mean_abs_ai_t1_tissue"] > FLAG_T1_AI: flags.append("high_t1_ai")
    if row["mean_tccbf_pos"] < FLAG_CBF: flags.append("low_cbf")
    row["flag"] = ";".join(flags)

    if not make_png:
        return row
    cuts = (-30.0, -20.0, 0.0)   # sagittal x (left temporal), coronal y (hippocampus), axial z
    tp = panel_slices(tdat, aff, cuts)
    wp = panel_slices(t1w, aff, cuts)
    pp = panel_slices(perf, aff, cuts)
    mp = panel_slices(mask.astype(np.float32), aff, cuts)
    cbf_t1 = fit["cbf_masked"]
    st = cio.STAGED / sid
    t1n_img = nib.load(str(st / "T1_brain.nii.gz"))
    t1n = t1n_img.get_fdata(dtype=np.float32)
    nat_cuts = tuple(c for c in (t1n_img.affine @ np.array([*[s / 2 for s in t1n.shape], 1.0]))[:3])
    npnl = panel_slices(t1n, t1n_img.affine, nat_cuts)
    rpnl = panel_slices(cbf_t1, t1n_img.affine, nat_cuts) if cbf_t1 is not None else None

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10))
    titles = ["axial z=0", "coronal y=-20", "sagittal x=-30"]
    tv = np.percentile(tdat[mask], 99)
    wlev = [0.35 * np.percentile(t1w[t1w > 0], 98), 0.75 * np.percentile(t1w[t1w > 0], 98)] if (t1w > 0).any() else [1]
    pv = np.percentile(perf[pos], 98) if pos.any() else 1.0
    for c in range(3):
        ax = axes[0, c]
        ax.imshow(tp[c], cmap="gray", vmin=0, vmax=tv, origin="lower")
        if wp[c].max() > 0:
            ax.contour(wp[c], levels=wlev, colors=["r", "y"], linewidths=0.5)
        ax.set_title(f"{sid} T1->sym: template + warped-T1 contours ({titles[c]})", fontsize=7)
        ax = axes[1, c]
        ax.imshow(tp[c], cmap="gray", vmin=0, vmax=tv, origin="lower")
        over = np.ma.masked_where((pp[c] <= 0) | (mp[c] <= 0), pp[c])     # inside the template brain only
        ax.imshow(over, cmap="jet", vmin=0, vmax=pv, alpha=0.55, origin="lower")
        ax.set_title(f"TC-CBF on T_sym inside the template brain (0..{pv:.0f}) cov={row['coverage']:.2f}", fontsize=7)
        ax = axes[2, c]
        nv = np.percentile(t1n[t1n > 0], 99)
        ax.imshow(npnl[c], cmap="gray", vmin=0, vmax=nv, origin="lower")
        if rpnl is not None and rpnl[c].max() > 0:
            ax.imshow(np.ma.masked_where(rpnl[c] < 10, rpnl[c]), cmap="hot", vmin=10, vmax=80, alpha=0.5, origin="lower")
        ax.set_title(f"ASL->T1 ({row['asl_t1_method']}): native T1_brain + CBF >= 10, brain-masked", fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{sid} ({group})  NCC={row['ncc_t1_template']:.3f}  |AI|T1={row['mean_abs_ai_t1']:.1f}%  "
                 f"meanCBF={row['mean_tccbf_pos']:.1f}  cov L/R={row['coverage_left']:.2f}/{row['coverage_right']:.2f}  "
                 f"ASL->T1 GM corr={row['asl_t1_gm_corr']} vent/cortex={row['asl_t1_vent_ratio']}  {row['flag']}",
                 fontsize=8.5)
    fig.tight_layout()
    cio.QC.mkdir(parents=True, exist_ok=True)
    fig.savefig(cio.QC / f"{sid}.png", dpi=110)
    plt.close(fig)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()
    groups = {}
    man = cio.META / "staging_manifest.csv"
    if man.exists():
        with open(man) as fh:
            groups = {r["id"]: r["group"] for r in csv.DictReader(fh)}
    tmpl2 = cio.ensure_template_2mm()
    ids = a.ids or sorted(p.name[: -len("_perf_sym.nii.gz")] for p in cio.SYM_PERF.glob("*_perf_sym.nii.gz"))
    out_csv = cio.QC / "registration_qc.csv"
    rows = {}
    if out_csv.exists():
        with open(out_csv) as fh:
            rows = {r["id"]: r for r in csv.DictReader(fh)}
    cols = ["id", "group", "ncc_t1_template", "mirror_dice_t1", "mirror_dice_perf", "mean_abs_ai_t1", "median_abs_ai_t1",
            "mean_abs_ai_t1_tissue", "mean_tccbf_pos",
            "mean_tccbf_mask", "coverage", "coverage_left", "coverage_right", "coverage_lr_ratio",
            "frac_negative", "asl_t1_method", "asl_t1_gm_corr", "asl_t1_vent_ratio", "flag"]
    for sid in ids:
        if not (cio.SYM_PERF / f"{sid}_t1_sym_2mm.nii.gz").exists():
            print(f"skip {sid}: no outputs")
            continue
        try:
            rows[sid] = qc_subject(sid, groups.get(sid, ""), tmpl2, make_png=not a.no_png)
            print(sid, {k: rows[sid][k] for k in ("ncc_t1_template", "mean_abs_ai_t1", "mean_tccbf_pos", "coverage", "coverage_lr_ratio", "flag")})
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {sid}: {type(exc).__name__}: {exc}")
    # Pre-specified flip-consistency rule over the whole set: mirror Dice below median - 3 MAD -> low_mirror_dice
    vals = {k: float(rows[k]["mirror_dice_t1"]) for k in rows if str(rows[k].get("mirror_dice_t1", "")) not in ("", "nan")}
    if vals:
        v = np.array(list(vals.values())); med = float(np.median(v)); mad = float(np.median(np.abs(v - med))) * 1.4826
        cut = med - 3 * mad
        for k, d in vals.items():
            fl = [f for f in str(rows[k].get("flag", "")).split(";") if f and f != "low_mirror_dice"]
            if d < cut:
                fl.append("low_mirror_dice")
            rows[k]["flag"] = ";".join(fl)
        print(f"mirror Dice (T1 tissue vs its flip): median {med:.4f}, MAD-SD {mad:.4f}, cut {cut:.4f}, "
              f"flagged {sum(d < cut for d in vals.values())}")
    cio.QC.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for sid in sorted(rows):
            w.writerow({c: rows[sid].get(c, "") for c in cols})
    print("wrote", out_csv, len(rows), "rows")


if __name__ == "__main__":
    main()
