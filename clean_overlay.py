#!/usr/bin/env python3
"""
clean_overlay.py — Island-free, direction-aware overlay visualization for the
AsymmetryAnalysis clinical ASL epilepsy pipeline.

THE PROBLEM
-----------
The per-voxel zAI map
  ``results_zscore/asymmetry/patients/<PID>/<PID>_asymmetry_zscore.nii.gz``
and the per-voxel z-score map
  ``results_zscore/patients/<PID>/<PID>_vs_<GROUP>_zscore.nii.gz``
are RAW: they are only thresholded for display, never cluster-filtered. Isolated
voxels ("islands") that pass ``|z| >= thr`` but are not part of a real cluster
appear as speckle in FSLeyes and publication figures. This module cleans them.

WHAT IT DOES
------------
``clean_overlay_map()`` thresholds the stat map (within an optional mask),
labels connected components, drops components smaller than ``min_cluster``, then
applies morphological binary opening (remove 1-2 voxel speckle) followed by
binary closing (fill pinholes). It preserves the original signed values inside
surviving clusters and reports how many islands / voxels were removed.

``render_overlay_montage()`` renders a feature-rich matplotlib montage:
  - grayscale anatomical underlay (mean_perfusion or patient T1w)
  - direction-aware diverging overlay (hyperperfusion warm / hypoperfusion cool),
    alpha-blended, zeros NaN-masked so the underlay shows through
  - 37-ROI clinical mask boundary contour
  - multi-slice axial montage spanning the cluster extent (+ coronal/sagittal
    through the peak cluster)
  - colorbar, slice-position labels, peak-cluster marker, rich title
  - a before/after QC panel (raw thresholded vs island-cleaned)

It reuses helpers from ``04_publication_figures.py`` and ``02_compute_zai.py``
(loaded dynamically because those filenames start with digits) rather than
reinventing them.

OUTPUTS
-------
  - ``results_zscore/asymmetry/patients/<PID>/<PID>_zai_cleaned.nii.gz``
    (so FSLeyes shows island-free overlays)
  - ``visual_analysis/<PID>_<space>_clean_overlay.png`` (montage)
  - ``visual_analysis/<PID>_<space>_clean_overlay_qc.png`` (before/after QC)

CLI
---
    python clean_overlay.py --patient P015
    python clean_overlay.py --patient P013 --threshold 3.0 --min-cluster 50
    python clean_overlay.py --all
    python clean_overlay.py --patient P015 --space zscore   # raw z-score map
    python clean_overlay.py --patient P015 --no-roi-mask
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results_zscore"
ASYMMETRY_DIR = RESULTS_DIR / "asymmetry"
DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR = BASE_DIR / "visual_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Dynamic import of digit-prefixed sibling modules (can't `import 04_...`)
# ----------------------------------------------------------------------------
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_pubfig = _load_module(BASE_DIR / "04_publication_figures.py", "pubfig_helpers")
# Reused helpers from 04_publication_figures.py
load_zai_map = _pubfig.load_zai_map
load_clinical_roi_mask = _pubfig.load_clinical_roi_mask
load_patient_demographics = _pubfig.load_patient_demographics
CLINICAL_ROI_LABELS = _pubfig.CLINICAL_ROI_LABELS

# Connectivity for connected-component labeling (matches the pipeline).
from src.thresholding import get_structure_3d


# ============================================================================
# GROUP RESOLUTION
# ============================================================================
def resolve_patient_group(patient_id: str) -> str | None:
    """Determine the patient's canonical control group.

    Each patient is matched to one FM_* age band (the canonical groups per
    CLAUDE.md). We read the band from the per-patient z-score filenames in
    ``results_zscore/patients/<PID>/`` and prefer an FM_* group when present.
    """
    pdir = RESULTS_DIR / "patients" / patient_id
    if not pdir.exists():
        return None
    groups = []
    for f in pdir.glob(f"{patient_id}_vs_*_zscore.nii.gz"):
        stem = f.name.replace(f"{patient_id}_vs_", "").replace("_zscore.nii.gz", "")
        groups.append(stem)
    if not groups:
        return None
    fm = [g for g in groups if g.startswith("FM_")]
    return sorted(fm)[0] if fm else sorted(groups)[0]


def load_group_perfusion(group: str):
    """Load per-voxel control perfusion mean + SD for a group (for direction
    inference) plus the anatomical mean_perfusion underlay and brain mask."""
    gdir = RESULTS_DIR / "groups" / group
    mean_perf = nib.load(str(gdir / "mean_perfusion.nii.gz")).get_fdata(dtype=np.float32)
    sd_perf = nib.load(str(gdir / "sd_perfusion.nii.gz")).get_fdata(dtype=np.float32)
    asym_mask = ASYMMETRY_DIR / "groups" / group / "brain_mask.nii.gz"
    if asym_mask.exists():
        mask = nib.load(str(asym_mask)).get_fdata().astype(bool)
    else:
        mask = nib.load(str(gdir / "brain_mask.nii.gz")).get_fdata().astype(bool)
    return mean_perf, sd_perf, mask


# ============================================================================
# CORE: ISLAND CLEANING
# ============================================================================
def clean_overlay_map(stat_map, threshold, min_cluster=50, mask=None,
                      open_iter=1, close_iter=1):
    """Remove speckle "islands" from a thresholded statistical map.

    Pipeline:
      1. binary threshold ``|stat_map| >= threshold`` (within ``mask`` if given)
      2. connected-component label (pipeline connectivity from
         ``get_structure_3d()``)
      3. drop components with < ``min_cluster`` voxels
      4. morphological ``binary_opening(open_iter)`` — removes 1-2 voxel
         speckle islands that survived only because they touched a larger blob
      5. morphological ``binary_closing(close_iter)`` — fills pinholes inside
         surviving clusters
      6. re-label and drop any components that fell below ``min_cluster`` after
         the morphology step (opening can split clusters)

    The returned ``cleaned`` map keeps the ORIGINAL signed values inside the
    surviving mask and zeros everywhere else.

    Parameters
    ----------
    stat_map : np.ndarray
        Signed statistical map (zAI or z-score).
    threshold : float
        Magnitude threshold (|x| >= threshold survives).
    min_cluster : int
        Minimum surviving connected-component size in voxels.
    mask : np.ndarray of bool, optional
        Spatial restriction (e.g. 37-ROI clinical mask). Voxels outside are
        never considered.
    open_iter, close_iter : int
        Morphological iterations. Set to 0 to skip that step.

    Returns
    -------
    cleaned : np.ndarray
        Signed stat map, island-free.
    report : dict
        Diagnostics: raw_suprathreshold_voxels, raw_cluster_count,
        n_islands_removed, voxels_removed, surviving_cluster_count,
        surviving_voxels, min_surviving_cluster_size.
    """
    stat_map = np.asarray(stat_map, dtype=np.float32)
    supra = np.abs(stat_map) >= threshold
    if mask is not None:
        supra = supra & mask.astype(bool)

    raw_supra_vox = int(supra.sum())
    structure = get_structure_3d()

    # --- raw connected components (pre-cleaning) ---
    raw_labeled, raw_n = ndimage.label(supra, structure=structure)
    if raw_n > 0:
        raw_sizes = ndimage.sum(supra, raw_labeled, range(1, raw_n + 1))
    else:
        raw_sizes = np.array([])
    raw_cluster_count = int(raw_n)

    # --- step 1: size filter ---
    keep = np.zeros_like(supra)
    for i, s in enumerate(raw_sizes, start=1):
        if s >= min_cluster:
            keep |= (raw_labeled == i)

    # --- step 2: morphological opening (speckle removal) then closing ---
    cleaned_mask = keep
    if open_iter and open_iter > 0:
        cleaned_mask = ndimage.binary_opening(
            cleaned_mask, structure=structure, iterations=int(open_iter))
    if close_iter and close_iter > 0:
        cleaned_mask = ndimage.binary_closing(
            cleaned_mask, structure=structure, iterations=int(close_iter))

    # --- step 3: re-enforce min size (opening can fragment clusters) ---
    final_labeled, final_n = ndimage.label(cleaned_mask, structure=structure)
    final = np.zeros_like(cleaned_mask)
    surviving_sizes = []
    if final_n > 0:
        fsizes = ndimage.sum(cleaned_mask, final_labeled, range(1, final_n + 1))
        for i, s in enumerate(fsizes, start=1):
            if s >= min_cluster:
                final |= (final_labeled == i)
                surviving_sizes.append(int(s))

    # --- assemble signed output ---
    # Keep original signed values inside the surviving mask. Pinhole voxels
    # added by binary_closing that were exactly 0 in the original are nudged to
    # the local cluster sign * threshold so the saved NIfTI's nonzero support
    # equals the morphological mask (no re-introduced holes / phantom islands).
    cleaned = np.zeros_like(stat_map)
    cleaned[final] = stat_map[final]
    filled = final & (stat_map == 0)
    if filled.any():
        # assign sign from nearest nonzero neighbour via cluster mean sign
        lbl, ln = ndimage.label(final, structure=structure)
        for ci in range(1, ln + 1):
            cm = lbl == ci
            vals = stat_map[cm & (stat_map != 0)]
            if vals.size:
                s = np.sign(vals.mean()) or 1.0
                holes = cm & filled
                cleaned[holes] = s * threshold

    surviving_vox = int(final.sum())
    report = {
        "threshold": float(threshold),
        "min_cluster": int(min_cluster),
        "open_iter": int(open_iter),
        "close_iter": int(close_iter),
        "raw_suprathreshold_voxels": raw_supra_vox,
        "raw_cluster_count": raw_cluster_count,
        "raw_islands_below_min": int((raw_sizes < min_cluster).sum()) if raw_n else 0,
        "surviving_cluster_count": len(surviving_sizes),
        "surviving_voxels": surviving_vox,
        "voxels_removed": raw_supra_vox - surviving_vox,
        "n_islands_removed": raw_cluster_count - len(surviving_sizes),
        "min_surviving_cluster_size": min(surviving_sizes) if surviving_sizes else 0,
    }
    return cleaned, report


def count_subminsize_components(cleaned, min_cluster):
    """QC: count connected components below ``min_cluster`` in the cleaned map.

    Counts components of the cleaned map's NONZERO support (not a re-threshold):
    every surviving voxel in the cleaned NIfTI is part of a real, retained
    cluster, so this directly proves the saved NIfTI is island-free. Used to
    show the cleaned NIfTI has zero sub-min-size components.
    """
    support = np.asarray(cleaned) != 0
    structure = get_structure_3d()
    labeled, n = ndimage.label(support, structure=structure)
    if n == 0:
        return 0, 0
    sizes = ndimage.sum(support, labeled, range(1, n + 1))
    return int((sizes < min_cluster).sum()), int(n)


# ============================================================================
# DIRECTION-AWARE COLORING
# ============================================================================
def perfusion_direction_map(cleaned, mean_perf, sd_perf, brain_mask,
                            z_thr=2.0):
    """Per-voxel direction-of-abnormality (+1 hyper / -1 hypo / 0 ambiguous).

    For each surviving zAI voxel we look at the per-side control perfusion
    z-score ``z = (patient - mean) / sd`` — but the patient perfusion isn't
    available here voxel-wise without reloading; instead we approximate the
    direction using the SIGN of the control-referenced perfusion deviation
    encoded by the cluster_report when available. As a robust fallback used by
    the renderer, the diverging colormap simply maps zAI sign, and the
    cluster-level direction summary (from the CSV) is shown in the title.

    This function returns the cleaned zAI itself; direction coloring in the
    montage is driven by the cluster_report ``direction_class`` for the title
    and by zAI sign for the per-voxel diverging map. Kept as a hook for future
    voxelwise direction overlays.
    """
    return cleaned


def load_direction_summary(patient_id: str):
    """Read direction_class / ez_side_pred from the asymmetry cluster report.

    Returns (summary_str, dominant_dir, csv_path) or (None, None, None).
    """
    csv_path = (ASYMMETRY_DIR / "patients" / patient_id /
                f"{patient_id}_asymmetry_cluster_report.csv")
    if not csv_path.exists():
        return None, None, None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None, None, csv_path
    if "direction_class" not in df.columns:
        return None, None, csv_path
    # Weight direction by cluster size.
    sizes = df.get("size_voxels", pd.Series([1] * len(df)))
    counts = {}
    for cls, sz in zip(df["direction_class"], sizes):
        counts[cls] = counts.get(cls, 0) + int(sz)
    if not counts:
        return None, None, csv_path
    n_hyper = (df["direction_class"] == "hyper").sum()
    n_hypo = (df["direction_class"] == "hypo").sum()
    dominant = "hyper" if n_hyper >= n_hypo else "hypo"
    parts = [f"{k}:{v}" for k, v in sorted(counts.items(),
                                           key=lambda kv: -kv[1])]
    summary = (f"clusters hyper={n_hyper} hypo={n_hypo} "
               f"mixed={(df['direction_class'] == 'mixed').sum()} "
               f"subtle={(df['direction_class'] == 'subtle').sum()}")
    # Side prediction (size-weighted majority of ez_side_pred).
    if "ez_side_pred" in df.columns:
        side_w = {}
        for side, sz in zip(df["ez_side_pred"], sizes):
            side_w[side] = side_w.get(side, 0) + int(sz)
        if side_w:
            ez_side = max(side_w, key=side_w.get)
            summary += f" | EZ side pred={ez_side}"
    return summary, dominant, csv_path


# ============================================================================
# RENDERING
# ============================================================================
def _peak_cluster_centroid(cleaned):
    """Return (i,j,k) of the peak |value| voxel and the bounding slice range."""
    absmap = np.abs(cleaned)
    if absmap.max() == 0:
        c = tuple(s // 2 for s in cleaned.shape)
        return c, (0, cleaned.shape[2])
    peak = np.unravel_index(np.argmax(absmap), absmap.shape)
    nz = np.where(absmap > 0)
    kmin, kmax = int(nz[2].min()), int(nz[2].max())
    return tuple(int(x) for x in peak), (kmin, kmax)


def _ortho(img, axis, idx):
    """Extract an oriented 2-D slice for display (radiological-ish)."""
    idx = int(np.clip(idx, 0, img.shape[axis] - 1))  # guard against off-by-one / shape mismatch
    if axis == 2:      # axial
        sl = img[:, :, idx]
    elif axis == 1:    # coronal
        sl = img[:, idx, :]
    else:              # sagittal
        sl = img[idx, :, :]
    return np.rot90(sl)


def render_overlay_montage(patient_id, cleaned, raw_thresholded, anatomy,
                           roi_mask, group, threshold, report,
                           direction_summary=None, dominant_dir=None,
                           display_max=None, out_png=None, qc_png=None,
                           space="zai"):
    """Render the multi-slice montage + before/after QC panel.

    Parameters
    ----------
    cleaned : np.ndarray         island-free signed stat map
    raw_thresholded : np.ndarray raw thresholded signed stat map (QC)
    anatomy : np.ndarray         grayscale underlay
    roi_mask : np.ndarray|None   37-ROI boolean mask (contour)
    group : str                  control group key (for title)
    """
    demos = load_patient_demographics()
    label = demos.get(patient_id, patient_id)

    # Display range: post-SD-fix peak |zAI| ~9-10; default to a sensible cap.
    if display_max is None:
        amax = float(np.abs(cleaned).max())
        display_max = max(threshold + 1.0, min(amax, 10.0)) if amax > 0 else threshold + 1.0
    norm = TwoSlopeNorm(vmin=-display_max, vcenter=0.0, vmax=display_max)
    # Direction-aware diverging cmap: + (left-dominant) red/warm, - cool/blue.
    cmap = plt.get_cmap("RdBu_r").copy()

    peak, (kmin, kmax) = _peak_cluster_centroid(cleaned)

    # Pick 12 evenly spaced axial slices spanning the cluster extent.
    n_axial = 12
    if kmax - kmin < n_axial:
        kmin = max(0, peak[2] - 6)
        kmax = min(cleaned.shape[2] - 1, peak[2] + 6)
    slice_ks = np.linspace(kmin, kmax, n_axial).astype(int)

    anat_disp = anatomy.copy()
    a_lo, a_hi = np.percentile(anat_disp[anat_disp > 0], [1, 99]) if (anat_disp > 0).any() else (0, 1)

    # ---------- main montage: 3 rows x 4 cols axial + ortho row ----------
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.1], hspace=0.12, wspace=0.04)

    overlay = np.where(np.abs(cleaned) > 0, cleaned, np.nan)

    def _draw(ax, axis, idx, tag):
        a = _ortho(anat_disp, axis, idx)
        o = _ortho(overlay, axis, idx)
        ax.imshow(a, cmap="gray", vmin=a_lo, vmax=a_hi, origin="upper")
        ax.imshow(o, cmap=cmap, norm=norm, alpha=0.78, origin="upper",
                  interpolation="nearest")
        if roi_mask is not None:
            rm = _ortho(roi_mask.astype(float), axis, idx)
            if rm.any():
                ax.contour(rm, levels=[0.5], colors="lime",
                           linewidths=0.6, alpha=0.7)
        ax.set_title(tag, fontsize=9, color="white",
                     backgroundcolor="black", pad=1)
        ax.axis("off")

    # axial montage (first 3 rows = 12 slices)
    for n, k in enumerate(slice_ks):
        ax = fig.add_subplot(gs[n // 4, n % 4])
        _draw(ax, 2, k, f"z={k}")

    # bottom row: coronal + sagittal through peak, peak axial w/ marker, cbar
    ax_cor = fig.add_subplot(gs[3, 0])
    _draw(ax_cor, 1, peak[1], f"coronal y={peak[1]}")
    ax_sag = fig.add_subplot(gs[3, 1])
    _draw(ax_sag, 0, peak[0], f"sagittal x={peak[0]}")

    ax_peak = fig.add_subplot(gs[3, 2])
    _draw(ax_peak, 2, peak[2], f"PEAK z={peak[2]}")
    # mark peak location on the rotated axial slice
    a_disp = _ortho(anat_disp, 2, peak[2])
    pj = a_disp.shape[0] - 1 - peak[1]
    pi = peak[0]
    ax_peak.plot(pi, pj, marker="o", mfc="none", mec="yellow",
                 mew=1.8, ms=16)
    ax_peak.annotate(f"|zAI|={np.abs(cleaned).max():.1f}",
                     (pi, pj), color="yellow", fontsize=8,
                     xytext=(4, 4), textcoords="offset points")

    # colorbar + legend cell
    ax_cb = fig.add_subplot(gs[3, 3])
    ax_cb.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_cb, fraction=0.5, pad=0.02)
    cbar.set_label("zAI  (+ left-dominant / - right-dominant)", fontsize=9)
    legend_handles = [
        Patch(facecolor=cmap(0.92), label="+ zAI  (L>R asymmetry)"),
        Patch(facecolor=cmap(0.08), label="- zAI  (R>L asymmetry)"),
        Patch(facecolor="none", edgecolor="lime", label="37-ROI clinical mask"),
    ]
    ax_cb.legend(handles=legend_handles, loc="lower center", fontsize=8,
                 frameon=True)

    stat_name = "zAI (control-referenced asymmetry)" if space == "zai" else "z-score (vs controls)"
    title = (f"{label} — island-cleaned {stat_name}   |   group={group}   "
             f"|x|>={threshold}, min_cluster={report['min_cluster']}\n"
             f"islands removed: {report['n_islands_removed']} "
             f"({report['voxels_removed']:,} vox)  ->  "
             f"{report['surviving_cluster_count']} clusters / "
             f"{report['surviving_voxels']:,} vox kept")
    if direction_summary:
        title += f"\ndirection: {direction_summary}"
        if dominant_dir:
            title += f"  (dominant: {dominant_dir}perfusion)"
    fig.suptitle(title, fontsize=12, y=0.995)

    if out_png is None:
        out_png = OUTPUT_DIR / f"{patient_id}_{space}_clean_overlay.png"
    fig.savefig(str(out_png), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---------- before/after QC panel ----------
    raw_overlay = np.where(np.abs(raw_thresholded) > 0, raw_thresholded, np.nan)
    fig2, axes = plt.subplots(2, 4, figsize=(18, 9))
    qc_ks = np.linspace(kmin, kmax, 4).astype(int)
    for col, k in enumerate(qc_ks):
        a = _ortho(anat_disp, 2, k)
        # raw (top)
        axes[0, col].imshow(a, cmap="gray", vmin=a_lo, vmax=a_hi)
        axes[0, col].imshow(_ortho(raw_overlay, 2, k), cmap=cmap, norm=norm,
                            alpha=0.78, interpolation="nearest")
        axes[0, col].set_title(f"RAW thresholded  z={k}", fontsize=10)
        axes[0, col].axis("off")
        # cleaned (bottom)
        axes[1, col].imshow(a, cmap="gray", vmin=a_lo, vmax=a_hi)
        axes[1, col].imshow(_ortho(overlay, 2, k), cmap=cmap, norm=norm,
                            alpha=0.78, interpolation="nearest")
        if roi_mask is not None:
            rm = _ortho(roi_mask.astype(float), 2, k)
            if rm.any():
                axes[1, col].contour(rm, levels=[0.5], colors="lime",
                                     linewidths=0.6, alpha=0.7)
        axes[1, col].set_title(f"ISLAND-CLEANED  z={k}", fontsize=10)
        axes[1, col].axis("off")
    fig2.suptitle(
        f"{label} — before/after island cleaning  ({space}, |x|>={threshold}, "
        f"min_cluster={report['min_cluster']})\n"
        f"raw: {report['raw_cluster_count']} components / "
        f"{report['raw_suprathreshold_voxels']:,} vox   ->   "
        f"cleaned: {report['surviving_cluster_count']} clusters / "
        f"{report['surviving_voxels']:,} vox   "
        f"({report['n_islands_removed']} islands, "
        f"{report['voxels_removed']:,} vox removed)",
        fontsize=12)
    if qc_png is None:
        qc_png = OUTPUT_DIR / f"{patient_id}_{space}_clean_overlay_qc.png"
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(str(qc_png), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    return str(out_png), str(qc_png)


# ============================================================================
# PER-PATIENT DRIVER
# ============================================================================
def load_stat_map(patient_id, space, group):
    """Load the raw per-voxel stat map and its affine for the requested space."""
    if space == "zai":
        path = (ASYMMETRY_DIR / "patients" / patient_id /
                f"{patient_id}_asymmetry_zscore.nii.gz")
    else:  # zscore (vs controls)
        path = (RESULTS_DIR / "patients" / patient_id /
                f"{patient_id}_vs_{group}_zscore.nii.gz")
    if not path.exists():
        # Fallback: any available asymmetry_zscore map.
        alt = (ASYMMETRY_DIR / "patients" / patient_id /
               f"{patient_id}_asymmetry_zscore.nii.gz")
        if alt.exists():
            print(f"  [fallback] {path.name} missing; using {alt.name}")
            path = alt
        else:
            return None, None, None
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine, path


def process_patient(patient_id, threshold=3.0, min_cluster=50,
                    open_iter=1, close_iter=1, space="zai", use_roi_mask=True):
    print(f"\n=== {patient_id} ({space}) ===")
    group = resolve_patient_group(patient_id)
    if group is None:
        print(f"  Could not resolve control group for {patient_id}; skipping.")
        return None

    stat_map, affine, src_path = load_stat_map(patient_id, space, group)
    if stat_map is None:
        print(f"  No stat map available for {patient_id} ({space}); skipping.")
        return None
    print(f"  group={group}  source={src_path.name}")

    # 37-ROI clinical mask (per-patient parcellation).
    roi_mask = None
    if use_roi_mask:
        roi_mask = load_clinical_roi_mask(patient_id, ref_shape=stat_map.shape)
        if roi_mask is None:
            print("  [warn] 37-ROI clinical mask unavailable; proceeding maskless.")

    # Clean.
    cleaned, report = clean_overlay_map(
        stat_map, threshold, min_cluster=min_cluster, mask=roi_mask,
        open_iter=open_iter, close_iter=close_iter)

    # Raw thresholded (for QC, same mask, no cleaning).
    raw_thresholded = np.where(
        (np.abs(stat_map) >= threshold) &
        (roi_mask if roi_mask is not None else np.ones_like(stat_map, bool)),
        stat_map, 0.0).astype(np.float32)

    # QC: prove zero sub-min-size components in cleaned map.
    sub_n, total_n = count_subminsize_components(cleaned, min_cluster)

    print(f"  raw suprathreshold voxels : {report['raw_suprathreshold_voxels']:,}")
    print(f"  raw cluster count         : {report['raw_cluster_count']}")
    print(f"  islands removed           : {report['n_islands_removed']}")
    print(f"  voxels removed            : {report['voxels_removed']:,}")
    print(f"  surviving clusters        : {report['surviving_cluster_count']}")
    print(f"  surviving voxels          : {report['surviving_voxels']:,}")
    print(f"  min surviving cluster size: {report['min_surviving_cluster_size']}")
    print(f"  cleaned sub-min components: {sub_n} (of {total_n} total)  "
          f"{'OK' if sub_n == 0 else 'FAIL'}")
    report["cleaned_subminsize_components"] = sub_n

    # Save cleaned NIfTI (zai space -> _zai_cleaned.nii.gz).
    suffix = "zai_cleaned" if space == "zai" else f"{group}_zscore_cleaned"
    out_nii = (ASYMMETRY_DIR / "patients" / patient_id /
               f"{patient_id}_{suffix}.nii.gz") if space == "zai" else \
              (RESULTS_DIR / "patients" / patient_id /
               f"{patient_id}_{suffix}.nii.gz")
    nib.save(nib.Nifti1Image(cleaned.astype(np.float32), affine), str(out_nii))
    print(f"  cleaned NIfTI -> {out_nii}")
    report["cleaned_nifti"] = str(out_nii)

    # Anatomy underlay: prefer mean_perfusion (shared geometry); fall back T1w.
    mean_perf, sd_perf, brain_mask = load_group_perfusion(group)
    anatomy = mean_perf
    if anatomy.shape != stat_map.shape:
        t1 = DATASET_DIR / patient_id / "T1w_acpc_dc_restore.nii.gz"
        if t1.exists():
            anatomy = nib.load(str(t1)).get_fdata(dtype=np.float32)

    direction_summary, dominant_dir, _ = load_direction_summary(patient_id)

    out_png, qc_png = render_overlay_montage(
        patient_id, cleaned, raw_thresholded, anatomy, roi_mask, group,
        threshold, report, direction_summary=direction_summary,
        dominant_dir=dominant_dir, space=space)
    print(f"  montage -> {out_png}")
    print(f"  QC      -> {qc_png}")
    report["montage_png"] = out_png
    report["qc_png"] = qc_png
    report["group"] = group
    return report


def discover_patients():
    pdir = ASYMMETRY_DIR / "patients"
    if not pdir.exists():
        return []
    return sorted([
        d.name for d in pdir.iterdir()
        if d.is_dir() and (d / f"{d.name}_asymmetry_zscore.nii.gz").exists()
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient", help="patient ID (e.g. P015)")
    ap.add_argument("--all", action="store_true", help="process all patients")
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--min-cluster", type=int, default=50)
    ap.add_argument("--open", dest="open_iter", type=int, default=1,
                    help="binary_opening iterations (speckle removal)")
    ap.add_argument("--close", dest="close_iter", type=int, default=1,
                    help="binary_closing iterations (pinhole fill)")
    ap.add_argument("--space", choices=["zai", "mni", "zscore"], default="zai",
                    help="'zai'/'mni' = asymmetry zAI map; 'zscore' = vs-control z map")
    ap.add_argument("--roi-mask", dest="roi_mask", action="store_true", default=True)
    ap.add_argument("--no-roi-mask", dest="roi_mask", action="store_false")
    ap.add_argument("--sym", action="store_true",
                    help="zAI maps are in symmetric-template space; use the symmetric "
                         "per-patient parcellation for the 37-ROI clinical mask.")
    args = ap.parse_args()

    if args.sym:
        _pubfig.SYM_MODE = True
        print("  [SYM] symmetric-space parcellation for clinical ROI mask.")

    space = "zai" if args.space in ("zai", "mni") else "zscore"

    if args.all:
        patients = discover_patients()
    elif args.patient:
        patients = [args.patient]
    else:
        ap.error("specify --patient <PID> or --all")

    results = []
    for pid in patients:
        r = process_patient(pid, threshold=args.threshold,
                            min_cluster=args.min_cluster,
                            open_iter=args.open_iter, close_iter=args.close_iter,
                            space=space, use_roi_mask=args.roi_mask)
        if r:
            results.append((pid, r))

    print("\n=== SUMMARY ===")
    for pid, r in results:
        print(f"{pid}: {r['n_islands_removed']} islands removed, "
              f"{r['surviving_cluster_count']} clusters kept, "
              f"sub-min-components={r['cleaned_subminsize_components']}")


if __name__ == "__main__":
    main()
