#!/usr/bin/env python3
"""Manuscript figures (Clinical Neurophysiology submission).

Fig. 1  Pipeline overview (schematic).
Fig. 2  Calibration of the voxel-wise asymmetry z-score before and after
        left-right symmetric registration (per patient).
Fig. 3  Representative cases: clusters surviving the clinical tier
        (|zAI| >= 4, >= 100 voxels, 37-ROI scope) coloured by the
        direction-of-abnormality class, on the symmetric T1 template.

Outputs to manuscript/figures/.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from src.thresholding import get_structure_3d  # noqa: E402
from ez_ground_truth import EZ_GROUND_TRUTH  # noqa: E402

_spec = importlib.util.spec_from_file_location("ci", BASE / "06_clinical_interpretation.py")
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)

FIG = BASE / "manuscript/figures"
FIG.mkdir(exist_ok=True, parents=True)
PAT = BASE / "results_zscore/asymmetry/patients"
T1 = BASE / "symreg/templates/T1_sym.nii.gz"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


# --------------------------------------------------------------------------
def fig1_schematic():
    """Pipeline overview: inputs -> common preprocessing -> two pipelines -> outputs -> reference."""
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                         "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold"})
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    EC, INK, MUTED = "#4a4a4a", "#1f1f1f", "#555555"

    def box(x, y, w, h, title, lines, fc, hc, fs=7.4, tfs=8.0, align="left", pad=1.6):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                             fc=fc, ec=EC, lw=0.8, zorder=2))
        if title:
            th = 5.2
            ax.add_patch(mpatches.FancyBboxPatch((x, y + h - th), w, th, boxstyle="round,pad=0,rounding_size=1.2",
                                                 fc=hc, ec="none", zorder=3, clip_on=False))
            ax.add_patch(mpatches.Rectangle((x, y + h - th), w, th / 2, fc=hc, ec="none", zorder=3))
            ax.text(x + pad, y + h - th / 2, title, ha="left", va="center", fontsize=tfs,
                    fontweight="bold", color=INK, zorder=4)
            top = y + h - th
        else:
            top = y + h
        body = "\n".join(lines)
        if align == "left":
            ax.text(x + pad, (top + y) / 2, body, ha="left", va="center", fontsize=fs, color=INK,
                    linespacing=1.45, zorder=4)
        else:
            ax.text(x + w / 2, (top + y) / 2, body, ha="center", va="center", fontsize=fs, color=INK,
                    linespacing=1.45, zorder=4)

    def arrow(x0, y0, x1, y1, text=None, tx=None, ty=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=EC, shrinkA=0, shrinkB=0,
                                    mutation_scale=8))
        if text:
            ax.text(tx, ty, text, fontsize=6.6, color=MUTED, ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)

    def line(x0, y0, x1, y1):
        ax.plot([x0, x1], [y0, y1], color=EC, lw=0.9, zorder=1, solid_capstyle="round")

    BLUE, BLUE_H = "#e4edf7", "#c6d8ec"
    AMBER, AMBER_H = "#fbf1dc", "#f1dcad"
    TEAL, TEAL_H = "#e1f0ec", "#b9dcd2"
    GREEN, GREEN_H = "#e8f1df", "#c7ddb4"
    ROSE, ROSE_H = "#f9e6e0", "#efc5b8"
    GREY, GREY_H = "#f3f3f3", "#dcdcdc"

    # ---- row 1: inputs and common processing ------------------------------
    box(2, 84, 27, 14, "Patients  (n = 17)",
        ["drug-resistant focal epilepsy", "13 unilateral MDT hypothesis (8 L, 5 R)",
         "multi-PLD pCASL 2.5 mm + T1w, 3 T"], BLUE, BLUE_H)
    box(71, 84, 27, 14, "Healthy controls  (n = 180)",
        ["HCP Lifespan multi-PLD pCASL", "three age bands, 60 each",
         "20\u201339, 40\u201359, 60\u201379 years"], BLUE, BLUE_H)
    box(33.5, 84, 33, 14, "Identical processing",
        ["HCP-ASL: calibrated CBF (mL/100 g/min)", "FSL flirt + fnirt to a left\u2013right",
         "symmetric T1 template (0.8 mm)"], AMBER, AMBER_H)
    arrow(29, 91, 33.5, 91); arrow(71, 91, 66.5, 91)

    # split to the two pipelines
    line(50, 84, 50, 79.5); line(24, 79.5, 76, 79.5)
    arrow(24, 79.5, 24, 75.5); arrow(76, 79.5, 76, 75.5)

    # ---- row 2: pipelines ----------------------------------------------------
    box(2, 49, 45, 26.5, "A   Regional pipeline",
        ["37 homologous region pairs: hippocampus, amygdala,",
         "thalamus and 34 Desikan\u2013Killiany cortical regions;",
         "mean CBF per region and side",
         "LI$_i$ = (CBF$_{L,i}$ \u2212 CBF$_{R,i}$) / (CBF$_{L,i}$ + CBF$_{R,i}$) and 14 further indices",
         "9 aggregation scores \u2192 one lateralization score per patient"],
        TEAL, TEAL_H, fs=7.1)
    box(53, 49, 45, 26.5, "B   Voxel-wise pipeline",
        ["AI(v) = 200 [CBF(v) \u2212 CBF(v\u0304)] / [CBF(v) + CBF(v\u0304)],  v\u0304 = mirror voxel",
         "$z_{\\mathrm{AI}}$(v) = [AI(v) \u2212 \u03bc$_{\\mathrm{AI}}$(v)] / \u03c3$_{\\mathrm{AI}}$(v);  age-band controls,",
         "\u03c3 floored at 5 %; gray matter, 37-region scope",
         "clusters: clinical tier |$z_{\\mathrm{AI}}$| \u2265 4, \u2265 100 voxels (51 mm\u00b3);",
         "research \u2265 3 / \u2265 50; surgical \u2265 5 / \u2265 200; native \u2265 20 voxels"],
        GREEN, GREEN_H, fs=7.1)

    # ---- row 3: direction inference (B only) ---------------------------------
    arrow(76, 49, 76, 44.5)
    box(53, 29, 45, 15.5, "Direction of abnormality (per cluster)",
        ["per-side perfusion z vs controls; |z| \u2265 2 = deviating side",
         "hypo  /  hyper  /  mixed  /  subtle",
         "predicted side = deviating hemisphere; mirrored pairs counted once"],
        ROSE, ROSE_H, fs=7.1)

    # ---- row 4: outputs ----------------------------------------------------
    arrow(24, 49, 24, 23.5); arrow(76, 29, 76, 23.5)
    box(2, 14, 45, 9.5, None,
        ["Lateralization score per patient", "ROC, bootstrap CI, leave-one-out cross-validation"],
        "white", GREY_H, align="center", fs=7.3)
    box(53, 14, 45, 9.5, None,
        ["Patient report: side, call strength (strong / moderate / weak),", "lobe of largest clusters, direction class"],
        "white", GREY_H, align="center", fs=7.3)

    # ---- row 5: reference standard ------------------------------------------
    line(24, 14, 24, 11); line(76, 14, 76, 11); line(24, 11, 76, 11); arrow(50, 11, 50, 8.4)
    box(6, 0.8, 88, 7.6, None,
        ["Reference standard: multidisciplinary-team (MDT) side in 13 unilateral patients (8 L, 5 R); agreement and balanced",
         "accuracy vs. the always-left baseline (8 of 13, 0.50). External test: 99 Chalfont patients, 66 same-scanner controls, same pipeline"],
        GREY, GREY_H, align="center", fs=7.0)

    fig.savefig(FIG / "fig1_pipeline_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / "fig1_pipeline_overview.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig2_calibration():
    rows = []
    # Output of `python symreg/check_full_calibration.py > manuscript/figures/calibration_source.txt`
    txt = (BASE / "manuscript/figures/calibration_source.txt")
    if not txt.exists():
        import subprocess
        txt.write_text(subprocess.run([sys.executable, str(BASE / "symreg/check_full_calibration.py")],
                                      capture_output=True, text=True, check=True).stdout)
    text = txt.read_text()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].startswith("P0") and parts[2].endswith("%"):
            rows.append((parts[0], float(parts[2].rstrip("%")), float(parts[3].rstrip("%"))))
    df = pd.DataFrame(rows, columns=["patient", "sym", "orig"]).sort_values("patient")
    txt.write_text(text)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x - 0.2, df.orig, 0.4, color="#b0b0b0", label="Standard MNI space (naive mirror)")
    ax.bar(x + 0.2, df.sym, 0.4, color="#2b6ca3", label="Symmetric-template space")
    ax.axhline(5, color="k", ls=":", lw=0.8)
    ax.axhline(6.9, color="#2b6ca3", ls="--", lw=0.8)
    ax.text(len(df) - 0.4, 2.2, "nominal two-tailed 5 %", ha="right", fontsize=7)
    ax.text(len(df) - 0.4, 22.5, "control leave-one-out: 6.9 %", ha="right", fontsize=7, color="#2b6ca3")
    ax.set_xticks(x); ax.set_xticklabels(df.patient, rotation=60, fontsize=7.5)
    ax.set_ylabel("Grey-matter voxels with |zAI| >= 1.96 (%)")
    ax.set_ylim(0, 55)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_calibration.png", dpi=300)
    fig.savefig(FIG / "fig2_calibration.pdf")
    plt.close(fig)
    print(df.to_string(index=False), "\nmean orig %.1f  sym %.1f" % (df.orig.mean(), df.sym.mean()))


# --------------------------------------------------------------------------
def cluster_class_volume(pid: str, peak=4.0, size=100, root: Path | None = None):
    """Rebuild (class volume, z volume) for in-scope clusters at the tier, one member per mirrored
    pair (the member in the hemisphere assigned by the direction rule). class: 1 = hypo, 2 = hyper,
    3 = subtle/mixed; z = z_AI at those voxels (signed). root = .../results_zscore/asymmetry/patients."""
    d = (root or PAT) / pid
    rep = pd.read_csv(d / f"{pid}_asymmetry_cluster_report.csv")
    normed = rep["primary_region"].apply(ci._normalize_region)
    rep["canon"] = normed.apply(lambda t: t[1])
    rep["in_scope"] = rep["canon"].apply(lambda n: bool(n) and n in ci.CLINICAL_NAMES)
    struct = get_structure_3d()
    zmap = nib.load(str(d / f"{pid}_asymmetry_zscore.nii.gz")).get_fdata(dtype=np.float32)
    out = None; zout = np.zeros(zmap.shape, np.float32)
    for direction, fname in (("left-dominant", "left_dominant"), ("right-dominant", "right_dominant")):
        img = nib.load(str(d / f"{pid}_asymmetry_{fname}.nii.gz"))
        mask = img.get_fdata() > 0
        lab, n = ndimage.label(mask, structure=struct)
        if out is None:
            out = np.zeros(mask.shape, np.uint8)
        sub = rep[rep.direction == direction].set_index("cluster_id")
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        cid = 0
        for i, sz in enumerate(sizes, 1):
            if sz < 50:
                continue
            cid += 1
            if cid not in sub.index:
                continue
            r = sub.loc[cid]
            if abs(r.peak_z) < peak or sz < size or not r.in_scope:
                continue
            if str(r.cluster_hemi) != str(r.ez_side_pred):
                continue
            cls = {"hypo": 1, "hyper": 2}.get(r.direction_class, 3)
            m = lab == i
            out[m] = cls; zout[m] = zmap[m]
    return out, zout, rep


def case_montage(cases, out_name, root: Path | None = None, z_mm=(-24, -12, 0, 12, 30), y_mm=-20,
                 zmax=8.0, title=None):
    """Lightbox montage in the style of PET/ASL z-map papers: axial slices at fixed MNI z (mm) plus one
    coronal slice through the hippocampi, T1 background, the signed z_AI of surviving clinical-tier
    clusters shown as a continuous colour (cool = hypoperfused side, warm = hyperperfused side,
    grey = subtle), colour bars, L/R markers, neurological orientation (image left = left hemisphere).
    cases: list of (patient_id, row_label)."""
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"]})
    img = nib.load(str(T1)); aff = img.affine; t1 = img.get_fdata(); t1 = t1 / np.percentile(t1[t1 > 0], 99)
    inv = np.linalg.inv(aff)
    def idx(xyz):
        return np.round(inv @ np.array([*xyz, 1]))[:3].astype(int)
    zi = [int(idx((0, 0, z))[2]) for z in z_mm]; yi = int(idx((0, y_mm, 0))[1])
    n = len(cases); ncol = len(z_mm) + 1
    fig = plt.figure(figsize=(7.2, 1.42 * n + 0.75))
    gs = fig.add_gridspec(n, ncol, left=0.075, right=0.90, top=0.95 if title is None else 0.92, bottom=0.07, wspace=0.03, hspace=0.06)
    cool = plt.get_cmap("Blues"); warm = plt.get_cmap("Reds"); UP = 3
    def rgba_of(cls2d, z2d):
        rgba = np.zeros(cls2d.shape + (4,))
        a = np.clip((np.abs(z2d) - 3.0) / (zmax - 3.0), 0.25, 1.0)      # colour intensity from |z| 3 .. zmax
        m1, m2, m3 = cls2d == 1, cls2d == 2, cls2d == 3
        rgba[m1] = cool(0.45 + 0.55 * a[m1]); rgba[m2] = warm(0.45 + 0.55 * a[m2]); rgba[m3] = (0.6, 0.6, 0.6, 1)
        rgba[..., 3] = (cls2d > 0).astype(float)
        return rgba
    for r, (pid, label) in enumerate(cases):
        vol, zv, _ = cluster_class_volume(pid, root=root)
        for c in range(ncol):
            ax = fig.add_subplot(gs[r, c])
            if c < len(z_mm):
                bg = t1[::-1, :, zi[c]].T; cl = vol[::-1, :, zi[c]].T; zz = zv[::-1, :, zi[c]].T
            else:
                bg = t1[::-1, yi, :].T; cl = vol[::-1, yi, :].T; zz = zv[::-1, yi, :].T
            # upsample the slices (3x) so that the embedded images are sharp in the PDF/PNG:
            # bilinear for the anatomy, nearest for the cluster labels and their z values
            bg = ndimage.zoom(bg, UP, order=1); cl = ndimage.zoom(cl, UP, order=0); zz = ndimage.zoom(zz, UP, order=0)
            ax.imshow(bg, cmap="gray", origin="lower", vmin=0, vmax=1, interpolation="bilinear")
            ax.imshow(rgba_of(cl, zz), origin="lower", interpolation="nearest")
            ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if r == 0:
                ax.set_title(f"z = {z_mm[c]:+d} mm" if c < len(z_mm) else f"y = {y_mm:+d} mm", fontsize=6.5, pad=2)
            ax.text(0.03, 0.04, "L", color="white", fontsize=6, transform=ax.transAxes, ha="left", va="bottom")
            ax.text(0.97, 0.04, "R", color="white", fontsize=6, transform=ax.transAxes, ha="right", va="bottom")
            if c == 0:
                ax.text(-0.06, 0.5, label, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6.3)
    # colour bars
    import matplotlib as mpl
    cax1 = fig.add_axes([0.915, 0.55, 0.012, 0.32]); cax2 = fig.add_axes([0.915, 0.12, 0.012, 0.32])
    mpl.colorbar.ColorbarBase(cax1, cmap=mpl.colors.ListedColormap(warm(np.linspace(0.45, 1.0, 64))), norm=mpl.colors.Normalize(3, zmax), orientation="vertical")
    mpl.colorbar.ColorbarBase(cax2, cmap=mpl.colors.ListedColormap(cool(np.linspace(0.45, 1.0, 64))), norm=mpl.colors.Normalize(3, zmax), orientation="vertical")
    cax1.set_title("hyper\n$|z_{\\mathrm{AI}}|$", fontsize=5.8, pad=3); cax2.set_title("hypo\n$|z_{\\mathrm{AI}}|$", fontsize=5.8, pad=3)
    for cax in (cax1, cax2):
        cax.tick_params(labelsize=5.5, length=2)
    fig.text(0.5, 0.015, "Neurological convention (image left = left hemisphere); clinical tier (peak $|z_{\\mathrm{AI}}| \\geq 4$, $\\geq$ 100 voxels, 37 region pairs); one member per mirrored pair, drawn in the hemisphere assigned by the direction rule; grey = subtle",
             ha="center", fontsize=5.8, color="#444444")
    if title:
        fig.suptitle(title, fontsize=8, y=0.985)
    fig.savefig(FIG / f"{out_name}.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG / f"{out_name}.pdf", bbox_inches="tight", dpi=400)
    plt.close(fig)


def fig3_cases(cases):
    """Fig. 3: four representative in-house cases (kept for the main text)."""
    case_montage(cases, "fig3_representative_cases")


# --------------------------------------------------------------------------
def fig4_enhanced():
    """Fig. 4: calibrated regional normative statistic (mean z of LI over the temporal
    pairs) per patient, by reference side, for the Chalfont and in-house cohorts, with
    the leave-one-out control distribution and the calibrated call threshold
    (95th percentile of |statistic| in controls). Data: ez_analysis_mdt/v2/<dataset>/."""
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"]})
    V2 = BASE / "ez_analysis_mdt/v2"
    COL = {"L": "#2b6cb0", "R": "#c05621"}; MARK = {"L": "o", "R": "s"}
    INK, MUTED = "#1f1f1f", "#6b6b6b"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [2.4, 1]})
    for ax, (ds, title) in zip(axes, [("chalfont", "Chalfont: 93 unilateral patients, 66 same-scanner controls"),
                                      ("inhouse", "In-house: 13 patients, HCP controls")]):
        pat = pd.read_csv(V2 / ds / "R_patients.csv"); loo = pd.read_csv(V2 / ds / "R_loo_controls.csv")
        summ = pd.read_csv(V2 / ds / "summary.csv").set_index("statistic")
        thr = float(summ.loc["R_temporal", "threshold"])
        pat = pat[pat.ez.isin(["L", "R"])]
        rng = np.random.default_rng(0)
        rows = [("controls\n(leave-one-out)", loo["R_temporal"].values, None),
                ("reference R", pat[pat.ez == "R"]["R_temporal"].values, "R"),
                ("reference L", pat[pat.ez == "L"]["R_temporal"].values, "L")]
        for y, (lab, vals, side) in enumerate(rows):
            jit = rng.uniform(-0.18, 0.18, len(vals))
            if side is None:
                ax.scatter(vals, y + jit, s=9, color="#9a9a9a", alpha=0.7, linewidths=0, zorder=2)
            else:
                ax.scatter(vals, y + jit, s=16, color=COL[side], marker=MARK[side], alpha=0.85,
                           linewidths=0.4, edgecolors="white", zorder=3)
                ok = (vals < 0) if side == "L" else (vals > 0)
                called = np.abs(vals) >= thr; okc = ok & called
                ax.text(5.05, y, f"{ok.sum()}/{len(vals)} by sign\n{okc.sum()}/{called.sum()} called",
                        fontsize=6.3, color=INK, va="center", ha="left")
        ax.axvline(0, color="#bdbdbd", lw=0.8, zorder=1)
        for t in (-thr, thr):
            ax.axvline(t, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.axvspan(-thr, thr, color="#f0f0f0", zorder=0)
        ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
        ax.set_xlim(-5, 5); ax.set_ylim(-0.6, 2.6)
        ax.set_xlabel("regional temporal statistic  $\\bar{z}_{\\mathrm{temporal}}$  (control SD units)", fontsize=7)
        ax.set_title(title, fontsize=7.2, loc="left", color=INK, pad=4)
        ax.tick_params(labelsize=6.5, length=2, color="#9a9a9a")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color("#9a9a9a")
        ax.text(-thr - 0.1, 2.55, f"call threshold \u00b1{thr:.2f}\n(95th pct. of controls)", fontsize=6, color=MUTED, ha="right", va="top")
        if ds == "chalfont":
            ax.text(-4.9, -0.5, "\u2190 left lower (predicts left EZ)", fontsize=6, color=MUTED, ha="left")
            ax.text(4.9, -0.5, "right lower (predicts right EZ) \u2192", fontsize=6, color=MUTED, ha="right")
        else:
            ax.text(-4.9, -0.5, "\u2190 left lower", fontsize=6, color=MUTED, ha="left")
            ax.text(4.9, -0.5, "right lower \u2192", fontsize=6, color=MUTED, ha="right")
    h = [plt.Line2D([], [], marker="o", color=COL["L"], ls="", ms=4, label="reference side left"),
         plt.Line2D([], [], marker="s", color=COL["R"], ls="", ms=4, label="reference side right"),
         plt.Line2D([], [], marker="o", color="#9a9a9a", ls="", ms=3, label="healthy control")]
    fig.legend(handles=h, fontsize=6.3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, handletextpad=0.3, columnspacing=1.4)
    plt.tight_layout(w_pad=1.5, rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig4_enhanced_estimator.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig4_enhanced_estimator.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig5_external_cohort():
    """Fig. 5: the external (Chalfont) cohort and pipeline, in the style of Fig. 1."""
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                         "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold"})
    fig = plt.figure(figsize=(7.2, 5.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    EC, INK, MUTED = "#4a4a4a", "#1f1f1f", "#555555"

    def box(x, y, w, h, title, lines, fc, hc, fs=6.8, tfs=7.6, align="left", pad=1.5):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=EC, lw=0.8, zorder=2))
        top = y + h
        if title:
            th = 5.0
            ax.add_patch(mpatches.FancyBboxPatch((x, y + h - th), w, th, boxstyle="round,pad=0,rounding_size=1.2", fc=hc, ec="none", zorder=3))
            ax.add_patch(mpatches.Rectangle((x, y + h - th), w, th / 2, fc=hc, ec="none", zorder=3))
            ax.text(x + pad, y + h - th / 2, title, ha="left", va="center", fontsize=tfs, fontweight="bold", color=INK, zorder=4)
            top = y + h - th
        body = "\n".join(lines)
        if align == "left":
            ax.text(x + pad, (top + y) / 2, body, ha="left", va="center", fontsize=fs, color=INK, linespacing=1.4, zorder=4)
        else:
            ax.text(x + w / 2, (top + y) / 2, body, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.4, zorder=4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=EC, shrinkA=0, shrinkB=0, mutation_scale=8))

    def line(x0, y0, x1, y1):
        ax.plot([x0, x1], [y0, y1], color=EC, lw=0.9, zorder=1, solid_capstyle="round")

    BLUE, BLUE_H = "#e4edf7", "#c6d8ec"; AMBER, AMBER_H = "#fbf1dc", "#f1dcad"
    TEAL, TEAL_H = "#e1f0ec", "#b9dcd2"; GREEN, GREEN_H = "#e8f1df", "#c7ddb4"
    ROSE, ROSE_H = "#f9e6e0", "#efc5b8"; GREY, GREY_H = "#f3f3f3", "#dcdcdc"

    # ---- row 1: source and acquisition ------------------------------------------------
    box(2, 86, 46, 12.5, "Source: Chalfont Centre for Epilepsy (UCL), retrospective",
        ["3 T GE scanner, same scanner for patients and controls",
         "multi-delay enhanced ASL (eASL): transit-time-corrected CBF (primary),",
         "uncorrected flow (sensitivity); 1.875 \u00d7 1.875 \u00d7 4 mm; MPRAGE T1"], BLUE, BLUE_H)
    box(52, 86, 46, 12.5, "Complete imaging available",
        ["controls 70  |  patients with a discrete MRI lesion 47  |  MRI-negative 53",
         "clinical lateralization from resection, lesion + MDT conclusion,",
         "or presurgical hypothesis; no postoperative outcome"], BLUE, BLUE_H)

    # ---- row 2: QC ---------------------------------------------------------------------
    arrow(25, 86, 25, 82); arrow(75, 86, 75, 82)
    box(2, 66.5, 46, 15.5, "Pre-specified quality rule (before any result was read)",
        ["gray/white ratio \u2265 1.2; GM CBF 15\u2013100 mL/100 g/min; saturation \u2264 1 %;",
         "controls only: left/right GM ratio 0.80\u20131.25",
         "excluded: 3 controls (+1 retest duplicate), 1 patient (technical)",
         "flagged, kept: 2 patients with global L/R imbalance; 3 registration flags"], AMBER, AMBER_H)
    box(52, 66.5, 46, 15.5, "Analyzed cohort",
        ["66 controls: age 41 \u00b1 13 y (19\u201366; recorded for 63), 12 women / 13 men (41 not recorded)",
         "99 patients: age 35 \u00b1 9 y (19\u201360; recorded for 85), 31 women / 37 men (31 n.r.); lesional 47 / MRI-negative 52",
         "side L 43 / R 50 / bilateral 6;  lobe: temporal 51, frontal 22,",
         "multilobar 24, parietal 1, occipital 1;  hippocampal sclerosis 22",
         "reference: resected 12 / lesion + MDT 35 / clinical hypothesis 52"], GREEN, GREEN_H)

    # ---- row 3: processing --------------------------------------------------------------
    line(25, 66.5, 25, 62.5); line(75, 66.5, 75, 62.5); line(25, 62.5, 75, 62.5); arrow(50, 62.5, 50, 59.5)
    box(2, 46.5, 96, 13, "Identical processing to the in-house pipeline",
        ["ASL \u2192 T1 rigid (FSL flirt)  \u2192  T1 \u2192 left\u2013right symmetric template (flirt + fnirt, 0.8 mm)  \u2192  CBF resampled once",
         "no FreeSurfer: gray matter and the 37 region pairs from a fixed template atlas (voxel-wise mode of the 180 HCP control parcellations)",
         "normative reference: the 66 controls pooled (too few for age bands); every control also scored leave-one-out as a pseudo-patient"], AMBER, AMBER_H)

    # ---- row 4: analyses --------------------------------------------------------------
    arrow(24, 46.5, 24, 42.5); arrow(76, 46.5, 76, 42.5)
    box(2, 22.5, 46, 20, "A   Pre-specified primary endpoint (as implemented)",
        ["voxel-wise $z_{\\mathrm{AI}}$ clusters, production rule, clinical tier, 37-region scope",
         "metric: balanced accuracy vs. majority baseline (0.54), Wilson CI;",
         "bilateral / no call = disagreement",
         "secondary: control false-positive rate of strong/moderate calls",
         "sensitivity: rule \u00d7 tier \u00d7 scope grid, uncorrected flow, QC strata"], TEAL, TEAL_H)
    box(52, 22.5, 46, 20, "B   Enhanced estimators (post hoc, identical on both cohorts)",
        ["regional normative $\\bar{z}_{\\mathrm{LI}}$ (temporal, mesial, all 37) from the atlas regions",
         "8-mm smoothed voxel-wise $z_{\\mathrm{AI}}$ on a mirror-symmetric 2.4-mm grid,",
         "hypoperfused z-mass per hemisphere (each pair counted once)",
         "call threshold = 95th percentile of leave-one-out controls (5 % FP)",
         "multi-metric comparison: call rate, PPV, sensitivity L/R, AUC, LR$^+$, strata"], GREEN, GREEN_H)

    # ---- row 5: outputs ----------------------------------------------------------------
    line(24, 22.5, 24, 18.5); line(76, 22.5, 76, 18.5); line(24, 18.5, 76, 18.5); arrow(50, 18.5, 50, 15.5)
    box(6, 2, 88, 13.5, "Results used in this article",
        ["primary rule 58 of 93 (balanced accuracy 0.63; strong calls 16 of 20);  regional temporal statistic 61 of 93 (0.65, AUC 0.69),",
         "calibrated calls 12 of 13;  calibration: 6.1 % of gray matter beyond |$z_{\\mathrm{AI}}$| \u2265 1.96 in controls, 9.6 % in patients",
         "(in-house: 6.9 % / 12.9 %);  an antisymmetric multivariate classifier gave no further gain"], GREY, GREY_H)

    fig.savefig(FIG / "fig5_external_cohort.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / "fig5_external_cohort.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# --------------------------------------------------------------------------
def figS_workflow():
    """Supplementary workflow figure: every script, its inputs and outputs, in execution order."""
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"]})
    fig = plt.figure(figsize=(7.2, 9.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    EC, INK = "#4a4a4a", "#1f1f1f"
    def box(x, y, w, h, title, lines, fc, hc, fs=5.5, tfs=6.4, pad=1.2):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=EC, lw=0.7, zorder=2))
        top = y + h
        if title:
            th = 3.2
            ax.add_patch(mpatches.FancyBboxPatch((x, y + h - th), w, th, boxstyle="round,pad=0,rounding_size=1.0", fc=hc, ec="none", zorder=3))
            ax.add_patch(mpatches.Rectangle((x, y + h - th), w, th / 2, fc=hc, ec="none", zorder=3))
            ax.text(x + pad, y + h - th / 2, title, ha="left", va="center", fontsize=tfs, fontweight="bold", color=INK, zorder=4)
            top = y + h - th
        ax.text(x + pad, (top + y) / 2, "\n".join(lines), ha="left", va="center", fontsize=fs, color=INK, linespacing=1.35, zorder=4)
    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=EC, shrinkA=0, shrinkB=0, mutation_scale=7))
    def line(x0, y0, x1, y1):
        ax.plot([x0, x1], [y0, y1], color=EC, lw=0.8, zorder=1)
    BLUE, BLUE_H = "#e4edf7", "#c6d8ec"; AMBER, AMBER_H = "#fbf1dc", "#f1dcad"; TEAL, TEAL_H = "#e1f0ec", "#b9dcd2"
    GREEN, GREEN_H = "#e8f1df", "#c7ddb4"; ROSE, ROSE_H = "#f9e6e0", "#efc5b8"; GREY, GREY_H = "#f3f3f3", "#dcdcdc"; VIOLET, VIOLET_H = "#ece6f4", "#d2c6e6"
    box(2, 90, 47, 9.5, "Inputs, in-house",
        ["Dataset/<P>/: T1w, aparc+aseg, HCP-ASL CBF (MNI, 0.8 mm)",
         "DatasetControls/FM_*/<HC>/: 180 HCP controls, 3 age bands",
         "ez_ground_truth.py (MDT side, certainty); clinical_spreadsheet.xlsx"], BLUE, BLUE_H)
    box(51, 90, 47, 9.5, "Inputs, external (Chalfont)",
        ["<chalfont>/{controls,discrete,negative}/<ID>/: MPRAGE, eASL Flow,",
         "   transit-corrected Flow, GIF parcellation (read-only)",
         "scripts/chalfont/stage_chalfont.py -> metadata/subjects.csv, groups.json"], BLUE, BLUE_H)
    arrow(25, 90, 25, 85.5); arrow(75, 90, 75, 85.5)
    box(2, 75, 47, 10.5, "Symmetric-template registration (symreg/)",
        ["build_sym_template.py -> templates/T1_sym.nii.gz (flip-symmetric T1)",
         "make_jobs.py | xargs -P 8 reg_subject.sh: flirt + fnirt, 197 subjects",
         "-> sym_perf/<ID>_perf_sym.nii.gz, <ID>_aparc_sym.nii.gz"], AMBER, AMBER_H)
    box(51, 75, 47, 10.5, "Registration and QC (scripts/chalfont/)",
        ["asl_quality.py -> qc/asl_quality_perf.csv, metadata/qc_exclusions.json",
         "build_template_atlas.py -> atlas/consensus_parcellation_sym.nii.gz",
         "run_registration.sh / reg_chalfont_subject.sh: ASL->T1 rigid, T1->T1_sym"], AMBER, AMBER_H)
    line(25, 75, 25, 70.5); line(75, 75, 75, 70.5); line(25, 70.5, 75, 70.5); arrow(50, 70.5, 50, 67.5)
    box(2, 47.5, 96, 20, "Pipeline B, voxel-wise  (symreg/run_sym_pipeline.sh | scripts/chalfont/run_chalfont_pipeline.sh; --dataset via dataset_config.py)",
        ["01_build_control_normative.py --sym [--rebuild-group]: control CBF mean/SD per band (Welford), consensus parcellation,",
         "   patient CBF z  ->  results_zscore/groups/<band>/, results_zscore/patients/<ID>/",
         "02_compute_zai.py --sym --group <band>: AI = 200 [CBF(v) - CBF(v_bar)] / [CBF(v) + CBF(v_bar)]; control mean/SD of AI (5 % floor);",
         "   z_AI; clusters at 1.96 / 50 voxels; direction class from per-side CBF z  ->  results_zscore/asymmetry/{groups,patients}/",
         "   (--pseudo-controls: each control scored leave-one-out)",
         "03_clinical_maps.py --sym: gray-matter maps, 37-ROI scope, voxel threshold 3 / 50  ->  results_zscore/clinical/<ID>/",
         "06_clinical_interpretation.py --all: tiers (peak >= 4, extent >= 100 at 1.96), production rule, call strength",
         "   ->  results_zscore/clinical/cohort_concordance.csv and per-patient reports",
         "scripts/clinical_stratified_analysis.py: rule x tier x scope grid  ->  ez_analysis_mdt/clinical_*.csv",
         "scripts/rerun_thresholding_methods.py --sym: methods M1-M13  ->  asymmetry/summary/all_patients_method_summary.csv",
         "Chalfont: scripts/chalfont/calibration_loo.py, collect_summaries.py, evaluate_lateralization.py  ->  lateralization/eval_*.csv"],
        GREEN, GREEN_H, fs=5.2)
    arrow(20, 47.5, 20, 43); arrow(66, 47.5, 66, 43)
    box(2, 27, 37, 16, "Pipeline A, regional (original space)",
        ["scripts/build_comprehensive_asymmetry.py:",
         "   37 ROI means, 15 indices  ->  results_analysis/<P>/",
         "05_roi_discrimination.py: 9 aggregation scores,",
         "   ROC vs MDT  ->  ez_analysis_mdt/",
         "scripts/regional_loocv.py, aggregation_separability.py,",
         "   per_index_discrimination.py"], TEAL, TEAL_H)
    box(41, 27, 57, 16, "Enhanced estimators, v2 (post hoc; scripts/v2/)",
        ["prep_subject_features.py --dataset {inhouse,chalfont}: ROI means (own parcellation",
         "   or atlas), 8-mm smoothed CBF on a mirror-symmetric 2.4-mm grid  ->  v2_features/",
         "analyze_v2.py: regional z_LI, z-bar_S; smoothed z_AI, hypoperfused z-mass LI_M;",
         "   leave-one-out calibration (95th percentile)  ->  ez_analysis_mdt/v2/<dataset>/*.csv",
         "multimetric_comparison.py: call rate, PPV, sensitivities, AUC, LR+, kappa, FP, strata",
         "advanced_lateralization.py (spike): covariate model, antisymmetric classifier, lobes"], VIOLET, VIOLET_H, fs=5.2)
    line(20, 27, 20, 22.5); line(66, 27, 66, 22.5); line(20, 22.5, 66, 22.5); arrow(43, 22.5, 43, 19.5)
    box(2, 8, 47, 11.5, "Audit (scripts/audit/, docs/audit/)",
        ["voxel_noise_budget.py [fwhm]: control SD of AI by region and scale",
         "regional_normative_prototype.py: ROI LI SD, leave-one-out false positives",
         "symreg/check_full_calibration.py: GM % |z_AI| >= 1.96 per patient",
         "docs/audit/METHODS_AUDIT_2026-10-20.md: findings and decisions"], ROSE, ROSE_H)
    box(51, 8, 47, 11.5, "Manuscript outputs (scripts/make_manuscript_figures.py)",
        ["Fig. 1 framework; Fig. 2 calibration; Fig. 3 cases (asymmetry/patients",
         "   + T1_sym); Fig. 4 enhanced estimator (v2 CSVs); Fig. 5 external cohort;",
         "Figs S1-S3 (appendix); tables from the CSVs above",
         "manuscript/Manuscript_Complete.tex  ->  tectonic  ->  PDF"], GREY, GREY_H)
    line(25, 8, 25, 4.5); line(75, 8, 75, 4.5); line(25, 4.5, 75, 4.5); arrow(50, 4.5, 50, 3.6)
    box(8, 0.4, 84, 3.2, None,
        ["Run control: scripts/run_capped.sh <name> <MemoryMax> <log> -- <cmd> (memory-capped, detached);  tests: python -m pytest -q tests;  code: github.com/aminjalali-research/AsymmetryAnalysis"],
        GREY, GREY_H, fs=5.2)
    fig.savefig(FIG / "figS_workflow.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / "figS_workflow.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    fig1_schematic()
    figS_workflow()
    fig4_enhanced()
    fig5_external_cohort()
    fig2_calibration()
    fig3_cases([("P029", "left TLE\nhyperperfusion, concordant"),
                ("P016", "right TLE, MRI-negative\nhypoperfusion, concordant"),
                ("P027", "right TLE\nhypoperfusion, discordant"),
                ("P015", "left TLE\nhyperperfusion, discordant")])
    print("done")
