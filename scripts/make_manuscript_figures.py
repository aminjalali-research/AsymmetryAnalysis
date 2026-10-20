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
def cluster_class_volume(pid: str, peak=4.0, size=100):
    """Rebuild a label volume: 1 = hypo, 2 = hyper, 3 = subtle/mixed, for in-scope clusters at the tier."""
    d = PAT / pid
    rep = pd.read_csv(d / f"{pid}_asymmetry_cluster_report.csv")
    normed = rep["primary_region"].apply(ci._normalize_region)
    rep["canon"] = normed.apply(lambda t: t[1])
    rep["in_scope"] = rep["canon"].apply(lambda n: bool(n) and n in ci.CLINICAL_NAMES)
    struct = get_structure_3d()
    out = None
    for direction, fname in (("left-dominant", "left_dominant"), ("right-dominant", "right_dominant")):
        img = nib.load(str(d / f"{pid}_asymmetry_{fname}.nii.gz"))
        mask = img.get_fdata() > 0
        lab, n = ndimage.label(mask, structure=struct)
        if out is None:
            out = np.zeros(mask.shape, np.uint8)
        sub = rep[rep.direction == direction].set_index("cluster_id")
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        cid = 0
        for i, s in enumerate(sizes, 1):
            if s < 50:
                continue
            cid += 1
            if cid not in sub.index:
                continue
            r = sub.loc[cid]
            assert int(r.size_voxels) == int(s), (pid, direction, cid, r.size_voxels, s)
            if abs(r.peak_z) < peak or s < size or not r.in_scope:
                continue
            # The asymmetry map is antisymmetric, so every finding appears as a
            # mirrored pair of clusters; draw only the member that lies in the
            # hemisphere the direction inference assigns the abnormality to.
            if str(r.cluster_hemi) != str(r.ez_side_pred):
                continue
            cls = {"hypo": 1, "hyper": 2}.get(r.direction_class, 3)
            out[lab == i] = cls
    return out, rep


def fig3_cases(cases):
    # NOTE (2026-10-20): the symmetric template affine has a negative x step (index 0 = x = +90 mm, right
    # hemisphere); axis 0 is reversed for display so that image-left = left hemisphere (neurological).
    t1 = nib.load(str(T1)).get_fdata()
    t1 = t1 / np.percentile(t1[t1 > 0], 99)
    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(7.2, 1.9 * n))
    cmap = {1: "#1f77e0", 2: "#d62728", 3: "#999999"}
    for row, (pid, note) in enumerate(cases):
        vol, rep = cluster_class_volume(pid)
        # choose slices: z of the three largest classified (hypo/hyper) in-scope clusters, else largest overall
        lab, k = ndimage.label(vol > 0)
        sizes = ndimage.sum(vol > 0, lab, range(1, k + 1))
        order = np.argsort(sizes)[::-1]
        zs = []
        for i in order:
            com = ndimage.center_of_mass(lab == i + 1)
            z = int(round(com[2]))
            if all(abs(z - zz) > 6 for zz in zs):
                zs.append(z)
            if len(zs) == 3:
                break
        while len(zs) < 3:
            zs.append(zs[-1] + 10 if zs else 110)
        for col, z in enumerate(zs):
            ax = axes[row, col]
            ax.imshow(t1[::-1, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
            rgba = np.zeros(vol[::-1, :, z].T.shape + (4,))
            for cls, colr in cmap.items():
                m = vol[::-1, :, z].T == cls
                rgba[m] = matplotlib.colors.to_rgba(colr, alpha=0.85)
            ax.imshow(rgba, origin="lower", interpolation="nearest")
            ax.set_title(f"z = {z}", fontsize=7, pad=2)
            ax.axis("off")
        # coronal through the largest cluster
        com = ndimage.center_of_mass(lab == order[0] + 1) if k else (113, 136, 113)
        y = int(round(com[1]))
        ax = axes[row, 3]
        ax.imshow(t1[::-1, y, :].T, cmap="gray", origin="lower", vmin=0, vmax=1)
        rgba = np.zeros(vol[::-1, y, :].T.shape + (4,))
        for cls, colr in cmap.items():
            m = vol[::-1, y, :].T == cls
            rgba[m] = matplotlib.colors.to_rgba(colr, alpha=0.85)
        ax.imshow(rgba, origin="lower", interpolation="nearest")
        ax.set_title(f"coronal y = {y}", fontsize=7, pad=2); ax.axis("off")
        axes[row, 0].text(-0.06, 0.5, f"{pid}: {note}", transform=axes[row, 0].transAxes, rotation=90,
                          ha="center", va="center", fontsize=6.5, linespacing=1.1)
    handles = [mpatches.Patch(color=cmap[1], label="hypoperfusion on cluster side"),
               mpatches.Patch(color=cmap[2], label="hyperperfusion on cluster side"),
               mpatches.Patch(color=cmap[3], label="subtle (no per-side deviation)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.995, "Neurological convention: left of image = left hemisphere", ha="center", va="top", fontsize=7.5)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.98))
    fig.savefig(FIG / "fig3_representative_cases.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig3_representative_cases.pdf", bbox_inches="tight")
    plt.close(fig)


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
        ["66 controls: age 40 \u00b1 13 y (19\u201366), 12 women",
         "99 patients: age 35 \u00b1 9 y (19\u201360), 31 women; lesional 47 / MRI-negative 52",
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


if __name__ == "__main__":
    fig1_schematic()
    fig4_enhanced()
    fig5_external_cohort()
    fig2_calibration()
    fig3_cases([("P029", "left TLE\nhyperperfusion, concordant"),
                ("P016", "right TLE, MRI-negative\nhypoperfusion, concordant"),
                ("P027", "right TLE\nhypoperfusion, discordant"),
                ("P015", "left TLE\nhyperperfusion, discordant")])
    print("done")
