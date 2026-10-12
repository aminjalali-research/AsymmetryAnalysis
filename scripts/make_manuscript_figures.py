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
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, text, fc="#f2f2f2", fs=8.5, bold=False):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                             fc=fc, ec="#444444", lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="#444444"))

    # inputs
    box(2, 78, 30, 16, "Patients (n = 17)\nmulti-PLD pCASL + T1w\nHCP-ASL processing", fc="#dbe9f6")
    box(68, 78, 30, 16, "Healthy controls (n = 180)\nHCP Lifespan ASL\nthree age bands of 60", fc="#dbe9f6")
    box(35, 78, 30, 16, "Left-right symmetric\nT1 template\n(FSL flirt + fnirt)", fc="#fff2cc")
    arrow(32, 86, 35, 86); arrow(68, 86, 65, 86)

    # pipeline A
    box(2, 46, 44, 24, "Pipeline A  (regional)\n37 homologous region pairs\n(Desikan-Killiany + hippocampus,\namygdala, thalamus)\n15 asymmetry indices, 4 aggregations", fc="#e2f0d9")
    # pipeline B
    box(54, 46, 44, 24, "Pipeline B  (voxel-wise)\nAI(v) = 100 (L - R) / mean(L, R)\nzAI(v) = (AI - mean_ctrl) / SD_ctrl\nbilateral SD pooling, 5 % SD floor\nclusters |zAI| >= 3, >= 50 voxels", fc="#e2f0d9")
    arrow(17, 78, 17, 70); arrow(50, 78, 76, 70); arrow(83, 78, 83, 70)

    # direction inference
    box(54, 22, 44, 18, "Direction-of-abnormality inference\nper-side perfusion z vs controls:\nhypo / hyper / mixed / subtle\npredicted side = deviating hemisphere", fc="#fce4d6")
    arrow(76, 46, 76, 40)

    # outputs
    box(2, 4, 44, 12, "Lateralization score per patient\nROC vs MDT reference (n = 13)", fc="#ededed")
    box(54, 4, 44, 12, "Patient report: side, strength,\nlobe-level clusters, direction class", fc="#ededed")
    arrow(24, 46, 24, 16); arrow(76, 22, 76, 16)
    ax.text(50, 1, "Both outputs are compared with the multidisciplinary-team (MDT) lateralization",
            ha="center", va="bottom", fontsize=8, style="italic")
    fig.savefig(FIG / "fig1_pipeline_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig1_pipeline_overview.pdf", bbox_inches="tight")
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
            ax.imshow(t1[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
            rgba = np.zeros(vol[:, :, z].T.shape + (4,))
            for cls, colr in cmap.items():
                m = vol[:, :, z].T == cls
                rgba[m] = matplotlib.colors.to_rgba(colr, alpha=0.85)
            ax.imshow(rgba, origin="lower", interpolation="nearest")
            ax.set_title(f"z = {z}", fontsize=7, pad=2)
            ax.axis("off")
        # coronal through the largest cluster
        com = ndimage.center_of_mass(lab == order[0] + 1) if k else (113, 136, 113)
        y = int(round(com[1]))
        ax = axes[row, 3]
        ax.imshow(t1[:, y, :].T, cmap="gray", origin="lower", vmin=0, vmax=1)
        rgba = np.zeros(vol[:, y, :].T.shape + (4,))
        for cls, colr in cmap.items():
            m = vol[:, y, :].T == cls
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


if __name__ == "__main__":
    fig1_schematic()
    fig2_calibration()
    fig3_cases([("P029", "left TLE\nhyperperfusion, concordant"),
                ("P016", "right TLE, MRI-negative\nhypoperfusion, concordant"),
                ("P027", "right TLE\nhypoperfusion, discordant"),
                ("P015", "left TLE\nhyperperfusion, discordant")])
    print("done")
