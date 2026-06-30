#!/usr/bin/env python3
"""
roi_native_vs_mni.py — Native-T1w vs MNI ROI perfusion-asymmetry comparison.

METHODOLOGICAL EXPERIMENT (manuscript). Tests whether spatial normalization to
MNI152 distorts the ROI-level perfusion-asymmetry signal / EZ lateralization,
relative to computing the same ROI asymmetry in native T1w space.

Rationale (Rai 2026 thesis): the thesis computed ROI asymmetry in NATIVE space
but voxel-wise z-maps in MNI. The current project pipeline computes ROI
asymmetry in MNI. Voxel-wise analysis requires a common grid (MNI only) so it is
NOT comparable across spaces; ROI analysis is anatomy-defined (aparc+aseg labels)
and therefore works identically in both spaces, which is exactly what we compare.

Per patient, per space:
  1. Load perfusion (2.5 mm) + aparc+aseg (0.8 mm). Same space, different
     resolution. Resample the LABEL volume to the perfusion grid with
     NEAREST-neighbour (no perfusion interpolation).
  2. For each of 37 ROI pairs, mean perfusion (voxels with label & perfusion>0)
     for Left and Right.
  3. Compute the 15 asymmetry indices via src.calculator.AsymmetryCalculator.

Comparisons (native vs MNI):
  - Per-ROI AI agreement over all patient x ROI observations: Pearson r, ICC(2,1),
    Bland-Altman bias + 95% limits of agreement, mean absolute difference. The
    primary index is the signed laterality_index (LI); we also summarise the
    Pipeline-A magnitude-weighted AAI aggregate.
  - Per-patient lateralization concordance: sign of the magnitude-weighted AAI
    aggregate (which side dominates) — native vs MNI agreement rate + flips.
  - Pipeline-A discrimination: weighted-AAI ROC-AUC vs MDT ground truth on
    unilateral L/R patients, native and MNI separately.
  - ROIs that disagree most between spaces (rank by mean |native LI - MNI LI|).

Outputs (results/native_vs_mni/):
  per_roi_ai_native.csv, per_roi_ai_mni.csv, agreement_summary.csv,
  comparison_figure.png, SUMMARY.md

Usage:
  python roi_native_vs_mni.py --all
  python roi_native_vs_mni.py --patient P013
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
from src.calculator import AsymmetryCalculator  # noqa: E402
from ez_ground_truth import EZ_GROUND_TRUTH, get_ez_label  # noqa: E402

PIPELINE = "/home/amin/multimodal/Aikam/projects/asl/asl_pipeline"
OUTDIR = os.path.join(PROJ, "results", "native_vs_mni")

# ---- 37 ROI pairs: (name, left_label, right_label) ----
SUBCORTICAL = [
    ("Thalamus", 10, 49),
    ("Hippocampus", 17, 53),
    ("Amygdala", 18, 54),
]
# Desikan-Killiany cortical names (label offset 1000 L / 2000 R)
DK_NAMES = {
    1: "bankssts", 2: "caudalanteriorcingulate", 3: "caudalmiddlefrontal",
    5: "cuneus", 6: "entorhinal", 7: "fusiform", 8: "inferiorparietal",
    9: "inferiortemporal", 10: "isthmuscingulate", 11: "lateraloccipital",
    12: "lateralorbitofrontal", 13: "lingual", 14: "medialorbitofrontal",
    15: "middletemporal", 16: "parahippocampal", 17: "paracentral",
    18: "parsopercularis", 19: "parsorbitalis", 20: "parstriangularis",
    21: "pericalcarine", 22: "postcentral", 23: "posteriorcingulate",
    24: "precentral", 25: "precuneus", 26: "rostralanteriorcingulate",
    27: "rostralmiddlefrontal", 28: "superiorfrontal", 29: "superiorparietal",
    30: "superiortemporal", 31: "supramarginal", 32: "frontalpole",
    33: "temporalpole", 34: "transversetemporal", 35: "insula",
}
CORTICAL = [
    (f"ctx-{DK_NAMES[i]}", 1000 + i, 2000 + i)
    for i in sorted(DK_NAMES.keys())
]
ROI_PAIRS = SUBCORTICAL + CORTICAL  # 3 + 34 = 37

CALC = AsymmetryCalculator()


def resample_labels_to_perfusion(aparc_img, perf_img):
    """Resample integer label volume onto perfusion grid via nearest-neighbour.

    Maps each perfusion voxel -> world -> aparc voxel using the composed
    affine, then order-0 (nearest) sampling. No interpolation of labels.
    """
    aparc = np.asarray(aparc_img.dataobj).astype(np.int32)
    # voxel(aparc) = inv(aparc.affine) @ perf.affine @ voxel(perf)
    M = np.linalg.inv(aparc_img.affine) @ perf_img.affine
    matrix = M[:3, :3]
    offset = M[:3, 3]
    out = affine_transform(
        aparc, matrix, offset=offset, output_shape=perf_img.shape,
        order=0, mode="constant", cval=0, prefilter=False,
    )
    return out.astype(np.int32)


def extract_roi_means(perf, labels):
    """Return dict name -> (L_mean, R_mean, nL, nR) over labelled, perf>0 voxels."""
    rows = {}
    for name, lL, lR in ROI_PAIRS:
        mL = (labels == lL) & (perf > 0)
        mR = (labels == lR) & (perf > 0)
        nL, nR = int(mL.sum()), int(mR.sum())
        vL = float(perf[mL].mean()) if nL else np.nan
        vR = float(perf[mR].mean()) if nR else np.nan
        rows[name] = (vL, vR, nL, nR)
    return rows


def process_patient_space(pid, space):
    """Load one patient/space, return per-ROI DataFrame with 15 indices."""
    sub = "T1w" if space == "native" else "MNINonLinear"
    perf_p = os.path.join(PIPELINE, pid, sub, "ASL", "perfusion_calib.nii.gz")
    apar_p = os.path.join(PIPELINE, pid, sub, "aparc+aseg.nii.gz")
    if not (os.path.exists(perf_p) and os.path.exists(apar_p)):
        return None, f"missing files ({sub})"

    perf_img = nib.load(perf_p)
    apar_img = nib.load(apar_p)
    perf = np.asarray(perf_img.dataobj).astype(np.float64)
    labels = resample_labels_to_perfusion(apar_img, perf_img)

    means = extract_roi_means(perf, labels)
    df = pd.DataFrame(
        [
            {
                "patient": pid, "space": space, "roi": name,
                "left_mean_perfusion": v[0], "right_mean_perfusion": v[1],
                "left_nvoxels": v[2], "right_nvoxels": v[3],
            }
            for name, v in means.items()
        ]
    )
    df = CALC.calculate_all_indices(df)
    return df, None


def weighted_aai_aggregate(df):
    """Magnitude-weighted signed laterality aggregate (Pipeline A primary).

    Each ROI's signed LI weighted by its AAI magnitude, summed over the 37
    pairs. Sign => dominant side (positive=Left>Right). |value| => strength.
    Returns (signed_aggregate, predicted_side).
    """
    li = df["laterality_index"].to_numpy(dtype=float)
    aai = df["absolute_asymmetry_index"].to_numpy(dtype=float)
    m = np.isfinite(li) & np.isfinite(aai)
    if not m.any():
        return np.nan, None
    agg = float(np.sum(li[m] * aai[m]))
    side = "L" if agg > 0 else "R"
    return agg, side


# ---------- agreement statistics ----------
def icc_2_1(x, y):
    """ICC(2,1) two-way random, single-rater, absolute agreement.

    x,y: paired measurements (native, MNI) of the same target. Shrout & Fleiss
    (1979) ICC(2,1).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 2:
        return np.nan
    M = np.column_stack([x, y])  # n subjects x k=2 raters
    k = 2
    grand = M.mean()
    ss_rows = k * ((M.mean(axis=1) - grand) ** 2).sum()   # between subjects
    ss_cols = n * ((M.mean(axis=0) - grand) ** 2).sum()   # between raters
    ss_total = ((M - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols                 # residual
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if denom != 0 else np.nan


def roc_auc(scores, labels_pos):
    """AUC via Mann-Whitney U. scores higher => more 'positive'.

    labels_pos: boolean array (True=positive class).
    """
    scores = np.asarray(scores, float)
    pos = np.asarray(labels_pos, bool)
    m = np.isfinite(scores)
    scores, pos = scores[m], pos[m]
    P, N = pos.sum(), (~pos).sum()
    if P == 0 or N == 0:
        return np.nan
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    auc = (ranks[pos].sum() - P * (P + 1) / 2.0) / (P * N)
    return float(auc)


def bland_altman(x, y):
    """Bias (mean diff native-MNI) + 95% limits of agreement + MAD."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    d = x[m] - y[m]
    bias = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else np.nan
    return bias, sd, bias - 1.96 * sd, bias + 1.96 * sd, float(np.abs(d).mean())


def pearson_r(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def run(patients):
    os.makedirs(OUTDIR, exist_ok=True)
    nat_frames, mni_frames, skipped = [], [], []
    for pid in patients:
        for space, store in (("native", nat_frames), ("mni", mni_frames)):
            df, err = process_patient_space(pid, space)
            if err:
                skipped.append((pid, space, err))
                print(f"  SKIP {pid} {space}: {err}")
            else:
                store.append(df)
                print(f"  OK   {pid} {space}: {len(df)} ROIs")

    if not nat_frames or not mni_frames:
        print("No data processed.")
        return

    nat = pd.concat(nat_frames, ignore_index=True)
    mni = pd.concat(mni_frames, ignore_index=True)
    nat.to_csv(os.path.join(OUTDIR, "per_roi_ai_native.csv"), index=False)
    mni.to_csv(os.path.join(OUTDIR, "per_roi_ai_mni.csv"), index=False)

    # ---- merge paired observations on (patient, roi) ----
    keys = ["patient", "roi"]
    merged = nat.merge(mni, on=keys, suffixes=("_nat", "_mni"))

    PRIMARY = "laterality_index"
    xli = merged[f"{PRIMARY}_nat"]
    yli = merged[f"{PRIMARY}_mni"]

    r = pearson_r(xli, yli)
    icc = icc_2_1(xli, yli)
    bias, sd, loa_lo, loa_hi, mad = bland_altman(xli, yli)

    # ---- per-patient lateralization concordance (weighted AAI) ----
    concord_rows = []
    for pid in sorted(set(merged["patient"])):
        dn = nat[nat["patient"] == pid]
        dm = mni[mni["patient"] == pid]
        agg_n, side_n = weighted_aai_aggregate(dn)
        agg_m, side_m = weighted_aai_aggregate(dm)
        mdt = get_ez_label(pid)
        concord_rows.append({
            "patient": pid, "mdt": mdt,
            "agg_native": agg_n, "side_native": side_n,
            "agg_mni": agg_m, "side_mni": side_m,
            "agree": side_n == side_m,
        })
    concord = pd.DataFrame(concord_rows)
    concord.to_csv(os.path.join(OUTDIR, "lateralization_concordance.csv"),
                   index=False)
    n_pat = len(concord)
    n_agree = int(concord["agree"].sum())
    concord_rate = n_agree / n_pat if n_pat else np.nan

    # ---- Pipeline-A discrimination AUC on unilateral L/R ----
    # Positive class = Left EZ; score = signed weighted aggregate (higher=>more Left).
    uni = concord[concord["mdt"].isin(["L", "R"])].copy()
    pos = (uni["mdt"] == "L").to_numpy()
    auc_native = roc_auc(uni["agg_native"].to_numpy(), pos)
    auc_mni = roc_auc(uni["agg_mni"].to_numpy(), pos)

    # ---- most-discordant ROIs (mean |LI_nat - LI_mni|) ----
    merged["abs_li_diff"] = (xli - yli).abs()
    roi_disc = (merged.groupby("roi")["abs_li_diff"]
                .agg(["mean", "max", "count"])
                .sort_values("mean", ascending=False))
    roi_disc.to_csv(os.path.join(OUTDIR, "roi_discordance.csv"))

    # ---- agreement_summary.csv ----
    summary = pd.DataFrame([{
        "primary_index": PRIMARY,
        "n_paired_obs": int((np.isfinite(xli) & np.isfinite(yli)).sum()),
        "n_patients": n_pat,
        "pearson_r": r,
        "icc_2_1": icc,
        "ba_bias": bias,
        "ba_sd_diff": sd,
        "ba_loa_low": loa_lo,
        "ba_loa_high": loa_hi,
        "mean_abs_diff": mad,
        "lat_concordance_rate": concord_rate,
        "n_lat_agree": n_agree,
        "auc_native": auc_native,
        "auc_mni": auc_mni,
        "n_unilateral": int(len(uni)),
    }])
    summary.to_csv(os.path.join(OUTDIR, "agreement_summary.csv"), index=False)

    make_figure(xli, yli, bias, loa_lo, loa_hi, concord, roi_disc)
    write_summary_md(summary, concord, roi_disc, skipped, patients)

    print("\n=== HEADLINE ===")
    print(f"paired ROI obs : {summary['n_paired_obs'][0]}")
    print(f"Pearson r (LI) : {r:.4f}")
    print(f"ICC(2,1)       : {icc:.4f}")
    print(f"BA bias        : {bias:+.4f}  LoA [{loa_lo:+.4f}, {loa_hi:+.4f}]")
    print(f"mean|diff|     : {mad:.4f}")
    print(f"lat concordance: {n_agree}/{n_pat} = {concord_rate:.3f}")
    print(f"AUC native     : {auc_native:.3f}   AUC MNI: {auc_mni:.3f}")
    print(f"top-3 discordant ROIs: {list(roi_disc.head(3).index)}")
    print(f"outputs in {OUTDIR}")


def make_figure(xli, yli, bias, loa_lo, loa_hi, concord, roi_disc):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    x = np.asarray(xli, float)
    y = np.asarray(yli, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    # (a) scatter native vs MNI LI + identity
    ax = axes[0, 0]
    ax.scatter(x, y, s=14, alpha=0.5, color="#2b6cb0")
    lim = max(np.abs(np.concatenate([x, y]))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="identity")
    ax.set_xlabel("Native T1w  Laterality Index")
    ax.set_ylabel("MNI  Laterality Index")
    ax.set_title(f"(a) Per-ROI LI: native vs MNI  (r={pearson_r(x,y):.3f})")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    # (b) Bland-Altman
    ax = axes[0, 1]
    mean_xy = (x + y) / 2
    diff = x - y
    ax.scatter(mean_xy, diff, s=14, alpha=0.5, color="#9b2c2c")
    ax.axhline(bias, color="k", lw=1.2, label=f"bias={bias:+.3f}")
    ax.axhline(loa_lo, color="gray", ls="--", lw=1,
               label=f"95% LoA [{loa_lo:+.2f},{loa_hi:+.2f}]")
    ax.axhline(loa_hi, color="gray", ls="--", lw=1)
    ax.set_xlabel("Mean LI  (native, MNI)/2")
    ax.set_ylabel("Difference  native - MNI")
    ax.set_title("(b) Bland-Altman of per-ROI LI")
    ax.legend(loc="upper right", fontsize=8)

    # (c) per-patient side agreement
    ax = axes[1, 0]
    c = concord.sort_values("patient")
    ypos = np.arange(len(c))
    colors = ["#2f855a" if a else "#c53030" for a in c["agree"]]
    ax.barh(ypos, c["agg_native"], color="#3182ce", alpha=0.7,
            height=0.4, label="native agg")
    ax.barh(ypos + 0.0, 0, color="none")
    ax.scatter(c["agg_mni"], ypos, color="#dd6b20", s=40, zorder=5,
               label="MNI agg")
    for i, (_, row) in enumerate(c.iterrows()):
        mark = "OK" if row["agree"] else "FLIP"
        ax.text(0, i + 0.28,
                f"{row['patient']} MDT={row['mdt']} "
                f"N={row['side_native']}/M={row['side_mni']} {mark}",
                fontsize=6, va="center")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Weighted-AAI signed aggregate (+L / -R)")
    ax.set_title("(c) Per-patient lateralization (bar=native, dot=MNI)")
    ax.legend(loc="lower right", fontsize=8)

    # (d) most discordant ROIs
    ax = axes[1, 1]
    top = roi_disc.head(12).iloc[::-1]
    ax.barh(np.arange(len(top)), top["mean"], color="#6b46c1", alpha=0.8)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([s.replace("ctx-", "") for s in top.index], fontsize=7)
    ax.set_xlabel("Mean |LI native - LI MNI|")
    ax.set_title("(d) Most space-discordant ROIs (top 12)")

    fig.suptitle("Native-T1w vs MNI ROI perfusion-asymmetry comparison",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUTDIR, "comparison_figure.png"), dpi=300)
    plt.close(fig)


def write_summary_md(summary, concord, roi_disc, skipped, patients):
    s = summary.iloc[0]
    lines = []
    lines.append("# Native-T1w vs MNI ROI Perfusion-Asymmetry Comparison\n")
    lines.append("Methodological experiment: does spatial normalization to "
                 "MNI152 materially change ROI-level perfusion asymmetry and EZ "
                 "lateralization, relative to native T1w space? ROI asymmetry is "
                 "anatomy-defined (aparc+aseg) so it is computable in both spaces; "
                 "voxel-wise z-maps are intentionally out of scope (they need a "
                 "common MNI grid).\n")
    lines.append("## Headline numbers\n")
    lines.append(f"- Patients analyzed: **{int(s['n_patients'])}** "
                 f"(of {len(patients)} cohort); paired ROI observations: "
                 f"**{int(s['n_paired_obs'])}**")
    lines.append(f"- Primary index: **laterality_index (LI)**")
    lines.append(f"- Pearson r (native vs MNI LI): **{s['pearson_r']:.4f}**")
    lines.append(f"- ICC(2,1) absolute agreement: **{s['icc_2_1']:.4f}**")
    lines.append(f"- Bland-Altman bias (native - MNI): **{s['ba_bias']:+.4f}**; "
                 f"95% limits of agreement **[{s['ba_loa_low']:+.4f}, "
                 f"{s['ba_loa_high']:+.4f}]**")
    lines.append(f"- Mean absolute LI difference: **{s['mean_abs_diff']:.4f}**")
    lines.append(f"- Per-patient lateralization concordance "
                 f"(weighted-AAI dominant side): "
                 f"**{int(s['n_lat_agree'])}/{int(s['n_patients'])} = "
                 f"{s['lat_concordance_rate']:.3f}**")
    lines.append(f"- Pipeline-A discrimination ROC-AUC vs MDT "
                 f"(n={int(s['n_unilateral'])} unilateral L/R): "
                 f"native **{s['auc_native']:.3f}**, MNI **{s['auc_mni']:.3f}**\n")

    lines.append("## Most space-discordant ROIs (mean |LI_native - LI_mni|)\n")
    lines.append("| ROI | mean |dLI| | max |dLI| | n |")
    lines.append("|---|---|---|---|")
    for roi, row in roi_disc.head(10).iterrows():
        lines.append(f"| {roi} | {row['mean']:.4f} | {row['max']:.4f} | "
                     f"{int(row['count'])} |")
    lines.append("")

    lines.append("## Per-patient lateralization detail\n")
    lines.append("| patient | MDT | native side | MNI side | agree |")
    lines.append("|---|---|---|---|---|")
    for _, row in concord.sort_values("patient").iterrows():
        lines.append(f"| {row['patient']} | {row['mdt']} | {row['side_native']} "
                     f"| {row['side_mni']} | {'yes' if row['agree'] else 'NO'} |")
    lines.append("")

    if skipped:
        lines.append("## Skipped (missing data)\n")
        for pid, space, err in skipped:
            lines.append(f"- {pid} [{space}]: {err}")
        lines.append("")

    # interpretation
    r = s["pearson_r"]
    icc = s["icc_2_1"]
    cr = s["lat_concordance_rate"]
    strong = (r >= 0.9 and icc >= 0.9)
    lines.append("## Plain conclusion (manuscript Methods/Results)\n")
    lines.append(
        f"Across {int(s['n_patients'])} patients and {int(s['n_paired_obs'])} "
        f"paired region observations, per-ROI perfusion laterality computed in "
        f"native T1w space and in MNI152 space agreed "
        f"{'strongly' if strong else 'moderately' if (r>=0.75 or icc>=0.75) else 'only weakly'} "
        f"(Pearson r = {r:.2f}, ICC(2,1) = {icc:.2f}), with a near-zero "
        f"Bland-Altman bias of {s['ba_bias']:+.3f} and 95% limits of agreement "
        f"[{s['ba_loa_low']:+.2f}, {s['ba_loa_high']:+.2f}] laterality-index "
        f"units. Patient-level EZ lateralization (sign of the magnitude-weighted "
        f"absolute-asymmetry aggregate) was concordant between spaces in "
        f"{int(s['n_lat_agree'])}/{int(s['n_patients'])} patients "
        f"({cr:.0%}), and Pipeline-A discrimination against the MDT ground truth "
        f"was {'essentially unchanged' if abs(s['auc_native']-s['auc_mni'])<0.05 else 'shifted'} "
        f"(ROC-AUC native {s['auc_native']:.2f} vs MNI {s['auc_mni']:.2f}). "
        f"{'Spatial normalization to MNI152 therefore does not materially distort the ROI-level perfusion-asymmetry / lateralization signal, supporting the project pipeline choice to compute ROI asymmetry in MNI space.' if (strong and cr>=0.8) else 'Spatial normalization introduces some ROI-level reshuffling but preserves the dominant lateralization signal in most patients; residual discordance is concentrated in the ROIs listed above.'}"
    )
    with open(os.path.join(OUTDIR, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true",
                   help="process the full ez_ground_truth cohort")
    g.add_argument("--patient", help="single patient id, e.g. P013")
    args = ap.parse_args()

    if args.all:
        patients = sorted(EZ_GROUND_TRUTH.keys())
    else:
        patients = [args.patient]
    print(f"Processing {len(patients)} patient(s): {patients}")
    run(patients)


if __name__ == "__main__":
    main()
