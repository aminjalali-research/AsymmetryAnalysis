#!/usr/bin/env python3
"""
zAI Distribution Analysis (control-referenced asymmetry z-scores)

Summarises the voxelwise distribution of each patient's control-referenced
asymmetry z-score (zAI, Pipeline B, symmetric-template space) to check
calibration and to show where the clinical thresholds fall on the data.

Inputs (canonical symmetric-space outputs of 02_compute_zai.py --sym):
    results_zscore/asymmetry/patients/<PID>/<PID>_asymmetry_zscore.nii.gz
    results_zscore/asymmetry/groups/<band>/brain_mask.nii.gz
Each patient is evaluated inside the brain mask of its own age band
(band assignment as in symreg/check_full_calibration.py), restricted to
voxels where the zAI map is non-zero (i.e. where the patient has data).

Outputs:
    visual_analysis/zai_distributions.png   per-patient |zAI| histograms +
                                            cohort survival curve
    visual_analysis/zai_distribution_summary.csv
        per-patient n_voxels, fraction |zAI|>=1.96 / >=3 / >=4, p99.9 |zAI|

For a calibrated normative model the fraction |zAI|>=1.96 should be ~5-15%.

Usage:
    python analyze_zai_distributions.py                      # all 17 patients
    python analyze_zai_distributions.py --patients P013 P020
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ZAI_DIR = ROOT / "results_zscore" / "asymmetry" / "patients"
GROUP_DIR = ROOT / "results_zscore" / "asymmetry" / "groups"
OUT_DIR = ROOT / "visual_analysis"

# Patient -> normative age band (same as symreg/check_full_calibration.py PAT_BAND).
PAT_BAND = {
    "FM_20_39": ["P013", "P015", "P016", "P017", "P019", "P020", "P026",
                 "P028", "P029", "P030"],
    "FM_40_59": ["P018", "P021", "P022", "P023", "P024", "P027"],
    "FM_60_79": ["P025"],
}
BAND_OF = {p: b for b, ps in PAT_BAND.items() for p in ps}
THRESHOLDS = (1.96, 3.0, 4.0)


def load_abs_zai(pid: str, mask_cache: dict) -> np.ndarray | None:
    """Return |zAI| values for one patient inside its band brain mask."""
    band = BAND_OF.get(pid)
    zpath = ZAI_DIR / pid / f"{pid}_asymmetry_zscore.nii.gz"
    if band is None or not zpath.exists():
        return None
    if band not in mask_cache:
        mask_cache[band] = (
            nib.load(str(GROUP_DIR / band / "brain_mask.nii.gz")).get_fdata() > 0
        )
    z = nib.load(str(zpath)).get_fdata(dtype=np.float32)
    m = mask_cache[band] & (z != 0) & np.isfinite(z)
    return np.abs(z[m])


def summarise(pid: str, absz: np.ndarray) -> dict:
    row = {"patient": pid, "band": BAND_OF[pid], "n_voxels": int(absz.size)}
    for t in THRESHOLDS:
        row[f"frac_abs_zai_ge_{t:g}"] = float(np.mean(absz >= t))
    row["p99_9_abs_zai"] = float(np.percentile(absz, 99.9))
    return row


def plot(data: dict[str, np.ndarray], save_path: Path) -> None:
    pids = list(data)
    ncols = 6
    nrows = int(np.ceil(len(pids) / ncols)) + 1
    fig = plt.figure(figsize=(3.2 * ncols, 2.6 * nrows))
    bins = np.linspace(0, 10, 101)
    colors = ["#e6a100", "#d1495b", "#6a1b9a"]
    for i, pid in enumerate(pids):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.hist(np.clip(data[pid], 0, 10), bins=bins, color="#4a7fb0",
                density=True, log=True)
        for t, c in zip(THRESHOLDS, colors):
            ax.axvline(t, color=c, lw=1, ls="--")
        ax.set_title(f"{pid} ({BAND_OF[pid]})", fontsize=9)
        ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel("density (log)", fontsize=8)
    # cohort survival curve
    ax = fig.add_subplot(nrows, 1, nrows)
    xs = np.linspace(0, 8, 161)
    for pid in pids:
        a = np.sort(data[pid])
        surv = 1.0 - np.searchsorted(a, xs, side="left") / a.size
        ax.plot(xs, surv, lw=0.9, alpha=0.8, label=pid)
    for t, c in zip(THRESHOLDS, colors):
        ax.axvline(t, color=c, lw=1, ls="--", label=f"|zAI|={t:g}")
    ax.axhline(0.05, color="grey", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("|zAI| threshold")
    ax.set_ylabel("fraction of voxels >= threshold")
    ax.legend(ncol=10, fontsize=6, loc="lower left")
    fig.suptitle(f"Control-referenced zAI distributions (n={len(pids)} patients, "
                 "band brain mask)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=None,
                    help="Patient IDs (default: all 17 in PAT_BAND)")
    args = ap.parse_args()
    pids = args.patients or sorted(BAND_OF)

    mask_cache: dict = {}
    data, rows = {}, []
    for pid in pids:
        absz = load_abs_zai(pid, mask_cache)
        if absz is None or absz.size == 0:
            print(f"  {pid}: no zAI map / unknown band, skipped")
            continue
        data[pid] = absz
        rows.append(summarise(pid, absz))
    if not rows:
        print("No patient zAI maps found.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "zai_distribution_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.5f")
    fig_path = OUT_DIR / "zai_distributions.png"
    plot(data, fig_path)

    with pd.option_context("display.width", 140):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nCohort mean frac |zAI|>=1.96: {df['frac_abs_zai_ge_1.96'].mean():.3%}")
    print(f"Saved: {csv_path}\nSaved: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
