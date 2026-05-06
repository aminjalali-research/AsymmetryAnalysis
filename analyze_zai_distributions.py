#!/usr/bin/env python3
"""
|AI| Distribution Analysis for Statistical Threshold Selection

Generates publication-quality distribution charts of the Asymmetry Index (AI)
to inform and visualise where various thresholds fall on the data.

Figures produced
────────────────
1. Per-patient |AI| histogram with quality-mask subset, half-normal noise fit,
   and annotated threshold lines (percentile + TFCE effective cutoff).
2. Group-level overlay of all 15 patients' |AI| CDFs.
3. Threshold sensitivity curve: fraction of voxels retained vs. |AI| cutoff.
4. Noise-vs-signal decomposition (half-normal + empirical tail).

Usage:
    python analyze_zai_distributions.py                     # all patients
    python analyze_zai_distributions.py --patients P013 P020
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── globals ────────────────────────────────────────────────────────────
OUT_DIR = Path("results_voxel/distribution_analysis")
PERCENTILES_TO_SHOW = [80, 85, 90, 95, 97, 99]
BINS = 200  # histogram bins for 0–200 range


# ─── data loader ────────────────────────────────────────────────────────
def load_patient_data(pid: str):
    """Return dict with |AI| arrays and TFCE mask for one patient."""
    import nibabel as nib

    ai_path = Path(f"results_voxel/{pid}/{pid}_voxel_ai_map.nii.gz")
    mask_path = Path(f"results_voxel/{pid}/{pid}_voxel_brain_mask.nii.gz")
    perf_path = Path(f"Dataset MNI/{pid}/perfusion_calib.nii.gz")
    tfce_path = Path(
        f"results_voxel/{pid}/method_comparison/{pid}_M11_Quality_TFCE.nii.gz"
    )

    if not ai_path.exists():
        return None

    ai = nib.load(str(ai_path)).get_fdata()
    brain_mask = (
        nib.load(str(mask_path)).get_fdata().astype(bool)
        if mask_path.exists()
        else ai != 0
    )

    # Bilateral quality mask
    if perf_path.exists():
        perf = nib.load(str(perf_path)).get_fdata()
        perf_flip = np.flip(perf, axis=0)
        brain_perf = perf[brain_mask]
        median_cbf = (
            np.median(brain_perf[brain_perf > 0]) if (brain_perf > 0).any() else 1.0
        )
        min_cbf = 0.10 * median_cbf
        quality_mask = brain_mask & (perf > min_cbf) & (perf_flip > min_cbf)
    else:
        quality_mask = brain_mask & (np.abs(ai) < 195)

    tfce_mask = (
        nib.load(str(tfce_path)).get_fdata() > 0
        if tfce_path.exists()
        else np.zeros_like(ai, dtype=bool)
    )

    abs_ai_brain = np.abs(ai[brain_mask])
    abs_ai_quality = np.abs(ai[quality_mask])
    abs_ai_tfce = np.abs(ai[tfce_mask]) if tfce_mask.any() else np.array([])

    return dict(
        pid=pid,
        abs_ai_brain=abs_ai_brain,
        abs_ai_quality=abs_ai_quality,
        abs_ai_tfce=abs_ai_tfce,
        n_brain=int(brain_mask.sum()),
        n_quality=int(quality_mask.sum()),
        n_tfce=int(tfce_mask.sum()),
    )


# ─── half-normal noise model ───────────────────────────────────────────
def fit_half_normal(abs_ai: np.ndarray, use_fraction: float = 0.50):
    """Fit σ of a half-normal from the lower `use_fraction` of |AI|."""
    sorted_vals = np.sort(abs_ai)
    n_use = int(len(sorted_vals) * use_fraction)
    lower = sorted_vals[:n_use]
    # For half-normal, E[|X|] = σ √(2/π)  ⇒  σ = mean * √(π/2)
    sigma = float(np.mean(lower) * np.sqrt(np.pi / 2))
    return sigma


def half_normal_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    """Half-normal PDF: f(x) = (2/(σ√(2π))) exp(-x²/(2σ²)) for x≥0."""
    return (2 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x**2) / (2 * sigma**2))


# ─── figure 1: per-patient distribution ────────────────────────────────
def plot_single_patient(data: dict, save_path: Path):
    """Full |AI| distribution for one patient with annotations."""
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]}
    )
    pid = data["pid"]

    # ── Top panel: histogram + noise model ──
    ax = axes[0]
    bin_edges = np.linspace(0, 200, BINS + 1)

    # Brain mask histogram
    ax.hist(
        data["abs_ai_brain"],
        bins=bin_edges,
        density=True,
        alpha=0.25,
        color="gray",
        label=f"All brain voxels (n={data['n_brain']:,})",
    )
    # Quality mask histogram
    ax.hist(
        data["abs_ai_quality"],
        bins=bin_edges,
        density=True,
        alpha=0.5,
        color="steelblue",
        label=f"Quality-masked (n={data['n_quality']:,})",
    )

    # Half-normal noise fit (on quality-masked data)
    sigma = fit_half_normal(data["abs_ai_quality"])
    x_fit = np.linspace(0, 200, 500)
    y_fit = half_normal_pdf(x_fit, sigma)
    ax.plot(
        x_fit,
        y_fit,
        "r-",
        lw=2.5,
        label=f"Half-normal noise (σ={sigma:.1f})",
    )

    # Percentile lines
    colors_pct = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(PERCENTILES_TO_SHOW)))
    for pct, clr in zip(PERCENTILES_TO_SHOW, colors_pct):
        cutoff = np.percentile(data["abs_ai_quality"], pct)
        n_above = (data["abs_ai_quality"] >= cutoff).sum()
        pct_above = 100 * n_above / len(data["abs_ai_quality"])
        ax.axvline(
            cutoff,
            color=clr,
            ls="--",
            lw=1.5,
            alpha=0.85,
            label=f"P{pct} = {cutoff:.0f}  ({pct_above:.1f}% voxels)",
        )

    # TFCE effective cutoff
    if len(data["abs_ai_tfce"]) > 0:
        tfce_min = data["abs_ai_tfce"].min()
        n_tfce = data["n_tfce"]
        tfce_pct = 100 * n_tfce / data["n_quality"]
        ax.axvline(
            tfce_min,
            color="limegreen",
            ls="-",
            lw=3,
            alpha=0.9,
            label=f"TFCE effective cutoff = {tfce_min:.0f}  "
            f"({n_tfce:,} vox, {tfce_pct:.1f}%)",
        )
        # TFCE voxels highlighted
        ax.hist(
            data["abs_ai_tfce"],
            bins=bin_edges,
            density=True,
            alpha=0.45,
            color="limegreen",
            label=f"TFCE-selected (n={n_tfce:,})",
        )

    ax.set_xlim(0, 200)
    ax.set_xlabel("|Asymmetry Index|", fontsize=13)
    ax.set_ylabel("Probability density", fontsize=13)
    ax.set_title(
        f"{pid}  —  |AI| Distribution with Threshold Annotations",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # ── Bottom panel: survival function (1 - CDF) ──
    ax2 = axes[1]
    sorted_q = np.sort(data["abs_ai_quality"])
    survival = 1 - np.arange(1, len(sorted_q) + 1) / len(sorted_q)
    ax2.semilogy(sorted_q, survival, color="steelblue", lw=2, label="Quality-masked")

    # Noise survival
    from scipy.stats import halfnorm

    x_s = np.linspace(0, 200, 500)
    ax2.semilogy(
        x_s,
        halfnorm.sf(x_s, scale=sigma),
        "r--",
        lw=2,
        label=f"Half-normal (σ={sigma:.1f})",
    )

    for pct, clr in zip(PERCENTILES_TO_SHOW, colors_pct):
        cutoff = np.percentile(data["abs_ai_quality"], pct)
        ax2.axvline(cutoff, color=clr, ls="--", lw=1, alpha=0.7)

    if len(data["abs_ai_tfce"]) > 0:
        ax2.axvline(data["abs_ai_tfce"].min(), color="limegreen", ls="-", lw=2.5)

    ax2.set_xlim(0, 200)
    ax2.set_ylim(1e-4, 1)
    ax2.set_xlabel("|AI| threshold", fontsize=13)
    ax2.set_ylabel("Fraction ≥ threshold", fontsize=13)
    ax2.set_title("Survival Function (complementary CDF)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {save_path.name}")


# ─── figure 2: group-level CDF overlay ─────────────────────────────────
def plot_group_cdf(all_data: list[dict], save_path: Path):
    """Overlay survival functions for all patients."""
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(all_data)))

    sigmas = []
    for d, clr in zip(all_data, cmap):
        sorted_q = np.sort(d["abs_ai_quality"])
        survival = 1 - np.arange(1, len(sorted_q) + 1) / len(sorted_q)
        ax.semilogy(sorted_q, survival, color=clr, lw=1.8, alpha=0.8, label=d["pid"])
        sigmas.append(fit_half_normal(d["abs_ai_quality"]))

    # Group-mean half-normal
    from scipy.stats import halfnorm

    mean_sigma = np.mean(sigmas)
    x_s = np.linspace(0, 200, 500)
    ax.semilogy(
        x_s,
        halfnorm.sf(x_s, scale=mean_sigma),
        "k--",
        lw=3,
        label=f"Group noise (σ̄={mean_sigma:.1f})",
    )

    ax.set_xlim(0, 200)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("|Asymmetry Index| threshold", fontsize=14)
    ax.set_ylabel("Fraction of quality-masked voxels ≥ threshold", fontsize=14)
    ax.set_title(
        "Group-Level |AI| Survival Functions (all patients)",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.9)
    ax.grid(axis="both", alpha=0.3)

    # Annotate key percentile bands
    for pct in [90, 95, 99]:
        cutoffs = [np.percentile(d["abs_ai_quality"], pct) for d in all_data]
        mean_c = np.mean(cutoffs)
        ax.axvline(mean_c, color="gray", ls=":", lw=1, alpha=0.6)
        ax.text(mean_c + 1, 0.5, f"P{pct}\n≈{mean_c:.0f}", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {save_path.name}")


# ─── figure 3: noise-signal decomposition ──────────────────────────────
def plot_noise_signal_decomposition(all_data: list[dict], save_path: Path):
    """Show how noise and signal separate across patients."""
    fig, axes = plt.subplots(3, 5, figsize=(22, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, d in enumerate(all_data):
        ax = axes_flat[idx]
        bin_edges = np.linspace(0, 200, BINS + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Empirical histogram (density)
        counts_q, _ = np.histogram(d["abs_ai_quality"], bins=bin_edges, density=True)

        # Noise model
        sigma = fit_half_normal(d["abs_ai_quality"])
        noise_pdf = half_normal_pdf(bin_centers, sigma)

        # Signal = empirical - noise (clipped at 0)
        signal_pdf = np.maximum(counts_q - noise_pdf, 0)

        ax.fill_between(
            bin_centers, noise_pdf, alpha=0.4, color="cornflowerblue", label="Noise"
        )
        ax.fill_between(
            bin_centers,
            noise_pdf,
            noise_pdf + signal_pdf,
            alpha=0.5,
            color="tomato",
            label="Signal",
        )
        ax.plot(bin_centers, counts_q, "k-", lw=1, alpha=0.6)

        # TFCE line
        if len(d["abs_ai_tfce"]) > 0:
            tfce_min = d["abs_ai_tfce"].min()
            ax.axvline(tfce_min, color="limegreen", ls="-", lw=2)

        ax.set_title(d["pid"], fontsize=11, fontweight="bold")
        ax.set_xlim(0, 200)
        if idx == 0:
            ax.legend(fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("|AI|", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density", fontsize=10)

    fig.suptitle(
        "Noise vs Signal Decomposition of |AI|  (blue=half-normal noise, red=signal excess, green=TFCE cutoff)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {save_path.name}")


# ─── figure 4: threshold sensitivity ───────────────────────────────────
def plot_threshold_sensitivity(all_data: list[dict], save_path: Path):
    """How many voxels survive at different |AI| cutoffs, per patient."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(all_data)))
    cutoffs = np.arange(0, 201, 1)

    for d, clr in zip(all_data, cmap):
        vals = d["abs_ai_quality"]
        n_q = len(vals)
        frac = np.array([(vals >= c).sum() / n_q * 100 for c in cutoffs])
        ax1.plot(cutoffs, frac, color=clr, lw=1.5, alpha=0.7, label=d["pid"])

        # Also number of clusters at each cutoff (sampled)
        # (skip for speed — just the voxel fraction is enough)

    ax1.set_xlabel("|AI| cutoff", fontsize=13)
    ax1.set_ylabel("% quality-masked voxels retained", fontsize=13)
    ax1.set_title(
        "Threshold Sensitivity: Voxel Retention", fontsize=14, fontweight="bold"
    )
    ax1.axhline(3.0, color="green", ls="--", lw=2, label="~3% target (TFCE)")
    ax1.axhline(10.0, color="orange", ls="--", lw=1.5, label="10% (percentile P90)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 50)
    ax1.grid(alpha=0.3)

    # Right panel: zoom into 0-10% range
    for d, clr in zip(all_data, cmap):
        vals = d["abs_ai_quality"]
        n_q = len(vals)
        frac = np.array([(vals >= c).sum() / n_q * 100 for c in cutoffs])
        ax2.plot(cutoffs, frac, color=clr, lw=1.5, alpha=0.7, label=d["pid"])

    ax2.set_xlabel("|AI| cutoff", fontsize=13)
    ax2.set_ylabel("% quality-masked voxels retained", fontsize=13)
    ax2.set_title("Zoomed: 0–10% Retention Range", fontsize=14, fontweight="bold")
    ax2.axhline(3.0, color="green", ls="--", lw=2, label="~3% (TFCE)")
    ax2.axhline(5.0, color="orange", ls="--", lw=1.5, label="5%")
    ax2.axhline(1.0, color="red", ls="--", lw=1.5, label="1%")
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_xlim(50, 200)
    ax2.set_ylim(0, 10)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {save_path.name}")


# ─── figure 5: summary statistics table figure ─────────────────────────
def plot_summary_table(all_data: list[dict], save_path: Path):
    """Create a summary statistics figure comparing key numbers per patient."""
    import pandas as pd

    rows = []
    for d in all_data:
        sigma = fit_half_normal(d["abs_ai_quality"])
        tfce_min = d["abs_ai_tfce"].min() if len(d["abs_ai_tfce"]) > 0 else np.nan
        tfce_pct = 100 * d["n_tfce"] / d["n_quality"] if d["n_quality"] > 0 else 0

        row = {
            "Patient": d["pid"],
            "Brain voxels": d["n_brain"],
            "Quality-masked": d["n_quality"],
            "Noise σ": f"{sigma:.1f}",
            "P90": f"{np.percentile(d['abs_ai_quality'], 90):.0f}",
            "P95": f"{np.percentile(d['abs_ai_quality'], 95):.0f}",
            "P99": f"{np.percentile(d['abs_ai_quality'], 99):.0f}",
            "TFCE cutoff": f"{tfce_min:.0f}" if not np.isnan(tfce_min) else "N/A",
            "TFCE voxels": d["n_tfce"],
            "TFCE %": f"{tfce_pct:.1f}%",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "distribution_summary_statistics.csv", index=False)

    # Table figure
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(range(len(df.columns)))
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif key[0] % 2 == 0:
            cell.set_facecolor("#D9E2F3")

    ax.set_title(
        "|AI| Distribution Summary Statistics per Patient",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {save_path.name}")


# ─── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="|AI| Distribution Analysis")
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Patient IDs to include (default: all in results_voxel/)",
    )
    args = parser.parse_args()

    # Discover patients
    if args.patients:
        patients = args.patients
    else:
        patients = sorted(
            p.name
            for p in Path("results_voxel").iterdir()
            if p.is_dir() and p.name.startswith("P")
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("|AI| DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Load all
    all_data = []
    for pid in patients:
        d = load_patient_data(pid)
        if d is None:
            print(f"  ⚠️  {pid}: data not found, skipping")
            continue
        print(
            f"  ✅ {pid}: brain={d['n_brain']:,}  quality={d['n_quality']:,}  "
            f"TFCE={d['n_tfce']:,}"
        )
        all_data.append(d)

    if not all_data:
        print("❌ No patient data found.")
        sys.exit(1)

    # Generate per-patient figures
    print(f"\n📊 Generating per-patient distribution charts...")
    for d in all_data:
        plot_single_patient(d, OUT_DIR / f"{d['pid']}_ai_distribution.png")

    # Group-level figures
    print(f"\n📊 Generating group-level figures...")
    plot_group_cdf(all_data, OUT_DIR / "group_survival_functions.png")
    plot_noise_signal_decomposition(
        all_data, OUT_DIR / "noise_signal_decomposition.png"
    )
    plot_threshold_sensitivity(all_data, OUT_DIR / "threshold_sensitivity.png")
    plot_summary_table(all_data, OUT_DIR / "summary_statistics_table.png")

    print(f"\n✅ All figures saved to {OUT_DIR}/")
    print(f"   {len(all_data)} per-patient charts + 4 group-level figures")


if __name__ == "__main__":
    main()
