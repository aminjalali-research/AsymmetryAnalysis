#!/usr/bin/env python3
"""
Visualize zAI (Control-Referenced Asymmetry Z-Score) Results

Generates publication-quality figures from the zAI pipeline outputs in
results_zscore/asymmetry/, following the visual style of
visual_comparative_analysis.py. Outputs are numbered and saved to
visual_analysis/ alongside existing figures.

Inputs (produced by 02_compute_zai.py):
  - results_zscore/asymmetry/patients/{pid}/{pid}_asymmetry_zscore.nii.gz
  - results_zscore/asymmetry/patients/{pid}/{pid}_asymmetry_cluster_report.csv
  - results_zscore/asymmetry/groups/{group}/brain_mask.nii.gz
  - results_zscore/groups/{group}/mean_perfusion.nii.gz   (anatomical bg)
  - results_zscore/groups/{group}/consensus_parcellation.nii.gz

Each voxel's zAI value answers: "Is this spot's LEFT-vs-RIGHT perfusion
difference abnormal compared to healthy controls?" rather than the raw
"is this spot's blood flow abnormal?" question.

Sign convention:
  - zAI > 0: abnormally LEFT-dominant perfusion at this voxel
  - zAI < 0: abnormally RIGHT-dominant perfusion at this voxel

**2026-05-05 37-ROI clinical scope policy (DEFAULT):** Voxel-level figures
(Figure 10 zAI overview, Figure 13 top clusters, Figure 14 multipatient
axial) are restricted to the 37 clinically-relevant ROI pairs by default,
matching `03_clinical_maps.py` and Pipeline A. Use `--no-clinical-mask` to
fall back to the full-brain (gray-matter-included) view for diagnostic
exploration. The mask is loaded per-patient from
`Dataset/<PID>/aparc+aseg.nii.gz`.

Usage:
    python 04_publication_figures.py                    # 37-ROI mask (default)
    python 04_publication_figures.py --no-clinical-mask # disable 37-ROI mask
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Set publication-quality style (matching visual_comparative_analysis.py)
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results_zscore"
ASYMMETRY_DIR = RESULTS_DIR / "asymmetry"
GROUP_KEY = "FM_20_39"   # default band (F_20_39 retired; per-patient bands resolved elsewhere)
GROUP_DIR = RESULTS_DIR / "groups" / GROUP_KEY
ASYM_GROUP_DIR = ASYMMETRY_DIR / "groups" / GROUP_KEY
DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR = BASE_DIR / "visual_analysis"

# Symmetric-space mode (2026-06-23 zAI fix): zAI maps are in the symmetric
# template space, so the per-patient clinical-ROI mask must use the symmetric
# parcellation (symreg/sym_perf/<PID>_aparc_sym.nii.gz). Set via --sym; also
# honoured by clean_overlay.py, which reuses load_clinical_roi_mask.
SYM_MODE = False
SYM_DIR = BASE_DIR / "symreg" / "sym_perf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ZAI_THRESHOLD = 1.96

# ----------------------------------------------------------------------------
# 37-ROI clinical scope (CLAUDE.md / DECISIONS.md 2026-05-05 policy)
# 76 FreeSurfer labels = 37 paired regions:
#   - 3 subcortical pairs: Thalamus (10/49), Hippocampus (17/53), Amygdala (18/54)
#   - 34 cortical Desikan-Killiany pairs: 1001-1035 (L) + 2001-2035 (R)
# Basal ganglia (caudate, putamen, pallidum) are intentionally excluded.
# ----------------------------------------------------------------------------
CLINICAL_SUBCORTICAL_LABELS = [10, 49, 17, 53, 18, 54]
CLINICAL_CORTICAL_LABELS = list(range(1001, 1036)) + list(range(2001, 2036))
CLINICAL_ROI_LABELS = set(CLINICAL_SUBCORTICAL_LABELS) | set(CLINICAL_CORTICAL_LABELS)

CLINICAL_MASK_FOOTER = (
    "Restricted to 37 clinical ROI pairs (3 subcortical + 34 Desikan-Killiany "
    "cortical) per CLAUDE.md 2026-05-05 policy."
)

# -- Region name mappings (abbreviated for figure labels) --

CORTICAL_REGIONS = {
    1001: "lh-bankssts", 2001: "rh-bankssts",
    1002: "lh-caud.ant.cing", 2002: "rh-caud.ant.cing",
    1003: "lh-caud.mid.front", 2003: "rh-caud.mid.front",
    1005: "lh-cuneus", 2005: "rh-cuneus",
    1006: "lh-entorhinal", 2006: "rh-entorhinal",
    1007: "lh-fusiform", 2007: "rh-fusiform",
    1008: "lh-inf.parietal", 2008: "rh-inf.parietal",
    1009: "lh-inf.temporal", 2009: "rh-inf.temporal",
    1010: "lh-isthmus.cing", 2010: "rh-isthmus.cing",
    1011: "lh-lat.occipital", 2011: "rh-lat.occipital",
    1012: "lh-lat.orbitofr", 2012: "rh-lat.orbitofr",
    1013: "lh-lingual", 2013: "rh-lingual",
    1014: "lh-med.orbitofr", 2014: "rh-med.orbitofr",
    1015: "lh-mid.temporal", 2015: "rh-mid.temporal",
    1016: "lh-parahippo", 2016: "rh-parahippo",
    1017: "lh-paracentral", 2017: "rh-paracentral",
    1018: "lh-parsoperc", 2018: "rh-parsoperc",
    1019: "lh-parsorbit", 2019: "rh-parsorbit",
    1020: "lh-parstriang", 2020: "rh-parstriang",
    1021: "lh-pericalcarine", 2021: "rh-pericalcarine",
    1022: "lh-postcentral", 2022: "rh-postcentral",
    1023: "lh-post.cing", 2023: "rh-post.cing",
    1024: "lh-precentral", 2024: "rh-precentral",
    1025: "lh-precuneus", 2025: "rh-precuneus",
    1026: "lh-rost.ant.cing", 2026: "rh-rost.ant.cing",
    1027: "lh-rost.mid.front", 2027: "rh-rost.mid.front",
    1028: "lh-sup.frontal", 2028: "rh-sup.frontal",
    1029: "lh-sup.parietal", 2029: "rh-sup.parietal",
    1030: "lh-sup.temporal", 2030: "rh-sup.temporal",
    1031: "lh-supramarginal", 2031: "rh-supramarginal",
    1032: "lh-frontalpole", 2032: "rh-frontalpole",
    1033: "lh-temporalpole", 2033: "rh-temporalpole",
    1034: "lh-transv.temp", 2034: "rh-transv.temp",
    1035: "lh-insula", 2035: "rh-insula",
}

SUBCORTICAL_REGIONS = {
    10: "lh-Thalamus", 49: "rh-Thalamus",
    17: "lh-Hippocampus", 53: "rh-Hippocampus",
    18: "lh-Amygdala", 54: "rh-Amygdala",
    11: "lh-Caudate", 50: "rh-Caudate",
    12: "lh-Putamen", 51: "rh-Putamen",
    13: "lh-Pallidum", 52: "rh-Pallidum",
}

ALL_REGIONS = {**CORTICAL_REGIONS, **SUBCORTICAL_REGIONS}


# ============================================================================
# DATA HELPERS
# ============================================================================

def discover_patients():
    """Find all patients with zAI maps produced by 02_compute_zai.py."""
    patients_dir = ASYMMETRY_DIR / "patients"
    if not patients_dir.exists():
        return []
    return sorted([
        d.name for d in patients_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}_asymmetry_zscore.nii.gz").exists()
    ])


def load_clinical_roi_mask(patient_id, ref_shape=None):
    """Load the per-patient 37-ROI clinical mask from Dataset/<PID>/aparc+aseg.nii.gz.

    Returns a boolean numpy array, or None if unavailable.
    Per CLAUDE.md / DECISIONS.md 2026-05-05 policy: 76 FreeSurfer labels =
    37 paired regions (Thalamus/Hippocampus/Amygdala + 34 Desikan-Killiany
    cortical).
    """
    if SYM_MODE:
        parc_path = SYM_DIR / f"{patient_id}_aparc_sym.nii.gz"
    else:
        parc_path = DATASET_DIR / patient_id / "aparc+aseg.nii.gz"
    if not parc_path.exists():
        return None
    parc = nib.load(str(parc_path)).get_fdata().astype(np.int32)
    if ref_shape is not None and parc.shape != ref_shape:
        return None
    mask = np.zeros(parc.shape, dtype=bool)
    for label in CLINICAL_ROI_LABELS:
        mask |= (parc == label)
    return mask


def load_zai_map(patient_id):
    """Load the patient's zAI (control-referenced asymmetry z-score) map."""
    path = (ASYMMETRY_DIR / "patients" / patient_id /
            f"{patient_id}_asymmetry_zscore.nii.gz")
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def load_group_data():
    """Load the anatomical background, brain mask, and parcellation.

    Background uses mean_perfusion.nii.gz (raw-perfusion control mean) as a
    visually informative anatomical reference; the asymmetry-pipeline brain
    mask (from results_zscore/asymmetry/groups/) is preferred for masking
    zAI maps when available.
    """
    mean_img = nib.load(str(GROUP_DIR / "mean_perfusion.nii.gz")).get_fdata(dtype=np.float32)
    asym_mask_path = ASYM_GROUP_DIR / "brain_mask.nii.gz"
    if asym_mask_path.exists():
        mask = nib.load(str(asym_mask_path)).get_fdata().astype(bool)
    else:
        mask = nib.load(str(GROUP_DIR / "brain_mask.nii.gz")).get_fdata().astype(bool)
    parc = nib.load(str(GROUP_DIR / "consensus_parcellation.nii.gz")).get_fdata().astype(np.int32)
    return mean_img, mask, parc


def load_patient_demographics():
    demos = {}
    spreadsheet = BASE_DIR / "clinical_spreadsheet.xlsx"
    if spreadsheet.exists():
        try:
            df = pd.read_excel(str(spreadsheet))
            df.columns = [c.strip().upper() for c in df.columns]
            id_col = [c for c in df.columns if "ID" in c][0]
            age_col = [c for c in df.columns if "AGE" in c][0]
            sex_col = [c for c in df.columns if "SEX" in c or "GENDER" in c][0]
            for _, row in df.iterrows():
                pid = str(row[id_col]).strip()
                if pid.startswith("sub-"):
                    pid = pid[4:]
                demos[pid] = f"{pid} ({row[sex_col]},{int(row[age_col])})"
        except Exception:
            pass
    return demos


def compute_region_matrix(patients, mask, parc, regions_dict):
    """Compute mean zAI per region per patient → (n_patients, n_regions) matrix."""
    region_labels = sorted(regions_dict.keys())
    region_names = [regions_dict[r] for r in region_labels]
    data = np.full((len(patients), len(region_labels)), np.nan)
    for i, pid in enumerate(patients):
        zai_map = load_zai_map(pid)
        valid = mask & (zai_map != 0)
        for j, rl in enumerate(region_labels):
            rmask = (parc == rl) & valid
            if rmask.sum() > 10:
                data[i, j] = zai_map[rmask].mean()
    return data, region_labels, region_names


# ============================================================================
# FIGURE 10: zAI DISTRIBUTION OVERVIEW
# ============================================================================

class ZAIVisualAnalyzer:
    """Publication-quality visualizations for zAI control-referenced asymmetry results.

    Parameters
    ----------
    apply_clinical_mask : bool
        When True (default), voxel-level figures (Figure 10, 13, 14) are
        restricted to the 37 clinical ROI pairs per CLAUDE.md /
        DECISIONS.md 2026-05-05 policy. When False, the full gray-matter
        scope is used (legacy behavior).
    """

    def __init__(self, apply_clinical_mask=True):
        self.patients = discover_patients()
        self.demos = load_patient_demographics()
        self.mean_img, self.mask, self.parc = load_group_data()
        self.apply_clinical_mask = apply_clinical_mask
        # Per-patient 37-ROI masks, lazily cached
        self._clinical_masks = {}
        # Audit which patients had a clinical mask successfully loaded
        self._clinical_mask_status = {}

    def _label(self, pid):
        return self.demos.get(pid, pid)

    def _patient_clinical_mask(self, pid, ref_shape):
        """Return per-patient 37-ROI mask (cached). None if unavailable."""
        if pid in self._clinical_masks:
            return self._clinical_masks[pid]
        m = load_clinical_roi_mask(pid, ref_shape=ref_shape)
        self._clinical_masks[pid] = m
        self._clinical_mask_status[pid] = (m is not None)
        if m is None:
            print(f"   ! No 37-ROI clinical mask available for {pid} "
                  f"(falling back to gray-matter scope for this patient).")
        return m

    def _effective_mask(self, pid, zai_map):
        """Combine the group brain mask with the per-patient 37-ROI mask
        (when --no-clinical-mask is unset and the mask is available).
        """
        if not self.apply_clinical_mask:
            return self.mask
        clin = self._patient_clinical_mask(pid, ref_shape=zai_map.shape)
        if clin is None:
            return self.mask
        return self.mask & clin

    def _add_clinical_footer(self, fig, y=0.005):
        """Add the 37-ROI policy footer to a figure (only if mask is on)."""
        if self.apply_clinical_mask:
            fig.text(0.5, y, CLINICAL_MASK_FOOTER, ha="center", va="bottom",
                     fontsize=8, style="italic", color="dimgray")

    # ------------------------------------------------------------------
    def plot_zai_overview(self):
        """Figure 10: zAI distribution dashboard (violin + classification)."""
        print("\n📊 Generating zAI overview dashboard...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        scope_suffix = " (37-ROI clinical scope)" if self.apply_clinical_mask else ""
        fig.suptitle(
            f"Control-Referenced Asymmetry zAI Overview (vs {GROUP_KEY})"
            f"{scope_suffix}",
            fontsize=16, fontweight="bold",
        )

        all_z = []
        patient_labels = []
        left_pcts = []
        right_pcts = []
        means = []
        stds = []

        for pid in self.patients:
            zai_map = load_zai_map(pid)
            eff_mask = self._effective_mask(pid, zai_map)
            valid = eff_mask & (zai_map != 0)
            z = zai_map[valid]

            rng = np.random.default_rng(42)
            idx = rng.choice(len(z), size=min(50000, len(z)), replace=False)
            all_z.append(z[idx])
            patient_labels.append(self._label(pid))

            n = len(z)
            left_pcts.append(100 * (z > ZAI_THRESHOLD).sum() / n)
            right_pcts.append(100 * (z < -ZAI_THRESHOLD).sum() / n)
            means.append(z.mean())
            stds.append(z.std())

        # -- Panel 1: Violin plot --
        ax1 = axes[0, 0]
        parts = ax1.violinplot(all_z, positions=range(len(self.patients)),
                               showmeans=False, showmedians=False, showextrema=False)
        palette = sns.color_palette("husl", len(self.patients))
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(palette[i])
            pc.set_alpha(0.6)
        bp = ax1.boxplot(all_z, positions=range(len(self.patients)),
                         widths=0.15, showfliers=False, patch_artist=True,
                         medianprops=dict(color="black", linewidth=1.5),
                         whiskerprops=dict(color="gray"), capprops=dict(color="gray"))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(palette[i])
            patch.set_alpha(0.8)
        ax1.axhline(ZAI_THRESHOLD, color="red", linestyle="--", alpha=0.5)
        ax1.axhline(-ZAI_THRESHOLD, color="blue", linestyle="--", alpha=0.5)
        ax1.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax1.set_xticks(range(len(self.patients)))
        ax1.set_xticklabels(patient_labels, fontsize=9)
        ax1.set_ylabel("zAI")
        ax1.set_title("Voxel-wise zAI Distributions")
        ax1.set_ylim(-10, 15)
        ax1.grid(True, alpha=0.3)

        # -- Panel 2: Mean zAI per patient --
        ax2 = axes[0, 1]
        colors = ["red" if m > 0 else "blue" for m in means]
        ax2.bar(patient_labels, means, color=colors, alpha=0.7, edgecolor="black")
        ax2.axhline(0, color="black", linestyle="-", linewidth=1)
        ax2.set_ylabel("Mean zAI")
        ax2.set_title("Global Asymmetry Shift\n(+ left-dominant, - right-dominant relative to controls)")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # -- Panel 3: Stacked classification bar --
        ax3 = axes[1, 0]
        normal_pcts = [100 - h - l for h, l in zip(left_pcts, right_pcts)]
        x = np.arange(len(self.patients))
        ax3.bar(x, normal_pcts, color="lightgray", label="Normal", edgecolor="white")
        ax3.bar(x, left_pcts, bottom=normal_pcts, color="indianred",
                label=f"Left-dominant (zAI>{ZAI_THRESHOLD})", edgecolor="white")
        bottoms = [n + h for n, h in zip(normal_pcts, left_pcts)]
        ax3.bar(x, right_pcts, bottom=bottoms, color="steelblue",
                label=f"Right-dominant (zAI<-{ZAI_THRESHOLD})", edgecolor="white")
        ax3.set_xticks(x)
        ax3.set_xticklabels(patient_labels, fontsize=9)
        ax3.set_ylabel("% of Brain Voxels")
        ax3.set_title("Voxel Classification per Patient")
        ax3.legend(fontsize=8, loc="lower right")
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3, axis="y")
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        # -- Panel 4: Variability (Std) vs Mean --
        ax4 = axes[1, 1]
        sig_counts = [h + l for h, l in zip(left_pcts, right_pcts)]
        scatter = ax4.scatter(means, stds, s=[s * 10 for s in sig_counts],
                              c=sig_counts, cmap="YlOrRd", alpha=0.7,
                              edgecolors="black", linewidth=1.5)
        for i, pid in enumerate(self.patients):
            ax4.annotate(self._label(pid), (means[i], stds[i]),
                         fontsize=8, ha="center", fontweight="bold")
        ax4.set_xlabel("Mean zAI (global asymmetry shift)")
        ax4.set_ylabel("Std Dev of zAI (heterogeneity)")
        ax4.set_title("Asymmetry Shift vs Heterogeneity\n(Size = % significant voxels)")
        ax4.axvline(0, color="gray", linestyle="-", linewidth=0.5)
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label("% Significant", rotation=270, labelpad=15)

        plt.tight_layout()
        self._add_clinical_footer(fig)
        output_file = OUTPUT_DIR / "10_zai_control_overview.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {output_file}")
        plt.close()

    # ------------------------------------------------------------------
    def plot_region_heatmap(self):
        """Figure 11: Regional zAI heatmap, patients x brain regions."""
        print("\n📊 Generating regional zAI heatmap...")

        data, region_labels, region_names = compute_region_matrix(
            self.patients, self.mask, self.parc, ALL_REGIONS)

        # Split into lh and rh
        lh_idx = [j for j, r in enumerate(region_labels)
                  if ALL_REGIONS[r].startswith("lh")]
        rh_idx = [j for j, r in enumerate(region_labels)
                  if ALL_REGIONS[r].startswith("rh")]
        order = lh_idx + rh_idx
        data_sorted = data[:, order]
        names_sorted = [region_names[j] for j in order]

        fig, ax = plt.subplots(figsize=(22, 8))

        norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
        im = ax.imshow(data_sorted, cmap="RdBu_r", norm=norm, aspect="auto",
                       interpolation="nearest")

        patient_labels = [self._label(pid) for pid in self.patients]
        ax.set_yticks(range(len(self.patients)))
        ax.set_yticklabels(patient_labels, fontsize=10)
        ax.set_xticks(range(len(names_sorted)))
        ax.set_xticklabels(names_sorted, rotation=90, fontsize=7, ha="center")

        # Hemisphere divider
        divider = len(lh_idx) - 0.5
        ax.axvline(divider, color="black", linewidth=2)
        ax.text(divider / 2, -1.5, "Left Hemisphere",
                ha="center", fontsize=11, fontweight="bold")
        ax.text(divider + len(rh_idx) / 2, -1.5, "Right Hemisphere",
                ha="center", fontsize=11, fontweight="bold")

        # Grid
        ax.set_xticks(np.arange(len(names_sorted)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(self.patients)) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3, alpha=0.2)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean zAI", rotation=270, labelpad=20, fontsize=11)

        ax.set_xlabel("Brain Region", fontsize=12)
        ax.set_ylabel("Patient ID", fontsize=12)
        ax.set_title(
            f"Regional Asymmetry zAI vs {GROUP_KEY} Controls\n"
            f"(Red = abnormally LEFT-dominant, Blue = abnormally RIGHT-dominant)",
            fontsize=14, fontweight="bold", pad=20,
        )

        plt.tight_layout()
        output_file = OUTPUT_DIR / "11_zai_regional_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {output_file}")
        plt.close()

    # ------------------------------------------------------------------
    def plot_subcortical_focus(self):
        """Figure 12: Subcortical structure zAI (clinical epilepsy focus)."""
        print("\n📊 Generating subcortical focus plot...")

        data, region_labels, region_names = compute_region_matrix(
            self.patients, self.mask, self.parc, SUBCORTICAL_REGIONS)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                 gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle(
            f"Subcortical Asymmetry zAI vs {GROUP_KEY} Controls\n"
            f"(Dashed lines = significance threshold |zAI| = {ZAI_THRESHOLD})",
            fontsize=14, fontweight="bold",
        )

        # -- Panel 1: Grouped bar chart --
        ax = axes[0]
        x = np.arange(len(region_names))
        width = 0.14
        offsets = np.linspace(-width * (len(self.patients) - 1) / 2,
                              width * (len(self.patients) - 1) / 2, len(self.patients))
        palette = sns.color_palette("husl", len(self.patients))

        for i, pid in enumerate(self.patients):
            vals = np.nan_to_num(data[i])
            colors = []
            for v in vals:
                if v > ZAI_THRESHOLD:
                    colors.append("indianred")
                elif v < -ZAI_THRESHOLD:
                    colors.append("steelblue")
                else:
                    colors.append(palette[i])
            ax.bar(x + offsets[i], vals, width, color=colors,
                   edgecolor="white", linewidth=0.5, label=self._label(pid))

        ax.axhline(ZAI_THRESHOLD, color="red", linestyle="--", alpha=0.4)
        ax.axhline(-ZAI_THRESHOLD, color="blue", linestyle="--", alpha=0.4)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(region_names, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Mean zAI", fontsize=11)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3, axis="y")

        # -- Panel 2: Heatmap version --
        ax2 = axes[1]
        norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
        im = ax2.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto",
                        interpolation="nearest")
        ax2.set_yticks(range(len(self.patients)))
        ax2.set_yticklabels([self._label(pid) for pid in self.patients], fontsize=10)
        ax2.set_xticks(range(len(region_names)))
        ax2.set_xticklabels(region_names, rotation=45, ha="right", fontsize=9)

        # Annotate cells with values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    color = "white" if abs(data[i, j]) > 2 else "black"
                    ax2.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                             fontsize=8, color=color, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Mean zAI", rotation=270, labelpad=15)

        plt.tight_layout()
        output_file = OUTPUT_DIR / "12_zai_subcortical_focus.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {output_file}")
        plt.close()

    # ------------------------------------------------------------------
    def plot_top_clusters(self):
        """Figure 13: Top significant cortical zAI clusters per patient.

        When `apply_clinical_mask` is True (default), clusters are filtered
        to those whose `primary_region` falls within the 37-ROI clinical
        scope (CLAUDE.md / DECISIONS.md 2026-05-05 policy).
        """
        print("\n📊 Generating top clusters summary...")

        # Build the set of region-name keywords matching the 37-ROI scope.
        # Cluster reports use REGION_NAMES strings produced upstream. We use
        # substring keyword matching to handle naming variants from different
        # producers (e.g. "lh-Hippocampus", "L-superiortemporal", etc.).
        clinical_keywords = {
            "thalamus", "hippocampus", "amygdala",
            "bankssts", "caud.ant.cing", "caud.mid.front", "cuneus",
            "entorhinal", "fusiform", "inf.parietal", "inf.temporal",
            "isthmus.cing", "lat.occipital", "lat.orbitofr", "lingual",
            "med.orbitofr", "mid.temporal", "parahippo", "paracentral",
            "parsoperc", "parsorbit", "parstriang", "pericalcarine",
            "postcentral", "post.cing", "precentral", "precuneus",
            "rost.ant.cing", "rost.mid.front", "sup.frontal", "sup.parietal",
            "sup.temporal", "supramarginal", "frontalpole", "temporalpole",
            "transv.temp", "insula",
            # Alternate naming variants
            "superiortemporal", "middletemporal", "inferiortemporal",
            "superiorfrontal", "middlefrontal", "inferiorparietal",
            "superiorparietal", "lateraloccipital", "lateralorbitofrontal",
            "medialorbitofrontal", "rostralanteriorcingulate",
            "rostralmiddlefrontal", "caudalanteriorcingulate",
            "caudalmiddlefrontal", "isthmuscingulate", "posteriorcingulate",
            "parahippocampal", "parsopercularis", "parsorbitalis",
            "parstriangularis", "transversetemporal",
        }

        def _in_clinical_scope(region_name):
            if not isinstance(region_name, str):
                return False
            name_low = region_name.lower()
            return any(kw in name_low for kw in clinical_keywords)

        top_n = 15
        fig, axes = plt.subplots(len(self.patients), 1,
                                 figsize=(14, 3.5 * len(self.patients)), sharex=False)
        if len(self.patients) == 1:
            axes = [axes]

        scope_suffix = " - 37-ROI clinical scope" if self.apply_clinical_mask else ""
        fig.suptitle(
            f"Top {top_n} Cortical zAI Clusters per Patient (excl. White Matter)"
            f"{scope_suffix}\n"
            f"(vs {GROUP_KEY} controls, |zAI| > {ZAI_THRESHOLD})",
            fontsize=14, fontweight="bold", y=1.01,
        )

        for idx, pid in enumerate(self.patients):
            ax = axes[idx]
            report_path = (ASYMMETRY_DIR / "patients" / pid /
                           f"{pid}_asymmetry_cluster_report.csv")
            if not report_path.exists():
                ax.text(0.5, 0.5, f"No cluster report for {pid}",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            df = pd.read_csv(str(report_path))
            df_cortical = df[
                ~df["primary_region"].str.contains(
                    "White-Matter|Unknown|Ventricle", na=False)
            ]

            # Apply 37-ROI scope filter when enabled
            if self.apply_clinical_mask:
                df_cortical = df_cortical[
                    df_cortical["primary_region"].apply(_in_clinical_scope)
                ]

            top_left = df_cortical[df_cortical["direction"] == "left-dominant"].nlargest(
                top_n, "size_voxels")
            top_right = df_cortical[df_cortical["direction"] == "right-dominant"].nlargest(
                top_n, "size_voxels")
            combined = pd.concat([top_left, top_right]).sort_values(
                "size_voxels", ascending=True)

            if combined.empty:
                msg = (f"No cortical clusters for {pid} in 37-ROI scope"
                       if self.apply_clinical_mask
                       else f"No cortical clusters for {pid}")
                ax.text(0.5, 0.5, msg,
                        ha="center", va="center", transform=ax.transAxes)
                continue

            colors = ["indianred" if d == "left-dominant" else "steelblue"
                      for d in combined["direction"]]
            y_pos = range(len(combined))
            ax.barh(y_pos, combined["size_mm3"], color=colors, alpha=0.7,
                    edgecolor="black", linewidth=0.3)

            for bar_idx, (_, row) in enumerate(combined.iterrows()):
                ax.text(combined["size_mm3"].max() * 0.01, bar_idx,
                        f"  {row['primary_region']}  (peak zAI={row['peak_z']:.1f})",
                        va="center", fontsize=8)

            ax.set_title(f"{self._label(pid)}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Cluster Size (mm\u00b3)", fontsize=10)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis="x")

            if idx == 0:
                ax.legend(handles=[
                    Patch(facecolor="indianred", label="LEFT-dominant (abnormal)"),
                    Patch(facecolor="steelblue", label="RIGHT-dominant (abnormal)"),
                ], loc="lower right", fontsize=9)

        plt.tight_layout()
        self._add_clinical_footer(fig)
        output_file = OUTPUT_DIR / "13_zai_top_clusters.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {output_file}")
        plt.close()

    # ------------------------------------------------------------------
    def plot_multipatient_axial(self):
        """Figure 14: Side-by-side axial zAI comparison across patients.

        When `apply_clinical_mask` is True (default), each patient's
        overlay is restricted to that patient's 37-ROI clinical mask.
        """
        print("\n📊 Generating multi-patient axial comparison...")

        z_indices = np.where(self.mask.any(axis=(0, 1)))[0]
        n_slices = 6
        slice_positions = np.linspace(
            z_indices[0] + 10, z_indices[-1] - 10, n_slices, dtype=int)

        fig, axes = plt.subplots(len(self.patients), n_slices,
                                 figsize=(3 * n_slices, 3.5 * len(self.patients)))

        scope_suffix = " - 37-ROI clinical scope" if self.apply_clinical_mask else ""
        fig.suptitle(
            f"Axial zAI Maps vs {GROUP_KEY} Controls{scope_suffix}\n"
            f"Red = abnormally LEFT-dominant (zAI>{ZAI_THRESHOLD}), "
            f"Blue = abnormally RIGHT-dominant (zAI<-{ZAI_THRESHOLD})",
            fontsize=14, fontweight="bold", y=1.01,
        )

        vmax_bg = np.percentile(self.mean_img[self.mask], 95)

        for row, pid in enumerate(self.patients):
            zai_map = load_zai_map(pid)
            eff_mask = self._effective_mask(pid, zai_map)

            for col, sl in enumerate(slice_positions):
                ax = axes[row, col] if len(self.patients) > 1 else axes[col]
                bg = self.mean_img[:, :, sl].T
                ax.imshow(bg, cmap="gray", origin="lower", aspect="equal",
                          vmin=0, vmax=vmax_bg)

                zslice = zai_map[:, :, sl].T
                mslice = eff_mask[:, :, sl].T
                sig = mslice & (np.abs(zslice) >= ZAI_THRESHOLD) & (zslice != 0)
                overlay = np.ma.masked_where(~sig, zslice)
                ax.imshow(overlay, cmap="RdBu_r", origin="lower", aspect="equal",
                          vmin=-6, vmax=6, alpha=0.75)

                ax.axis("off")
                if row == 0:
                    ax.set_title(f"z={sl}", fontsize=10, fontweight="bold")
                if col == 0:
                    ax.text(-5, bg.shape[0] / 2, self._label(pid), rotation=90,
                            va="center", ha="right", fontsize=10, fontweight="bold")

        plt.tight_layout()
        self._add_clinical_footer(fig)
        output_file = OUTPUT_DIR / "14_zai_multipatient_axial.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {output_file}")
        plt.close()

    # ------------------------------------------------------------------
    def plot_regional_consistency(self):
        """Figure 15: Which regions are consistently asymmetric across patients."""
        print("\n📊 Generating regional consistency analysis...")

        data, region_labels, region_names = compute_region_matrix(
            self.patients, self.mask, self.parc, ALL_REGIONS)

        region_stats = []
        for j, rname in enumerate(region_names):
            vals = data[:, j]
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) < 2:
                continue
            region_stats.append({
                "region": rname,
                "mean_z": np.mean(valid_vals),
                "std_z": np.std(valid_vals),
                "n_left": (valid_vals > ZAI_THRESHOLD).sum(),
                "n_right": (valid_vals < -ZAI_THRESHOLD).sum(),
                "n_significant": (np.abs(valid_vals) > ZAI_THRESHOLD).sum(),
            })

        rdf = pd.DataFrame(region_stats)
        rdf = rdf.sort_values("mean_z", key=abs, ascending=False)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Regional Asymmetry zAI Consistency vs {GROUP_KEY} Controls",
            fontsize=14, fontweight="bold",
        )

        # -- Top regions by mean zAI --
        ax1 = axes[0, 0]
        top = rdf.head(20)
        colors = ["red" if x > 0 else "blue" for x in top["mean_z"]]
        y_pos = np.arange(len(top))
        ax1.barh(y_pos, top["mean_z"], color=colors, alpha=0.7, edgecolor="black",
                 linewidth=0.3)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top["region"], fontsize=9)
        ax1.set_xlabel("Mean zAI", fontsize=11)
        ax1.set_title("Top 20 Regions by Average Asymmetry Deviation\n"
                      "(Red=LEFT-dominant, Blue=RIGHT-dominant)")
        ax1.axvline(0, color="black", linestyle="-", linewidth=1)
        ax1.grid(True, alpha=0.3, axis="x")

        # -- Most consistently abnormal --
        ax2 = axes[0, 1]
        consistent = rdf.nlargest(20, "n_significant")
        x2 = np.arange(len(consistent))
        width = 0.35
        ax2.barh(x2 - width / 2, consistent["n_left"], width,
                 label="LEFT-dominant", color="indianred", alpha=0.7)
        ax2.barh(x2 + width / 2, consistent["n_right"], width,
                 label="RIGHT-dominant", color="steelblue", alpha=0.7)
        ax2.set_yticks(x2)
        ax2.set_yticklabels(consistent["region"], fontsize=9)
        ax2.set_xlabel("Number of Patients", fontsize=11)
        ax2.set_title(f"Most Consistently Asymmetric Regions\n"
                      f"(# patients with |zAI|>{ZAI_THRESHOLD})")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="x")

        # -- Variability vs mean --
        ax3 = axes[1, 0]
        scatter = ax3.scatter(rdf["mean_z"], rdf["std_z"],
                              s=rdf["n_significant"] * 40 + 10,
                              c=rdf["n_significant"], cmap="YlOrRd",
                              alpha=0.6, edgecolors="black")
        ax3.set_xlabel("Mean zAI", fontsize=11)
        ax3.set_ylabel("Std Dev of zAI", fontsize=11)
        ax3.set_title(f"Regional Asymmetry: Mean vs Variability\n"
                      f"(Size = N patients |zAI|>{ZAI_THRESHOLD})")
        ax3.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax3.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label("N Significant", rotation=270, labelpad=15)

        # -- Distribution of regional means --
        ax4 = axes[1, 1]
        ax4.hist(rdf["mean_z"], bins=30, color="steelblue", alpha=0.7,
                 edgecolor="black")
        ax4.axvline(rdf["mean_z"].mean(), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {rdf['mean_z'].mean():.3f}")
        ax4.axvline(0, color="black", linestyle="-", linewidth=1)
        ax4.set_xlabel("Mean zAI", fontsize=11)
        ax4.set_ylabel("Number of Regions", fontsize=11)
        ax4.set_title("Distribution of Regional Mean zAI")
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_file = OUTPUT_DIR / "15_zai_regional_consistency.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {output_file}")
        plt.close()

    # ------------------------------------------------------------------
    def plot_patient_clustering(self):
        """Figure 16: Hierarchical clustering of patients by zAI patterns."""
        print("\n📊 Generating patient clustering analysis...")

        data, region_labels, region_names = compute_region_matrix(
            self.patients, self.mask, self.parc, ALL_REGIONS)
        matrix = np.nan_to_num(data, nan=0.0)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            "Patient Clustering Based on Asymmetry zAI Patterns",
            fontsize=14, fontweight="bold",
        )

        # -- Dendrogram --
        ax1 = axes[0]
        patient_labels = [self._label(pid) for pid in self.patients]
        linkage = hierarchy.linkage(matrix, method="ward")
        dendro = hierarchy.dendrogram(linkage, labels=patient_labels,
                                      ax=ax1, leaf_font_size=10, color_threshold=0)
        ax1.set_xlabel("Patient ID", fontsize=11)
        ax1.set_ylabel("Distance", fontsize=11)
        ax1.set_title("Hierarchical Clustering Dendrogram\n(Ward Linkage)", fontsize=12)
        ax1.grid(True, alpha=0.3, axis="y")

        # -- Clustered heatmap --
        ax2 = axes[1]
        idx = dendro["leaves"]
        matrix_sorted = matrix[idx, :]
        norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
        im = ax2.imshow(matrix_sorted, cmap="RdBu_r", norm=norm,
                        aspect="auto", interpolation="nearest")
        ax2.set_yticks(range(len(self.patients)))
        ax2.set_yticklabels([patient_labels[i] for i in idx], fontsize=10)
        ax2.set_xlabel("Brain Regions", fontsize=11)
        ax2.set_ylabel("Patient ID (Clustered)", fontsize=11)
        ax2.set_title("Clustered Asymmetry Deviation Patterns", fontsize=12)
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Mean zAI")

        plt.tight_layout()
        output_file = OUTPUT_DIR / "16_zai_patient_clustering.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {output_file}")
        plt.close()

    # ------------------------------------------------------------------
    def run_all(self):
        """Generate all zAI visualizations."""
        print("\n" + "=" * 60)
        print("📊 zAI CONTROL-REFERENCED ASYMMETRY — VISUALIZATION SUITE")
        print("=" * 60)
        print(f"   Patients: {self.patients}")
        print(f"   Group: {GROUP_KEY}")
        print(f"   Output: {OUTPUT_DIR}")

        self.plot_zai_overview()              # 10
        self.plot_region_heatmap()            # 11
        self.plot_subcortical_focus()         # 12
        self.plot_top_clusters()              # 13
        self.plot_multipatient_axial()        # 14
        self.plot_regional_consistency()      # 15
        self.plot_patient_clustering()        # 16

        print(f"\n✅ All zAI visualizations saved to: {OUTPUT_DIR}")
        print("=" * 60 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate publication figures for zAI results "
                    "(default: 37-ROI clinical mask).")
    parser.add_argument(
        "--no-clinical-mask", dest="apply_clinical_mask",
        action="store_false", default=True,
        help="Disable the 37-ROI clinical mask. Falls back to the full "
             "gray-matter scope (legacy behavior). "
             "By default, voxel-level figures (10, 13, 14) are restricted "
             "to the 37-ROI clinical scope per CLAUDE.md / DECISIONS.md "
             "2026-05-05 policy.")
    parser.add_argument("--sym", action="store_true",
                        help="zAI maps are in symmetric-template space; use the "
                             "symmetric per-patient parcellation for the clinical ROI mask.")
    args = parser.parse_args()

    if args.sym:
        SYM_MODE = True
        print("→ [SYM] symmetric-space parcellation for clinical ROI mask.")

    if args.apply_clinical_mask:
        print("→ 37-ROI clinical mask: ON (default).")
        print("  Voxel-level figures (10, 13, 14) restricted to the 37 "
              "clinical ROI pairs.")
    else:
        print("→ 37-ROI clinical mask: OFF (--no-clinical-mask).")

    analyzer = ZAIVisualAnalyzer(apply_clinical_mask=args.apply_clinical_mask)
    analyzer.run_all()
