#!/usr/bin/env python3
"""
Control-Referenced Asymmetry Z-Score Analysis

Compares each patient's LEFT-vs-RIGHT perfusion asymmetry against the
normal asymmetry pattern of healthy controls.

The key difference from 01_build_control_normative.py:
  - 01_build_control_normative.py: z = (patient_perfusion - mean_perfusion) / sd
    → "Is this spot's blood flow abnormal?" (scattered results everywhere)

  - THIS script: z = (patient_AI - mean_AI) / sd_AI
    → "Is this spot's LEFT-RIGHT DIFFERENCE abnormal?" (lateralized results)

Where AI = 100 * (Left - Right) / ((Left + Right) / 2)

This directly answers: "Where does this patient have abnormal asymmetry
compared to healthy people?" — which is what matters for presurgical
epilepsy lateralization.

Result interpretation:
  - z > 0: Patient has abnormally LEFT-dominant perfusion (left higher than right)
  - z < 0: Patient has abnormally RIGHT-dominant perfusion (right higher than left)
  - z ≈ 0: Patient's asymmetry is normal at this location

Usage:
    python 02_compute_zai.py
    python 02_compute_zai.py --patient P013
    python 02_compute_zai.py --threshold 3.0 --min-cluster 50
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage

# Thresholding methods + helpers are now centralised in src/thresholding.py
# (extracted from control_zscore_asymmetry_clustering.py during the
# 2026-05-04 cleanup phase).  We reuse the connectivity constant and the
# 3-D structuring-element helper here; the local cluster_filter() below
# retains its own implementation because it returns per-cluster metadata
# (id/size/mask) that the registry version does not provide.
from src.thresholding import (
    METHOD_REGISTRY,
    CLUSTER_CONNECTIVITY,
    get_structure_3d,
)

assert len(METHOD_REGISTRY) == 13, "thresholding registry should expose 13 methods"

# ============================================================================
# CONFIGURATION
# ============================================================================

ZSCORE_THRESHOLD = 1.96      # |z| >= 1.96 (p < 0.05 two-tailed)
MIN_CLUSTER_SIZE = 50        # voxels
BRAIN_MASK_COVERAGE = 0.75
VOXEL_VOL_MM3 = 0.8 ** 3

BASE_DIR = Path(__file__).parent
CONTROLS_BASE_DIR = BASE_DIR / "DatasetControls"
PATIENT_DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR = BASE_DIR / "results_zscore" / "asymmetry"
GROUP_KEY = "FM_20_39"

# Symmetric-space mode (2026-06-22 zAI over-dispersion fix). When enabled, the
# mirror-AI is computed on perfusion registered to a left-right symmetric
# template (symreg/sym_perf/), so np.flip(axis=0) compares homologous voxels.
SYM_MODE = False
SYM_DIR = BASE_DIR / "symreg" / "sym_perf"

GROUP_DEFINITIONS = {
    "F_20_39": {"subdirs": ["20_29F", "30_39F"]},
    "F_40_59": {"subdirs": ["40_49F", "50_59F"]},
    "F_60_79": {"subdirs": ["60_69F", "70_79F"]},
    "M_20_39": {"subdirs": ["20_29M", "30_39M"]},
    "M_40_59": {"subdirs": ["40_49M", "50_59M"]},
    "M_60_79": {"subdirs": ["60_69M", "70_79M"]},
    # Thesis comparator groups: sex-pooled, stored flat in DatasetControls/FM_*/.
    "FM_20_39": {"subdirs": ["FM_20_39"]},
    "FM_40_59": {"subdirs": ["FM_40_59"]},
    "FM_60_79": {"subdirs": ["FM_60_79"]},
}

# Gray matter labels
CORTICAL_LABELS = set(range(1001, 1036)) | set(range(2001, 2036))
SUBCORTICAL_LABELS = {10, 11, 12, 13, 17, 18, 26, 28,
                      49, 50, 51, 52, 53, 54, 58, 60}
GRAY_MATTER_LABELS = CORTICAL_LABELS | SUBCORTICAL_LABELS

REGION_NAMES = {
    10: "L-Thalamus", 49: "R-Thalamus",
    17: "L-Hippocampus", 53: "R-Hippocampus",
    18: "L-Amygdala", 54: "R-Amygdala",
    11: "L-Caudate", 50: "R-Caudate",
    12: "L-Putamen", 51: "R-Putamen",
    13: "L-Pallidum", 52: "R-Pallidum",
    26: "L-Accumbens", 58: "R-Accumbens",
    28: "L-VentralDC", 60: "R-VentralDC",
    1001: "L-bankssts", 2001: "R-bankssts",
    1002: "L-caud.ant.cing", 2002: "R-caud.ant.cing",
    1003: "L-caud.mid.front", 2003: "R-caud.mid.front",
    1005: "L-cuneus", 2005: "R-cuneus",
    1006: "L-entorhinal", 2006: "R-entorhinal",
    1007: "L-fusiform", 2007: "R-fusiform",
    1008: "L-inf.parietal", 2008: "R-inf.parietal",
    1009: "L-inf.temporal", 2009: "R-inf.temporal",
    1010: "L-isthmus.cing", 2010: "R-isthmus.cing",
    1011: "L-lat.occipital", 2011: "R-lat.occipital",
    1012: "L-lat.orbitofr", 2012: "R-lat.orbitofr",
    1013: "L-lingual", 2013: "R-lingual",
    1014: "L-med.orbitofr", 2014: "R-med.orbitofr",
    1015: "L-mid.temporal", 2015: "R-mid.temporal",
    1016: "L-parahippo", 2016: "R-parahippo",
    1017: "L-paracentral", 2017: "R-paracentral",
    1018: "L-parsoperc", 2018: "R-parsoperc",
    1019: "L-parsorbit", 2019: "R-parsorbit",
    1020: "L-parstriang", 2020: "R-parstriang",
    1021: "L-pericalcarine", 2021: "R-pericalcarine",
    1022: "L-postcentral", 2022: "R-postcentral",
    1023: "L-post.cing", 2023: "R-post.cing",
    1024: "L-precentral", 2024: "R-precentral",
    1025: "L-precuneus", 2025: "R-precuneus",
    1026: "L-rost.ant.cing", 2026: "R-rost.ant.cing",
    1027: "L-rost.mid.front", 2027: "R-rost.mid.front",
    1028: "L-sup.frontal", 2028: "R-sup.frontal",
    1029: "L-sup.parietal", 2029: "R-sup.parietal",
    1030: "L-sup.temporal", 2030: "R-sup.temporal",
    1031: "L-supramarginal", 2031: "R-supramarginal",
    1032: "L-frontalpole", 2032: "R-frontalpole",
    1033: "L-temporalpole", 2033: "R-temporalpole",
    1034: "L-transv.temp", 2034: "R-transv.temp",
    1035: "L-insula", 2035: "R-insula",
}


# ============================================================================
# ASYMMETRY COMPUTATION
# ============================================================================

def compute_asymmetry_map(perfusion):
    """Compute voxel-wise asymmetry: AI = 100 * (L - R) / ((L + R) / 2).

    L is the original image. R is the left-right flipped version.
    The result is a full-brain map where each voxel represents
    how much the LEFT side dominates over the RIGHT at that location.
    """
    left = perfusion.astype(np.float64)
    right = np.flip(left, axis=0)  # mirror across midline

    mean_lr = (left + right) / 2.0
    valid = mean_lr > 0.001  # both sides must have signal

    ai = np.zeros_like(left)
    ai[valid] = 100.0 * (left[valid] - right[valid]) / mean_lr[valid]

    return ai, valid


# ============================================================================
# GROUP STATISTICS (Welford's on asymmetry maps)
# ============================================================================

def discover_subjects(group_key):
    """Find control subject directories."""
    subjects = []
    for subdir_name in GROUP_DEFINITIONS[group_key]["subdirs"]:
        subdir = CONTROLS_BASE_DIR / subdir_name
        if not subdir.exists():
            continue
        for entry in sorted(subdir.iterdir()):
            if entry.is_dir():
                subjects.append(entry)
    return subjects


def find_perfusion_file(subject_dir):
    """Find the upsampled MNI-space perfusion NIfTI."""
    if SYM_MODE:
        p = SYM_DIR / f"{subject_dir.name}_perf_sym.nii.gz"
        if not p.exists():
            raise FileNotFoundError(f"No symmetric perfusion for {subject_dir.name}: {p}")
        return p
    candidates = list(subject_dir.glob("*_MNISpace_perfusion_calib_upsampled.nii.gz"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Expected 1 perfusion file in {subject_dir}, found {len(candidates)}")


def build_group_asymmetry_stats(group_key, force_rebuild=False):
    """Build voxel-wise mean and SD of asymmetry across controls."""
    group_dir = OUTPUT_DIR / "groups" / group_key
    group_dir.mkdir(parents=True, exist_ok=True)
    meta_path = group_dir / "metadata.json"

    if meta_path.exists() and not force_rebuild:
        print(f"\n  Loading cached asymmetry stats for {group_key}...")
        mean_ai = nib.load(str(group_dir / "mean_asymmetry.nii.gz")).get_fdata(dtype=np.float32)
        sd_ai = nib.load(str(group_dir / "sd_asymmetry.nii.gz")).get_fdata(dtype=np.float32)
        mask = nib.load(str(group_dir / "brain_mask.nii.gz")).get_fdata().astype(bool)
        ref_img = nib.load(str(group_dir / "mean_asymmetry.nii.gz"))
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"    {meta['n_subjects']} subjects, {meta['brain_mask_voxels']:,} mask voxels")
        return mean_ai, sd_ai, mask, ref_img

    subjects = discover_subjects(group_key)
    n = len(subjects)
    print(f"\n{'='*70}")
    print(f"  Building ASYMMETRY group stats for {group_key} ({n} subjects)")
    print(f"{'='*70}")

    # Welford's online algorithm on asymmetry maps
    mean = None
    M2 = None
    count = None
    ref_img = None
    subject_ids = []

    for i, subj_dir in enumerate(subjects):
        subj_id = subj_dir.name
        subject_ids.append(subj_id)
        perf_path = find_perfusion_file(subj_dir)
        img = nib.load(str(perf_path))
        perfusion = img.get_fdata(dtype=np.float32)

        # Compute this subject's asymmetry map
        ai, valid = compute_asymmetry_map(perfusion)

        if mean is None:
            shape = ai.shape
            ref_img = img
            mean = np.zeros(shape, dtype=np.float64)
            M2 = np.zeros(shape, dtype=np.float64)
            count = np.zeros(shape, dtype=np.int32)

        count[valid] += 1
        delta = np.zeros(shape, dtype=np.float64)
        delta[valid] = ai[valid] - mean[valid]
        mean[valid] += delta[valid] / count[valid]
        delta2 = np.zeros(shape, dtype=np.float64)
        delta2[valid] = ai[valid] - mean[valid]
        M2[valid] += delta[valid] * delta2[valid]

        print(f"  [{i+1:3d}/{n}] {subj_id}  valid: {valid.sum():,}")

    # Sample SD
    sd = np.zeros_like(mean)
    valid_sd = count >= 2
    sd[valid_sd] = np.sqrt(M2[valid_sd] / (count[valid_sd] - 1))

    # Bilateral SD pooling: left and right anatomical mirror voxels share
    # the elementwise-max SD. This eliminates the L-vs-R SD asymmetry that
    # would otherwise bias clusters systematically toward whichever
    # hemisphere happens to have lower control SD (e.g. dorsal convexity).
    sd_mirror = np.flip(sd, axis=0)
    sd = np.maximum(sd, sd_mirror)

    # Physical SD floor in percent-AI units. The previous floor of 0.01
    # admitted voxels with effectively zero control variance, producing
    # spurious clip-to-20 zAI at brain convexity. 5% is a conservative
    # biologically-meaningful minimum for healthy-control AI dispersion.
    SD_FLOOR_PCT = 5.0
    sd = np.maximum(sd, SD_FLOOR_PCT)

    # Brain mask
    coverage = count / n
    brain_mask = (coverage >= BRAIN_MASK_COVERAGE)
    mean[~brain_mask] = 0
    sd[~brain_mask] = 0

    # Save
    affine = ref_img.affine
    header = ref_img.header.copy()

    def _save(arr, fname, dtype=np.float32):
        out = nib.Nifti1Image(arr.astype(dtype), affine, header)
        nib.save(out, str(group_dir / fname))

    _save(mean, "mean_asymmetry.nii.gz")
    _save(sd, "sd_asymmetry.nii.gz")
    _save(brain_mask, "brain_mask.nii.gz", dtype=np.uint8)
    _save(count, "count_map.nii.gz", dtype=np.int32)

    meta = {
        "group_key": group_key,
        "type": "asymmetry",
        "n_subjects": n,
        "subject_ids": subject_ids,
        "brain_mask_voxels": int(brain_mask.sum()),
        "mean_ai_in_mask": float(mean[brain_mask].mean()) if brain_mask.any() else 0,
        "built_at": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Brain mask: {brain_mask.sum():,} voxels")
    print(f"  Mean AI in mask: {meta['mean_ai_in_mask']:.3f}")
    print(f"  SD range: [{sd[brain_mask].min():.3f}, {sd[brain_mask].max():.3f}]")

    return mean.astype(np.float32), sd.astype(np.float32), brain_mask, ref_img


# ============================================================================
# PATIENT PROCESSING
# ============================================================================

def find_patient_perfusion(patient_dir):
    if SYM_MODE:
        p = SYM_DIR / f"{patient_dir.name}_perf_sym.nii.gz"
        if not p.exists():
            raise FileNotFoundError(f"No symmetric perfusion for {patient_dir.name}: {p}")
        return p
    candidates = list(patient_dir.glob("*_perfusion_calib_resampled_to_T1w.nii.gz"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Expected 1 perfusion file in {patient_dir}, found {len(candidates)}")


def cluster_filter(binary_map, min_size):
    """Local cluster_filter that ALSO returns per-cluster metadata.

    Uses ``src.thresholding.get_structure_3d()`` so the connectivity choice
    matches the registry methods.  Returns ``(filtered_mask, clusters)``
    where each cluster is a dict ``{id, size, mask}`` — used downstream to
    build the cluster-region report CSV.  ``src.thresholding.cluster_filter``
    only returns the mask, so we keep this richer wrapper.
    """
    structure = get_structure_3d()
    labeled, n = ndimage.label(binary_map, structure=structure)
    filtered = np.zeros_like(binary_map)
    clusters = []
    cid = 0
    for i in range(1, n + 1):
        cmask = labeled == i
        size = int(cmask.sum())
        if size >= min_size:
            cid += 1
            filtered |= cmask
            clusters.append({"id": cid, "size": size, "mask": cmask})
    return filtered, clusters


def _classify_direction(z_perf_cluster: float, z_perf_mirror: float,
                        cluster_hemi: str, mirror_hemi: str,
                        z_thr: float = 2.0) -> tuple[str, str, str]:
    """
    Direction-of-abnormality classification for a single zAI cluster.

    Replaces the legacy polarity-flip rule (which assumed all asymmetries
    arose from contralateral hypoperfusion). Instead, we use per-side
    perfusion z-scores against controls to identify which side actually
    deviates from healthy norms, and predict the EZ on the deviating side
    regardless of direction (hypoperfusion or hyperperfusion).

    Parameters
    ----------
    z_perf_cluster : float
        Median per-voxel perfusion z-score at the cluster mask.
    z_perf_mirror : float
        Median per-voxel perfusion z-score at the mirrored (contralateral)
        cluster mask.
    cluster_hemi, mirror_hemi : str
        "L" or "R" — physical hemisphere of the cluster and its mirror.
    z_thr : float, optional (default 2.0)
        Per-side deviation threshold; |z_perf| >= z_thr counts as deviating.

    Returns
    -------
    direction_class : str
        One of "hyper", "hypo", "mixed", "subtle". For hyper/hypo, the
        direction refers to the deviating side (whichever has |z| >= z_thr).
        "mixed" = both sides deviate; "subtle" = neither deviates strongly.
    ez_side_pred : str
        "L" or "R" — predicted EZ side (always the deviating side).
    deviating_side : str
        "L" or "R" — which side has the larger |z_perf|.
    """
    cluster_dev = abs(z_perf_cluster) >= z_thr
    mirror_dev = abs(z_perf_mirror) >= z_thr

    larger_side = (cluster_hemi if abs(z_perf_cluster) >= abs(z_perf_mirror)
                   else mirror_hemi)

    if cluster_dev and not mirror_dev:
        return ("hyper" if z_perf_cluster > 0 else "hypo",
                cluster_hemi, cluster_hemi)
    if mirror_dev and not cluster_dev:
        return ("hyper" if z_perf_mirror > 0 else "hypo",
                mirror_hemi, mirror_hemi)
    if cluster_dev and mirror_dev:
        return "mixed", larger_side, larger_side
    return "subtle", larger_side, larger_side


def process_patient(patient_id, mean_ai, sd_ai, brain_mask, parc, gm_mask,
                    affine, threshold, min_cluster,
                    mean_perf=None, sd_perf=None):
    """Compute asymmetry z-score for one patient.

    Parameters
    ----------
    mean_perf, sd_perf : numpy.ndarray, optional
        Per-voxel control perfusion mean and standard deviation
        (from `01_build_control_normative.py`). When both are provided,
        per-cluster direction-of-abnormality classification is computed
        and added to the cluster_report CSV (columns: cluster_hemi,
        z_perf_cluster_side, z_perf_mirror_side, direction_class,
        ez_side_pred). When omitted, the legacy `direction` column
        (left-dominant / right-dominant) is the only side-related output.
    """
    patient_dir = PATIENT_DATASET_DIR / patient_id
    if not patient_dir.exists():
        print(f"  Patient not found: {patient_dir}")
        return None

    perf_path = find_patient_perfusion(patient_dir)
    perfusion = nib.load(str(perf_path)).get_fdata(dtype=np.float32)

    # Patient's asymmetry map
    patient_ai, patient_valid = compute_asymmetry_map(perfusion)

    # Mask: brain coverage + gray matter + patient has data.
    # SD is already floored at 5% (percent-AI units) and bilaterally
    # pooled by the build_control_normative.py step, so no second SD
    # gate is needed here.
    valid = brain_mask & gm_mask & patient_valid & (sd_ai > 0)

    # Z-score of asymmetry
    z_asym = np.zeros_like(patient_ai)
    z_asym[valid] = (patient_ai[valid] - mean_ai[valid]) / sd_ai[valid]
    z_asym = np.clip(z_asym, -20, 20)

    # Threshold: positive z = abnormally left-dominant, negative = right-dominant
    left_dom = (z_asym > threshold) & valid   # abnormal LEFT > RIGHT
    right_dom = (z_asym < -threshold) & valid  # abnormal RIGHT > LEFT

    # Cluster filter
    left_filt, left_clusters = cluster_filter(left_dom, min_cluster)
    right_filt, right_clusters = cluster_filter(right_dom, min_cluster)

    # Significant z-scores only
    sig_z = np.zeros_like(z_asym)
    sig_z[left_filt] = z_asym[left_filt]
    sig_z[right_filt] = z_asym[right_filt]

    # Save
    out_dir = OUTPUT_DIR / "patients" / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{patient_id}_asymmetry"

    def _save(arr, name, dtype=np.float32):
        img = nib.Nifti1Image(arr.astype(dtype), affine)
        nib.save(img, str(out_dir / name))

    _save(z_asym, f"{prefix}_zscore.nii.gz")
    _save(sig_z, f"{prefix}_significant.nii.gz")
    _save(left_filt, f"{prefix}_left_dominant.nii.gz", dtype=np.uint8)
    _save(right_filt, f"{prefix}_right_dominant.nii.gz", dtype=np.uint8)

    # Per-voxel perfusion z-volume vs controls (used for direction inference).
    # Only computed when control perfusion stats are available.
    if mean_perf is not None and sd_perf is not None:
        sd_perf_safe = np.maximum(sd_perf, 0.01)
        z_perf_volume = (perfusion - mean_perf) / sd_perf_safe
        z_perf_valid = brain_mask & (perfusion > 0)
    else:
        z_perf_volume = None
        z_perf_valid = None

    # Cluster report with region mapping
    rows = []
    for direction, clusters_list in [("left-dominant", left_clusters),
                                      ("right-dominant", right_clusters)]:
        for cl in clusters_list:
            mask = cl["mask"]
            z_vals = z_asym[mask]
            com = ndimage.center_of_mass(mask)
            centroid_mni = affine @ np.array([*[round(c) for c in com], 1])

            labels_in = parc[mask]
            labels_in = labels_in[labels_in > 0]
            if len(labels_in) > 0:
                unique, counts = np.unique(labels_in, return_counts=True)
                primary_label = unique[np.argmax(counts)]
                primary_region = REGION_NAMES.get(int(primary_label), f"Label-{primary_label}")
            else:
                primary_region = "Unknown"

            # ----------------------------------------------------------------
            # Direction-of-abnormality inference (replaces polarity-flip rule)
            # ----------------------------------------------------------------
            # Cluster physical hemisphere from MNI x-coordinate (RAS: x>0 = R)
            cluster_hemi = "R" if centroid_mni[0] > 0 else "L"
            mirror_hemi = "L" if cluster_hemi == "R" else "R"

            if z_perf_volume is not None:
                mirror_mask = np.flip(mask, axis=0)
                cluster_voxels = mask & z_perf_valid
                mirror_voxels = mirror_mask & z_perf_valid
                if cluster_voxels.sum() > 0:
                    z_perf_cluster = float(np.median(z_perf_volume[cluster_voxels]))
                else:
                    z_perf_cluster = 0.0
                if mirror_voxels.sum() > 0:
                    z_perf_mirror = float(np.median(z_perf_volume[mirror_voxels]))
                else:
                    z_perf_mirror = 0.0
                direction_class, ez_side_pred, _ = _classify_direction(
                    z_perf_cluster, z_perf_mirror, cluster_hemi, mirror_hemi)
            else:
                z_perf_cluster = float("nan")
                z_perf_mirror = float("nan")
                direction_class = ""
                ez_side_pred = ""

            rows.append({
                "patient_id": patient_id,
                "direction": direction,
                "cluster_id": cl["id"],
                "size_voxels": cl["size"],
                "size_mm3": round(cl["size"] * VOXEL_VOL_MM3, 1),
                "mean_z": round(float(z_vals.mean()), 2),
                "peak_z": round(float(z_vals[np.argmax(np.abs(z_vals))]), 2),
                "centroid_x": round(float(centroid_mni[0]), 1),
                "centroid_y": round(float(centroid_mni[1]), 1),
                "centroid_z_coord": round(float(centroid_mni[2]), 1),
                "primary_region": primary_region,
                # Direction-of-abnormality inference columns
                "cluster_hemi": cluster_hemi,
                "z_perf_cluster_side": (round(z_perf_cluster, 2)
                                        if not np.isnan(z_perf_cluster) else ""),
                "z_perf_mirror_side": (round(z_perf_mirror, 2)
                                       if not np.isnan(z_perf_mirror) else ""),
                "direction_class": direction_class,
                "ez_side_pred": ez_side_pred,
            })

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / f"{prefix}_cluster_report.csv", index=False)

    # Summary
    n_valid = int(valid.sum())
    n_left = int(left_filt.sum())
    n_right = int(right_filt.sum())

    summary = {
        "patient_id": patient_id,
        "threshold": threshold,
        "min_cluster": min_cluster,
        "valid_gm_voxels": n_valid,
        "left_dominant_voxels": n_left,
        "right_dominant_voxels": n_right,
        "left_dominant_clusters": len(left_clusters),
        "right_dominant_clusters": len(right_clusters),
        "pct_abnormal_asymmetry": round(100 * (n_left + n_right) / n_valid, 1) if n_valid > 0 else 0,
    }
    with open(out_dir / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Montage
    plot_montage(patient_id, sig_z, valid, nib.load(str(
        OUTPUT_DIR / "groups" / GROUP_KEY / "mean_asymmetry.nii.gz")).get_fdata(),
        brain_mask, out_dir, prefix)

    return summary, rows


def plot_montage(patient_id, sig_z, mask, mean_ai, brain_mask, out_dir, prefix):
    """Axial montage of asymmetry z-scores."""
    z_indices = np.where(mask.any(axis=(0, 1)))[0]
    n_slices = 12
    selected = np.linspace(z_indices[0] + 5, z_indices[-1] - 5, n_slices, dtype=int)

    # Use the group mean perfusion as background if available
    mean_perf_path = BASE_DIR / "results_zscore" / "groups" / GROUP_KEY / "mean_perfusion.nii.gz"
    if mean_perf_path.exists():
        bg_vol = nib.load(str(mean_perf_path)).get_fdata(dtype=np.float32)
    else:
        bg_vol = np.abs(mean_ai)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        f"{patient_id} — Asymmetry Z-Score vs Controls (Gray Matter)\n"
        f"Red/warm = abnormally LEFT-dominant | Blue/cool = abnormally RIGHT-dominant\n"
        f"Threshold: |z| >= {ZSCORE_THRESHOLD}, Min cluster: {MIN_CLUSTER_SIZE} voxels",
        fontsize=12, fontweight="bold",
    )

    vmax_bg = np.percentile(bg_vol[brain_mask], 95) if brain_mask.any() else 1

    for idx, ax in enumerate(axes.flat):
        if idx >= n_slices:
            ax.axis("off")
            continue
        sl = selected[idx]
        bg = bg_vol[:, :, sl].T
        ax.imshow(bg, cmap="gray", origin="lower", aspect="equal",
                  vmin=0, vmax=vmax_bg)

        zslice = sig_z[:, :, sl].T
        has_signal = zslice != 0
        if has_signal.any():
            overlay = np.ma.masked_where(~has_signal, zslice)
            ax.imshow(overlay, cmap="RdBu_r", origin="lower", aspect="equal",
                      vmin=-8, vmax=8, alpha=0.85)

        ax.set_title(f"z = {sl}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(out_dir / f"{prefix}_montage.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    global ZSCORE_THRESHOLD, MIN_CLUSTER_SIZE, SYM_MODE

    parser = argparse.ArgumentParser(
        description="Control-referenced asymmetry z-score analysis")
    parser.add_argument("--patient", "-p", type=str, default=None)
    parser.add_argument("--group", "-g", type=str, default=GROUP_KEY)
    parser.add_argument("--threshold", "-t", type=float, default=ZSCORE_THRESHOLD)
    parser.add_argument("--min-cluster", type=int, default=MIN_CLUSTER_SIZE)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--sym", action="store_true",
                        help="Use symmetric-template-registered perfusion (symreg/sym_perf/) "
                             "so the mirror-AI compares homologous voxels (zAI over-dispersion fix).")
    args = parser.parse_args()

    ZSCORE_THRESHOLD = args.threshold
    MIN_CLUSTER_SIZE = args.min_cluster
    group_key = args.group
    SYM_MODE = args.sym
    if SYM_MODE:
        print("  [SYM] Using symmetric-template-registered perfusion from symreg/sym_perf/")

    print("\n" + "=" * 70)
    print("  CONTROL-REFERENCED ASYMMETRY Z-SCORE ANALYSIS")
    print("=" * 70)
    print(f"  Compares: Patient's L-R asymmetry vs Controls' L-R asymmetry")
    print(f"  Threshold: |z| >= {ZSCORE_THRESHOLD}")
    print(f"  Min cluster: {MIN_CLUSTER_SIZE} voxels")
    print(f"  Group: {group_key}")

    # Build/load group asymmetry stats
    mean_ai, sd_ai, brain_mask, ref_img = build_group_asymmetry_stats(
        group_key, force_rebuild=args.rebuild)
    affine = ref_img.affine

    # Load per-voxel control perfusion mean/SD for direction-of-abnormality
    # inference (computed by 01_build_control_normative.py). When these
    # files are missing, we fall back to the legacy direction column only.
    mean_perf_path = BASE_DIR / "results_zscore" / "groups" / group_key / "mean_perfusion.nii.gz"
    sd_perf_path = BASE_DIR / "results_zscore" / "groups" / group_key / "sd_perfusion.nii.gz"
    if mean_perf_path.exists() and sd_perf_path.exists():
        mean_perf = nib.load(str(mean_perf_path)).get_fdata(dtype=np.float32)
        sd_perf = nib.load(str(sd_perf_path)).get_fdata(dtype=np.float32)
        print(f"  Loaded control perfusion stats for direction inference: "
              f"{mean_perf.shape}")
    else:
        mean_perf = None
        sd_perf = None
        print("  Warning: control perfusion stats not found at "
              f"{mean_perf_path.parent}; direction inference will be skipped.")

    # Load consensus parcellation for region mapping
    parc_path = BASE_DIR / "results_zscore" / "groups" / group_key / "consensus_parcellation.nii.gz"
    if parc_path.exists():
        parc = nib.load(str(parc_path)).get_fdata().astype(np.int32)
    else:
        print("  Warning: No consensus parcellation found. Run 01_build_control_normative.py first.")
        parc = np.zeros(brain_mask.shape, dtype=np.int32)

    # Gray matter mask
    gm_mask = np.zeros(brain_mask.shape, dtype=bool)
    for label in GRAY_MATTER_LABELS:
        gm_mask |= (parc == label)

    # Find patients
    if args.patient:
        patients = [args.patient]
    else:
        # Use demographics to find matching patients
        spreadsheet = BASE_DIR / "clinical_spreadsheet.xlsx"
        if spreadsheet.exists():
            df = pd.read_excel(str(spreadsheet))
            df.columns = [c.strip().upper() for c in df.columns]
            id_col = [c for c in df.columns if "ID" in c][0]
            age_col = [c for c in df.columns if "AGE" in c][0]
            sex_col = [c for c in df.columns if "SEX" in c or "GENDER" in c][0]
            # Match patients to THIS group's sex + age band, parsed from group_key
            # (e.g. FM_20_39 -> sex=FM, ages 20-39). "FM" is sex-pooled (any sex).
            gp = group_key.split("_")
            grp_sex, grp_lo, grp_hi = gp[0], int(gp[1]), int(gp[2])
            patients = []
            for _, row in df.iterrows():
                pid = str(row[id_col]).strip()
                if pid.startswith("sub-"):
                    pid = pid[4:]
                age = int(row[age_col])
                sex = str(row[sex_col]).strip().upper()[0]
                sex_ok = (grp_sex == "FM") or (sex == grp_sex)
                if sex_ok and grp_lo <= age <= grp_hi:
                    if not (PATIENT_DATASET_DIR / pid).exists():
                        continue
                    # In symmetric-space mode, skip patients without a registered
                    # symmetric perfusion (e.g. P014, excluded from the cohort).
                    if SYM_MODE and not (SYM_DIR / f"{pid}_perf_sym.nii.gz").exists():
                        continue
                    patients.append(pid)
        else:
            patients = sorted([d.name for d in PATIENT_DATASET_DIR.iterdir() if d.is_dir()])

    print(f"  Patients: {patients}")
    print("=" * 70)

    all_summaries = []
    for pid in patients:
        print(f"\n  Processing {pid}...")
        result = process_patient(pid, mean_ai, sd_ai, brain_mask, parc, gm_mask,
                                 affine, ZSCORE_THRESHOLD, MIN_CLUSTER_SIZE,
                                 mean_perf=mean_perf, sd_perf=sd_perf)
        if result is None:
            continue
        summary, cluster_rows = result
        all_summaries.append(summary)

        print(f"    Valid GM voxels: {summary['valid_gm_voxels']:,}")
        print(f"    Abnormal asymmetry: {summary['pct_abnormal_asymmetry']:.1f}%")
        print(f"    Left-dominant: {summary['left_dominant_clusters']} clusters "
              f"({summary['left_dominant_voxels']:,} voxels)")
        print(f"    Right-dominant: {summary['right_dominant_clusters']} clusters "
              f"({summary['right_dominant_voxels']:,} voxels)")

        if cluster_rows:
            print(f"\n    {'Dir':<16} {'#':>3} {'Size':>7} {'Mean z':>7} {'Peak z':>7} {'Region'}")
            print(f"    {'─'*65}")
            for row in sorted(cluster_rows, key=lambda x: -x["size_voxels"])[:15]:
                print(f"    {row['direction']:<16} {row['cluster_id']:>3} "
                      f"{row['size_voxels']:>7} {row['mean_z']:>7.2f} "
                      f"{row['peak_z']:>7.2f} {row['primary_region']}")

    # FSLeyes commands
    if all_summaries:
        mean_perf = BASE_DIR / "results_zscore" / "groups" / group_key / "mean_perfusion.nii.gz"
        print(f"\n{'='*70}")
        print("  FSLeyes commands")
        print(f"{'='*70}")
        for s in all_summaries:
            pid = s["patient_id"]
            pdir = OUTPUT_DIR / "patients" / pid
            left_f = pdir / f"{pid}_asymmetry_left_dominant.nii.gz"
            right_f = pdir / f"{pid}_asymmetry_right_dominant.nii.gz"
            sig_f = pdir / f"{pid}_asymmetry_significant.nii.gz"

            print(f"\n  # {pid} — lateralized abnormal asymmetry:")
            print(f"  fsleyes '{mean_perf}' \\")
            if left_f.exists():
                print(f"    '{left_f}' -cm red -dr 0.5 1 -a 90 "
                      f"-n 'LEFT dominant (abnormal)' \\")
            if right_f.exists():
                print(f"    '{right_f}' -cm blue -dr 0.5 1 -a 90 "
                      f"-n 'RIGHT dominant (abnormal)' &")

    print(f"\n{'='*70}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
