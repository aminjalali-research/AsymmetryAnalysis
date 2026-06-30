#!/usr/bin/env python3
"""
Generate Clinical-Grade zAI Maps for Presurgical Planning

**2026-05-04 methodology pivot:** This script now consumes zAI maps produced
by `02_compute_zai.py` (Approach B: asymmetry-then-z-score, the
literature-standard zAI of Shang 2021 / Boscolo Galazzo 2016 / Gennari 2025)
instead of the raw perfusion z-score maps from `01_build_control_normative.py`
(Approach A). The audit at
`docs/superpowers/audits/2026-05-04-control_asymmetry_zscore-audit.md`
confirms `02_compute_zai.py` correctly implements zAI.

**2026-05-05 37-ROI clinical scope policy (DEFAULT):** Surgeon-facing output is
restricted to the 37 clinically-relevant ROI pairs (76 FreeSurfer labels)
per CLAUDE.md / `.claude/DECISIONS.md` 2026-05-05 entry:
  - 3 subcortical pairs: Thalamus (10/49), Hippocampus (17/53), Amygdala (18/54)
  - 34 cortical Desikan-Killiany regions: 1001-1035 (L) + 2001-2035 (R)
Excluded basal ganglia (caudate, putamen, pallidum) are intentional and match
Pipeline A / manuscript scope. The mask is loaded from the patient's
`Dataset/<PID>/aparc+aseg.nii.gz`. Use `--no-clinical-mask` to disable the
mask and fall back to the previous gray-matter-only behavior. By default,
the script also writes diagnostic `*_clinical_zai_unmasked_*.nii.gz` variants
alongside the masked surgeon-facing outputs for QC/exploratory use.

The raw zAI maps highlight too many voxels because they cover the full
brain mask, including white matter, ventricles, and unlabeled regions.
For epilepsy presurgical planning, clinicians need to see only gray matter
(cortical + subcortical) with strict thresholding and meaningful cluster
sizes.

This script produces clean NIfTI maps showing only focal lateralization
abnormalities in clinically relevant tissue.

Interpretation of zAI values (unchanged from the upstream script):
  - zAI > 0: abnormally LEFT-dominant perfusion (left higher than right)
  - zAI < 0: abnormally RIGHT-dominant perfusion (right higher than left)

Note: positive/negative no longer means "hyper/hypo" — it means the side
that is MORE perfused relative to its contralateral mirror, expressed as
a deviation from healthy controls. We retain the variable names
`hyper`/`hypo` only for backward-compatible output filenames; semantically
they now mean "left-dominant" / "right-dominant" for zAI inputs.

Usage:
    python 03_clinical_maps.py                    # 37-ROI mask (default)
    python 03_clinical_maps.py --patient P013
    python 03_clinical_maps.py --threshold 3.0 --min-cluster 50
    python 03_clinical_maps.py --no-clinical-mask # disable 37-ROI mask
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

# ============================================================================
# CONFIGURATION — tuned for presurgical planning
# ============================================================================

# Stricter threshold: |zAI| >= 3.0 corresponds to p < 0.003
# Much more specific than 1.96 for voxel-wise comparisons
CLINICAL_ZAI_THRESHOLD = 3.0

# Larger cluster size: 50 voxels = ~25.6 mm3 at 0.8mm isotropic
# Removes scattered noise, keeps focal abnormalities
CLINICAL_MIN_CLUSTER = 50

# 18-connected (faces + edges)
CLUSTER_CONNECTIVITY = 2

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results_zscore"
GROUP_KEY = "FM_20_39"   # fallback only; the band is resolved PER PATIENT below
GROUP_DIR = RESULTS_DIR / "groups" / GROUP_KEY


def resolve_patient_group(pid):
    """Resolve the age-matched control band for a patient from the per-patient
    z-map filename written by 01_build_control_normative.py
    (``<pid>_vs_<GROUP>_zscore.nii.gz``). Falls back to GROUP_KEY. This avoids
    silently referencing a single fixed (and possibly stale) control group for
    patients matched to different FM_* bands."""
    zdir = RESULTS_DIR / "patients" / pid
    if zdir.exists():
        for c in sorted(zdir.glob(f"{pid}_vs_*_zscore.nii.gz")):
            stem = c.name[len(f"{pid}_vs_"):-len("_zscore.nii.gz")]
            if stem:
                return stem
    return GROUP_KEY
# zAI inputs come from 02_compute_zai.py:
#   results_zscore/asymmetry/patients/{pid}/{pid}_asymmetry_zscore.nii.gz
ZAI_PATIENTS_DIR = RESULTS_DIR / "asymmetry" / "patients"
OUTPUT_DIR = RESULTS_DIR / "clinical"

# Gray matter labels from FreeSurfer aparc+aseg
# Cortical: 1001-1035 (lh), 2001-2035 (rh)
# Subcortical: thalamus, hippocampus, amygdala, caudate, putamen, pallidum, accumbens
CORTICAL_LABELS = set(range(1001, 1036)) | set(range(2001, 2036))
SUBCORTICAL_LABELS = {10, 11, 12, 13, 17, 18, 26, 28,  # left
                      49, 50, 51, 52, 53, 54, 58, 60}  # right
GRAY_MATTER_LABELS = CORTICAL_LABELS | SUBCORTICAL_LABELS

# ----------------------------------------------------------------------------
# 37-ROI clinical scope (CLAUDE.md / DECISIONS.md 2026-05-05 policy)
# 76 FreeSurfer labels = 37 paired regions:
#   - 3 subcortical pairs: Thalamus (10/49), Hippocampus (17/53), Amygdala (18/54)
#   - 34 cortical Desikan-Killiany pairs: 1001-1035 (L) + 2001-2035 (R)
# Basal ganglia (caudate 11/50, putamen 12/51, pallidum 13/52) are intentionally
# excluded — matches Pipeline A / manuscript reported analysis scope.
# ----------------------------------------------------------------------------
CLINICAL_SUBCORTICAL_LABELS = [10, 49, 17, 53, 18, 54]
CLINICAL_CORTICAL_LABELS = list(range(1001, 1036)) + list(range(2001, 2036))
CLINICAL_ROI_LABELS = set(CLINICAL_SUBCORTICAL_LABELS) | set(CLINICAL_CORTICAL_LABELS)
DATASET_DIR = BASE_DIR / "Dataset"

# Symmetric-space mode (2026-06-22 zAI fix): the zAI maps are now built in the
# left-right symmetric template space, so the per-patient clinical-ROI mask must
# use the symmetric-space parcellation (symreg/sym_perf/{pid}_aparc_sym.nii.gz).
SYM_MODE = False
SYM_DIR = BASE_DIR / "symreg" / "sym_perf"

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


def build_gray_matter_mask(parcellation):
    """Create a boolean mask of gray matter voxels from parcellation."""
    gm = np.zeros(parcellation.shape, dtype=bool)
    for label in GRAY_MATTER_LABELS:
        gm |= (parcellation == label)
    return gm


def _build_clinical_roi_mask(patient_id, ref_shape=None, ref_affine=None):
    """Build the 37-ROI clinical surgeon-facing mask for a given patient.

    Per CLAUDE.md / .claude/DECISIONS.md 2026-05-05 policy, restricts
    surgeon-facing zAI output to 76 FreeSurfer labels (37 paired regions):
      - 3 subcortical pairs: Thalamus (10/49), Hippocampus (17/53),
        Amygdala (18/54)
      - 34 cortical Desikan-Killiany regions: 1001-1035 (L) + 2001-2035 (R)

    Loads from `Dataset/<PID>/aparc+aseg.nii.gz` (which is in MNI space at
    227x272x227, 0.8mm isotropic — same geometry as the zAI map).

    Parameters
    ----------
    patient_id : str
        Patient ID, e.g. "P015".
    ref_shape, ref_affine : optional
        If provided, sanity-checked against the loaded parcellation.

    Returns
    -------
    np.ndarray of bool with shape == parcellation.shape, or None if the
    file is missing.
    """
    if SYM_MODE:
        parc_path = SYM_DIR / f"{patient_id}_aparc_sym.nii.gz"
    else:
        parc_path = DATASET_DIR / patient_id / "aparc+aseg.nii.gz"
    if not parc_path.exists():
        print(f"  ! Cannot build clinical ROI mask for {patient_id}: "
              f"{parc_path} not found.")
        return None

    parc_img = nib.load(str(parc_path))
    parc = parc_img.get_fdata().astype(np.int32)

    if ref_shape is not None and parc.shape != ref_shape:
        print(f"  ! Shape mismatch for {patient_id}: parcellation "
              f"{parc.shape} vs zAI {ref_shape}. Skipping clinical mask.")
        return None

    mask = np.zeros(parc.shape, dtype=bool)
    for label in CLINICAL_ROI_LABELS:
        mask |= (parc == label)
    return mask


def cluster_filter(binary_map, min_size):
    """Connected component labeling with size filter."""
    structure = ndimage.generate_binary_structure(3, CLUSTER_CONNECTIVITY)
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
    # Re-label consecutively
    relabeled, _ = ndimage.label(filtered, structure=structure)
    return filtered, relabeled, clusters


def _compute_clinical_outputs(zai_map, base_mask, threshold, min_cluster):
    """Compute the family of clinical output arrays for a given mask.

    `base_mask` is the boolean voxel set that defines the analysis scope
    (e.g. brain ∩ gray-matter ∩ patient-has-data, optionally further
    restricted by the 37-ROI clinical mask).

    Returns a dict of arrays + cluster lists ready to be saved.
    """
    # Apply threshold inside the base mask
    # left_dom = abnormally LEFT-dominant (positive zAI)
    # right_dom = abnormally RIGHT-dominant (negative zAI)
    # Variable names "hyper"/"hypo" retained for backward-compatible filenames.
    hyper_raw = (zai_map > threshold) & base_mask
    hypo_raw = (zai_map < -threshold) & base_mask

    hyper_filt, _, hyper_clusters = cluster_filter(hyper_raw, min_cluster)
    hypo_filt, _, hypo_clusters = cluster_filter(hypo_raw, min_cluster)

    # 1. Clinical zAI (zeroed outside base mask)
    clinical_zai = np.zeros_like(zai_map)
    clinical_zai[base_mask] = zai_map[base_mask]

    # 2. Significant clusters only (zAI values preserved within surviving clusters)
    sig_zai = np.zeros_like(zai_map)
    sig_zai[hyper_filt] = zai_map[hyper_filt]
    sig_zai[hypo_filt] = zai_map[hypo_filt]

    # 3. Combined cluster label map (+N = left-dominant N, -N = right-dominant N)
    cluster_map = np.zeros(zai_map.shape, dtype=np.int32)
    for cl in hyper_clusters:
        cluster_map[cl["mask"]] = cl["id"]
    for cl in hypo_clusters:
        cluster_map[cl["mask"]] = -cl["id"]

    # 4. Lateralized: keep only the more abnormal side at each voxel pair
    zai_mirror = np.flip(clinical_zai, axis=0)
    dominant_mask = np.abs(clinical_zai) > np.abs(zai_mirror)

    lat_hyper = hyper_filt & dominant_mask
    lat_hypo = hypo_filt & dominant_mask
    lat_hyper, _, lat_hyper_cl = cluster_filter(lat_hyper, min_cluster)
    lat_hypo, _, lat_hypo_cl = cluster_filter(lat_hypo, min_cluster)

    lat_sig_zai = np.zeros_like(zai_map)
    lat_sig_zai[lat_hyper] = zai_map[lat_hyper]
    lat_sig_zai[lat_hypo] = zai_map[lat_hypo]

    return {
        "base_mask": base_mask,
        "clinical_zai": clinical_zai,
        "sig_zai": sig_zai,
        "cluster_map": cluster_map,
        "hyper_filt": hyper_filt,
        "hypo_filt": hypo_filt,
        "hyper_clusters": hyper_clusters,
        "hypo_clusters": hypo_clusters,
        "lat_hyper": lat_hyper,
        "lat_hypo": lat_hypo,
        "lat_hyper_cl": lat_hyper_cl,
        "lat_hypo_cl": lat_hypo_cl,
        "lat_sig_zai": lat_sig_zai,
    }


def _build_cluster_rows(patient_id, parc, zai_map, affine, hyper_clusters, hypo_clusters):
    """Build the per-cluster CSV rows with region mapping."""
    voxel_vol = 0.8 ** 3
    rows = []
    for direction, clusters_list in [("left-dominant", hyper_clusters),
                                      ("right-dominant", hypo_clusters)]:
        for cl in clusters_list:
            zai_vals = zai_map[cl["mask"]]
            com = ndimage.center_of_mass(cl["mask"])
            centroid_mni = affine @ np.array([*[round(c) for c in com], 1])

            labels_in = parc[cl["mask"]]
            labels_in = labels_in[labels_in > 0]
            if len(labels_in) > 0:
                unique, counts = np.unique(labels_in, return_counts=True)
                primary_label = unique[np.argmax(counts)]
                primary_region = REGION_NAMES.get(int(primary_label), f"Label-{primary_label}")
            else:
                primary_region = "Unknown"

            rows.append({
                "patient_id": patient_id,
                "direction": direction,
                "cluster_id": cl["id"],
                "size_voxels": cl["size"],
                "size_mm3": round(cl["size"] * voxel_vol, 1),
                "mean_zai": round(float(zai_vals.mean()), 2),
                "peak_zai": round(float(zai_vals[np.argmax(np.abs(zai_vals))]), 2),
                "centroid_x": round(float(centroid_mni[0]), 1),
                "centroid_y": round(float(centroid_mni[1]), 1),
                "centroid_z": round(float(centroid_mni[2]), 1),
                "primary_region": primary_region,
            })
    return rows


def process_patient(patient_id, brain_mask, parc, gm_mask, mean_img, affine,
                    threshold, min_cluster, apply_clinical_mask=True):
    """Generate clinical zAI maps for one patient.

    Parameters
    ----------
    apply_clinical_mask : bool
        When True (default), surgeon-facing outputs are restricted to the
        37-ROI clinical scope (CLAUDE.md / DECISIONS.md 2026-05-05). The
        masked outputs take the canonical `*_clinical_zai_*.nii.gz`
        filenames; an additional set of unmasked diagnostic variants
        `*_clinical_zai_unmasked_*.nii.gz` is written alongside.
        When False (legacy behavior, --no-clinical-mask), only the
        unmasked variant is produced and written under the canonical
        `*_clinical_zai_*.nii.gz` filenames.
    """
    # Load zAI map produced by 02_compute_zai.py
    zai_path = (ZAI_PATIENTS_DIR / patient_id /
                f"{patient_id}_asymmetry_zscore.nii.gz")
    if not zai_path.exists():
        print(f"  X No zAI map for {patient_id} (expected at {zai_path})")
        return None

    zai_map = nib.load(str(zai_path)).get_fdata(dtype=np.float32)

    # ------------------------------------------------------------------
    # Build masks. The unmasked baseline (gray-matter only) is always
    # computed; the 37-ROI clinical mask is computed when requested.
    # ------------------------------------------------------------------
    unmasked_base = brain_mask & gm_mask & (zai_map != 0)

    clinical_roi_mask = None
    if apply_clinical_mask:
        clinical_roi_mask = _build_clinical_roi_mask(
            patient_id, ref_shape=zai_map.shape, ref_affine=affine)
        if clinical_roi_mask is None:
            print(f"  ! Falling back to gray-matter-only for {patient_id} "
                  f"(clinical ROI mask unavailable).")

    # The "primary" outputs (those that get the canonical names) reflect
    # whether the clinical mask is being applied.
    if apply_clinical_mask and clinical_roi_mask is not None:
        primary_base = unmasked_base & clinical_roi_mask
        primary_scope = "37-ROI clinical mask"
    else:
        primary_base = unmasked_base
        primary_scope = "gray-matter-only (no 37-ROI mask)"

    primary = _compute_clinical_outputs(zai_map, primary_base, threshold, min_cluster)

    # When the clinical mask is applied (default), also write an unmasked
    # diagnostic variant alongside for QC/exploratory use.
    write_diagnostic_unmasked = apply_clinical_mask and clinical_roi_mask is not None
    if write_diagnostic_unmasked:
        diagnostic = _compute_clinical_outputs(
            zai_map, unmasked_base, threshold, min_cluster)
    else:
        diagnostic = None

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    out_dir = OUTPUT_DIR / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{patient_id}_clinical_zai"

    def _save(arr, name, dtype=np.float32):
        img = nib.Nifti1Image(arr.astype(dtype), affine)
        nib.save(img, str(out_dir / name))

    def _write_outputs(p, file_prefix):
        _save(p["clinical_zai"], f"{file_prefix}_gm.nii.gz")
        _save(p["sig_zai"], f"{file_prefix}_significant_clusters.nii.gz")
        _save(p["cluster_map"], f"{file_prefix}_cluster_labels.nii.gz", dtype=np.int32)
        _save(p["hyper_filt"], f"{file_prefix}_left_dominant.nii.gz", dtype=np.uint8)
        _save(p["hypo_filt"], f"{file_prefix}_right_dominant.nii.gz", dtype=np.uint8)
        _save(p["lat_hyper"], f"{file_prefix}_lateralized_left_dominant.nii.gz", dtype=np.uint8)
        _save(p["lat_hypo"], f"{file_prefix}_lateralized_right_dominant.nii.gz", dtype=np.uint8)
        _save(p["lat_sig_zai"], f"{file_prefix}_lateralized_significant.nii.gz")

    # Primary (canonical filenames — masked by default)
    _write_outputs(primary, prefix)

    # Diagnostic unmasked variants (only when 37-ROI mask is applied)
    if diagnostic is not None:
        _write_outputs(diagnostic, f"{prefix}_unmasked")

    # ------------------------------------------------------------------
    # Cluster reports (CSV) and summary JSON — based on the primary outputs
    # ------------------------------------------------------------------
    primary_rows = _build_cluster_rows(
        patient_id, parc, zai_map,
        affine, primary["hyper_clusters"], primary["hypo_clusters"])
    if primary_rows:
        pd.DataFrame(primary_rows).to_csv(
            out_dir / f"{prefix}_cluster_report.csv", index=False)

    if diagnostic is not None:
        diagnostic_rows = _build_cluster_rows(
            patient_id, parc, zai_map,
            affine, diagnostic["hyper_clusters"], diagnostic["hypo_clusters"])
        if diagnostic_rows:
            pd.DataFrame(diagnostic_rows).to_csv(
                out_dir / f"{prefix}_unmasked_cluster_report.csv", index=False)

    # Summary
    n_base = int(primary_base.sum())
    n_left = int(primary["hyper_filt"].sum())
    n_right = int(primary["hypo_filt"].sum())
    n_sig = n_left + n_right
    n_lat_left = int(primary["lat_hyper"].sum())
    n_lat_right = int(primary["lat_hypo"].sum())
    n_lat_sig = n_lat_left + n_lat_right

    summary = {
        "patient_id": patient_id,
        "input_metric": "zAI (asymmetry z-score vs controls)",
        "threshold": threshold,
        "min_cluster": min_cluster,
        "clinical_mask_applied": bool(apply_clinical_mask and clinical_roi_mask is not None),
        "scope": primary_scope,
        "scope_voxels": n_base,
        # Backward-compatible field name (was "gray_matter_voxels"); now
        # reflects the scope mask (37-ROI when applied, else gray matter).
        "gray_matter_voxels": n_base,
        "left_dominant_voxels": n_left,
        "right_dominant_voxels": n_right,
        "pct_significant": round(100 * n_sig / n_base, 1) if n_base > 0 else 0,
        "left_dominant_clusters": len(primary["hyper_clusters"]),
        "right_dominant_clusters": len(primary["hypo_clusters"]),
        "lateralized_left_dominant_voxels": n_lat_left,
        "lateralized_right_dominant_voxels": n_lat_right,
        "lateralized_left_dominant_clusters": len(primary["lat_hyper_cl"]),
        "lateralized_right_dominant_clusters": len(primary["lat_hypo_cl"]),
        "pct_lateralized": round(100 * n_lat_sig / n_base, 1) if n_base > 0 else 0,
    }
    if diagnostic is not None:
        n_diag_base = int(unmasked_base.sum())
        n_diag_left = int(diagnostic["hyper_filt"].sum())
        n_diag_right = int(diagnostic["hypo_filt"].sum())
        summary["diagnostic_unmasked"] = {
            "scope": "gray-matter-only (no 37-ROI mask)",
            "scope_voxels": n_diag_base,
            "left_dominant_voxels": n_diag_left,
            "right_dominant_voxels": n_diag_right,
            "left_dominant_clusters": len(diagnostic["hyper_clusters"]),
            "right_dominant_clusters": len(diagnostic["hypo_clusters"]),
        }

    with open(out_dir / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plain-language README for surgeons / non-imaging clinicians.
    # One file per patient, regenerated each run so timestamps stay current.
    _write_layman_readme(out_dir, patient_id, threshold, min_cluster,
                         summary)

    return summary, primary_rows


def _write_layman_readme(out_dir, patient_id, threshold, min_cluster, summary):
    """Write a plain-language README.txt explaining each NIfTI in `out_dir`.

    Surgeons reading these files in 3D Slicer / FSLeyes shouldn't have to
    consult a methods document to understand the sign convention or which
    file means what. The README is regenerated on every run so the
    timestamp always reflects the current pipeline run.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    pid = patient_id
    pct_lat = summary.get("pct_lateralized", 0.0)
    n_lat_left = summary.get("lateralized_left_dominant_voxels", 0)
    n_lat_right = summary.get("lateralized_right_dominant_voxels", 0)
    scope = summary.get("scope", "(unknown scope)")

    text = (
        f"What's in this folder\n"
        f"=====================\n"
        f"Patient ID: {pid}\n"
        f"Pipeline:   Voxel zAI lateralization (Pipeline B)\n"
        f"Date:       {timestamp}\n"
        f"Thresholds: |zAI| >= {threshold} AND cluster size >= {min_cluster} voxels\n"
        f"Scope:      {scope}\n"
        f"Summary:    {pct_lat:.1f}% of in-scope voxels lateralized\n"
        f"            ({n_lat_left:,} left-dominant + {n_lat_right:,} "
        f"right-dominant)\n"
        f"\n"
        f"Sign convention (read this once, then ignore)\n"
        f"---------------------------------------------\n"
        f"  'Left-dominant'  = LEFT side has MORE blood flow than expected.\n"
        f"                     Counterintuitively: in interictal epilepsy,\n"
        f"                     this usually points AWAY from the EZ, since\n"
        f"                     the EZ side is hypoperfused.\n"
        f"  'Right-dominant' = RIGHT side has MORE blood flow than expected.\n"
        f"                     Look at temporal-lobe clusters here for\n"
        f"                     likely LEFT EZ.\n"
        f"\n"
        f"Files\n"
        f"-----\n"
        f"  {pid}_clinical_zai_lateralized_left_dominant.nii.gz\n"
        f"    Voxels where the LEFT side has abnormally HIGH blood flow vs\n"
        f"    healthy control baseline. Counterintuitively: in interictal\n"
        f"    epilepsy, this usually points AWAY from the EZ, since the EZ\n"
        f"    is hypoperfused. Look at temporal-lobe clusters here for\n"
        f"    likely RIGHT EZ.\n"
        f"\n"
        f"  {pid}_clinical_zai_lateralized_right_dominant.nii.gz\n"
        f"    Voxels where the RIGHT side has abnormally HIGH blood flow.\n"
        f"    Look at temporal-lobe clusters here for likely LEFT EZ.\n"
        f"\n"
        f"  {pid}_clinical_zai_lateralized_significant.nii.gz\n"
        f"    Continuous zAI values (positive = L>R, negative = R>L) kept\n"
        f"    only inside surviving lateralized clusters. Use a diverging\n"
        f"    colormap (e.g. brain_colours_diverging_bwr) to view.\n"
        f"\n"
        f"  {pid}_clinical_zai_cluster_report.csv\n"
        f"    Per-cluster table: size, peak |zAI|, primary FreeSurfer\n"
        f"    region, direction. Use 06_clinical_interpretation.py to\n"
        f"    digest this into a surgeon-facing summary.\n"
        f"\n"
        f"  {pid}_clinical_zai_gm.nii.gz\n"
        f"    Continuous zAI map masked to gray matter (and the 37-ROI\n"
        f"    clinical scope by default; pass --no-clinical-mask to\n"
        f"    disable).\n"
        f"\n"
        f"  {pid}_clinical_zai_significant_clusters.nii.gz\n"
        f"    Continuous zAI values inside ALL surviving clusters\n"
        f"    (both sides), not restricted to the dominant-side-only view.\n"
        f"\n"
        f"  {pid}_clinical_zai_cluster_labels.nii.gz\n"
        f"    Integer cluster ID per voxel: positive IDs = left-dominant,\n"
        f"    negative IDs = right-dominant. Useful for highlighting a\n"
        f"    single cluster (e.g. the top-N peak |zAI| view).\n"
        f"\n"
        f"  {pid}_clinical_zai_left_dominant.nii.gz /\n"
        f"  {pid}_clinical_zai_right_dominant.nii.gz\n"
        f"    Bilateral (not 'lateralized') versions of the dominance\n"
        f"    masks: a voxel can appear in both files if both halves of a\n"
        f"    bilateral pair are abnormal. The 'lateralized_*' versions\n"
        f"    above keep only the more abnormal side at each voxel pair\n"
        f"    and are the recommended surgeon-facing view.\n"
        f"\n"
        f"  {pid}_clinical_zai_montage.png\n"
        f"    Axial montage of significant lateralized voxels overlaid on\n"
        f"    the group mean perfusion. Quick look without launching\n"
        f"    FSLeyes.\n"
        f"\n"
        f"  {pid}_clinical_zai_summary.json\n"
        f"    Machine-readable summary of voxel/cluster counts and the\n"
        f"    scope mask used.\n"
        f"\n"
        f"  {pid}_clinical_zai_unmasked_*.nii.gz / *_unmasked_cluster_report.csv\n"
        f"    Diagnostic variants computed without the 37-ROI clinical\n"
        f"    mask (gray matter only). Same naming scheme as above. Use\n"
        f"    these for QC / exploratory review only; the unmasked outputs\n"
        f"    include cerebellum, brainstem and basal ganglia which the\n"
        f"    surgeon-facing pipeline intentionally suppresses.\n"
    )
    (out_dir / "README.txt").write_text(text)


def plot_clinical_montage(patient_id, mean_img, brain_mask, gm_mask, affine):
    """Clean axial montage showing only significant gray matter zAI clusters."""
    out_dir = OUTPUT_DIR / patient_id
    prefix = f"{patient_id}_clinical_zai"

    sig_zai = nib.load(str(out_dir / f"{prefix}_lateralized_significant.nii.gz")).get_fdata()

    z_indices = np.where(gm_mask.any(axis=(0, 1)))[0]
    n_slices = 12
    selected = np.linspace(z_indices[0] + 5, z_indices[-1] - 5, n_slices, dtype=int)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        f"{patient_id} - Clinical zAI Map (Gray Matter Only)\n"
        f"Threshold: |zAI| >= {CLINICAL_ZAI_THRESHOLD}, "
        f"Min cluster: {CLINICAL_MIN_CLUSTER} voxels\n"
        f"Red = abnormally LEFT-dominant, Blue = abnormally RIGHT-dominant",
        fontsize=13, fontweight="bold",
    )

    vmax_bg = np.percentile(mean_img[brain_mask], 95)

    for idx, ax in enumerate(axes.flat):
        if idx >= n_slices:
            ax.axis("off")
            continue
        sl = selected[idx]
        bg = mean_img[:, :, sl].T
        ax.imshow(bg, cmap="gray", origin="lower", aspect="equal",
                  vmin=0, vmax=vmax_bg)

        zai_slice = sig_zai[:, :, sl].T
        has_signal = zai_slice != 0
        if has_signal.any():
            overlay = np.ma.masked_where(~has_signal, zai_slice)
            ax.imshow(overlay, cmap="RdBu_r", origin="lower", aspect="equal",
                      vmin=-8, vmax=8, alpha=0.85)

        ax.set_title(f"z = {sl}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    fig_path = out_dir / f"{prefix}_montage.png"
    plt.savefig(str(fig_path), dpi=200, bbox_inches="tight")
    plt.close()
    return fig_path


def main():
    global CLINICAL_ZAI_THRESHOLD, CLINICAL_MIN_CLUSTER, SYM_MODE

    parser = argparse.ArgumentParser(
        description="Generate clinical-grade zAI maps "
                    "(default: 37-ROI clinical mask)")
    parser.add_argument("--sym", action="store_true",
                        help="zAI maps are in symmetric-template space; use the "
                             "symmetric per-patient parcellation for the clinical ROI mask.")
    parser.add_argument("--patient", "-p", type=str, default=None)
    parser.add_argument("--threshold", "-t", type=float, default=CLINICAL_ZAI_THRESHOLD,
                        help=f"zAI threshold (default: {CLINICAL_ZAI_THRESHOLD})")
    parser.add_argument("--min-cluster", type=int, default=CLINICAL_MIN_CLUSTER,
                        help=f"Min cluster size in voxels (default: {CLINICAL_MIN_CLUSTER})")
    parser.add_argument(
        "--no-clinical-mask", dest="apply_clinical_mask",
        action="store_false", default=True,
        help="Disable the 37-ROI clinical mask. Falls back to "
             "gray-matter-only output (legacy behavior). "
             "By default the surgeon-facing output is restricted to the "
             "37-ROI clinical scope per CLAUDE.md / DECISIONS.md "
             "2026-05-05 policy, and an unmasked diagnostic variant is "
             "written alongside.")
    args = parser.parse_args()

    CLINICAL_ZAI_THRESHOLD = args.threshold
    CLINICAL_MIN_CLUSTER = args.min_cluster
    SYM_MODE = args.sym
    if SYM_MODE:
        print("  [SYM] Using symmetric-space per-patient parcellation for clinical ROI mask")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    if args.apply_clinical_mask:
        print("  CLINICAL zAI MAPS - 37-ROI CLINICAL SCOPE (default)")
    else:
        print("  CLINICAL zAI MAPS - GRAY MATTER ONLY (--no-clinical-mask)")
    print("=" * 70)
    print(f"  Input:        zAI maps from 02_compute_zai.py")
    print(f"                ({ZAI_PATIENTS_DIR}/{{pid}}/{{pid}}_asymmetry_zscore.nii.gz)")
    print(f"  Threshold:    |zAI| >= {CLINICAL_ZAI_THRESHOLD}")
    print(f"  Min cluster:  {CLINICAL_MIN_CLUSTER} voxels "
          f"({CLINICAL_MIN_CLUSTER * 0.512:.0f} mm3)")
    if args.apply_clinical_mask:
        print(f"  Tissue:       37 clinical ROI pairs "
              f"(3 subcortical + 34 Desikan-Killiany cortical)")
        print(f"                Per CLAUDE.md / DECISIONS.md 2026-05-05 policy")
        print(f"                Diagnostic unmasked variants also written "
              f"(*_unmasked_*.nii.gz)")
    else:
        print(f"  Tissue:       Cortical + subcortical gray matter only "
              f"(--no-clinical-mask)")
    print(f"  Group:        resolved per-patient (age-matched FM_* band)")

    # Find patients with zAI maps available
    if args.patient:
        patients = [args.patient]
    else:
        if ZAI_PATIENTS_DIR.exists():
            patients = sorted([
                d.name for d in ZAI_PATIENTS_DIR.iterdir()
                if d.is_dir() and
                (d / f"{d.name}_asymmetry_zscore.nii.gz").exists()
            ])
        else:
            patients = []

    print(f"  Patients:     {patients}")
    print("=" * 70)

    all_summaries = []

    for pid in patients:
        # Resolve + load THIS patient's age-matched control band (not a fixed
        # group) so 40-59 / 60-79 patients are not referenced against FM_20_39.
        group = resolve_patient_group(pid)
        gdir = RESULTS_DIR / "groups" / group
        if not (gdir / "consensus_parcellation.nii.gz").exists():
            print(f"  ⚠ {pid}: control band '{group}' not found at {gdir}; skipping.")
            continue
        mean_img = nib.load(str(gdir / "mean_perfusion.nii.gz")).get_fdata(dtype=np.float32)
        brain_mask = nib.load(str(gdir / "brain_mask.nii.gz")).get_fdata().astype(bool)
        parc = nib.load(str(gdir / "consensus_parcellation.nii.gz")).get_fdata().astype(np.int32)
        affine = nib.load(str(gdir / "mean_perfusion.nii.gz")).affine
        gm_mask = build_gray_matter_mask(parc)
        print(f"  [{pid}] band = {group}")
        result = process_patient(
            pid, brain_mask, parc, gm_mask, mean_img,
            affine, CLINICAL_ZAI_THRESHOLD, CLINICAL_MIN_CLUSTER,
            apply_clinical_mask=args.apply_clinical_mask)
        if result is None:
            continue
        summary, cluster_rows = result
        all_summaries.append(summary)

        # Print summary
        print(f"\n  {pid}:")
        print(f"    Gray matter voxels: {summary['gray_matter_voxels']:,}")
        print(f"    Bilateral (both sides): {summary['pct_significant']:.1f}% "
              f"({summary['left_dominant_voxels'] + summary['right_dominant_voxels']:,} voxels, "
              f"{summary['left_dominant_clusters']} L-dom + {summary['right_dominant_clusters']} R-dom clusters)")
        print(f"    Lateralized (dominant side only): {summary['pct_lateralized']:.1f}% "
              f"({summary['lateralized_left_dominant_voxels'] + summary['lateralized_right_dominant_voxels']:,} voxels, "
              f"{summary['lateralized_left_dominant_clusters']} L-dom + "
              f"{summary['lateralized_right_dominant_clusters']} R-dom clusters)")

        if cluster_rows:
            print(f"\n    {'Dir':<14} {'#':>3} {'Size':>7} {'mm3':>8} "
                  f"{'Mean zAI':>9} {'Peak zAI':>9} {'Region'}")
            print(f"    {'-'*70}")
            for row in sorted(cluster_rows, key=lambda x: -x["size_voxels"])[:20]:
                print(f"    {row['direction']:<14} {row['cluster_id']:>3} "
                      f"{row['size_voxels']:>7} {row['size_mm3']:>8.1f} "
                      f"{row['mean_zai']:>9.2f} {row['peak_zai']:>9.2f} "
                      f"{row['primary_region']}")

        # Montage
        fig_path = plot_clinical_montage(pid, mean_img, brain_mask, gm_mask, affine)
        print(f"    Montage: {fig_path.name}")

    # FSLeyes commands
    if all_summaries:
        print(f"\n{'='*70}")
        print("  FSLeyes commands - CLINICAL zAI MAPS")
        print(f"{'='*70}")
        mean_f = GROUP_DIR / "mean_perfusion.nii.gz"
        for s in all_summaries:
            pid = s["patient_id"]
            pdir = OUTPUT_DIR / pid
            lat_left_f = pdir / f"{pid}_clinical_zai_lateralized_left_dominant.nii.gz"
            lat_right_f = pdir / f"{pid}_clinical_zai_lateralized_right_dominant.nii.gz"
            lat_sig_f = pdir / f"{pid}_clinical_zai_lateralized_significant.nii.gz"
            gm_zai_f = pdir / f"{pid}_clinical_zai_gm.nii.gz"

            print(f"\n  # {pid} - lateralized clusters (dominant hemisphere only):")
            print(f"  # red = LEFT side high blood flow (suggests R EZ if interictal)")
            print(f"  # blue = RIGHT side high blood flow (suggests L EZ if interictal)")
            print(f"  fsleyes '{mean_f}' \\")
            if lat_left_f.exists():
                print(f"    '{lat_left_f}' -cm red -dr 0.5 1 -a 90 "
                      f"-n 'Left-dominant (L>R; suggests R EZ if interictal)' \\")
            if lat_right_f.exists():
                print(f"    '{lat_right_f}' -cm blue -dr 0.5 1 -a 90 "
                      f"-n 'Right-dominant (R>L; suggests L EZ if interictal)' &")

            print(f"\n  # {pid} - lateralized zAI heatmap:")
            print(f"  fsleyes '{mean_f}' \\")
            print(f"    '{lat_sig_f}' -cm brain_colours_diverging_bwr -dr -6 6 -a 80 "
                  f"-n 'Lateralized zAI' &")

    print(f"\n{'='*70}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
