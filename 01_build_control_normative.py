#!/usr/bin/env python3
"""
Voxel-Wise Z-Score Control Comparison Pipeline

Compares each epilepsy patient's ASL perfusion scan against an age/sex-matched
healthy control group.  For every voxel the z-score is:

    z = (patient - group_mean) / group_sd

Significant voxels (|z| >= 1.96, p < 0.05 two-tailed) are grouped into
spatially contiguous clusters and mapped to FreeSurfer brain regions via a
consensus parcellation built from the control subjects.

Usage:
    python 01_build_control_normative.py                          # all matchable patients
    python 01_build_control_normative.py --patient P013           # single patient
    python 01_build_control_normative.py --group F_20_39          # specific group
    python 01_build_control_normative.py --rebuild-group          # force rebuild group stats
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage, stats

# ============================================================================
# CONFIGURATION
# ============================================================================

ZSCORE_THRESHOLD = 1.96       # |z| >= 1.96  →  p < 0.05 two-tailed
BRAIN_MASK_COVERAGE = 0.75    # require 75% of controls to have signal
MIN_CLUSTER_SIZE = 10         # minimum voxels per cluster
CLUSTER_CONNECTIVITY = 2      # 18-connected (faces + edges)
SD_FLOOR = 1e-6               # minimum SD to prevent division by zero
VOXEL_VOL_MM3 = 0.8 ** 3     # 0.512 mm³ per voxel at 0.8 mm isotropic

BASE_DIR = Path(__file__).parent
CONTROLS_BASE_DIR = BASE_DIR / "DatasetControls"
PATIENT_DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR = BASE_DIR / "results_zscore"
CLINICAL_SPREADSHEET = BASE_DIR / "clinical_spreadsheet.xlsx"

# Symmetric-space mode (2026-06-22 zAI over-dispersion fix). When enabled,
# perfusion and parcellation are read from the left-right symmetric template
# space produced by symreg/reg_subject.sh (flirt+fnirt onto a flip-symmetric
# T1 template), so the voxel mirror-AI in 02_compute_zai.py compares
# anatomically homologous voxels. Output paths are unchanged: symmetric space
# becomes the new canonical (original-space outputs archived separately).
SYM_MODE = False
SYM_DIR = BASE_DIR / "symreg" / "sym_perf"

# ============================================================================
# GROUP DEFINITIONS  –  extend as new controls arrive
# ============================================================================

GROUP_DEFINITIONS = {
    "F_20_39": {
        "subdirs": ["20_29F", "30_39F"],
        "sex": "F",
        "age_range": (20, 39),
        "description": "Females aged 20-39",
    },
    "F_40_59": {
        "subdirs": ["40_49F", "50_59F"],
        "sex": "F",
        "age_range": (40, 59),
        "description": "Females aged 40-59",
    },
    "F_60_79": {
        "subdirs": ["60_69F", "70_79F"],
        "sex": "F",
        "age_range": (60, 79),
        "description": "Females aged 60-79",
    },
    "M_20_39": {
        "subdirs": ["20_29M", "30_39M"],
        "sex": "M",
        "age_range": (20, 39),
        "description": "Males aged 20-39",
    },
    "M_40_59": {
        "subdirs": ["40_49M", "50_59M"],
        "sex": "M",
        "age_range": (40, 59),
        "description": "Males aged 40-59",
    },
    "M_60_79": {
        "subdirs": ["60_69M", "70_79M"],
        "sex": "M",
        "age_range": (60, 79),
        "description": "Males aged 60-79",
    },
    # Thesis comparator groups (Rai 2026, Table 10): 3 sex-pooled 20-year bands,
    # n=60 each. The new HCP-Aging/Development controls live FLAT under these dirs
    # (DatasetControls/FM_20_39/<ID>/ ...), so each group reads a single subdir.
    "FM_20_39": {
        "subdirs": ["FM_20_39"],
        "sex": "FM",
        "age_range": (20, 39),
        "description": "All subjects aged 20-39 (n=60, thesis Group 1)",
    },
    "FM_40_59": {
        "subdirs": ["FM_40_59"],
        "sex": "FM",
        "age_range": (40, 59),
        "description": "All subjects aged 40-59 (n=60, thesis Group 2)",
    },
    "FM_60_79": {
        "subdirs": ["FM_60_79"],
        "sex": "FM",
        "age_range": (60, 79),
        "description": "All subjects aged 60-79 (n=60, thesis Group 3)",
    },
}

# NOTE: the sex-specific F_*/M_* groups above point at per-sex decade dirs
# (e.g. 20_29F) that are NOT present in the current dataset — the new controls
# are stored sex-pooled. Those groups are therefore reported as "unavailable"
# (discover_subjects skips missing subdirs). A sex-specific sensitivity analysis
# would require per-control sex labels to split the FM_* dirs.

# ============================================================================
# FreeSurfer region names  (from voxel_roi_extraction.py)
# ============================================================================

REGION_NAMES = {
    10: "Left-Thalamus-Proper", 49: "Right-Thalamus-Proper",
    17: "Left-Hippocampus", 53: "Right-Hippocampus",
    18: "Left-Amygdala", 54: "Right-Amygdala",
    11: "Left-Caudate", 50: "Right-Caudate",
    12: "Left-Putamen", 51: "Right-Putamen",
    13: "Left-Pallidum", 52: "Right-Pallidum",
    26: "Left-Accumbens-area", 58: "Right-Accumbens-area",
    28: "Left-VentralDC", 60: "Right-VentralDC",
    4: "Left-Lateral-Ventricle", 43: "Right-Lateral-Ventricle",
    2: "Left-Cerebral-White-Matter", 41: "Right-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex", 42: "Right-Cerebral-Cortex",
    7: "Left-Cerebellum-White-Matter", 46: "Right-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex", 47: "Right-Cerebellum-Cortex",
    16: "Brain-Stem",
    1001: "ctx-lh-bankssts", 2001: "ctx-rh-bankssts",
    1002: "ctx-lh-caudalanteriorcingulate", 2002: "ctx-rh-caudalanteriorcingulate",
    1003: "ctx-lh-caudalmiddlefrontal", 2003: "ctx-rh-caudalmiddlefrontal",
    1005: "ctx-lh-cuneus", 2005: "ctx-rh-cuneus",
    1006: "ctx-lh-entorhinal", 2006: "ctx-rh-entorhinal",
    1007: "ctx-lh-fusiform", 2007: "ctx-rh-fusiform",
    1008: "ctx-lh-inferiorparietal", 2008: "ctx-rh-inferiorparietal",
    1009: "ctx-lh-inferiortemporal", 2009: "ctx-rh-inferiortemporal",
    1010: "ctx-lh-isthmuscingulate", 2010: "ctx-rh-isthmuscingulate",
    1011: "ctx-lh-lateraloccipital", 2011: "ctx-rh-lateraloccipital",
    1012: "ctx-lh-lateralorbitofrontal", 2012: "ctx-rh-lateralorbitofrontal",
    1013: "ctx-lh-lingual", 2013: "ctx-rh-lingual",
    1014: "ctx-lh-medialorbitofrontal", 2014: "ctx-rh-medialorbitofrontal",
    1015: "ctx-lh-middletemporal", 2015: "ctx-rh-middletemporal",
    1016: "ctx-lh-parahippocampal", 2016: "ctx-rh-parahippocampal",
    1017: "ctx-lh-paracentral", 2017: "ctx-rh-paracentral",
    1018: "ctx-lh-parsopercularis", 2018: "ctx-rh-parsopercularis",
    1019: "ctx-lh-parsorbitalis", 2019: "ctx-rh-parsorbitalis",
    1020: "ctx-lh-parstriangularis", 2020: "ctx-rh-parstriangularis",
    1021: "ctx-lh-pericalcarine", 2021: "ctx-rh-pericalcarine",
    1022: "ctx-lh-postcentral", 2022: "ctx-rh-postcentral",
    1023: "ctx-lh-posteriorcingulate", 2023: "ctx-rh-posteriorcingulate",
    1024: "ctx-lh-precentral", 2024: "ctx-rh-precentral",
    1025: "ctx-lh-precuneus", 2025: "ctx-rh-precuneus",
    1026: "ctx-lh-rostralanteriorcingulate", 2026: "ctx-rh-rostralanteriorcingulate",
    1027: "ctx-lh-rostralmiddlefrontal", 2027: "ctx-rh-rostralmiddlefrontal",
    1028: "ctx-lh-superiorfrontal", 2028: "ctx-rh-superiorfrontal",
    1029: "ctx-lh-superiorparietal", 2029: "ctx-rh-superiorparietal",
    1030: "ctx-lh-superiortemporal", 2030: "ctx-rh-superiortemporal",
    1031: "ctx-lh-supramarginal", 2031: "ctx-rh-supramarginal",
    1032: "ctx-lh-frontalpole", 2032: "ctx-rh-frontalpole",
    1033: "ctx-lh-temporalpole", 2033: "ctx-rh-temporalpole",
    1034: "ctx-lh-transversetemporal", 2034: "ctx-rh-transversetemporal",
    1035: "ctx-lh-insula", 2035: "ctx-rh-insula",
}


# ============================================================================
# CONTROL GROUP BUILDER
# ============================================================================

class ControlGroupBuilder:
    """Build voxel-wise mean/SD maps and consensus parcellation from controls."""

    def __init__(self, group_key):
        if group_key not in GROUP_DEFINITIONS:
            raise ValueError(
                f"Unknown group '{group_key}'. "
                f"Available: {list(GROUP_DEFINITIONS.keys())}"
            )
        self.group_key = group_key
        self.group_def = GROUP_DEFINITIONS[group_key]
        self.output_dir = OUTPUT_DIR / "groups" / group_key
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def discover_subjects(self):
        """Find all control subject directories for this group."""
        subjects = []
        for subdir_name in self.group_def["subdirs"]:
            subdir = CONTROLS_BASE_DIR / subdir_name
            if not subdir.exists():
                print(f"  ⚠  Subdirectory not found (skipping): {subdir}")
                continue
            for entry in sorted(subdir.iterdir()):
                if entry.is_dir():
                    subjects.append(entry)
        return subjects

    # ------------------------------------------------------------------
    @staticmethod
    def _find_perfusion_file(subject_dir):
        """Return path to the MNI-space upsampled perfusion NIfTI."""
        if SYM_MODE:
            p = SYM_DIR / f"{subject_dir.name}_perf_sym.nii.gz"
            if not p.exists():
                raise FileNotFoundError(f"No symmetric perfusion for {subject_dir.name}: {p}")
            return p
        candidates = list(subject_dir.glob("*_MNISpace_perfusion_calib_upsampled.nii.gz"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No MNISpace upsampled perfusion file in {subject_dir}"
            )
        raise FileNotFoundError(
            f"Multiple MNISpace upsampled perfusion files in {subject_dir}: {candidates}"
        )

    @staticmethod
    def _find_parcellation_file(subject_dir):
        """Return path to the aparc+aseg NIfTI."""
        if SYM_MODE:
            p = SYM_DIR / f"{subject_dir.name}_aparc_sym.nii.gz"
            if not p.exists():
                raise FileNotFoundError(f"No symmetric aparc for {subject_dir.name}: {p}")
            return p
        parc = subject_dir / "aparc+aseg.nii.gz"
        if parc.exists():
            return parc
        raise FileNotFoundError(f"No aparc+aseg.nii.gz in {subject_dir}")

    # ------------------------------------------------------------------
    def build_group_stats(self):
        """Compute mean, SD, count, and brain mask using Welford's algorithm."""
        subjects = self.discover_subjects()
        n_subjects = len(subjects)
        if n_subjects == 0:
            raise RuntimeError(
                f"No subjects found for group '{self.group_key}'. "
                f"Looked in: {self.group_def['subdirs']}"
            )
        print(f"\n{'='*70}")
        print(f"Building group statistics for {self.group_key}")
        print(f"  {self.group_def['description']}")
        print(f"  Subjects found: {n_subjects}")
        print(f"{'='*70}")

        # -- Welford's online algorithm --
        reference_img = None
        mean = None
        M2 = None
        count = None
        subject_ids = []

        for i, subj_dir in enumerate(subjects):
            subj_id = subj_dir.name
            subject_ids.append(subj_id)
            perf_path = self._find_perfusion_file(subj_dir)
            img = nib.load(str(perf_path))
            data = img.get_fdata(dtype=np.float32)

            if mean is None:
                shape = data.shape
                reference_img = img
                mean = np.zeros(shape, dtype=np.float64)
                M2 = np.zeros(shape, dtype=np.float64)
                count = np.zeros(shape, dtype=np.int32)

            valid = data > 0
            count[valid] += 1

            delta = np.zeros(shape, dtype=np.float64)
            delta[valid] = data[valid].astype(np.float64) - mean[valid]
            mean[valid] += delta[valid] / count[valid]

            delta2 = np.zeros(shape, dtype=np.float64)
            delta2[valid] = data[valid].astype(np.float64) - mean[valid]
            M2[valid] += delta[valid] * delta2[valid]

            print(f"  [{i+1:3d}/{n_subjects}] {subj_id}  "
                  f"non-zero voxels: {valid.sum():,}")

        # -- Compute sample SD --
        sd = np.zeros_like(mean)
        valid_for_sd = count >= 2
        sd[valid_for_sd] = np.sqrt(M2[valid_for_sd] / (count[valid_for_sd] - 1))

        # -- Brain mask: coverage >= threshold AND SD > floor --
        coverage_fraction = count / n_subjects
        brain_mask = (coverage_fraction >= BRAIN_MASK_COVERAGE) & (sd > SD_FLOOR)

        # -- Zero out outside mask --
        mean[~brain_mask] = 0
        sd[~brain_mask] = 0

        # -- Save outputs --
        affine = reference_img.affine
        header = reference_img.header.copy()

        def _save_nifti(arr, fname, dtype=np.float32, descrip=""):
            out = nib.Nifti1Image(arr.astype(dtype), affine, header)
            out.header["descrip"] = descrip
            nib.save(out, str(self.output_dir / fname))

        _save_nifti(mean, "mean_perfusion.nii.gz",
                    descrip=f"Group mean CBF - {self.group_key}")
        _save_nifti(sd, "sd_perfusion.nii.gz",
                    descrip=f"Group SD CBF - {self.group_key}")
        _save_nifti(brain_mask, "brain_mask.nii.gz", dtype=np.uint8,
                    descrip=f"Brain mask (>={BRAIN_MASK_COVERAGE*100:.0f}pct coverage)")
        _save_nifti(count, "count_map.nii.gz", dtype=np.int32,
                    descrip="Number of subjects contributing per voxel")

        # -- Metadata --
        metadata = {
            "group_key": self.group_key,
            "description": self.group_def["description"],
            "n_subjects": n_subjects,
            "subject_ids": subject_ids,
            "brain_mask_coverage_threshold": BRAIN_MASK_COVERAGE,
            "sd_floor": SD_FLOOR,
            "brain_mask_voxels": int(brain_mask.sum()),
            "total_voxels": int(np.prod(brain_mask.shape)),
            "mean_cbf_in_mask": float(mean[brain_mask].mean()) if brain_mask.any() else 0,
            "built_at": datetime.now().isoformat(),
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # -- Summary --
        mask_pct = 100 * brain_mask.sum() / brain_mask.size
        print(f"\n  ✓ Group stats saved to: {self.output_dir}")
        print(f"    Brain mask: {brain_mask.sum():,} voxels ({mask_pct:.1f}%)")
        print(f"    Mean CBF in mask: {metadata['mean_cbf_in_mask']:.2f} ml/100g/min")
        print(f"    SD range in mask: [{sd[brain_mask].min():.2f}, {sd[brain_mask].max():.2f}]")

        return mean, sd, brain_mask, count, reference_img

    # ------------------------------------------------------------------
    def build_consensus_parcellation(self):
        """Majority-vote parcellation across all control subjects."""
        subjects = self.discover_subjects()
        n_subjects = len(subjects)
        print(f"\n  Building consensus parcellation from {n_subjects} subjects...")

        all_parcs = []
        for subj_dir in subjects:
            parc_path = self._find_parcellation_file(subj_dir)
            parc_data = nib.load(str(parc_path)).get_fdata().astype(np.int16)
            all_parcs.append(parc_data)

        stacked = np.stack(all_parcs, axis=-1)  # (227,272,227,N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            consensus, _ = stats.mode(stacked, axis=-1, keepdims=False)
        consensus = np.squeeze(consensus).astype(np.int32)

        # Save using affine from first subject
        ref_img = nib.load(str(self._find_parcellation_file(subjects[0])))
        out_img = nib.Nifti1Image(consensus, ref_img.affine, ref_img.header)
        out_img.header["descrip"] = f"Consensus aparc+aseg - {self.group_key}"
        out_path = self.output_dir / "consensus_parcellation.nii.gz"
        nib.save(out_img, str(out_path))

        n_labels = len(np.unique(consensus[consensus > 0]))
        print(f"  ✓ Consensus parcellation saved: {n_labels} unique labels")
        return consensus

    # ------------------------------------------------------------------
    def load_or_build(self, force_rebuild=False):
        """Load cached group stats or build from scratch."""
        meta_path = self.output_dir / "metadata.json"
        if meta_path.exists() and not force_rebuild:
            print(f"\n  Loading cached group stats for {self.group_key}...")
            mean = nib.load(str(self.output_dir / "mean_perfusion.nii.gz"))
            sd_img = nib.load(str(self.output_dir / "sd_perfusion.nii.gz"))
            mask_img = nib.load(str(self.output_dir / "brain_mask.nii.gz"))

            mean_data = mean.get_fdata(dtype=np.float32)
            sd_data = sd_img.get_fdata(dtype=np.float32)
            mask_data = mask_img.get_fdata().astype(bool)

            with open(meta_path) as f:
                metadata = json.load(f)
            print(f"    {metadata['n_subjects']} subjects, "
                  f"{metadata['brain_mask_voxels']:,} mask voxels")

            return mean_data, sd_data, mask_data, mean
        else:
            mean_data, sd_data, mask_data, _, ref_img = self.build_group_stats()
            # Build consensus parcellation too
            if not (self.output_dir / "consensus_parcellation.nii.gz").exists() or force_rebuild:
                self.build_consensus_parcellation()
            return mean_data, sd_data, mask_data, ref_img


# ============================================================================
# CLUSTER ANALYZER
# ============================================================================

class ClusterAnalyzer:
    """Connected-component cluster analysis on thresholded z-score maps."""

    def __init__(self, min_size=MIN_CLUSTER_SIZE, connectivity=CLUSTER_CONNECTIVITY):
        self.min_size = min_size
        self.structure = ndimage.generate_binary_structure(3, connectivity)

    def find_clusters(self, binary_map, zscore_map, affine):
        """Label connected components and compute per-cluster statistics.

        Returns list of dicts, one per surviving cluster.
        """
        labeled, n_raw = ndimage.label(binary_map, structure=self.structure)

        clusters = []
        for cid in range(1, n_raw + 1):
            mask_c = labeled == cid
            size = int(mask_c.sum())
            if size < self.min_size:
                continue

            z_vals = zscore_map[mask_c]
            com = ndimage.center_of_mass(mask_c)
            centroid_ijk = tuple(int(round(c)) for c in com)
            centroid_mni = self._ijk_to_mni(centroid_ijk, affine)

            peak_idx = np.argmax(np.abs(z_vals))
            peak_coords = np.argwhere(mask_c)
            peak_ijk = tuple(peak_coords[peak_idx])
            peak_mni = self._ijk_to_mni(peak_ijk, affine)

            clusters.append({
                "cluster_id": len(clusters) + 1,
                "size_voxels": size,
                "size_mm3": round(size * VOXEL_VOL_MM3, 2),
                "centroid_ijk": centroid_ijk,
                "centroid_mni": tuple(round(c, 1) for c in centroid_mni),
                "peak_z": round(float(z_vals[peak_idx]), 3),
                "mean_z": round(float(z_vals.mean()), 3),
                "peak_ijk": peak_ijk,
                "peak_mni": tuple(round(c, 1) for c in peak_mni),
                "mask": mask_c,
            })

        return clusters

    @staticmethod
    def _ijk_to_mni(ijk, affine):
        """Convert voxel indices to MNI coordinates."""
        coord = affine @ np.array([*ijk, 1])
        return tuple(coord[:3])

    def create_cluster_label_map(self, clusters, shape):
        """Build a 3D volume with cluster IDs."""
        label_map = np.zeros(shape, dtype=np.int32)
        for cl in clusters:
            label_map[cl["mask"]] = cl["cluster_id"]
        return label_map


# ============================================================================
# REGION MAPPER
# ============================================================================

class RegionMapper:
    """Map clusters to FreeSurfer brain regions via a parcellation."""

    def __init__(self, parcellation_data):
        self.parc = parcellation_data

    def map_cluster(self, cluster_mask):
        """Return region overlap for a single cluster.

        Returns list of (label, name, n_voxels, pct) sorted by overlap.
        """
        labels_in_cluster = self.parc[cluster_mask]
        labels_in_cluster = labels_in_cluster[labels_in_cluster > 0]
        if len(labels_in_cluster) == 0:
            return [(-1, "Unknown", int(cluster_mask.sum()), 100.0)]

        unique, counts = np.unique(labels_in_cluster, return_counts=True)
        total = counts.sum()
        result = []
        for lab, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
            name = REGION_NAMES.get(int(lab), f"Label-{int(lab)}")
            result.append((int(lab), name, int(cnt), round(100 * cnt / total, 1)))
        return result

    def primary_region(self, cluster_mask):
        """Return the name of the dominant region for a cluster."""
        overlap = self.map_cluster(cluster_mask)
        return overlap[0][1] if overlap else "Unknown"


# ============================================================================
# PATIENT Z-SCORE ANALYZER
# ============================================================================

class PatientZScoreAnalyzer:
    """Compute z-score maps for a patient against a control group."""

    def __init__(self, group_mean, group_sd, brain_mask, ref_img,
                 parcellation_data, group_key):
        self.group_mean = group_mean.astype(np.float64)
        self.group_sd = group_sd.astype(np.float64)
        self.brain_mask = brain_mask
        self.ref_img = ref_img
        self.affine = ref_img.affine if hasattr(ref_img, 'affine') else ref_img.header.get_best_affine()
        self.group_key = group_key
        self.cluster_analyzer = ClusterAnalyzer()
        self.region_mapper = RegionMapper(parcellation_data)

    @staticmethod
    def _find_patient_perfusion(patient_dir):
        """Find the patient's perfusion NIfTI (resampled-to-T1w, actually MNI)."""
        if SYM_MODE:
            p = SYM_DIR / f"{patient_dir.name}_perf_sym.nii.gz"
            if not p.exists():
                raise FileNotFoundError(f"No symmetric perfusion for {patient_dir.name}: {p}")
            return p
        candidates = list(patient_dir.glob("*_perfusion_calib_resampled_to_T1w.nii.gz"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No perfusion_calib_resampled_to_T1w file in {patient_dir}"
            )
        raise FileNotFoundError(
            f"Multiple perfusion files in {patient_dir}: {candidates}"
        )

    def process_patient(self, patient_id):
        """Full pipeline for one patient: z-score → threshold → cluster → regions."""
        patient_dir = PATIENT_DATASET_DIR / patient_id
        if not patient_dir.exists():
            print(f"  ✗ Patient directory not found: {patient_dir}")
            return None

        out_dir = OUTPUT_DIR / "patients" / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Processing {patient_id} vs {self.group_key}...")

        # --- Load patient perfusion ---
        perf_path = self._find_patient_perfusion(patient_dir)
        patient_data = nib.load(str(perf_path)).get_fdata(dtype=np.float32).astype(np.float64)

        # --- Compute z-score ---
        # Only compute where: brain mask is valid, SD is sufficient,
        # AND patient has non-zero perfusion (avoids false hypo-perfusion
        # at voxels where patient coverage is missing but controls have signal)
        zscore_map = np.zeros_like(patient_data)
        valid = self.brain_mask & (self.group_sd > SD_FLOOR) & (patient_data > 0)
        zscore_map[valid] = (patient_data[valid] - self.group_mean[valid]) / self.group_sd[valid]

        # Clip extreme z-scores (from voxels with very small SD)
        zscore_map = np.clip(zscore_map, -20, 20)

        # --- Threshold ---
        hyper_mask = (zscore_map > ZSCORE_THRESHOLD) & valid
        hypo_mask = (zscore_map < -ZSCORE_THRESHOLD) & valid

        # --- Cluster analysis ---
        hyper_clusters = self.cluster_analyzer.find_clusters(hyper_mask, zscore_map, self.affine)
        hypo_clusters = self.cluster_analyzer.find_clusters(hypo_mask, zscore_map, self.affine)

        # --- Region mapping ---
        all_cluster_rows = []
        all_region_rows = []

        for direction, clusters in [("hyper", hyper_clusters), ("hypo", hypo_clusters)]:
            for cl in clusters:
                primary = self.region_mapper.primary_region(cl["mask"])
                all_cluster_rows.append({
                    "patient_id": patient_id,
                    "group": self.group_key,
                    "direction": direction,
                    "cluster_id": cl["cluster_id"],
                    "size_voxels": cl["size_voxels"],
                    "size_mm3": cl["size_mm3"],
                    "mean_z": cl["mean_z"],
                    "peak_z": cl["peak_z"],
                    "centroid_mni_x": cl["centroid_mni"][0],
                    "centroid_mni_y": cl["centroid_mni"][1],
                    "centroid_mni_z": cl["centroid_mni"][2],
                    "peak_mni_x": cl["peak_mni"][0],
                    "peak_mni_y": cl["peak_mni"][1],
                    "peak_mni_z": cl["peak_mni"][2],
                    "primary_region": primary,
                })

                overlap = self.region_mapper.map_cluster(cl["mask"])
                for lab, name, nvox, pct in overlap:
                    all_region_rows.append({
                        "patient_id": patient_id,
                        "direction": direction,
                        "cluster_id": cl["cluster_id"],
                        "region_label": lab,
                        "region_name": name,
                        "overlap_voxels": nvox,
                        "overlap_pct": pct,
                    })

        # --- Save NIfTI outputs ---
        def _save(arr, fname, dtype=np.float32, descrip=""):
            img = nib.Nifti1Image(arr.astype(dtype), self.affine)
            img.header["descrip"] = descrip
            nib.save(img, str(out_dir / fname))

        prefix = f"{patient_id}_vs_{self.group_key}"
        _save(zscore_map, f"{prefix}_zscore.nii.gz",
              descrip=f"Z-score map {patient_id} vs {self.group_key}")
        _save(hyper_mask, f"{prefix}_hyper_perfusion.nii.gz", dtype=np.uint8,
              descrip=f"Hyper-perfusion z>{ZSCORE_THRESHOLD}")
        _save(hypo_mask, f"{prefix}_hypo_perfusion.nii.gz", dtype=np.uint8,
              descrip=f"Hypo-perfusion z<-{ZSCORE_THRESHOLD}")

        if hyper_clusters:
            hyper_label = self.cluster_analyzer.create_cluster_label_map(
                hyper_clusters, zscore_map.shape)
            _save(hyper_label, f"{prefix}_hyper_clusters.nii.gz", dtype=np.int32,
                  descrip="Hyper-perfusion cluster labels")

        if hypo_clusters:
            hypo_label = self.cluster_analyzer.create_cluster_label_map(
                hypo_clusters, zscore_map.shape)
            _save(hypo_label, f"{prefix}_hypo_clusters.nii.gz", dtype=np.int32,
                  descrip="Hypo-perfusion cluster labels")

        # --- Save CSV reports ---
        if all_cluster_rows:
            pd.DataFrame(all_cluster_rows).to_csv(
                out_dir / f"{prefix}_cluster_report.csv", index=False)
        if all_region_rows:
            pd.DataFrame(all_region_rows).to_csv(
                out_dir / f"{prefix}_region_overlap.csv", index=False)

        # --- Save summary JSON ---
        z_in_mask = zscore_map[valid]
        summary = {
            "patient_id": patient_id,
            "group": self.group_key,
            "zscore_threshold": ZSCORE_THRESHOLD,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "mask_voxels": int(valid.sum()),
            "patient_nonzero_in_mask": int((patient_data[valid] > 0).sum()),
            "zscore_mean": round(float(z_in_mask.mean()), 4),
            "zscore_std": round(float(z_in_mask.std()), 4),
            "zscore_min": round(float(z_in_mask.min()), 4),
            "zscore_max": round(float(z_in_mask.max()), 4),
            "hyper_voxels_raw": int(hyper_mask.sum()),
            "hypo_voxels_raw": int(hypo_mask.sum()),
            "hyper_clusters": len(hyper_clusters),
            "hypo_clusters": len(hypo_clusters),
            "hyper_cluster_voxels": sum(c["size_voxels"] for c in hyper_clusters),
            "hypo_cluster_voxels": sum(c["size_voxels"] for c in hypo_clusters),
            "processed_at": datetime.now().isoformat(),
        }
        with open(out_dir / f"{prefix}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # --- Print summary ---
        print(f"    Z-score range: [{summary['zscore_min']:.2f}, {summary['zscore_max']:.2f}]")
        print(f"    Mean z in mask: {summary['zscore_mean']:.3f}")
        print(f"    Hyper-perfusion: {summary['hyper_voxels_raw']:,} voxels → "
              f"{len(hyper_clusters)} clusters ({summary['hyper_cluster_voxels']:,} voxels)")
        print(f"    Hypo-perfusion:  {summary['hypo_voxels_raw']:,} voxels → "
              f"{len(hypo_clusters)} clusters ({summary['hypo_cluster_voxels']:,} voxels)")

        if all_cluster_rows:
            print(f"\n    {'Dir':<6} {'#':>3} {'Size':>8} {'Mean z':>8} {'Peak z':>8} {'Primary Region'}")
            print(f"    {'─'*60}")
            for row in all_cluster_rows:
                print(f"    {row['direction']:<6} {row['cluster_id']:>3} "
                      f"{row['size_voxels']:>7}v {row['mean_z']:>8.3f} "
                      f"{row['peak_z']:>8.3f} {row['primary_region']}")

        # --- Visualization ---
        self._plot_zscore_montage(zscore_map, valid, patient_id, out_dir, prefix)

        return summary

    def _plot_zscore_montage(self, zscore_map, mask, patient_id, out_dir, prefix):
        """Save axial slice montage of z-scores overlaid on group mean."""
        # Find slices that contain brain
        z_indices = np.where(mask.any(axis=(0, 1)))[0]
        if len(z_indices) == 0:
            return
        n_slices = min(12, len(z_indices))
        selected = np.linspace(z_indices[0], z_indices[-1], n_slices, dtype=int)

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f"{patient_id} vs {self.group_key} — Z-Score Map (threshold ±{ZSCORE_THRESHOLD})",
                     fontsize=14, fontweight="bold")

        for idx, ax in enumerate(axes.flat):
            if idx >= n_slices:
                ax.axis("off")
                continue
            sl = selected[idx]
            bg = self.group_mean[:, :, sl].T
            zs = zscore_map[:, :, sl].T
            m = mask[:, :, sl].T

            ax.imshow(bg, cmap="gray", origin="lower", aspect="equal")
            zs_masked = np.ma.masked_where(~m | (np.abs(zs) < ZSCORE_THRESHOLD), zs)
            ax.imshow(zs_masked, cmap="RdBu_r", origin="lower", aspect="equal",
                      vmin=-4, vmax=4, alpha=0.7)
            ax.set_title(f"z = {sl}", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        fig_path = out_dir / f"{prefix}_zscore_montage.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    ✓ Montage saved: {fig_path.name}")


# ============================================================================
# TOP-LEVEL ORCHESTRATOR
# ============================================================================

class ControlZScoreAnalysis:
    """Orchestrate the full z-score comparison pipeline."""

    def __init__(self, group_key=None, patient_id=None, rebuild=False):
        self.rebuild = rebuild
        self.specific_patient = patient_id
        self.demographics = self._load_demographics()

        if group_key:
            self.groups_to_run = [group_key]
        elif patient_id:
            self.groups_to_run = self._groups_for_patient(patient_id)
        else:
            self.groups_to_run = self._available_groups()

    def _load_demographics(self):
        """Load patient demographics from clinical spreadsheet."""
        if not CLINICAL_SPREADSHEET.exists():
            print(f"  ⚠  Clinical spreadsheet not found: {CLINICAL_SPREADSHEET}")
            return {}
        try:
            df = pd.read_excel(str(CLINICAL_SPREADSHEET))
            # Normalize column names
            df.columns = [c.strip().upper() for c in df.columns]
            id_col = [c for c in df.columns if "ID" in c or "PATIENT" in c or "SUBJECT" in c]
            age_col = [c for c in df.columns if "AGE" in c]
            sex_col = [c for c in df.columns if "SEX" in c or "GENDER" in c]

            if not id_col or not age_col or not sex_col:
                print(f"  ⚠  Could not identify ID/AGE/SEX columns in spreadsheet")
                return {}

            demos = {}
            for _, row in df.iterrows():
                pid = str(row[id_col[0]]).strip()
                # Normalize ID: "sub-P013" → "P013"
                if pid.startswith("sub-"):
                    pid = pid[4:]
                age = int(row[age_col[0]])
                sex = str(row[sex_col[0]]).strip().upper()
                if sex.startswith("F"):
                    sex = "F"
                elif sex.startswith("M"):
                    sex = "M"
                demos[pid] = {"age": age, "sex": sex}
            return demos
        except Exception as e:
            print(f"  ⚠  Error reading demographics: {e}")
            return {}

    def _groups_for_patient(self, patient_id):
        """Determine which control group(s) apply for a patient."""
        if patient_id not in self.demographics:
            print(f"  ⚠  No demographics for {patient_id}, defaulting to all available groups")
            return self._available_groups()

        demo = self.demographics[patient_id]
        groups = []
        for gkey, gdef in GROUP_DEFINITIONS.items():
            lo, hi = gdef["age_range"]
            sex_match = gdef["sex"] == "FM" or gdef["sex"] == demo["sex"]
            age_match = lo <= demo["age"] <= hi
            if sex_match and age_match:
                groups.append(gkey)
        return groups if groups else self._available_groups()

    def _available_groups(self):
        """Return group keys whose control subdirectories actually exist."""
        available = []
        for gkey, gdef in GROUP_DEFINITIONS.items():
            has_data = any(
                (CONTROLS_BASE_DIR / sd).exists() and
                any((CONTROLS_BASE_DIR / sd).iterdir())
                for sd in gdef["subdirs"]
                if (CONTROLS_BASE_DIR / sd).exists()
            )
            if has_data:
                available.append(gkey)
        return available

    def _patients_for_group(self, group_key):
        """Return patient IDs that match this control group."""
        if self.specific_patient:
            return [self.specific_patient]

        gdef = GROUP_DEFINITIONS[group_key]
        lo, hi = gdef["age_range"]
        matched = []
        for pid, demo in self.demographics.items():
            patient_dir = PATIENT_DATASET_DIR / pid
            if not patient_dir.exists():
                continue
            sex_match = gdef["sex"] == "FM" or gdef["sex"] == demo["sex"]
            age_match = lo <= demo["age"] <= hi
            if sex_match and age_match:
                # In symmetric-space mode, skip patients without a registered
                # symmetric perfusion (e.g. P014, excluded from the cohort).
                if SYM_MODE and not (SYM_DIR / f"{pid}_perf_sym.nii.gz").exists():
                    continue
                matched.append(pid)
        return sorted(matched)

    def run(self):
        """Execute the full pipeline."""
        print("\n" + "=" * 70)
        print("  VOXEL-WISE Z-SCORE CONTROL COMPARISON")
        print("=" * 70)
        print(f"  Threshold: |z| >= {ZSCORE_THRESHOLD} (p < 0.05)")
        print(f"  Min cluster size: {MIN_CLUSTER_SIZE} voxels")
        print(f"  Groups to process: {self.groups_to_run}")

        if not self.groups_to_run:
            print("\n  ✗ No control groups available. Check DatasetControls directory.")
            return

        all_summaries = []

        for group_key in self.groups_to_run:
            # -- Step 1 & 2: Build or load group stats --
            builder = ControlGroupBuilder(group_key)
            mean_data, sd_data, mask_data, ref_img = builder.load_or_build(
                force_rebuild=self.rebuild)

            # Load consensus parcellation
            parc_path = builder.output_dir / "consensus_parcellation.nii.gz"
            if parc_path.exists():
                parc_data = nib.load(str(parc_path)).get_fdata().astype(np.int32)
            else:
                print("  Building consensus parcellation...")
                parc_data = builder.build_consensus_parcellation()

            # -- Step 3-6: Process patients --
            patients = self._patients_for_group(group_key)
            if not patients:
                print(f"\n  No matchable patients for {group_key}")
                continue

            print(f"\n  Patients for {group_key}: {patients}")

            analyzer = PatientZScoreAnalyzer(
                mean_data, sd_data, mask_data, ref_img, parc_data, group_key)

            for pid in patients:
                summary = analyzer.process_patient(pid)
                if summary:
                    all_summaries.append(summary)

        # -- Summary CSV --
        if all_summaries:
            summary_dir = OUTPUT_DIR / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(all_summaries)
            csv_path = summary_dir / "all_patients_zscore_summary.csv"
            df.to_csv(str(csv_path), index=False)
            print(f"\n  ✓ Summary saved: {csv_path}")

        # -- FSLeyes commands --
        self._print_fsleyes_commands(all_summaries)

        print(f"\n{'='*70}")
        print(f"  DONE — Results in: {OUTPUT_DIR}")
        print(f"{'='*70}\n")

    def _print_fsleyes_commands(self, summaries):
        """Print FSLeyes commands for viewing results."""
        if not summaries:
            return

        print(f"\n{'='*70}")
        print("  FSLeyes visualization commands")
        print(f"{'='*70}")

        for s in summaries:
            pid = s["patient_id"]
            gkey = s["group"]
            prefix = f"{pid}_vs_{gkey}"
            pdir = OUTPUT_DIR / "patients" / pid
            gdir = OUTPUT_DIR / "groups" / gkey

            mean_f = gdir / "mean_perfusion.nii.gz"
            zscore_f = pdir / f"{prefix}_zscore.nii.gz"
            hyper_f = pdir / f"{prefix}_hyper_clusters.nii.gz"
            hypo_f = pdir / f"{prefix}_hypo_clusters.nii.gz"

            cmd = f"  fsleyes '{mean_f}' \\\n"
            cmd += f"    '{zscore_f}' -cm red-yellow -nr -dr -4 4 -a 60 -n 'Z-score' \\\n"
            if hyper_f.exists():
                cmd += f"    '{hyper_f}' -cm hot -dr 0.5 10 -a 80 -n 'Hyper clusters' \\\n"
            if hypo_f.exists():
                cmd += f"    '{hypo_f}' -cm cool -dr 0.5 10 -a 80 -n 'Hypo clusters' \\\n"
            cmd = cmd.rstrip(" \\\n") + " &"

            print(f"\n  # {pid} vs {gkey}")
            print(cmd)


# ============================================================================
# MAIN
# ============================================================================

def main():
    global ZSCORE_THRESHOLD, MIN_CLUSTER_SIZE, SYM_MODE

    parser = argparse.ArgumentParser(
        description="Voxel-wise z-score comparison: patients vs age/sex-matched controls")
    parser.add_argument("--sym", action="store_true",
                        help="Use left-right symmetric-template-registered perfusion/parcellation "
                             "(symreg/sym_perf/) — required for the corrected voxel mirror-AI.")
    parser.add_argument("--patient", "-p", type=str, default=None,
                        help="Process a single patient (e.g., P013)")
    parser.add_argument("--group", "-g", type=str, default=None,
                        help="Process a specific control group (e.g., F_20_39)")
    parser.add_argument("--rebuild-group", action="store_true",
                        help="Force rebuild group statistics even if cached")
    parser.add_argument("--threshold", "-t", type=float, default=None,
                        help=f"Z-score threshold (default: {ZSCORE_THRESHOLD})")
    parser.add_argument("--min-cluster", type=int, default=None,
                        help=f"Minimum cluster size in voxels (default: {MIN_CLUSTER_SIZE})")
    args = parser.parse_args()

    if args.threshold is not None:
        ZSCORE_THRESHOLD = args.threshold
    if args.min_cluster is not None:
        MIN_CLUSTER_SIZE = args.min_cluster
    SYM_MODE = args.sym
    if SYM_MODE:
        print("  [SYM] Using symmetric-template-registered inputs from symreg/sym_perf/")

    pipeline = ControlZScoreAnalysis(
        group_key=args.group,
        patient_id=args.patient,
        rebuild=args.rebuild_group,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
