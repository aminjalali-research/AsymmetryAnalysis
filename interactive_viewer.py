#!/usr/bin/env python3
"""
Interactive Overlay Viewer — zAI canonical (post-2026-05-04 pivot)

Menu-driven launcher for FSLeyes that visualizes the canonical
post-cleanup pipeline outputs:

  - Anatomy (Dataset, Dataset MNI)
  - Pipeline B Step 1-2: raw voxel z-score maps (results_zscore/patients/)
  - Pipeline B Step 3: voxel zAI maps (results_zscore/asymmetry/patients/)
  - Pipeline B Step 3 (clinical): gray-matter zAI maps (results_zscore/clinical/)

All references to deleted directories (results/, results_voxel/,
results_voxel_roi/, laterality_maps/, *_ez_analysis/, etc.) have been
removed in the Phase 2 cleanup pass.
"""

import subprocess
from pathlib import Path
import time
import tempfile

try:
    from ez_ground_truth import EZ_GROUND_TRUTH
except Exception:
    EZ_GROUND_TRUTH = {}

# Lazy imports for the new clinical-mask / top-N options. These are only
# required when the user picks the new menu options (10 / 11). Pulled in
# at call-time so the legacy menu options keep zero-dep.
try:
    import numpy as _np
    import nibabel as _nib
    import pandas as _pd
    _HAVE_SCI = True
except Exception:
    _HAVE_SCI = False


# ---------------------------------------------------------------------------
# Clinical ROI labels (Desikan-Killiany; 34 cortical pairs + 3 subcortical
# pairs = 37 paired regions, mirrored L/R = 76 FreeSurfer label IDs).
#
# Source: CLAUDE.md "34 Cortical + 6 Subcortical Paired Regions" (the 6
# subcortical labels are 3 pairs: thalamus, hippocampus, amygdala). The
# remaining 3 pairs in the CLAUDE.md doc (caudate / putamen / pallidum) are
# basal ganglia and not part of the surgeon-relevant EZ region set, so they
# are intentionally NOT included here.
# ---------------------------------------------------------------------------
SUBCORTICAL_LABELS = [10, 49, 17, 53, 18, 54]                 # thalamus, hippocampus, amygdala (L+R)
CORTICAL_LABELS = list(range(1001, 1036)) + list(range(2001, 2036))  # 34 cortical L+R
CLINICAL_ROI_LABELS = SUBCORTICAL_LABELS + CORTICAL_LABELS    # 76 labels = 37 ROI pairs


class InteractiveOverlayViewer:
    """Interactive viewer for canonical pipeline outputs."""

    def __init__(self, patient_id=None):
        self.base_dir = Path(".")
        self.patient_id = patient_id

        if patient_id:
            self._set_patient_files(patient_id)
        else:
            # Default to P013 if no patient specified
            self._set_patient_files("P013")

    def _set_patient_files(self, patient_id):
        """Set file paths for a specific patient (canonical paths only)."""
        self.patient_id = patient_id

        # Anatomy & raw inputs (Dataset/, Dataset MNI/)
        self.freesurfer_files = {
            "T1w": f"Dataset/{patient_id}/T1w_acpc_dc_restore.nii.gz",
            "Parcellation": f"Dataset/{patient_id}/aparc+aseg.nii.gz",
            "Perfusion_Original": f"Dataset/{patient_id}/perfusion_calib.nii.gz",
            "Perfusion_Resampled": f"Dataset/{patient_id}/{patient_id}_perfusion_calib_resampled_to_T1w.nii.gz",
        }

        # MNI-space anatomy (Dataset MNI/)
        self.mni_t1w = f"Dataset MNI/{patient_id}/T1w_restore.nii.gz"
        self.mni_perfusion = f"Dataset MNI/{patient_id}/perfusion_calib.nii.gz"
        self.mni_parcellation = f"Dataset MNI/{patient_id}/aparc+aseg.nii.gz"

        # Pipeline B — raw voxel z-score maps (results_zscore/patients/<PID>/)
        # File pattern: <PID>_vs_<group>_zscore.nii.gz, _hyper_*, _hypo_*
        self.zscore_control_files = {}
        zscore_patient_dir = Path(f"results_zscore/patients/{patient_id}")
        if zscore_patient_dir.exists():
            for f in sorted(zscore_patient_dir.glob(f"{patient_id}_vs_*_zscore.nii.gz")):
                fname = f.name
                group_key = fname.replace(f"{patient_id}_vs_", "").replace("_zscore.nii.gz", "")
                prefix = f"{patient_id}_vs_{group_key}"
                self.zscore_control_files[group_key] = {
                    "zscore": str(f),
                    "hyper": str(zscore_patient_dir / f"{prefix}_hyper_perfusion.nii.gz"),
                    "hypo": str(zscore_patient_dir / f"{prefix}_hypo_perfusion.nii.gz"),
                    "hyper_clusters": str(zscore_patient_dir / f"{prefix}_hyper_clusters.nii.gz"),
                    "hypo_clusters": str(zscore_patient_dir / f"{prefix}_hypo_clusters.nii.gz"),
                }
        self.zscore_group_dir = "results_zscore/groups"

        # Pipeline B — zAI voxel maps (results_zscore/asymmetry/patients/<PID>/)
        # Canonical filenames: <PID>_asymmetry_zscore.nii.gz,
        #                      <PID>_asymmetry_significant.nii.gz,
        #                      <PID>_asymmetry_left_dominant.nii.gz,
        #                      <PID>_asymmetry_right_dominant.nii.gz
        self.zai_dir = Path(f"results_zscore/asymmetry/patients/{patient_id}")
        self.zai_files = {
            "asymmetry_zscore": self.zai_dir / f"{patient_id}_asymmetry_zscore.nii.gz",
            "significant": self.zai_dir / f"{patient_id}_asymmetry_significant.nii.gz",
            "left_dominant": self.zai_dir / f"{patient_id}_asymmetry_left_dominant.nii.gz",
            "right_dominant": self.zai_dir / f"{patient_id}_asymmetry_right_dominant.nii.gz",
        }

        # Pipeline B — clinical (gray-matter) maps (results_zscore/clinical/<PID>/)
        self.clinical_dir = Path(f"results_zscore/clinical/{patient_id}")

    # ------------------------------------------------------------------
    # AVAILABILITY DETECTION
    # ------------------------------------------------------------------

    def _detect_available_data(self):
        """Return dict of which canonical data is available for this patient."""
        pid = self.patient_id
        zai_path = self.zai_dir / f"{pid}_asymmetry_zscore.nii.gz"
        clin_zai = self.clinical_dir / f"{pid}_clinical_zai_lateralized_significant.nii.gz"
        return {
            "anatomy": Path(self.freesurfer_files["T1w"]).exists(),
            "mni_anatomy": Path(self.mni_t1w).exists(),
            "raw_zscore": bool(self.zscore_control_files),
            "zai": zai_path.exists(),
            "clinical": clin_zai.exists() or self.clinical_dir.exists(),
        }

    def _availability_str(self):
        """Compact one-line availability summary for the menu header."""
        avail = self._detect_available_data()
        labels = {
            "anatomy": "anatomy",
            "mni_anatomy": "MNI anatomy",
            "raw_zscore": "raw z-score",
            "zai": "zAI",
            "clinical": "clinical",
        }
        parts = []
        for k, lbl in labels.items():
            mark = "[+]" if avail[k] else "[-]"
            parts.append(f"{mark} {lbl}")
        return "  ".join(parts)

    # ------------------------------------------------------------------
    # PATIENT SELECTION
    # ------------------------------------------------------------------

    def get_available_patients(self):
        """Scan canonical zAI directory for patients with data."""
        zai_root = Path("results_zscore/asymmetry/patients")
        anat_root = Path("Dataset")
        patients = set()

        if zai_root.exists():
            for d in sorted(zai_root.iterdir()):
                if d.is_dir() and d.name.startswith("P"):
                    if (d / f"{d.name}_asymmetry_zscore.nii.gz").exists():
                        patients.add(d.name)

        # Also include patients with anatomy-only available
        if anat_root.exists():
            for d in sorted(anat_root.iterdir()):
                if d.is_dir() and d.name.startswith("P"):
                    if (d / "T1w_acpc_dc_restore.nii.gz").exists():
                        patients.add(d.name)

        return sorted(patients)

    def select_patient(self):
        """Interactive patient selection."""
        patients = self.get_available_patients()

        if not patients:
            print("[!] No patients found in Dataset/ or results_zscore/asymmetry/patients/.")
            return None

        print("\n" + "=" * 70)
        print("AVAILABLE PATIENTS")
        print("=" * 70)

        zai_root = Path("results_zscore/asymmetry/patients")
        for i, patient in enumerate(patients, 1):
            tags = []
            if (zai_root / patient / f"{patient}_asymmetry_zscore.nii.gz").exists():
                tags.append("zAI")
            if Path(f"Dataset/{patient}/T1w_acpc_dc_restore.nii.gz").exists():
                tags.append("anat")
            ez = EZ_GROUND_TRUTH.get(patient, {})
            ez_tag = f"EZ={ez.get('ez', '?')}" if ez else "EZ=?"
            tag_str = f"[{', '.join(tags) or 'none'}]  {ez_tag}"
            print(f"   {i:2d}. {patient}   {tag_str}")

        print("\n   0. Exit")
        print("=" * 70)

        while True:
            choice = input(
                f"\nSelect patient (1-{len(patients)}) or 0 to exit: "
            ).strip()

            if choice == "0":
                return None

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(patients):
                    selected = patients[idx]
                    print(f"\n[+] Selected patient: {selected}")
                    self._set_patient_files(selected)
                    return selected
                else:
                    print(f"[!] Invalid choice. Please enter 1-{len(patients)} or 0")
            except ValueError:
                print("[!] Invalid input. Please enter a number")

    # ------------------------------------------------------------------
    # FILE-LIST UTILITIES
    # ------------------------------------------------------------------

    def check_files(self):
        """Check which canonical files exist for the current patient."""
        pid = self.patient_id
        all_ok = True

        print("[*] Checking canonical file availability:")

        print("\n  Anatomy (Dataset/):")
        for name, path in self.freesurfer_files.items():
            if Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024**2)
                print(f"     [+] {name}: {path} ({size_mb:.1f} MB)")
            else:
                print(f"     [-] {name}: {path} - NOT FOUND")

        print("\n  MNI anatomy (Dataset MNI/):")
        for name, path in [
            ("MNI T1w", self.mni_t1w),
            ("MNI Perfusion", self.mni_perfusion),
            ("MNI Parcellation", self.mni_parcellation),
        ]:
            if Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024**2)
                print(f"     [+] {name}: {path} ({size_mb:.1f} MB)")
            else:
                print(f"     [-] {name}: {path} - NOT FOUND")

        print("\n  Pipeline B Step 1-2 — raw voxel z-score (results_zscore/patients/):")
        if self.zscore_control_files:
            for group, files in self.zscore_control_files.items():
                z_path = files["zscore"]
                if Path(z_path).exists():
                    size_mb = Path(z_path).stat().st_size / (1024**2)
                    print(f"     [+] vs {group}: {z_path} ({size_mb:.1f} MB)")
        else:
            print(f"     [-] No raw z-score maps for {pid}")
            print("         Run: python 01_build_control_normative.py")
            all_ok = False

        print("\n  Pipeline B Step 3 — zAI maps (results_zscore/asymmetry/patients/):")
        zai_found = 0
        for name, path in self.zai_files.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024**2)
                print(f"     [+] {name}: {path} ({size_mb:.1f} MB)")
                zai_found += 1
            else:
                print(f"     [-] {name}: {path} - NOT FOUND")
        if zai_found == 0:
            print(f"         Run: python 02_compute_zai.py --patient {pid}")
            all_ok = False

        print("\n  Pipeline B Step 3 (clinical) — gray-matter zAI (results_zscore/clinical/):")
        if self.clinical_dir.exists():
            clin_files = sorted(self.clinical_dir.glob(f"{pid}_clinical*.nii.gz"))
            for f in clin_files:
                size_mb = f.stat().st_size / (1024**2)
                print(f"     [+] {f.name} ({size_mb:.1f} MB)")
            if not clin_files:
                print(f"     [-] No clinical NIfTI files in {self.clinical_dir}")
        else:
            print(f"     [-] {self.clinical_dir} not found")
            print(f"         Run: python 03_clinical_maps.py --patient {pid}")

        return all_ok

    # ------------------------------------------------------------------
    # FSLEYES INVOCATION HELPER
    # ------------------------------------------------------------------

    def run_fsleyes_overlay(self, title, base_file, overlay_configs, wait_time=2):
        """Run FSLeyes with specified overlay configuration."""

        print(f"\n[*] {title}")
        print("=" * 70)

        if not Path(base_file).exists():
            print(f"[!] Base image not found: {base_file}")
            return

        cmd = f"fsleyes '{base_file}'"

        for config in overlay_configs:
            file_path = config["file"]
            if not Path(file_path).exists():
                print(f"   [!] Overlay missing (skipped): {file_path}")
                continue
            colormap = config.get("colormap", "red-yellow")
            display_range = config.get("range", "-0.4 0.4")
            alpha = config.get("alpha", "70")
            name = config.get("name", Path(file_path).stem)

            cmd += f" '{file_path}' -cm {colormap} -dr {display_range} -a {alpha} -n \"{name}\""

        cmd += " &"

        print(f"Command: {cmd}")
        print("\nDescription: Opening FSLeyes with overlays...")

        try:
            subprocess.Popen(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"[+] {title} opened.")
            print("    Close the FSLeyes window when done viewing.")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[!] Error opening {title}: {e}")

    def _pick_zscore_group(self):
        """Pick a control group if multiple are available; return key or None."""
        groups = list(self.zscore_control_files.keys())
        if not groups:
            print("[!] No raw z-score maps found for this patient.")
            print(f"    Run: python 01_build_control_normative.py")
            return None
        if len(groups) == 1:
            return groups[0]
        print("\n  Available control groups:")
        for i, g in enumerate(groups, 1):
            print(f"    {i}. {g}")
        choice = input(f"  Select group [1-{len(groups)}, default=1]: ").strip()
        idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(groups) else 0
        return groups[idx]

    # ------------------------------------------------------------------
    # OPTION 1: Basic Anatomy
    # ------------------------------------------------------------------

    def view_basic_anatomy(self):
        """View T1w anatomy with parcellation overlay."""
        overlays = [
            {
                "file": self.freesurfer_files["Parcellation"],
                "colormap": "random",
                "range": "1 2035",
                "alpha": "30",
                "name": "Parcellation",
            }
        ]
        self.run_fsleyes_overlay(
            "Basic Anatomy: T1w + Parcellation",
            self.freesurfer_files["T1w"],
            overlays,
        )

    # ------------------------------------------------------------------
    # OPTION 2: Perfusion Comparison
    # ------------------------------------------------------------------

    def view_perfusion_comparison(self):
        """Compare T1w-space and MNI-space perfusion."""
        overlays = [
            {
                "file": self.freesurfer_files["Perfusion_Resampled"],
                "colormap": "hot",
                "range": "0 80",
                "alpha": "60",
                "name": "Perfusion (T1w-resampled)",
            },
        ]

        # If the MNI-space perfusion exists, append it (same MNI geometry as the
        # patient T1w — but for clinicians the resampled-to-T1w is the primary
        # blood-flow visual on top of T1w).
        if Path(self.mni_perfusion).exists():
            overlays.append({
                "file": self.mni_perfusion,
                "colormap": "cool",
                "range": "0 80",
                "alpha": "40",
                "name": "Perfusion (MNI raw)",
            })

        self.run_fsleyes_overlay(
            "Perfusion Comparison",
            self.freesurfer_files["T1w"],
            overlays,
        )

    # ------------------------------------------------------------------
    # OPTION 3: Raw Voxel z-Score Map (Pipeline B Step 1-2)
    # ------------------------------------------------------------------

    def view_zscore_control_map(self):
        """Patient vs controls — gray-matter z-score map (lateralized).

        Shows how this patient's blood flow in each brain region compares
        to the average of healthy controls. Red = higher than normal,
        blue = lower.
        """
        group = self._pick_zscore_group()
        if not group:
            return

        cdir = self.clinical_dir
        mean_file = f"{self.zscore_group_dir}/{group}/mean_perfusion.nii.gz"
        lat_sig_file = cdir / f"{self.patient_id}_clinical_lateralized_significant.nii.gz"
        gm_z_file = cdir / f"{self.patient_id}_clinical_gm_zscore.nii.gz"

        # Prefer lateralized; fall back to GM-only z-score; fall back to raw
        if lat_sig_file.exists():
            overlay_path = str(lat_sig_file)
            label = f"{self.patient_id} lateralized z-score"
        elif gm_z_file.exists():
            overlay_path = str(gm_z_file)
            label = f"{self.patient_id} GM z-score"
            print("  (Note: lateralized maps not found, using full GM z-score.)")
        else:
            overlay_path = self.zscore_control_files[group]["zscore"]
            label = f"{self.patient_id} raw z-score (no clinical map)"
            print("  (Note: clinical maps not found, using raw z-score map.)")
            print(f"  Run: python 03_clinical_maps.py --patient {self.patient_id}")

        print(f"\n[*] PATIENT vs CONTROLS — Z-Score Map")
        print(f"   Patient: {self.patient_id}  |  Controls: {group}")
        print(f"   Red = higher blood flow than controls")
        print(f"   Blue = lower blood flow than controls")

        cmd = f"fsleyes '{mean_file}' "
        cmd += f"'{overlay_path}' -cm brain_colours_diverging_bwr -dr -6 6 -a 80 "
        cmd += f"-n '{label}' &"

        print(f"\n  Command:\n  {cmd}\n")
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("  [+] FSLeyes launched")
        except Exception as e:
            print(f"  [!] Error: {e}")

    # ------------------------------------------------------------------
    # OPTION 4: Significant z-score Clusters Only
    # ------------------------------------------------------------------

    def view_zscore_clusters(self):
        """Patient vs controls — lateralized significant clusters.

        Shows ONLY the dominant-hemisphere clusters where blood flow
        is significantly different from controls. Cleanest view for
        presurgical lateralization.
        """
        group = self._pick_zscore_group()
        if not group:
            return

        cdir = self.clinical_dir
        mean_file = f"{self.zscore_group_dir}/{group}/mean_perfusion.nii.gz"
        lat_hyper = cdir / f"{self.patient_id}_clinical_lateralized_hyper.nii.gz"
        lat_hypo = cdir / f"{self.patient_id}_clinical_lateralized_hypo.nii.gz"

        if not lat_hyper.exists():
            lat_hyper = cdir / f"{self.patient_id}_clinical_hyper.nii.gz"
            lat_hypo = cdir / f"{self.patient_id}_clinical_hypo.nii.gz"
            if lat_hyper.exists():
                print("  (Note: lateralized maps not found, using bilateral.)")
            else:
                # Fall back to raw cluster maps
                lat_hyper = Path(self.zscore_control_files[group]["hyper_clusters"])
                lat_hypo = Path(self.zscore_control_files[group]["hypo_clusters"])
                print("  (Note: clinical maps not found, using raw cluster maps.)")
                print(f"  Run: python 03_clinical_maps.py --patient {self.patient_id}")

        print(f"\n[*] PATIENT vs CONTROLS — Significant Clusters")
        print(f"   Patient: {self.patient_id}  |  Controls: {group}")
        print(f"   Red = hyper-perfusion  |  Blue = hypo-perfusion")

        cmd = f"fsleyes '{mean_file}' "
        if lat_hyper.exists():
            cmd += f"'{lat_hyper}' -cm red -dr 0.5 1 -a 90 -n 'Hyper (dominant side)' "
        if lat_hypo.exists():
            cmd += f"'{lat_hypo}' -cm blue -dr 0.5 1 -a 90 -n 'Hypo (dominant side)' "
        cmd += "&"

        print(f"\n  Command:\n  {cmd}\n")
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("  [+] FSLeyes launched")
        except Exception as e:
            print(f"  [!] Error: {e}")

    # ------------------------------------------------------------------
    # OPTION 5: z-score Clusters + Brain Region Labels
    # ------------------------------------------------------------------

    def view_zscore_with_parcellation(self):
        """z-score clusters with brain region boundaries overlaid."""
        group = self._pick_zscore_group()
        if not group:
            return

        cdir = self.clinical_dir
        mean_file = f"{self.zscore_group_dir}/{group}/mean_perfusion.nii.gz"
        parc_file = f"{self.zscore_group_dir}/{group}/consensus_parcellation.nii.gz"
        lat_hyper = cdir / f"{self.patient_id}_clinical_lateralized_hyper.nii.gz"
        lat_hypo = cdir / f"{self.patient_id}_clinical_lateralized_hypo.nii.gz"

        if not lat_hyper.exists():
            lat_hyper = cdir / f"{self.patient_id}_clinical_hyper.nii.gz"
            lat_hypo = cdir / f"{self.patient_id}_clinical_hypo.nii.gz"

        print(f"\n[*] PATIENT vs CONTROLS — Clusters + Brain Region Labels")
        print(f"   Patient: {self.patient_id}  |  Controls: {group}")
        print(f"   Red = hyper-perfusion  |  Blue = hypo-perfusion")

        cmd = f"fsleyes '{mean_file}' "
        if Path(parc_file).exists():
            cmd += f"'{parc_file}' -cm random -dr 1 2035 -a 20 -n 'Brain Regions' "
        if lat_hyper.exists():
            cmd += f"'{lat_hyper}' -cm red -dr 0.5 1 -a 85 -n 'Hyper (dominant side)' "
        if lat_hypo.exists():
            cmd += f"'{lat_hypo}' -cm blue -dr 0.5 1 -a 85 -n 'Hypo (dominant side)' "
        cmd += "&"

        print(f"\n  Command:\n  {cmd}\n")
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("  [+] FSLeyes launched")
        except Exception as e:
            print(f"  [!] Error: {e}")

    # ------------------------------------------------------------------
    # OPTION 6: zAI Asymmetry Map (PRIMARY CLINICAL PRODUCT)
    # ------------------------------------------------------------------

    def _check_zai(self):
        """Verify zAI files exist; print remediation if not."""
        zai_path = self.zai_files["asymmetry_zscore"]
        if not zai_path.exists():
            print(f"[!] zAI map not found for {self.patient_id}.")
            print(f"    Expected: {zai_path}")
            print(f"    Run: python 02_compute_zai.py --patient {self.patient_id}")
            return False
        return True

    def view_zai_asymmetry_map(self):
        """zAI asymmetry map — L vs R deviation from controls.

        Each voxel shows whether the LEFT z-score is higher or lower
        than the RIGHT z-score at the mirror location. Primary
        lateralization map.
        Red/warm = left side more deviant from controls.
        Blue/cool = right side more deviant from controls.
        """
        if not self._check_zai():
            return
        group = self._pick_zscore_group() or "FM_20_39"

        mean_f = f"{self.zscore_group_dir}/{group}/mean_perfusion.nii.gz"
        ai_f = str(self.zai_files["asymmetry_zscore"])

        print(f"\n[*] zAI ASYMMETRY MAP — L vs R Deviation from Controls")
        print(f"   Patient: {self.patient_id}")
        print(f"   Red = left hemisphere MORE deviant from controls")
        print(f"   Blue = right hemisphere MORE deviant from controls")

        cmd = f"fsleyes '{mean_f}' "
        cmd += f"'{ai_f}' -cm brain_colours_diverging_bwr -dr -3 3 -a 75 "
        cmd += f"-n 'zAI (L-R)' &"

        print(f"\n  Command:\n  {cmd}\n")
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("  [+] FSLeyes launched")
        except Exception as e:
            print(f"  [!] Error: {e}")

    def view_zai_cleaned_overlay(self):
        """Island-cleaned zAI overlay (option 16).

        Generates an island-free cleaned zAI NIfTI + PNG montage via
        clean_overlay.py, then offers the cleaned NIfTI to FSLeyes so
        speckle islands no longer clutter the interactive overlay. Falls back
        gracefully if the module or maps are unavailable. Non-breaking add-on.
        """
        if not self._check_zai():
            return
        import importlib.util as _ilu
        import sys as _sys
        co_path = Path(__file__).parent / "clean_overlay.py"
        if not co_path.exists():
            print("[!] clean_overlay.py not found; cannot generate cleaned overlay.")
            return
        try:
            spec = _ilu.spec_from_file_location("clean_overlay", str(co_path))
            co = _ilu.module_from_spec(spec)
            _sys.modules["clean_overlay"] = co
            spec.loader.exec_module(co)
        except Exception as e:
            print(f"[!] Could not load clean_overlay.py: {e}")
            return

        print(f"\n[*] ISLAND-CLEANED zAI OVERLAY — {self.patient_id}")
        print("    Thresholding |zAI|>=3, min_cluster=50, opening+closing.")
        report = co.process_patient(self.patient_id, threshold=3.0,
                                    min_cluster=50, space="zai",
                                    use_roi_mask=True)
        if not report:
            print("[!] Cleaning failed; see messages above.")
            return
        cleaned_nii = report.get("cleaned_nifti")
        group = report.get("group", "FM_20_39")
        mean_f = f"{self.zscore_group_dir}/{group}/mean_perfusion.nii.gz"
        print(f"\n  Islands removed: {report['n_islands_removed']} "
              f"({report['voxels_removed']:,} vox)")
        print(f"  Montage: {report.get('montage_png')}")
        print(f"  QC:      {report.get('qc_png')}")
        if cleaned_nii and Path(cleaned_nii).exists():
            cmd = (f"fsleyes '{mean_f}' '{cleaned_nii}' "
                   f"-cm brain_colours_diverging_bwr -dr -8 8 -a 75 "
                   f"-n 'zAI cleaned' &")
            print(f"\n  Command:\n  {cmd}\n")
            try:
                subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                print("  [+] FSLeyes launched (island-free overlay)")
            except Exception as e:
                print(f"  [!] Error: {e}")

    # ------------------------------------------------------------------
    # OPTION 7: Lateralized Dominance Maps (Left-dominant vs Right-dominant)
    # ------------------------------------------------------------------

    def view_zai_dominance(self):
        """zAI lateralized dominance: red=Left dominant, blue=Right dominant.

        Shows EACH bilateral pair on its dominant side only (clean,
        clinically interpretable presurgical view).
        """
        if not self._check_zai():
            return
        group = self._pick_zscore_group() or "FM_20_39"

        mean_f = f"{self.zscore_group_dir}/{group}/mean_perfusion.nii.gz"
        left_f = self.zai_files["left_dominant"]
        right_f = self.zai_files["right_dominant"]

        if not left_f.exists() or not right_f.exists():
            print(f"[!] Dominance maps missing.")
            print(f"    Expected: {left_f}")
            print(f"    Expected: {right_f}")
            print(f"    Run: python 02_compute_zai.py --patient {self.patient_id}")
            return

        print(f"\n[*] zAI LATERALIZED DOMINANCE (one-sided per bilateral pair)")
        print(f"   Patient: {self.patient_id}")
        print(f"   Red = LEFT-dominant voxels (L > R; "
              f"interictally suggests RIGHT EZ)")
        print(f"   Blue = RIGHT-dominant voxels (R > L; "
              f"interictally suggests LEFT EZ)")

        cmd = f"fsleyes '{mean_f}' "
        cmd += (f"'{left_f}' -cm red -dr 0.5 3 -a 85 "
                f"-n 'Left-dominant (L>R; suggests R EZ if interictal)' ")
        cmd += (f"'{right_f}' -cm blue -dr 0.5 3 -a 85 "
                f"-n 'Right-dominant (R>L; suggests L EZ if interictal)' &")

        print(f"\n  Command:\n  {cmd}\n")
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("  [+] FSLeyes launched")
        except Exception as e:
            print(f"  [!] Error: {e}")

    # ------------------------------------------------------------------
    # OPTION 8: Significant zAI Clusters
    # ------------------------------------------------------------------

    def view_zai_significant(self):
        """Significant zAI clusters — most focal asymmetric findings.

        Shows voxels where L-R asymmetry exceeds the surgeon-facing
        clinical threshold |zAI| >= 4 (tightened 2026-05-05; was implicitly
        |zAI| >= 3 via the legacy |zAI|>=0.5 display range on the file's
        full continuous values, which let almost all gray-matter voxels
        show up as "significant"). The underlying file
        `<PID>_asymmetry_significant.nii.gz` actually stores continuous
        zAI values (range ~ -17..+17), not a binary mask, so we display
        it with a diverging colormap and a strict display range.
        """
        if not self._check_zai():
            return
        group = (self._pick_zscore_group() if len(self.zscore_control_files) > 1
                 else (next(iter(self.zscore_control_files), None) or "FM_20_39"))

        mean_f = f"{self.zscore_group_dir}/{group}/mean_perfusion.nii.gz"
        sig_f = self.zai_files["significant"]

        if not sig_f.exists():
            print(f"[!] Significance mask missing: {sig_f}")
            print(f"    Run: python 02_compute_zai.py --patient {self.patient_id}")
            return

        threshold = 4.0   # tightened 2026-05-05 for surgeon-facing focality
        print(f"\n[*] SIGNIFICANT zAI CLUSTERS (Focal, |zAI| >= {threshold:g})")
        print(f"   Patient: {self.patient_id}")
        print(f"   Shows: voxels where L-R asymmetry is at least {threshold:g} SDs")
        print(f"   (use option 6 to inspect the continuous map at lower thresholds)")

        cmd = f"fsleyes '{mean_f}' "
        cmd += (
            f"'{sig_f}' -cm brain_colours_diverging_bwr "
            f"-dr -{threshold:g} {threshold:g} -a 85 "
            f"-n 'zAI significant (|zAI|>={threshold:g})' &"
        )

        print(f"\n  Command:\n  {cmd}\n")
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)
            print("  [+] FSLeyes launched")
        except Exception as e:
            print(f"  [!] Error: {e}")

    # ------------------------------------------------------------------
    # OPTION 9: Clinical zAI (gray matter only, lateralized) — surgeon-facing
    # ------------------------------------------------------------------

    def view_zai_clinical(self):
        """Clinical-grade zAI — gray-matter only, lateralized dominance.

        This is the surgeon-facing version: white-matter noise removed,
        bilateral pairs shown only on the dominant side. Recommended
        for MDT presurgical review.

        Fixed 2026-05-05:
        - default control group to F_20_39 (no interactive prompt that
          could block / silently return None on EOF)
        - the "_clinical_zai_lateralized_significant" file is the
          *continuous* zAI map (range ~ -15..+17), NOT a binary mask, so
          we now display it with a diverging colormap and a clinically
          tightened threshold (|zAI| >= 4) instead of "-cm hot -dr 0.5 1"
          which clipped almost the entire signal range.
        - stderr is no longer redirected to DEVNULL on launch, so any
          FSLeyes parsing/runtime failure is now visible to the user.
        """
        cdir = self.clinical_dir
        if not cdir.exists():
            print(f"[!] Clinical maps directory not found: {cdir}")
            print(f"    Run: python 03_clinical_maps.py --patient {self.patient_id}")
            return

        # Default to F_20_39 silently; only prompt if the caller has more
        # than one group available (avoids blocking input() in the
        # surgeon-facing default flow).
        group = (self._pick_zscore_group() if len(self.zscore_control_files) > 1
                 else (next(iter(self.zscore_control_files), None) or "FM_20_39"))
        mean_f = f"{self.zscore_group_dir}/{group}/mean_perfusion.nii.gz"

        left_f = cdir / f"{self.patient_id}_clinical_zai_lateralized_left_dominant.nii.gz"
        right_f = cdir / f"{self.patient_id}_clinical_zai_lateralized_right_dominant.nii.gz"
        sig_f = cdir / f"{self.patient_id}_clinical_zai_lateralized_significant.nii.gz"

        if not left_f.exists() and not right_f.exists() and not sig_f.exists():
            print(f"[!] Clinical zAI maps not found in {cdir}")
            print(f"    Run: python 03_clinical_maps.py --patient {self.patient_id}")
            return

        threshold = 4.0   # |zAI| >= 4 (clinically focal); option 6 = continuous, no threshold

        print(f"\n[*] CLINICAL zAI (Gray Matter Only, Lateralized)")
        print(f"   Patient: {self.patient_id}  |  Controls: {group}")
        print(f"   Red = LEFT-dominant (L>R; suggests R EZ if interictal)")
        print(f"   Blue = RIGHT-dominant (R>L; suggests L EZ if interictal)")
        print(f"   Display threshold: |zAI| >= {threshold:g}  (use option 6 for continuous)")

        # NOTE: left_dominant / right_dominant are BINARY masks (value 1.0),
        # so -dr 0.5 1 is correct for those layers.
        # The "lateralized_significant" file is CONTINUOUS zAI values;
        # display it as a diverging map with the surgeon-facing threshold.
        cmd = f"fsleyes '{mean_f}' "
        if sig_f.exists():
            cmd += (
                f"'{sig_f}' -cm brain_colours_diverging_bwr "
                f"-dr -{threshold:g} {threshold:g} -a 65 "
                f"-n 'zAI continuous (|zAI|>={threshold:g})' "
            )
        if left_f.exists():
            cmd += (f"'{left_f}' -cm red -dr 0.5 1 -a 85 "
                    f"-n 'Left-dominant (L>R; suggests R EZ if interictal)' ")
        if right_f.exists():
            cmd += (f"'{right_f}' -cm blue -dr 0.5 1 -a 85 "
                    f"-n 'Right-dominant (R>L; suggests L EZ if interictal)' ")
        cmd += "&"

        print(f"\n  Command:\n  {cmd}\n")
        try:
            # Do NOT redirect stderr; any FSLeyes parsing error needs to
            # surface so the user can debug a "fails to launch" report.
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)
            print("  [+] FSLeyes launched (stderr left unredirected for visibility)")
        except Exception as e:
            print(f"  [!] Error: {e}")

    # ------------------------------------------------------------------
    # OPTION 10: Top-N Dominant Clusters (focal, surgeon-facing)
    # ------------------------------------------------------------------

    def view_zai_dominant_clusters(self, top_n=3, threshold=4.0):
        """Show ONLY the top-N clusters by peak |zAI| in FSLeyes.

        Motivation: the canonical zAI map for some patients (e.g. P013)
        contains 60K+-voxel clusters in non-EZ regions that visually drown
        out a smaller surgeon-relevant cluster. This view filters to the
        top-N highest-peak clusters and prints their anatomical regions
        + peak values to stdout for clinical context.

        Falls back gracefully if the cluster_labels NIfTI is missing
        (prints regions only).
        """
        if not self._check_zai():
            return
        if not _HAVE_SCI:
            print("[!] This option requires numpy / nibabel / pandas.")
            print("    Install with: pip install numpy nibabel pandas")
            return

        pid = self.patient_id

        # Prefer the gray-matter clinical cluster report (cleaner; matches
        # the surgeon-facing view); fall back to the asymmetry report if
        # the clinical one isn't present.
        clin_csv = self.clinical_dir / f"{pid}_clinical_zai_cluster_report.csv"
        asym_csv = self.zai_dir / f"{pid}_asymmetry_cluster_report.csv"

        if clin_csv.exists():
            csv_path = clin_csv
            peak_col, mean_col = "peak_zai", "mean_zai"
            tag = "clinical zAI (gray matter)"
        elif asym_csv.exists():
            csv_path = asym_csv
            peak_col, mean_col = "peak_z", "mean_z"
            tag = "asymmetry zAI (full brain)"
        else:
            print(f"[!] No cluster_report CSV found for {pid}.")
            print(f"    Expected: {clin_csv}")
            print(f"             or {asym_csv}")
            print(f"    Run: python 03_clinical_maps.py --patient {pid}")
            return

        try:
            df = _pd.read_csv(csv_path)
        except Exception as e:
            print(f"[!] Could not read {csv_path}: {e}")
            return

        if peak_col not in df.columns:
            print(f"[!] Expected column '{peak_col}' missing from {csv_path}.")
            print(f"    Found columns: {list(df.columns)}")
            return

        # Filter by absolute peak threshold and sort
        df["abs_peak"] = df[peak_col].abs()
        df_thr = df[df["abs_peak"] >= threshold].copy()
        if df_thr.empty:
            print(f"[!] No clusters with |peak zAI| >= {threshold:g} in {csv_path.name}.")
            print(f"    Lower the threshold or pick option 6 / 8 to inspect the full map.")
            return

        df_top = df_thr.nlargest(top_n, "abs_peak").copy()

        # Stdout summary for clinical context
        ez = EZ_GROUND_TRUTH.get(pid, {})
        ez_side = ez.get("ez", "?")
        ez_conf = ez.get("confidence", "?")

        print()
        print("=" * 70)
        print(f"  TOP-{top_n} DOMINANT zAI CLUSTERS for {pid}  (|zAI| >= {threshold:g})")
        print(f"  Source: {csv_path.name}  ({tag})")
        print(f"  MDT ground truth: EZ = {ez_side}  ({ez_conf} confidence)")
        print("  Legend: 'LEFT-dominant'  = L side has MORE blood flow than expected")
        print("                            (interictally suggests RIGHT EZ).")
        print("          'RIGHT-dominant' = R side has MORE blood flow than expected")
        print("                            (interictally suggests LEFT EZ).")
        print("=" * 70)
        for _, row in df_top.iterrows():
            mean_v = row[mean_col]
            side = "LEFT-dominant " if mean_v > 0 else "RIGHT-dominant"
            print(
                f"  #{int(row['cluster_id']):3d}  {side}  "
                f"peak={row[peak_col]:+6.2f}  size={int(row['size_voxels']):6d} vox  "
                f"region={row['primary_region']}"
            )
        print("=" * 70)
        print()

        # Build a binary mask of just those clusters using the cluster
        # labels NIfTI. Prefer clinical labels; fall back to asymmetry
        # labels (rarely present) — if neither, print regions and exit.
        clin_labels = self.clinical_dir / f"{pid}_clinical_zai_cluster_labels.nii.gz"
        asym_labels = self.zai_dir / f"{pid}_asymmetry_cluster_labels.nii.gz"

        labels_path = None
        if clin_labels.exists() and csv_path == clin_csv:
            labels_path = clin_labels
        elif asym_labels.exists() and csv_path == asym_csv:
            labels_path = asym_labels
        elif clin_labels.exists():
            labels_path = clin_labels
        elif asym_labels.exists():
            labels_path = asym_labels

        if labels_path is None:
            print("  [!] No cluster_labels NIfTI found — printed top-N regions only.")
            print(f"      To get a focal FSLeyes overlay, run option 9 (clinical zAI).")
            return

        # Build top-N binary mask
        try:
            labels_nii = _nib.load(str(labels_path))
            labels_data = labels_nii.get_fdata().astype(_np.int32)
        except Exception as e:
            print(f"  [!] Could not load cluster labels {labels_path}: {e}")
            return

        # cluster_id in the CSV indexes left-dominant clusters and right-
        # dominant clusters in separate id-spaces; the labels NIfTI may
        # use signed labels (positive = L-dom, negative = R-dom) or a
        # single id-space. Try both.
        top_ids = df_top["cluster_id"].astype(int).tolist()
        top_dirs = df_top["direction"].tolist() if "direction" in df_top.columns else [None] * len(top_ids)

        # Attempt 1: signed labels
        signed_ids = []
        for cid, direction in zip(top_ids, top_dirs):
            if direction == "right-dominant":
                signed_ids.append(-cid)
            else:
                signed_ids.append(cid)
        mask = _np.isin(labels_data, signed_ids)
        if mask.sum() == 0:
            # Attempt 2: unsigned labels
            mask = _np.isin(labels_data, top_ids)

        if mask.sum() == 0:
            print(f"  [!] None of the top-{top_n} cluster IDs were found in the labels NIfTI.")
            print(f"      Falling back to FSLeyes view of full clinical zAI (option 9).")
            self.view_zai_clinical()
            return

        # Save temp mask
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / f"{pid}_top{top_n}_zai_clusters.nii.gz"
        _nib.save(
            _nib.Nifti1Image(
                mask.astype(_np.uint8), labels_nii.affine, labels_nii.header
            ),
            str(tmp_path),
        )
        print(f"  [+] Top-{top_n} mask written: {tmp_path}  ({int(mask.sum())} voxels)")

        # Launch FSLeyes: T1w anatomy + continuous zAI (faded) + top-N mask (yellow)
        group = (next(iter(self.zscore_control_files), None) or "FM_20_39")
        mean_f = f"{self.zscore_group_dir}/{group}/mean_perfusion.nii.gz"
        ai_f = str(self.zai_files["asymmetry_zscore"])
        t1_f = self.freesurfer_files["T1w"]

        # Use T1w as background if available; group mean otherwise
        bg = t1_f if Path(t1_f).exists() else mean_f

        cmd = (
            f"fsleyes '{bg}' "
            f"'{ai_f}' -cm brain_colours_diverging_bwr -dr -{threshold:g} {threshold:g} "
            f"-a 35 -n 'zAI continuous (faded)' "
            f"'{tmp_path}' -cm yellow -dr 0.5 1 -a 90 -n 'Top-{top_n} clusters' &"
        )
        print(f"\n  Command:\n  {cmd}\n")
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)
            print("  [+] FSLeyes launched")
        except Exception as e:
            print(f"  [!] Error: {e}")

    # ------------------------------------------------------------------
    # OPTION 11: 37-ROI Clinical Mask View
    # ------------------------------------------------------------------

    def view_zai_clinical_roi_mask(self, threshold=4.0, top_n=5):
        """Restrict zAI display to the 37 clinically-relevant
        Desikan-Killiany regions (3 subcortical pairs + 34 cortical pairs).

        Useful when the surgeon wants to suppress non-cortical / non-EZ
        regions (e.g. cerebellum, brainstem, ventricles) that FSLeyes
        would otherwise show as noisy zAI signal.

        Also prints the top-N regions (by peak |zAI|) limited to the
        37-ROI mask, so the surgeon can read the focal asymmetry hits
        without having to scroll through cluster reports.
        """
        if not self._check_zai():
            return
        if not _HAVE_SCI:
            print("[!] This option requires numpy / nibabel / pandas.")
            print("    Install with: pip install numpy nibabel pandas")
            return

        pid = self.patient_id
        parc_path = Path(self.freesurfer_files["Parcellation"])
        if not parc_path.exists():
            print(f"[!] Parcellation not found: {parc_path}")
            print(f"    Cannot build 37-ROI clinical mask without aparc+aseg.")
            return

        ai_path = self.zai_files["asymmetry_zscore"]

        # Load parcellation + zAI map
        try:
            parc_nii = _nib.load(str(parc_path))
            parc_data = parc_nii.get_fdata().astype(_np.int32)
        except Exception as e:
            print(f"[!] Could not load parcellation {parc_path}: {e}")
            return
        try:
            ai_nii = _nib.load(str(ai_path))
            ai_data = ai_nii.get_fdata()
        except Exception as e:
            print(f"[!] Could not load zAI map {ai_path}: {e}")
            return

        if parc_data.shape != ai_data.shape:
            print(f"[!] Shape mismatch: parc={parc_data.shape}, zAI={ai_data.shape}")
            print("    The 37-ROI mask requires both volumes in the same space.")
            return

        # Build 37-ROI mask
        roi_mask = _np.isin(parc_data, CLINICAL_ROI_LABELS)
        n_roi_vox = int(roi_mask.sum())
        if n_roi_vox == 0:
            print("[!] 37-ROI mask is empty — parcellation labels not in expected ranges.")
            return

        # Apply mask
        ai_masked = _np.where(roi_mask, ai_data, 0.0)

        # Top-N regions by peak |zAI| within the 37-ROI mask
        ez = EZ_GROUND_TRUTH.get(pid, {})
        ez_side = ez.get("ez", "?")
        ez_conf = ez.get("confidence", "?")

        print()
        print("=" * 70)
        print(f"  37-ROI CLINICAL MASK VIEW for {pid}")
        print(f"  ROIs: {len(CLINICAL_ROI_LABELS)} FreeSurfer labels "
              f"(3 subcortical + 34 cortical pairs = 37)")
        print(f"  Mask voxels: {n_roi_vox:,}  ({100.0 * n_roi_vox / roi_mask.size:.1f}% of volume)")
        print(f"  Display threshold: |zAI| >= {threshold:g}")
        print(f"  MDT ground truth: EZ = {ez_side}  ({ez_conf} confidence)")
        print("=" * 70)

        # Compute peak |zAI| per FreeSurfer label and rank
        peak_per_label = []
        for lbl in CLINICAL_ROI_LABELS:
            sel = (parc_data == lbl)
            if not sel.any():
                continue
            vals = ai_data[sel]
            vals = vals[_np.isfinite(vals)]
            if vals.size == 0:
                continue
            peak_idx = _np.argmax(_np.abs(vals))
            peak_v = float(vals[peak_idx])
            mean_v = float(_np.mean(vals[_np.abs(vals) >= threshold])) if (_np.abs(vals) >= threshold).any() else 0.0
            n_thr = int((_np.abs(vals) >= threshold).sum())
            peak_per_label.append((lbl, peak_v, mean_v, n_thr, sel.sum()))

        # Sort by absolute peak descending
        peak_per_label.sort(key=lambda x: abs(x[1]), reverse=True)

        # Inline FreeSurfer Desikan-Killiany label-name table
        # (only the 37 ROI pairs we actually display).
        _DK_CORTICAL_NAMES = [
            "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal",
            "corpuscallosum", "cuneus", "entorhinal", "fusiform",
            "inferiorparietal", "inferiortemporal", "isthmuscingulate",
            "lateraloccipital", "lateralorbitofrontal", "lingual",
            "medialorbitofrontal", "middletemporal", "parahippocampal",
            "paracentral", "parsopercularis", "parsorbitalis",
            "parstriangularis", "pericalcarine", "postcentral",
            "posteriorcingulate", "precentral", "precuneus",
            "rostralanteriorcingulate", "rostralmiddlefrontal",
            "superiorfrontal", "superiorparietal", "superiortemporal",
            "supramarginal", "frontalpole", "temporalpole",
            "transversetemporal", "insula",
        ]
        _FS_NAMES = {
            10: "L-Thalamus-Proper",   49: "R-Thalamus-Proper",
            17: "L-Hippocampus",       53: "R-Hippocampus",
            18: "L-Amygdala",          54: "R-Amygdala",
        }
        for i, nm in enumerate(_DK_CORTICAL_NAMES):
            # FreeSurfer convention: 1001..1035 = ctx-lh-<name>, 2001..2035 = ctx-rh-<name>
            _FS_NAMES[1001 + i] = f"ctx-lh-{nm}"
            _FS_NAMES[2001 + i] = f"ctx-rh-{nm}"

        print(f"  Top {top_n} ROIs by peak |zAI|:")
        shown = 0
        for lbl, peak_v, mean_v, n_thr, n_vox in peak_per_label:
            if shown >= top_n:
                break
            if lbl in (10, 17, 18) or 1001 <= lbl <= 1035:
                side = "LEFT"
            else:  # 49, 53, 54 or 2001-2035
                side = "RIGHT"
            name = _FS_NAMES.get(lbl, f"label-{lbl}")
            tag = "(suprathr.)" if n_thr > 0 else "(subthr.)"
            print(
                f"    {side:5s}  peak={peak_v:+6.2f}  "
                f"#vox >= {threshold:g}: {n_thr:6d}  "
                f"{tag:12s}  {name} (label {lbl})"
            )
            shown += 1
        if shown == 0:
            print("    (no ROI had any non-zero zAI)")
        print("=" * 70)
        print()

        # Save masked zAI map
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / f"{pid}_zai_37roi_masked.nii.gz"
        _nib.save(
            _nib.Nifti1Image(ai_masked.astype(_np.float32), ai_nii.affine, ai_nii.header),
            str(tmp_path),
        )
        print(f"  [+] 37-ROI masked zAI written: {tmp_path}")

        # Launch FSLeyes
        t1_f = self.freesurfer_files["T1w"]
        bg = t1_f if Path(t1_f).exists() else (
            f"{self.zscore_group_dir}/FM_20_39/mean_perfusion.nii.gz"
        )

        cmd = (
            f"fsleyes '{bg}' "
            f"'{tmp_path}' -cm brain_colours_diverging_bwr "
            f"-dr -{threshold:g} {threshold:g} -a 80 -n 'zAI (37-ROI)' &"
        )
        print(f"\n  Command:\n  {cmd}\n")
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)
            print("  [+] FSLeyes launched")
        except Exception as e:
            print(f"  [!] Error: {e}")

    # ------------------------------------------------------------------
    # OPTION 12: Combined Presurgical View (recommended for review)
    # ------------------------------------------------------------------

    def view_combined_presurgical(self):
        """All canonical zAI layers in one FSLeyes window.

        Layer 1: group mean perfusion (gray background)
        Layer 2: continuous zAI map (semi-transparent, diverging)
        Layer 3: clinical lateralized dominance (red/blue, dominant-side-only)
        Layer 4: brain region boundaries (faint, anatomical reference)
        """
        if not self._check_zai():
            return
        group = self._pick_zscore_group() or "FM_20_39"

        mean_f = f"{self.zscore_group_dir}/{group}/mean_perfusion.nii.gz"
        parc_f = f"{self.zscore_group_dir}/{group}/consensus_parcellation.nii.gz"
        ai_f = str(self.zai_files["asymmetry_zscore"])

        # Prefer clinical lateralized maps over raw dominance (white-matter cleaner)
        cdir = self.clinical_dir
        clin_left = cdir / f"{self.patient_id}_clinical_zai_lateralized_left_dominant.nii.gz"
        clin_right = cdir / f"{self.patient_id}_clinical_zai_lateralized_right_dominant.nii.gz"

        if clin_left.exists() and clin_right.exists():
            left_f = clin_left
            right_f = clin_right
            tag = "clinical (gray matter)"
        else:
            left_f = self.zai_files["left_dominant"]
            right_f = self.zai_files["right_dominant"]
            tag = "raw (full brain)"

        print(f"\n[*] COMBINED PRESURGICAL VIEW — {tag}")
        print(f"   Patient: {self.patient_id}  |  Controls: {group}")
        print(f"   Layers: mean perfusion + zAI continuous + lateralized dominance + regions")
        print(f"   Toggle layers in FSLeyes (eye icon) to focus on specific information.")

        # Threshold tightened 2026-05-05 from |zAI|>=3 to |zAI|>=4 for
        # surgeon-facing focality (option 6 = continuous, no threshold).
        threshold = 4.0
        print(f"   Continuous zAI display threshold: |zAI| >= {threshold:g}")

        cmd = f"fsleyes '{mean_f}' "
        cmd += (
            f"'{ai_f}' -cm brain_colours_diverging_bwr "
            f"-dr -{threshold:g} {threshold:g} -a 30 -n 'zAI (continuous, |zAI|>={threshold:g})' "
        )
        if left_f.exists():
            cmd += (f"'{left_f}' -cm red -dr 0.5 1 -a 80 "
                    f"-n 'Left-dominant (L>R; suggests R EZ if interictal)' ")
        if right_f.exists():
            cmd += (f"'{right_f}' -cm blue -dr 0.5 1 -a 80 "
                    f"-n 'Right-dominant (R>L; suggests L EZ if interictal)' ")
        if Path(parc_f).exists():
            cmd += f"'{parc_f}' -cm random -dr 1 2035 -a 15 -n 'Brain Regions' "
        cmd += "&"

        print(f"\n  Command:\n  {cmd}\n")
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)
            print("  [+] FSLeyes launched")
        except Exception as e:
            print(f"  [!] Error: {e}")

    # ------------------------------------------------------------------
    # OPTION 13: Show File List
    # ------------------------------------------------------------------

    def show_file_list(self):
        """Show detailed canonical-file inventory for the current patient."""
        print("\n[*] CANONICAL FILE INVENTORY")
        print("=" * 70)
        self.check_files()

    # ------------------------------------------------------------------
    # OPTION 14: Show Quick FSLeyes Commands (canonical zAI paths)
    # ------------------------------------------------------------------

    def show_quick_commands(self):
        """Print canonical FSLeyes command snippets the user can copy/paste."""

        pid = self.patient_id
        print("\n[*] QUICK FSLEYES COMMANDS (canonical zAI paths)")
        print("=" * 70)
        print(f"Commands for Patient: {pid}")
        print("=" * 70)
        print()

        # ---- 1. Anatomy + parcellation ----
        print("# Anatomy + parcellation:")
        print(f"fsleyes Dataset/{pid}/T1w_acpc_dc_restore.nii.gz \\")
        print(f"        Dataset/{pid}/aparc+aseg.nii.gz \\")
        print("        -cm random -dr 1 2035 -a 30 -n 'Parcellation' &")
        print()

        # ---- 2. zAI overlay (canonical, primary clinical product) ----
        print("# zAI overlay (canonical, primary clinical product):")
        print(f"fsleyes Dataset/{pid}/T1w_acpc_dc_restore.nii.gz \\")
        print(f"        results_zscore/asymmetry/patients/{pid}/{pid}_asymmetry_zscore.nii.gz \\")
        print("        -cm brain_colours_diverging_bwr -dr -3 3 -a 75 -n 'zAI' &")
        print()

        # ---- 3. zAI lateralized dominance (raw) ----
        print("# zAI lateralized dominance (raw, full brain):")
        print("# (red = LEFT side has more blood flow; blue = RIGHT side has more)")
        print(f"fsleyes Dataset/{pid}/T1w_acpc_dc_restore.nii.gz \\")
        print(f"        results_zscore/asymmetry/patients/{pid}/{pid}_asymmetry_left_dominant.nii.gz \\")
        print("        -cm red -dr 0.5 3 -a 80 "
              "-n 'Left-dominant (L>R; suggests R EZ if interictal)' \\")
        print(f"        results_zscore/asymmetry/patients/{pid}/{pid}_asymmetry_right_dominant.nii.gz \\")
        print("        -cm blue -dr 0.5 3 -a 80 "
              "-n 'Right-dominant (R>L; suggests L EZ if interictal)' &")
        print()

        # ---- 4. Significant zAI clusters ----
        print("# Significant zAI clusters (statistical mask):")
        print(f"fsleyes Dataset/{pid}/T1w_acpc_dc_restore.nii.gz \\")
        print(f"        results_zscore/asymmetry/patients/{pid}/{pid}_asymmetry_significant.nii.gz \\")
        print("        -cm red-yellow -dr 0.5 1 -a 90 -n 'zAI significant' &")
        print()

        # ---- 5. Clinical lateralized clusters (surgeon-facing) ----
        if self.clinical_dir.exists():
            print("# Clinical lateralized clusters (gray-matter only, surgeon-facing):")
            print("# (red = LEFT side high blood flow; blue = RIGHT side high)")
            print(f"fsleyes Dataset/{pid}/T1w_acpc_dc_restore.nii.gz \\")
            print(f"        results_zscore/clinical/{pid}/{pid}_clinical_zai_lateralized_left_dominant.nii.gz \\")
            print("        -cm red -dr 0.5 1 -a 85 "
                  "-n 'Left-dominant (L>R; suggests R EZ if interictal)' \\")
            print(f"        results_zscore/clinical/{pid}/{pid}_clinical_zai_lateralized_right_dominant.nii.gz \\")
            print("        -cm blue -dr 0.5 1 -a 85 "
                  "-n 'Right-dominant (R>L; suggests L EZ if interictal)' &")
            print()

        # ---- 6. Combined presurgical (zAI + clinical + parcellation) ----
        print("# COMBINED presurgical (zAI continuous + clinical clusters + parcellation):")
        print(f"fsleyes Dataset/{pid}/T1w_acpc_dc_restore.nii.gz \\")
        print(f"        results_zscore/asymmetry/patients/{pid}/{pid}_asymmetry_zscore.nii.gz \\")
        print("        -cm brain_colours_diverging_bwr -dr -3 3 -a 60 -n 'zAI' \\")
        if self.clinical_dir.exists():
            print(f"        results_zscore/clinical/{pid}/{pid}_clinical_zai_lateralized_significant.nii.gz \\")
            print("        -cm hot -dr 0.5 1 -a 90 -n 'Significant' \\")
        print(f"        Dataset/{pid}/aparc+aseg.nii.gz \\")
        print("        -cm random -dr 1 2035 -a 25 -n 'Parcellation' &")
        print()

        # ---- 7. Patient vs controls — raw z-score (supplementary) ----
        if self.zscore_control_files:
            for group, files in self.zscore_control_files.items():
                print(f"# Patient vs {group} controls — raw z-score (supplementary):")
                print(f"fsleyes results_zscore/groups/{group}/mean_perfusion.nii.gz \\")
                print(f"        {files['zscore']} \\")
                print("        -cm brain_colours_diverging_bwr -dr -6 6 -a 70 -n 'Raw z-score' &")
                print()

    # ------------------------------------------------------------------
    # OPTION 15: Clinical Report (replaces former Custom Overlay Builder)
    # ------------------------------------------------------------------
    #
    # Surgeon-facing per-patient report. Imports the analyzer from
    # 06_clinical_interpretation.py, runs it for the current patient with
    # the canonical |zAI|>=4 / size>=100 thresholds, prints the report
    # to stdout, and (if FSLeyes is available + the cluster_labels NIfTI
    # is on disk) offers to launch the top-N cluster overlay (option 10
    # logic) so the surgeon can flip from the deterministic answer to
    # the spatial view in one keystroke.

    def view_clinical_report(self):
        """Run 06_clinical_interpretation.py for the current patient."""
        pid = self.patient_id

        # Lazy import — only needed for this option, and the script lives at
        # the repo root, so the module name is "06_clinical_interpretation"
        # but Python can't import a module that starts with a digit. Use
        # importlib.util to load it by file path.
        import importlib.util

        script_path = Path("06_clinical_interpretation.py").resolve()
        if not script_path.exists():
            print(f"[!] Script not found: {script_path}")
            print("    Cannot generate the clinical report.")
            return

        try:
            spec = importlib.util.spec_from_file_location(
                "_clin_interp", str(script_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[!] Could not import 06_clinical_interpretation.py: {e}")
            return

        ci = mod.ClinicalInterpreter(pid)
        if not ci.has_data():
            print(f"[!] No zAI cluster report on disk for {pid} "
                  f"(expected at {ci.cluster_csv}).")
            print(f"    Run: python 02_compute_zai.py")
            return

        report = ci.run()
        if report is None:
            print(f"[!] {pid}: cluster report exists but is empty.")
            return

        print()
        print(ci.render_text(report))

        # Offer to launch the spatial view (top-N dominant clusters).
        try:
            ans = input("\nLaunch FSLeyes for top-3 dominant clusters? [y/N]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans == "y":
            self.view_zai_dominant_clusters(top_n=3, threshold=4.0)

    # ------------------------------------------------------------------
    # MENU & MAIN LOOP
    # ------------------------------------------------------------------

    def show_menu(self):
        """Display the main visualization menu."""
        avail = self._detect_available_data()
        ez = EZ_GROUND_TRUTH.get(self.patient_id, {})

        print("\nINTERACTIVE OVERLAY VIEWER (zAI canonical, post-2026-05-04 pivot)")
        print("=" * 70)
        print(f"Current Patient: {self.patient_id}")
        if ez:
            print(f"MDT EZ: {ez.get('ez', '?')} ({ez.get('confidence', '?')} confidence)")
        else:
            print("MDT EZ: (not in ground-truth table)")
        print(f"Available data: {self._availability_str()}")
        print("Tip: 'right-dominant' cluster in temporal lobe usually means "
              "LEFT EZ")
        print("     (interictal hypoperfusion of EZ side -> opposite side "
              "looks 'dominant').")
        print("=" * 70)

        # zAI gating: if no zAI, mark options 6-10 unavailable
        zai_tag = "" if avail["zai"] else "  (no zAI for this patient)"
        clin_tag = "" if avail["clinical"] else "  (no clinical maps)"
        rawz_tag = "" if avail["raw_zscore"] else "  (no raw z-score)"

        print()
        print("Patient Selection:")
        print("   S. Switch Patient")
        print()
        print("Anatomy:")
        print("   1. Basic Anatomy (T1w + Parcellation)")
        print("   2. Perfusion Comparison (Dataset vs Dataset MNI)")
        print()
        print("Pipeline B - Voxel z-score vs Controls (Supplementary):")
        print(f"   3. Raw Voxel z-Score Map (gray-matter){rawz_tag}")
        print(f"   4. Significant Clusters Only{rawz_tag}")
        print(f"   5. Clusters + Brain Region Labels{rawz_tag}")
        print()
        print("Pipeline B - Voxel zAI vs Controls (PRIMARY CLINICAL PRODUCT):")
        print("   (default threshold |zAI|>=4 for clinical views; option 6 = continuous, no threshold)")
        print(f"   6. zAI Asymmetry Map (continuous, L vs R){zai_tag}")
        print(f"   7. Lateralized Dominance Maps "
              f"(red=LEFT high; blue=RIGHT high){zai_tag}")
        print(f"   8. Significant zAI Clusters (TFCE, |zAI|>=4){zai_tag}")
        print(f"   9. Clinical zAI (gray matter, surgeon-facing){clin_tag}")
        print(f"  10. Top-N Dominant Clusters * (focal: top 3 by peak |zAI|, region print){zai_tag}")
        print(f"  11. 37-ROI Clinical Mask View * (zAI restricted to Desikan-Killiany){zai_tag}")
        print(f"  12. Combined Presurgical View (RECOMMENDED){zai_tag}")
        print()
        print("Utilities:")
        print("  13. Show File List (canonical paths only)")
        print("  14. Show Quick FSLeyes Commands (canonical zAI paths)")
        print(f"  15. Clinical Report * (surgeon-facing, |zAI|>=4){zai_tag}")
        print(f"  16. Island-Cleaned zAI Overlay * (speckle-free NIfTI + montage){zai_tag}")
        print("   0. Exit")
        print()

        return input("Enter choice (0-16, S): ").strip().lower()

    def main_loop(self):
        """Main interactive loop."""

        print("Interactive Overlay Viewer for canonical zAI / clinical maps")
        print("=" * 70)

        # First, select patient
        if self.select_patient() is None:
            print("No patient selected. Exiting.")
            return

        # Show file inventory once at start
        self.check_files()

        dispatch = {
            "1": self.view_basic_anatomy,
            "2": self.view_perfusion_comparison,
            "3": self.view_zscore_control_map,
            "4": self.view_zscore_clusters,
            "5": self.view_zscore_with_parcellation,
            "6": self.view_zai_asymmetry_map,
            "7": self.view_zai_dominance,
            "8": self.view_zai_significant,
            "9": self.view_zai_clinical,
            "10": self.view_zai_dominant_clusters,
            "11": self.view_zai_clinical_roi_mask,
            "12": self.view_combined_presurgical,
            "13": self.show_file_list,
            "14": self.show_quick_commands,
            "15": self.view_clinical_report,
            "16": self.view_zai_cleaned_overlay,
        }

        # Options that print to stdout (no FSLeyes wait needed)
        no_pause_choices = {"0", "13", "14", "15", "s"}

        while True:
            choice = self.show_menu()

            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "s":
                if self.select_patient() is None:
                    print("Goodbye!")
                    break
                self.check_files()
                continue
            elif choice in dispatch:
                dispatch[choice]()
            else:
                print("[!] Invalid choice. Please enter 0-16 or S.")

            if choice not in no_pause_choices:
                input("\nPress Enter to return to menu...")


def main():
    """Main execution function."""
    try:
        viewer = InteractiveOverlayViewer()
        viewer.main_loop()
    except KeyboardInterrupt:
        print("\nViewer stopped by user. Goodbye!")


if __name__ == "__main__":
    main()
