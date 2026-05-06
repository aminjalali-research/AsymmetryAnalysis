# QC Tool: Interactive Overlay Viewer

> Optional QC and demo tool — menu-driven launcher for FSLeyes that visualizes the canonical post-cleanup pipeline outputs in 3D.

## Purpose

Provide a single command-line entry point for inspecting any pipeline output for any patient in 3D. The script presents a numbered menu of 15 options + `S` (switch patient) + `0` (exit) and, on selection, builds and runs the appropriate `fsleyes` command with the right NIfTI files, color maps, alpha values, and display ranges. It also auto-detects which canonical data layers are available for the current patient and gates options accordingly. Option 15 (Clinical Report) is the **deterministic surgeon-facing answer** and prints to stdout instead of launching FSLeyes; it offers a one-keystroke jump to the spatial top-3-cluster overlay if the surgeon wants the visual.

This tool is for **interactive QC**, ad hoc exploration, and live demos. It does not modify any pipeline output. All inputs must already exist (run steps 1–4 first).

## Script

`interactive_viewer.py` (canonical, at repo root)

## Inputs (canonical only)

- `Dataset/<PID>/` — T1w-space anatomy and perfusion (always present)
- `Dataset MNI/<PID>/` — MNI-space anatomy and perfusion (always present)
- `results_zscore/groups/<group>/` — control-group statistics (mean / SD / parcellation)
- `results_zscore/patients/<PID>/` — Pipeline B Step 1-2 raw voxel z-score maps (supplementary)
- `results_zscore/asymmetry/patients/<PID>/` — Pipeline B Step 3 zAI maps (PRIMARY CLINICAL PRODUCT)
- `results_zscore/clinical/<PID>/` — Pipeline B Step 3 gray-matter zAI maps (surgeon-facing)
- `ez_ground_truth.py` — MDT-noted EZ side displayed in the menu header

> All references to deleted directories (`results/`, `results_voxel/`, `results_voxel_roi/`, `laterality_maps/`, `*_ez_analysis/`, `nifti_extraction_results/`, etc.) were removed during the Phase 2 cleanup pass on 2026-05-04. The script no longer reads from any of those.

## Outputs

None directly — launches FSLeyes with appropriate overlays. Some options will fall back to alternative canonical files if a preferred file (e.g. lateralized clinical map) is missing, and will print the exact `0X_*.py` script the user needs to run to fill the gap.

## Usage

```bash
python interactive_viewer.py
# 1) Pick a patient from the list (every P0NN found in Dataset/ or
#    results_zscore/asymmetry/patients/ shows up; each row notes whether
#    zAI exists, and the MDT-noted EZ side).
# 2) Pick option 1-16, S, or 0.
```

The menu header shows:

```
Current Patient: P013
MDT EZ: L (High confidence)
Available data: [+] anatomy  [+] MNI anatomy  [+] raw z-score  [+] zAI  [+] clinical
```

Options that depend on missing data are tagged in the menu (e.g. `(no zAI for this patient)`) so the clinician knows before clicking.

## Menu structure

| # | Section | Option |
|---|---------|--------|
| S | Patient selection | Switch patient |
| 1 | Anatomy | Basic Anatomy (T1w + Parcellation) |
| 2 | Anatomy | Perfusion Comparison (Dataset vs Dataset MNI) |
| 3 | Pipeline B z-score (supplementary) | Raw Voxel z-Score Map (gray-matter) |
| 4 | Pipeline B z-score (supplementary) | Significant Clusters Only |
| 5 | Pipeline B z-score (supplementary) | Clusters + Brain Region Labels |
| 6 | Pipeline B zAI (PRIMARY) | zAI Asymmetry Map (continuous, L vs R) |
| 7 | Pipeline B zAI (PRIMARY) | Lateralized Dominance Maps (Red=L>R, Blue=R>L) |
| 8 | Pipeline B zAI (PRIMARY) | Significant zAI Clusters (TFCE, &#124;zAI&#124; >= 4) |
| 9 | Pipeline B zAI (PRIMARY) | Clinical zAI (gray matter, surgeon-facing) |
| 10 | Pipeline B zAI (PRIMARY) | **Top-N Dominant Clusters** (focal: top 3 by peak &#124;zAI&#124;, with region print) |
| 11 | Pipeline B zAI (PRIMARY) | **37-ROI Clinical Mask View** (zAI restricted to Desikan-Killiany clinical regions) |
| 12 | Pipeline B zAI (PRIMARY) | Combined Presurgical View (recommended for review) |
| 13 | Utilities | Show File List (canonical paths only) |
| 14 | Utilities | Show Quick FSLeyes Commands (canonical zAI paths) |
| 15 | Utilities | Custom Overlay Builder |
| 16 | Utilities | Quality Control View |
| 0 | Utilities | Exit |

Options 6–12 are the **primary clinical product** of this project. Option 12 is the multi-layer "mission control" overlay for an MDT presurgical review meeting (combines zAI continuous + lateralized clinical clusters + parcellation, all on the group mean perfusion underlay).

Options 10 and 11 (added 2026-05-05) are the **focal surgeon-facing views**:

- **Option 10 — Top-N Dominant Clusters.** Reads the per-patient cluster_report CSV (preferring the gray-matter clinical version `<PID>_clinical_zai_cluster_report.csv`), filters to clusters with peak |zAI| ≥ 4, sorts by absolute peak, and prints the top 3 to stdout (cluster id, side, peak, size, anatomical region). It then writes a binary mask of just those clusters to a temp NIfTI and launches FSLeyes with `T1w + faded continuous zAI + top-N mask in yellow`. Motivation: the canonical zAI map for some patients (e.g. P013) contains 60K+-voxel clusters in non-EZ regions that visually drown out a smaller surgeon-relevant cluster. This view filters to the focal hits and lets the surgeon read off the anatomical regions in plain text without scrolling cluster reports.
- **Option 11 — 37-ROI Clinical Mask View.** Loads `Dataset/<PID>/aparc+aseg.nii.gz`, builds a binary mask of the 37 clinically-relevant Desikan-Killiany pairs (3 subcortical pairs: thalamus / hippocampus / amygdala, plus 34 cortical pairs from FreeSurfer labels 1001–1035 and 2001–2035), multiplies the zAI map by that mask, and displays the result. Also prints the top 5 ROIs by peak |zAI| with their anatomical names. Useful for suppressing non-cortical / non-EZ regions (cerebellum, brainstem, ventricles, basal ganglia) that the bare zAI map otherwise shows as noisy signal.

Options 3–5 are **supplementary** — raw blood-flow z-score (no asymmetry computation). They show "where blood flow differs from controls", whereas options 6–12 show "where the L-R difference between the two hemispheres is itself unusual".

### Default thresholds

Surgeon-facing options (8, 9, 10, 11, 12) display zAI with a tightened threshold of **|zAI| ≥ 4** (changed 2026-05-05 from |zAI| ≥ 3, which let almost the entire gray-matter volume show up as "abnormal"). Option 6 keeps the continuous map with no threshold for diagnostic exploration.

## Auto-detection

Before showing the menu, the viewer probes for each of the five canonical data layers:

```python
{
  'anatomy':     Dataset/<PID>/T1w_acpc_dc_restore.nii.gz,
  'mni_anatomy': Dataset MNI/<PID>/T1w_restore.nii.gz,
  'raw_zscore':  results_zscore/patients/<PID>/<PID>_vs_<group>_zscore.nii.gz,
  'zai':         results_zscore/asymmetry/patients/<PID>/<PID>_asymmetry_zscore.nii.gz,
  'clinical':    results_zscore/clinical/<PID>/<PID>_clinical_zai_*.nii.gz,
}
```

The header line and per-option labels reflect what is actually on disk.

## Dependencies

- **Python:** `pathlib`, `subprocess` (stdlib only on the Python side); `ez_ground_truth.py` imported for MDT EZ display
- **Runtime tools (NOT Python packages):**
  - **FSLeyes** — image viewer, must be on `PATH`. Install via the FSL distribution.
  - **FSL** — only required to install FSLeyes; FSL itself is not invoked.
- Pipeline outputs from steps 1–4 (the viewer has no inputs of its own)

If FSLeyes is unavailable the script still prints the exact command for the user to run (or paste into a remote machine that has it).

## Limitations

- Uses FSLeyes-specific color names (`brain_colours_diverging_bwr`, `red`, `blue`); other viewers will not have these.
- Requires a desktop session with X / Wayland — does not work over a headless SSH connection unless X11-forwarded.
- Assumes outputs live in the project's standard directory layout (no command-line arguments to relocate them).

## Cross-references

- Pipeline outputs displayed: `results_zscore/patients/`, `results_zscore/asymmetry/patients/`, `results_zscore/clinical/`
- Related canonical scripts: `01_build_control_normative.py`, `02_compute_zai.py`, `03_clinical_maps.py`, `04_publication_figures.py`
- Install instructions: [`install.md`](install.md) covers FSLeyes setup
- Audit of cleanup that produced this script: `docs/superpowers/audits/2026-05-05-interactive-viewer-cleanup.md`
