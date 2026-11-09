# QC Tool: Interactive Overlay Viewer

> Optional QC and demo tool — menu-driven launcher for FSLeyes that visualizes the canonical post-cleanup pipeline outputs in 3D.

## Purpose

Provide a single command-line entry point for inspecting any pipeline output for any patient in 3D. The script presents a numbered menu of 18 options + `S` (switch patient) + `0` (exit) and, on selection, builds and runs the appropriate `fsleyes` command with the right NIfTI files, color maps, alpha values, and display ranges. It also auto-detects which canonical data layers are available for the current patient and gates options accordingly. Option 15 (Clinical Report) is the **deterministic surgeon-facing answer** and prints to stdout instead of launching FSLeyes; it offers a one-keystroke jump to the spatial top-3-cluster overlay if the surgeon wants the visual.

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

Mostly none: options 1–17 launch FSLeyes with the appropriate overlays. Option 18 and `--png` write
`visual_analysis/subject_views/<dataset>/<ID>.png` (gitignored). Some options will fall back to alternative canonical files if a preferred file (e.g. lateralized clinical map) is missing, and will print the exact `0X_*.py` script the user needs to run to fill the gap.

## Usage

```bash
python interactive_viewer.py
# 1) Pick a patient from the list (every P0NN found in Dataset/ or
#    results_zscore/asymmetry/patients/ shows up; each row notes whether
#    zAI exists, and the MDT-noted EZ side).
#    Type the list number or the ID itself (e.g. D002 with --dataset chalfont).
# 2) Pick option 1-18, S, or 0.

# Same all-layer view without the menu (options 17-18; dataset taken from the ID):
python interactive_viewer.py --subject D002               # FSLeyes
python interactive_viewer.py --subject D002 --png --open  # PNG, opened in VS Code
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
| 15 | Utilities | Patient Report (&#124;zAI&#124; >= 4, predicted side; prints to stdout) |
| 16 | Utilities | Island-Cleaned zAI Overlay (speckle-free NIfTI + montage) |
| 17 | All layers, one subject | **All layers in FSLeyes**: CBF, #3 perfusion z, #4 zAI, clinical clusters, atlas; centred on the peak cluster |
| 18 | All layers, one subject | **All layers as a PNG** for the IDE (sagittal / coronal / axial through the peak) |
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

## Looking at the results yourself (updated 2026-11-09)

### One subject, every layer: options 17–18 or `--subject` (terminal and IDE)

Options 17 and 18 show every symmetric-space layer of the current patient, centred on the strongest clinical
cluster: the largest |zAI| in the clinical map, or in the continuous zAI map when the clinical map is empty. The
same view runs without the menu through `--subject`. In that mode the dataset is taken from the ID: in-house `P0xx`,
Chalfont `C/D/N…`. Both routes first print one summary line: reference side, rule call and confidence, top lobe and
peak |zAI| (from `cohort_concordance.csv`).

```bash
conda activate Asym
python interactive_viewer.py --dataset chalfont                   # menu; pick D002, then 17 (FSLeyes) or 18 (PNG)
python interactive_viewer.py --subject D002                       # terminal / desktop: FSLeyes with every layer
python interactive_viewer.py --subject D002 --png --open          # IDE: writes a PNG and opens it (VS Code `code` if on PATH, else xdg-open)
python interactive_viewer.py --subject D002 --print               # print the FSLeyes command only (paste on another machine)
python interactive_viewer.py --subject P016 --at 60 150 90        # centre on a voxel of your choice instead of the peak
python interactive_viewer.py --dataset chalfont --all             # PNG for all 99 Chalfont patients (~1.5 s each)
python interactive_viewer.py --dataset inhouse --all              # PNG for the 17 in-house patients
```

PNGs go to `visual_analysis/subject_views/<dataset>/<ID>.png`; that folder is gitignored, so regenerate it on each
machine. In VS Code, click a PNG in the Explorer to open it, and use the arrow keys in the image preview to step
through the patients.

| Layer (bottom → top) | File | FSLeyes start state | PNG row |
|---|---|---|---|
| Template | `symreg/templates/T1_sym.nii.gz` | on | underlay |
| Atlas outline | in-house: `results_zscore/groups/<band>/consensus_parcellation.nii.gz`; Chalfont: `external_chalfont/atlas/consensus_parcellation_sym.nii.gz` | off | — |
| CBF | `symreg/sym_perf/<ID>_perf_sym.nii.gz` · `external_chalfont/sym_perf/<ID>_perf_sym.nii.gz` (TC-Flow); FSLeyes hot 0–80, PNG per-subject 2nd–98th percentile (shown in the row label) | off | 1 |
| #3 perfusion z vs controls | `<results>/patients/<ID>/<ID>_vs_<band>_zscore.nii.gz`, \|z\| 2–6 | off | 2 |
| #4 zAI, continuous | `<results>/asymmetry/patients/<ID>/<ID>_asymmetry_zscore.nii.gz`, \|z\| 2–8 | off | 3 |
| Clinical clusters | `<results>/clinical/<ID>/<ID>_clinical_zai_lateralized_significant.nii.gz`, \|z\| 2–8 | **on** | 4 |

Reading the colours: red-yellow means above the controls (#3) or above the mirror voxel (zAI); blue-lightblue means
below. Values with |z| under the lower bound are transparent. In FSLeyes, tick the eye icons in the Overlay list to
switch layers on.

Orientation differs between the two outputs. The PNG is neurological (L on the left, labelled); FSLeyes defaults to
radiological, and labels the sides itself. The PNG columns are sagittal, coronal and axial through the same voxel,
which the green crosshair marks.

Layers that do not exist are skipped and listed as `missing layers`. For example, Chalfont controls (`C…`) have CBF
only; for them use `fsleyes symreg/templates/T1_sym.nii.gz external_chalfont/sym_perf/C001_perf_sym.nii.gz -cm hot -dr 0 80 -a 60`.
Code: `AllLayersView` and `run_all_layers()` in `interactive_viewer.py`; tests: section f of `tests/test_regressions.py`.

### External cohort (Chalfont) in the menu viewer

```bash
python interactive_viewer.py --dataset chalfont     # D*/N* patients (and C* pseudo-controls) from external_chalfont/
```

Fixed 2026-11-09: every signed overlay the menu viewer launches (`brain_colours_diverging_bwr`) now carries
`-cr -0.001 0.001 -ic`, which hides exact zeros. Before this, the white centre of the colormap covered every
zero voxel and hid the anatomy. The same flags are in the command that `03_clinical_maps.py` prints.

`--dataset chalfont` points every `results_zscore/` path at `external_chalfont/results_zscore/`, takes the reference
side from `external_chalfont/metadata/subjects.csv` and uses the fixed template atlas
(`external_chalfont/atlas/consensus_parcellation_sym.nii.gz`) as the parcellation underlay. Options 1–2 (original
anatomy) are not available for Chalfont (no `Dataset/<ID>/`); options 3–12 work as for in-house patients.

### Without the viewer: the files to open

All symmetric-space maps share one grid, so any of them can be dropped on `symreg/templates/T1_sym.nii.gz` in
FSLeyes (index 0 of the x axis is the right hemisphere; FSLeyes shows this correctly from the affine).

| What | In-house | Chalfont |
|---|---|---|
| z of the voxel asymmetry (analysis 4), continuous | `results_zscore/asymmetry/patients/<ID>/<ID>_asymmetry_zscore.nii.gz` | `external_chalfont/results_zscore/asymmetry/patients/<ID>/…` |
| gray-matter clinical map, lateralized clusters, cluster table | `results_zscore/clinical/<ID>/<ID>_clinical_zai_*.nii.gz`, `…_cluster_report.csv`, `…_montage.png` | same under `external_chalfont/results_zscore/clinical/<ID>/` |
| perfusion z vs controls (analysis 3), hypo/hyper clusters | `results_zscore/patients/<ID>/<ID>_vs_<band>_zscore.nii.gz`, `…_hypo_clusters.nii.gz`, `…_zscore_montage.png` | `…/patients/<ID>/<ID>_vs_CH_ALL_*` |
| ready-made PNG montages (no viewer needed) | `results_zscore/clinical/<ID>/<ID>_clinical_zai_montage.png`, `results_zscore/asymmetry/patients/<ID>/<ID>_asymmetry_montage.png` | same paths under `external_chalfont/` |
| registration QC (warped T1 on the template, L/R coverage) | — | `external_chalfont/qc/<ID>.png`, `external_chalfont/qc/registration_qc.csv` |

Example (Chalfont patient D002, clinical map on the template; `python interactive_viewer.py --subject D002 --print` gives the full
command with all layers). Signed z-maps need a two-sided colormap with transparent zeros (`-cm red-yellow -nc
blue-lightblue -un`). A diverging map such as `brain_colours_diverging_bwr` paints every zero voxel white and hides the
anatomy.

```bash
fsleyes symreg/templates/T1_sym.nii.gz \
  external_chalfont/atlas/consensus_parcellation_sym.nii.gz -ot label -o \
  external_chalfont/results_zscore/clinical/D002/D002_clinical_zai_lateralized_significant.nii.gz -cm red-yellow -nc blue-lightblue -un -dr 2 8
```

### Tables that summarise lateralization and localization (open in a spreadsheet or pandas)

| Question | File |
|---|---|
| which side each method called, per patient (Chalfont, as-implemented rule) | `external_chalfont/results_zscore/lateralization/eval_production_rule_per_patient.csv`, `eval_per_patient.csv` |
| per-stratum accuracy, control false-positive calls | `external_chalfont/results_zscore/lateralization/eval_rule_tier_strata.csv`, `eval_control_false_positives.csv` |
| regional normative statistic and smoothed voxel mass (v2), per patient and leave-one-out control | `ez_analysis_mdt/v2/<dataset>/{R_patients,V_patients,R_loo_controls,V_loo_controls,summary,strata}.csv` |
| analysis 3 (perfusion z) lateralization and lobe localization | `ez_analysis_mdt/perfz/<dataset>/{summary,localization,localization_per_patient,localization_lobe_fraction}.csv` |
| sensitivity/specificity over all thresholds, best operating points | `ez_analysis_mdt/audit/sens_spec/`, `ez_analysis_mdt/audit/best_operating_points/summary.csv` |
| structural prior (FSL FIRST hippocampal/amygdalar volume AI) and its combination with perfusion | `ez_analysis_mdt/audit/first_chalfont/{volume_ai,results,results_flow}.csv` |
| does the index definition matter (ratio, log ratio, difference, z-difference) | `ez_analysis_mdt/audit/ai_definitions/{summary,per_subject_chalfont,per_subject_inhouse}.csv` |
| plain Flow instead of transit-corrected CBF (Chalfont) | `ez_analysis_mdt/audit/flow_variant/{summary,R_patients,V_patients}.csv` |
| in-house cohort concordance (primary rule, top lobe) | `results_zscore/clinical/cohort_concordance.csv`, `ez_analysis_mdt/clinical_*.csv` |

Localization is tabulated, not drawn: `localization_per_patient.csv` gives, per patient and method, the lobe with the
most abnormal voxels on the true and on the predicted side (`hit` = 1 when it is the clinical lobe). To see the voxels
behind a row, open the patient's clinical map (table above) on the template.

Figures that are regenerated from these tables: `python scripts/make_manuscript_figures.py` (Fig. 1–5, supplementary
case montages for both cohorts, `manuscript/figures/`) and `python 04_publication_figures.py --sym` (exploratory
figures 10–16 in `visual_analysis/`).

## Limitations

- Uses FSLeyes-specific color names (`red-yellow`, `blue-lightblue`, `red`, `blue`); other viewers will not have these.
- FSLeyes requires a desktop session with X / Wayland — it does not work over a headless SSH connection unless X11-forwarded. `interactive_viewer.py --subject <ID> --png` (and option 18) needs no display (matplotlib, Agg backend) and is the option for SSH, remote IDEs and batch use.
- Assumes outputs live in the project's standard directory layout; the only relocation switch is `--dataset chalfont`.

## Cross-references

- Pipeline outputs displayed: `results_zscore/patients/`, `results_zscore/asymmetry/patients/`, `results_zscore/clinical/`
- Related canonical scripts: `01_build_control_normative.py`, `02_compute_zai.py`, `03_clinical_maps.py`, `04_publication_figures.py`
- Install instructions: [`install.md`](install.md) covers FSLeyes setup
- Audit of cleanup that produced this script: `docs/superpowers/audits/2026-05-05-interactive-viewer-cleanup.md`
