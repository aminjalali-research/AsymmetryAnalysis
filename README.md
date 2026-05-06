# AsymmetryAnalysis

**Open-source toolkit for ASL perfusion epileptogenic-zone localization and lateralization in drug-resistant focal epilepsy.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-orange.svg)]()

## Table of contents

- [What this is](#what-this-is)
- [Scientific background](#scientific-background)
- [Two complementary pipelines](#two-complementary-pipelines)
- [Direction-of-abnormality inference](#direction-of-abnormality-inference)
- [Localization vs. lateralization](#localization-vs-lateralization)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Full pipeline walkthrough](#full-pipeline-walkthrough)
- [Data layout](#data-layout)
- [Results layout](#results-layout)
- [Asymmetry indices (15)](#asymmetry-indices-15)
- [Thresholding methods (13)](#thresholding-methods-13)
- [Surgical-grade thresholds and the 37-ROI clinical scope](#surgical-grade-thresholds-and-the-37-roi-clinical-scope)
- [Interactive viewer](#interactive-viewer)
- [Limitations](#limitations)
- [How to cite](#how-to-cite)
- [References](#references)
- [Related work and notes](#related-work-and-notes)
- [License](#license)

## What this is

AsymmetryAnalysis is a clinical-research toolkit for **localizing and lateralizing the epileptogenic zone (EZ)** from arterial spin labeling (ASL) perfusion MRI in patients with drug-resistant focal epilepsy. It is intended to **complement, not replace**, EEG, structural MRI, neuropsychology, and PET in multimodal presurgical evaluation. Outputs are designed for integration into multidisciplinary-team (MDT) review, with an explicit multi-modal-concordance requirement before any prediction is treated as a surgical target.

The toolkit answers two related but distinct clinical questions:

1. **Localization** — *where* in the brain is perfusion abnormal in this patient relative to healthy controls? (clinically primary)
2. **Lateralization** — *which hemisphere* contains the EZ? (an intermediate step that helps the surgeon, not a final answer)

It is open source under the MIT license and is the companion code for the manuscript *"Quantitative Assessment of Cerebral Perfusion Asymmetry Using Arterial Spin Labeling MRI"* (under review at *NeuroImage: Clinical*, 2026).

## Scientific background

The EZ characteristically demonstrates **interictal hypoperfusion** due to reduced neuronal metabolism and neurovascular uncoupling. However, in some patients — particularly those with FCD-II lesions, post-ictal imaging, or recent intense seizures — the same region shows **ipsilateral hyperperfusion** instead. A naïve asymmetry analysis that assumes hypoperfusion will systematically mis-localize the hyperperfused subset.

This toolkit explicitly handles both directions by referencing each hemisphere **independently** against a healthy-control normative database, classifying each surviving asymmetry as `L-hyper`, `L-hypo`, `R-hyper`, `R-hypo`, or `mixed`, and combining the direction with a per-cluster polarity rule to predict the EZ side.

The core of the framework is a **voxel-wise asymmetry z-score (zAI)** computed against a per-demographic-group healthy normative database, with thirteen statistical thresholding methods, optional direction-of-abnormality inference, and an explicit **surgical-grade reporting threshold** (`|zAI|≥4`, cluster ≥100 voxels, restricted to a 37-paired-ROI clinical scope).

The methodology builds on, and integrates findings from, the following recent literature:

- **Han et al., 2026, *CNS Neurosci Ther.*** — PET Z-map validated against SEEG in n=120 drug-resistant focal epilepsy patients. Reports moderate sensitivity (0.62), strong specificity (0.94), NPV (0.91), with marked regional heterogeneity (frontal/parietal κ≈0.65 vs. temporal/insular κ≈0.45). Anchors the **exclusion-utility framing**: the framework is most reliable for ruling out a hemisphere, not for confirming one.
- **Gennari et al., 2026, *Neurol Sci.* and 2025, *Epilepsia*** — head-to-head comparison of ASL voxel-based asymmetry-index analysis vs. FDG-PET in pediatric drug-resistant focal epilepsy. Validates the voxel-AI paradigm directly.
- **Shang et al., 2021, *Sci Rep.*** — voxel-based zAI for ASL EZ lateralization in MRI-negative TLE.
- **Boscolo Galazzo et al., 2016, *NeuroImage Clin.*** — simultaneous PET/ASL methodology, including the standard AI formula.
- **Sierra-Marcos et al., 2017, *Brain*** — postictal hypoperfusion as a SOZ-localizing signal in ASL; supports the time-since-last-seizure framing.
- **Ferrari et al., 2024, *Sci Rep.*** — direct evidence that 40% of FCD lesions are hyperperfused (vs. hypoperfused), correlating with EEG spike rate. Justifies the direction-inference logic.

## Two complementary pipelines

| | **Pipeline A — ROI** | **Pipeline B — Voxel** |
|---|---|---|
| **What it does** | Per-region asymmetry indices on 37 FreeSurfer ROI pairs | Per-voxel asymmetry z-score against healthy normative database |
| **Question answered** | Which hemisphere has more aggregate asymmetry? | Where in the brain are the abnormal asymmetries, and how big are they? |
| **Output** | Patient-level lateralization score, ROC vs. MDT | Thresholded zAI volume + cluster reports |
| **Pros** | Interpretable, statistically parsimonious | Spatially precise, suitable for surgical planning |
| **Cons** | Direction-blind; loses focal info | Higher dimensionality, more thresholding decisions |
| **Status** | Complete on full 15-patient cohort | Complete on F_20_39-matched subset (5/15 patients); other groups in processing |

The framework is *dual-pipeline by design*: ROI summaries support direct benchmarking and literature comparability; voxel maps support spatial localization. Both pipelines run independently on the same patient cohort and are cross-compared.

## Direction-of-abnormality inference

For each surviving zAI cluster, Pipeline B computes the **per-side perfusion z-score** against the same control cohort and classifies the cluster:

```
For each surviving zAI cluster:
    Compute  z_perf_A = median over cluster ∩ side-A  of (CBF_patient − μ_CBF_controls) / σ_CBF_controls
    Compute  z_perf_B = median over mirror voxels ∩ side-B of (same)

    If z_perf_A ≥ +z_thr  and  |z_perf_B| < z_thr:  A-hyper   →  EZ side = A
    If z_perf_A ≤ −z_thr  and  |z_perf_B| < z_thr:  A-hypo    →  EZ side = A  (deviating side IS EZ in both directions)
    If z_perf_B ≥ +z_thr  and  |z_perf_A| < z_thr:  B-hyper   →  EZ side = B
    If z_perf_B ≤ −z_thr  and  |z_perf_A| < z_thr:  B-hypo    →  EZ side = B
    Else:                                          mixed/subtle  →  flag for manual MDT review
```

with default `z_thr = 2`. **Key principle**: the EZ side is always the *deviating* side, regardless of whether the deviation is hyperperfusion or hypoperfusion. Direction (hyper vs hypo) is metadata about the underlying mechanism (interictal hypometabolism vs. FCD-II / post-ictal active hyperperfusion) — not a side-flipping switch. This **replaces** the previous polarity-flip rule, which implicitly assumed all asymmetries were hypoperfusion-driven and therefore mis-classified the FCD-II / post-ictal hyperperfused subset.

## Localization vs. lateralization

The toolkit reports **two separate performance dimensions** — never combined into a single number:

- **Lateralization performance** — left-vs-right AUC, sensitivity, specificity, accuracy. Useful for comparison with prior literature; this is the typical reporting in the field.
- **Localization performance** — *target Dice* (overlap between surviving clusters and MDT-defined target ROI), *target hit rate* (fraction of patients with ≥1 cluster within the target), *top-cluster accuracy* (whether the largest cluster is in or adjacent to the target lobe). This is the metric that actually answers the surgical question.

Critically, **the lateralization-AUC ranking and the localization-Dice ranking can disagree.** A method that produces large numbers in non-EZ regions can win on lateralization-AUC and lose on localization-Dice. We report both and recommend that downstream users prioritize localization for surgical decisions, lateralization for benchmarking.

## Installation

```bash
git clone https://github.com/aminjalali-research/AsymmetryAnalysis
cd AsymmetryAnalysis
pip install -e .
```

Runtime requirements (Python ≥ 3.10): NumPy, SciPy, Pandas, nibabel, matplotlib, seaborn, scikit-learn, openpyxl. See `requirements.txt` and `pyproject.toml` for pinned versions.

Optional:
- **FSLeyes** for the interactive 3D viewer (`interactive_viewer.py`).
- **HCP-ASL** ([physimals/hcp-asl](https://github.com/physimals/hcp-asl)) for processing raw ASL data into the MNI-space CBF maps consumed by this toolkit.

The toolkit assumes:
- All subjects (patients and controls) have been processed through HCP-ASL into MNI152 0.8mm isotropic space.
- FreeSurfer's `aparc+aseg.nii.gz` is available per subject.
- Patient demographics (`ID`, `AGE`, `SEX`) live in `clinical_spreadsheet.xlsx`.

## Quickstart

```bash
# Pipeline B — voxel zAI
python 01_build_control_normative.py --group F_20_39   # build mean/SD perfusion DB
python 02_compute_zai.py                                # patient zAI maps (control-referenced)
python 03_clinical_maps.py                              # gray-matter-masked clinical zAI
python 04_publication_figures.py                        # figures 10–16 for manuscript

# Pipeline A — ROI MDT discrimination
python 05_roi_discrimination.py                         # 15 indices × 4 aggregations × ROC

# Surgeon-facing report (Pipeline B with surgical-grade thresholds + 37-ROI scope)
python 06_clinical_interpretation.py --patient P015     # one patient
python 06_clinical_interpretation.py --all              # cohort summary

# Interactive QC (FSLeyes)
python interactive_viewer.py
```

## Full pipeline walkthrough

### Step 1 — Build control normative database (`01_build_control_normative.py`)
Loads all controls in the specified demographic group (e.g., `F_20_39`), streams Welford's online algorithm over per-voxel perfusion to produce numerically stable mean and SD volumes, then z-scores each patient's perfusion against the control mean/SD.

```bash
python 01_build_control_normative.py --group F_20_39
```

Outputs:
- `results_zscore/groups/F_20_39/mean_perfusion.nii.gz`
- `results_zscore/groups/F_20_39/sd_perfusion.nii.gz`
- `results_zscore/groups/F_20_39/consensus_parcellation.nii.gz`
- `results_zscore/patients/<PID>/<PID>_vs_F_20_39_zscore.nii.gz`

### Step 2 — Compute control-referenced zAI (`02_compute_zai.py`)
Computes per-voxel `AI(v) = 100 × (L − R) / ((L + R) / 2)` for every subject, builds a per-voxel control AI mean/SD database (Welford), then z-scores each patient AI map against it. **This is the canonical Pipeline B output** and now includes **per-cluster direction-of-abnormality inference** (new direction columns: `cluster_hemi`, `z_perf_cluster_side`, `z_perf_mirror_side`, `direction_class`, `ez_side_pred`).

```bash
python 02_compute_zai.py                                # all matched patients
python 02_compute_zai.py --patient P013                 # single patient
python 02_compute_zai.py --threshold 3.0 --min-cluster 50
```

Outputs (per patient):
- `results_zscore/asymmetry/patients/<PID>/<PID>_asymmetry_zscore.nii.gz` — zAI map
- `results_zscore/asymmetry/patients/<PID>/<PID>_asymmetry_left_dominant.nii.gz`, `_right_dominant.nii.gz`, `_significant.nii.gz`
- `results_zscore/asymmetry/patients/<PID>/<PID>_asymmetry_cluster_report.csv` — with direction-inference columns
- `results_zscore/asymmetry/patients/<PID>/<PID>_asymmetry_summary.json`

### Step 3 — Clinical-grade gray-matter zAI (`03_clinical_maps.py`)
Masks zAI to gray matter, applies stricter thresholds (`|zAI| ≥ 3`, cluster ≥ 50 voxels), produces lateralized maps for surgical review.

```bash
python 03_clinical_maps.py
```

### Step 4 — Publication figures (`04_publication_figures.py`)
Generates figures 10–16: zAI panels, AI distributions, cluster overlays. PNG, dpi=300.

### Step 5 — ROI MDT discrimination (`05_roi_discrimination.py`)
Pipeline A. Computes 15 asymmetry indices × 4 aggregation strategies on 37 ROI pairs, runs ROC + LOOCV against the MDT ground truth from `ez_ground_truth.py`.

```bash
python 05_roi_discrimination.py
```

Outputs in `ez_analysis_mdt/` — performance tables, per-patient prediction CSV, ROC plots.

### Step 6 — Surgeon-facing clinical interpretation (`06_clinical_interpretation.py`)
Default reporting tool. Applies surgical-grade thresholds (`|zAI| ≥ 4`, cluster ≥ 100 voxels), restricts to the 37-paired-ROI clinical scope, **uses direction-of-abnormality inference** for the side prediction (replacing the older polarity-flip rule, with a graceful fallback for legacy CSVs), and emits per-patient text + JSON reports plus a cohort concordance CSV.

```bash
python 06_clinical_interpretation.py --patient P015
python 06_clinical_interpretation.py --all
python 06_clinical_interpretation.py --peak-threshold 5 --size-threshold 200   # Bartolomei tier
```

Outputs:
- `results_zscore/clinical/<PID>/<PID>_surgeon_report.txt`
- `results_zscore/clinical/<PID>/<PID>_surgeon_report.json`
- `results_zscore/clinical/cohort_concordance.csv`

### Diagnostic / methodology aids
- `analyze_zai_distributions.py` — AI distribution histograms for QC.

## Data layout

```
AsymmetryAnalysis/
├── Dataset/                    # 15 patients (P013–P027)
│   └── P013/
│       ├── aparc+aseg.nii.gz
│       ├── T1w_acpc_dc_restore.nii.gz
│       └── P013_perfusion_calib_resampled_to_T1w.nii.gz   # MNI space despite filename
├── DatasetControls/            # Healthy controls organized by age/sex
│   ├── 20_29F/
│   │   └── <ctrl_id>/
│   │       ├── *_MNISpace_perfusion_calib_upsampled.nii.gz
│   │       └── aparc+aseg.nii.gz
│   ├── 30_39F/...
│   └── ...
├── clinical_spreadsheet.xlsx   # ID, AGE, SEX
└── ez_ground_truth.py          # MDT-reviewed labels
```

All volumes share MNI152 geometry (227 × 272 × 227 voxels, 0.8 mm isotropic) — no registration needed at analysis time. Patient and control data are kept private and excluded from the public repository (see `.gitignore`).

## Results layout

```
results_zscore/
├── groups/F_20_39/             # Group-level normative DB
│   ├── mean_perfusion.nii.gz
│   ├── sd_perfusion.nii.gz
│   └── consensus_parcellation.nii.gz
├── patients/<PID>/             # Patient-level zscore (perfusion)
│   └── <PID>_vs_F_20_39_zscore.nii.gz
├── asymmetry/
│   ├── groups/F_20_39/         # AI normative DB
│   │   ├── mean_asymmetry.nii.gz
│   │   └── sd_asymmetry.nii.gz
│   └── patients/<PID>/         # zAI maps + cluster reports (with direction columns)
└── clinical/<PID>/             # Gray-matter, surgical-grade
    ├── <PID>_clinical_*.nii.gz
    ├── <PID>_surgeon_report.txt
    └── <PID>_surgeon_report.json

ez_analysis_mdt/                # Pipeline A outputs (ROC, LOOCV, etc.)
visual_analysis/                # Publication figures
```

## Asymmetry indices (15)

All formulas literature-cited; canonical implementations in `src/calculator.py`.

| Category | Indices |
|---|---|
| **Ratio-based** | Laterality Index (LI), Absolute Asymmetry Index (AAI), Percent Difference, Log Ratio, Coefficient of Asymmetry |
| **Baselines** | Simple Difference, Ratio |
| **Standardized effect sizes** | Cohen's *d*, Hedges' *g*, Glass's Δ |
| **Distribution-based** | Z-score (LI), Robust Z-score (LI) |
| **Advanced** | CV Ratio, Hyperperfusion Ratio, Normalized Difference |

## Thresholding methods (13)

Implemented in `src/thresholding.py`. The **default reporting method is M11 (Quality + TFCE)** — a bilateral quality-mask pre-filter combined with Threshold-Free Cluster Enhancement.

| ID | Method | Notes |
|---|---|---|
| M1 | Fixed Z + cluster | Simple, default `|z| ≥ 1.64`, cluster ≥ 50 |
| M2 | Quality + percentile | Pre-mask + 90th-percentile cutoff |
| M3 | Benjamini–Hochberg FDR | `q = 0.05` |
| M4 | Bonferroni | `α = 0.05`, very conservative |
| M5 | TFCE | Smith & Nichols 2009 |
| M6 | Gaussian Random Fields | Worsley 1996 |
| M7 | Permutation (1000 sign-flips) | Maximal-statistic null |
| M8 | Gaussian Mixture Model | Decomposes into background/signal |
| M9 | Otsu | Bi-modal data-driven cutoff |
| M10 | Random baseline | Null-method reference |
| M11 | **Quality + TFCE** | Default; recommended |
| M12 | Quality + GMM | |
| M13 | Quality + Otsu | |

## Surgical-grade thresholds and the 37-ROI clinical scope

Research-grade thresholds (M1–M13 above) are appropriate for whole-brain exploratory analysis. **Surgical decisions require a stricter standard.** The toolkit therefore exposes three reporting tiers:

| Tier | `|zAI|` threshold | Cluster size | Source |
|---|---|---|---|
| Research / sensitivity | ≥ 3 | ≥ 50 voxels | Shang 2021 |
| **Standard clinical view (default)** | **≥ 4** | **≥ 100 voxels** | Boscolo Galazzo 2016 |
| Surgical confidence | ≥ 5 | ≥ 200 voxels | Bartolomei 2008 |
| Concordance-required | ≥ 4 + within 2 cm of MRI lesion | + EEG/neuropsych | Gennari 2025 |

The clinical-scope mask is **37 paired ROIs**: 3 subcortical pairs (thalamus, hippocampus, amygdala) + 34 Desikan–Killiany cortical pairs. Excluded: caudate, putamen, pallidum (non-typical resection targets), paracentral lobule, isthmus cingulate, occipital pole. Voxel-level zAI maps are computed and stored *unmasked* — the 37-ROI mask is applied at display and decision time only.

## Interactive viewer

`python interactive_viewer.py` opens a menu-driven FSLeyes-based interface with options 1–20:

- **Option 10** — top-N largest clusters
- **Option 11** — 37-ROI clinical-scope mask preview
- **Option 15** — clinical report (surgeon-facing, surgical-grade thresholds)

## Limitations

- Single-site, single-scanner (3T Siemens MAGNETOM Prisma), single-pipeline (HCP-ASL) cohort. External validation on independent public ASL data (OpenNeuro, ASL-BIDS) is the highest short-term priority.
- 15 patients is small relative to recent precedent (Han 2026 used n=120 for SEEG-validated PET Z-maps). Results are best interpreted as a methodological demonstration, not a definitive performance estimate.
- Pipeline A asymmetry indices are *direction-blind*; only Pipeline B with the direction-inference refinement explicitly distinguishes hypoperfusion from hyperperfusion.
- Lateralization-AUC and localization-Dice rankings can disagree; clinical decisions should prioritize localization, not lateralization-AUC ordering.
- MDT ground truth is the current standard but is not an outcome-validated gold standard; surgical-outcome (Engel-class) follow-up is in progress.
- No public-dataset external validation has yet been performed.
- Surface-based methods miss mesial structures; volumetric voxel-wise zAI (Pipeline B) helps but is not immune to partial-volume effects in the hippocampus and amygdala.

## How to cite

```bibtex
@article{jalali2026asl_localization,
  title  = {Quantitative Assessment of Cerebral Perfusion Asymmetry Using
            Arterial Spin Labeling MRI: An Open-Source Framework with
            Direction-of-Abnormality Inference for Epileptogenic Zone
            Localization and Lateralization},
  author = {Jalali, Amin and others},
  journal= {(under review at NeuroImage: Clinical)},
  year   = {2026},
  url    = {https://github.com/aminjalali-research/AsymmetryAnalysis}
}
```

## References

Key references for the methodology:

1. **Han et al., 2026** — Clinical Utility of Z-Score Distribution Mapping in EZ Localizations, *CNS Neurosci Ther*, doi: 10.1002/cns.70876.
2. **Gennari et al., 2026** — Head-to-head ASL vs FDG-PET in pediatric epilepsy, *Neurol Sci*.
3. **Gennari et al., 2025** — ASL predicts surgical outcome, *Epilepsia*, doi: 10.1111/epi.18375.
4. **Shang et al., 2021** — voxel-based zAI, *Sci Rep*, PMC8149682.
5. **Boscolo Galazzo et al., 2016** — simultaneous PET/ASL, *NeuroImage Clin*, PMC4872676.
6. **Sierra-Marcos et al., 2017** — postictal hypoperfusion ASL, *Brain*, doi: 10.1093/brain/awx241.
7. **Ferrari et al., 2024** — FCD hyperperfusion in pediatric ASL, *Sci Rep*, doi: 10.1038/s41598-024-58352-9.
8. **Smith & Nichols, 2009** — TFCE, *NeuroImage*, doi: 10.1016/j.neuroimage.2008.03.061.
9. **Bartolomei et al., 2008** — Epileptogenicity Index, *Brain*, doi: 10.1093/brain/awn111.
10. **Alsop et al., 2015** — ISMRM ASL recommended implementation, *Magn Reson Med*.

## Related work and notes

External pipelines and references explored during development:

- **HCP-ASL** ([physimals/hcp-asl](https://github.com/physimals/hcp-asl)) — the official Human Connectome Project ASL processing pipeline by the Oxford Physimals group; this is what we use to produce the MNI-space CBF maps consumed by Pipeline A and Pipeline B. Built on BASIL/`oxford_asl`.
- **ASLPrep** ([PennLINC/aslprep](https://github.com/PennLINC/aslprep)) — alternative BIDS-app ASL preprocessing pipeline; an option for users who prefer a fMRIPrep-style workflow over HCP-ASL.
- **NeuroSTORM** ([CUHK-AIM-Group/NeuroSTORM](https://github.com/CUHK-AIM-Group/NeuroSTORM)) — relevant for symmetrical voxel-based bilateral-pairing templates.
- **ASL-BIDS** ([standard](https://www.nature.com/articles/s41597-022-01615-9)) — the BIDS extension for ASL; required for portability of the framework to public datasets such as OpenNeuro `ds004199` (Presurgical MRI Epilepsy) and `ds005602` (IDEAS — Imaging Database for Epilepsy And Surgery).

A note on processing-template symmetry: the HCP-ASL pipeline transforms data to MNI152 standard space at 0.8 mm isotropic for final outputs, but uses MSMAll cortical-surface registration and preserves subject-specific anatomy on the way there — i.e. the templates used are **not** explicitly symmetrical. This is appropriate for asymmetry analysis: a symmetric template would erase precisely the bilateral-anatomy information that the asymmetry index relies on.

## License

MIT — see [`LICENSE`](LICENSE).

---

**Funding.** This work was supported by the SEAMO (Southeastern Ontario Academic Medical Organization) Innovation Fund.

**Acknowledgments.** We thank the multidisciplinary epilepsy team for clinical data review, the patients for their participation, and the Physimals group at the University of Oxford for developing and maintaining the open-source [HCP-ASL](https://github.com/physimals/hcp-asl) processing pipeline used in this work.
