# AsymmetryAnalysis

**Open-source toolkit for arterial spin labeling (ASL) perfusion asymmetry analysis in drug-resistant focal epilepsy.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-orange.svg)]()

## Table of contents

- [What this is](#what-this-is)
- [Main findings on the development cohort](#main-findings-on-the-development-cohort)
- [Two analysis pipelines](#two-analysis-pipelines)
- [Symmetric-template registration](#symmetric-template-registration)
- [Direction-of-abnormality inference](#direction-of-abnormality-inference)
- [Mirrored cluster pairs](#mirrored-cluster-pairs)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Full pipeline walkthrough](#full-pipeline-walkthrough)
- [Validation analyses](#validation-analyses)
- [External validation (Chalfont)](#external-validation-chalfont)
- [Enhanced estimators (v2)](#enhanced-estimators-v2)
- [Methods audit](#methods-audit)
- [Memory-safe execution](#memory-safe-execution)
- [Data layout](#data-layout)
- [Results layout](#results-layout)
- [Asymmetry indices (15)](#asymmetry-indices-15)
- [Thresholding methods (13)](#thresholding-methods-13)
- [Reporting tiers and the 37-region clinical scope](#reporting-tiers-and-the-37-region-clinical-scope)
- [Interactive viewer](#interactive-viewer)
- [Limitations](#limitations)
- [How to cite](#how-to-cite)
- [References](#references)
- [License](#license)

## What this is

AsymmetryAnalysis quantifies left-right perfusion asymmetry from ASL MRI and compares it with a healthy normative cohort, to support lateralization of the epileptogenic zone (EZ) in presurgical evaluation. It is a research tool. Its output is one input to multidisciplinary-team (MDT) review and must not be used alone to choose the side of surgery.

It is the companion code for the manuscript *Normative voxel-wise arterial spin labeling asymmetry for lateralizing the epileptogenic zone in drug-resistant focal epilepsy* (in preparation). The manuscript itself is not part of this repository.

## Main findings on the development cohort

Cohort: 17 adults (13 with a unilateral EZ hypothesis: 8 left, 5 right) and 180 healthy controls.

| Analysis | Result |
|---|---|
| Calibration, gray matter with \|zAI\| ≥ 1.96 | 31% in standard MNI space; 12.6% after symmetric registration; 6.9% for controls (leave-one-out) |
| Regional pipeline, best aggregation | AUC 0.60 (95% CI 0.28–0.90); 8/13 under leave-one-out, equal to always predicting left |
| Voxel-wise primary rule, clinical tier | 7/13 agree with MDT; balanced accuracy 0.55 |
| Best voxel-wise rule of all tested | 8/13; balanced accuracy 0.61 |
| Hyperperfusion-dominant patients | 3 of 13, but the class tracks each patient's global CBF offset from the control band (ρ = 0.87), so it is an artefact of off-site controls, not a phenotype |
| Noise budget | control SD of the voxel-wise AI is 55–70% in gray matter at 0.8 mm, so \|zAI\| ≥ 4 is unreachable at a typical voxel; region means give 9–10% (AI units) |

In this cohort no rule lateralized better than chance once the left-sided majority is taken into account. The 2026-10-20 audit (`docs/audit/METHODS_AUDIT_2026-10-20.md`) explains why: the voxel-wise test has no sensitivity to physiological asymmetries (10–20%), the direction class measures a global offset, and the sample cannot resolve a 0.65-vs-0.54 difference. The external cohort below shows what does work.

### External cohort (Chalfont, 99 patients, 66 same-scanner controls)

| Analysis | Result |
|---|---|
| Pre-specified voxel-wise rule, clinical tier | 58/93 unilateral patients (balanced accuracy 0.63; majority baseline 0.54), but 56% of healthy controls scored leave-one-out receive a strong or moderate call |
| Regional normative temporal statistic (v2) | 61/93 by sign (0.65; AUC 0.69); calibrated to a 5% control false-positive rate: 13 calls, 12 correct |
| Smoothed voxel-wise z-mass (v2) | 59/93 by sign (0.63); 8 calibrated calls, 8 correct |
| Antisymmetric multivariate classifier (exploratory) | 55–59/93 under nested leave-one-out: no gain over the fixed regional statistic |

## Two analysis pipelines

| | **Pipeline A: regional** | **Pipeline B: voxel-wise** |
|---|---|---|
| **Input** | Mean CBF in 37 homologous region pairs | Voxel-wise CBF after symmetric registration |
| **Statistic** | 15 asymmetry indices, 9 aggregation methods | Asymmetry z-score (zAI) against age-band controls |
| **Output** | One lateralization score per patient | Thresholded zAI maps, cluster reports, patient report |
| **Direction** | Direction-blind | Each cluster classed hypo- or hyperperfused |

## Symmetric-template registration

The voxel asymmetry index compares each voxel with its mirror across the midline. This is valid only if homologous structures sit at mirror-symmetric positions, which MNI152 does not guarantee. Every subject is therefore registered (FSL flirt + fnirt) to a study-specific template built as the average of the mean control T1 and its left-right flip (`symreg/`). Without this step, about 31% of gray matter exceeds |zAI| ≥ 1.96 in every patient.

## Direction-of-abnormality inference

An asymmetry index cannot tell left hyperperfusion from right hypoperfusion. For each cluster, `02_compute_zai.py` compares both hemispheres with the control perfusion mean and SD:

```
z_A = median perfusion z-score over the cluster (hemisphere A)
z_B = median perfusion z-score over its mirror (hemisphere B)

|z_A| >= 2 and |z_B| < 2   -> hypo or hyper on A   -> predicted side A
|z_B| >= 2 and |z_A| < 2   -> hypo or hyper on B   -> predicted side B
both >= 2                  -> mixed                -> flagged for review
neither >= 2               -> subtle               -> larger |z| side, low certainty
```

The predicted side is the deviating hemisphere, whether the deviation is hypo- or hyperperfusion. The direction class is reported for every cluster so that the MDT can weigh it against seizure timing and histology.

## Mirrored cluster pairs

The zAI map is antisymmetric: zAI at a voxel equals minus zAI at its mirror. Every abnormality therefore appears as two clusters, one in each hemisphere, with opposite sign, the same direction class and the same predicted side. Counts and votes must use one member per pair (rows where `cluster_hemi == ez_side_pred` in the cluster report). Rules that assign a side from the sign of the asymmetry alone cancel out if both members are counted.

## Installation

```bash
git clone https://github.com/aminjalali-research/AsymmetryAnalysis
cd AsymmetryAnalysis
pip install -e .
```

Requirements: Python ≥ 3.10 with NumPy, SciPy, pandas, nibabel, matplotlib, seaborn, scikit-learn and openpyxl (see `requirements.txt`). **FSL** (flirt, fnirt) is required for the symmetric registration in `symreg/`. Optional: FSLeyes for `interactive_viewer.py`; [HCP-ASL](https://github.com/physimals/hcp-asl) to produce the CBF maps from raw data.

Assumptions: every subject has been processed with HCP-ASL into MNI152 space at 0.8 mm, with a FreeSurfer `aparc+aseg.nii.gz`; patient demographics (ID, AGE, SEX) are in `clinical_spreadsheet.xlsx`.

## Quickstart

```bash
# Pipeline B (voxel-wise), symmetric space
python symreg/build_sym_template.py            # once: build the symmetric T1 template
python symreg/make_jobs.py full | tr '\t' ' ' | \
  xargs -P 8 -L 1 bash symreg/reg_subject.sh     # register 197 subjects (FSL flirt + fnirt)
scripts/run_capped.sh sym 34G symreg/logs/run.out -- bash symreg/run_sym_pipeline.sh   # 01 -> 02 (per band) -> 03 -> 06 -> thresholding, memory-capped
python symreg/check_full_calibration.py        # calibration check (target 5-15%)

# Pipeline A (regional)
python scripts/build_comprehensive_asymmetry.py   # per-patient ROI asymmetry tables
python 05_roi_discrimination.py                    # 15 indices x 9 aggregations, ROC

# Patient reports and cohort concordance
python 06_clinical_interpretation.py --patient P015
python 06_clinical_interpretation.py --all
```

## Full pipeline walkthrough

### Step 1: control normative database (`01_build_control_normative.py`)
Accumulates per-voxel perfusion mean and SD for an age band (Welford's algorithm) and z-scores each patient's perfusion against it.

```bash
python 01_build_control_normative.py --sym --rebuild-group               # all bands
python 01_build_control_normative.py --sym --group FM_20_39              # one band
```

### Step 2: asymmetry z-score (`02_compute_zai.py`)
Computes `AI(v) = 200 (CBF(v) − CBF(v̄)) / (CBF(v) + CBF(v̄))` per subject (v̄ = mirror voxel; defined only where both voxels have positive perfusion, since 2026-10-20), builds the control AI mean and SD (5% SD floor; the mirror max-pooling step in the code is a no-op because every AI map is exactly antisymmetric), and z-scores each patient. Clusters are formed at |zAI| ≥ 1.96 with ≥ 50 voxels; the reporting tiers in step 6 keep a cluster when its peak reaches the tier value and its extent at 1.96 reaches the tier size. Each run processes only the patients in the given band, so run it once per band.

```bash
for b in FM_20_39 FM_40_59 FM_60_79; do
  python 02_compute_zai.py --sym --rebuild --group $b
done
```

The cluster report includes `cluster_hemi`, `z_perf_cluster_side`, `z_perf_mirror_side`, `direction_class` and `ez_side_pred`.

### Step 3: gray-matter clinical maps (`03_clinical_maps.py`)
Masks zAI to gray matter within the 37-region clinical scope and writes lateralized maps.

```bash
python 03_clinical_maps.py --sym          # add --no-clinical-mask to keep all gray matter
```

### Step 4: figures (`04_publication_figures.py`)
Exploratory figures 10–16 (`python 04_publication_figures.py --sym`).

### Step 5: regional discrimination (`05_roi_discrimination.py`)
Reads `results_analysis/<PID>/<PID>_comprehensive_asymmetry_analysis.csv` and evaluates 9 aggregation methods against the MDT labels in `ez_ground_truth.py` (13 unilateral patients). Bilateral and unclear cases are scored by asymmetry magnitude, not forced onto a side.

### Step 6: patient report (`06_clinical_interpretation.py`)
Applies a reporting tier (default |zAI| ≥ 4, ≥ 100 voxels), restricts clusters to the 37-region scope, assigns sides with the direction rule, and writes a text and JSON report per patient plus `results_zscore/clinical/cohort_concordance.csv`.

```bash
python 06_clinical_interpretation.py --all
python 06_clinical_interpretation.py --all --peak-threshold 5 --size-threshold 200
```

## Validation analyses

| Script | Purpose |
|---|---|
| `scripts/clinical_stratified_analysis.py` | Decision rule × spatial scope × threshold tier; balanced accuracy; clinical strata; agreement between pipelines |
| `scripts/regional_loocv.py` | Leave-one-out accuracy of the regional scores |
| `scripts/aggregation_separability.py` | Paired bootstrap of AUC differences between aggregation methods |
| `scripts/per_index_discrimination.py` | ROC of each of the 15 indices |
| `scripts/rerun_thresholding_methods.py` | Coverage and cluster counts of the 13 thresholding methods |
| `symreg/check_full_calibration.py` | Gray-matter fraction with \|zAI\| ≥ 1.96 per patient |
| `scripts/v2/multimetric_comparison.py` | Call rate, PPV, per-side sensitivity, balanced accuracy, AUC, LR+, κ, control false-positive rate, strata and the call-rate × PPV Pareto front for every method |
| `scripts/audit/voxel_noise_budget.py [fwhm]` | Control SD of the asymmetry by region and scale (voxel, smoothed, regional) |
| `scripts/audit/regional_normative_prototype.py` | Regional LI SD per band and leave-one-out false positives |

Always report balanced accuracy and the majority-class baseline alongside the number of correctly lateralized patients, and never rank methods on one metric.

## External validation (Chalfont)

`dataset_config.py` adds `--dataset chalfont` to steps 01, 02, 03 and 06. The external cohort (3 T GE multi-delay eASL, transit-time-corrected CBF; 66 controls and 99 patients after a pre-specified quality rule) is registered to the same symmetric template; gray matter and the 37 region pairs come from a fixed template atlas (`scripts/chalfont/build_template_atlas.py`) because no FreeSurfer parcellation exists. The analysis plan with the primary endpoint was written before any result was read (`scripts/chalfont/CHALFONT_PLAN.md`).

```bash
python scripts/chalfont/stage_chalfont.py            # subjects.csv, groups.json (no PII copied)
python scripts/chalfont/asl_quality.py               # pre-specified QC -> qc_exclusions.json
python scripts/chalfont/build_template_atlas.py      # fixed DK atlas in template space
bash scripts/chalfont/run_registration.sh            # ASL -> T1 rigid, T1 -> T1_sym
scripts/run_capped.sh chalfont 34G external_chalfont/logs/run.out -- \
  env CHALFONT_JOBS=5 bash scripts/chalfont/run_chalfont_pipeline.sh tc all
```

Outputs go to `external_chalfont/results_zscore/` (gitignored), including `lateralization/eval_*.csv` with the primary endpoint, the rule × tier grid and the control false-positive rates.

## Enhanced estimators (v2)

Added after the primary analysis and applied identically to both cohorts (`scripts/v2/`):

```bash
python scripts/v2/prep_subject_features.py --dataset inhouse    # ROI means + 8-mm smoothed maps on a mirror-symmetric 2.4-mm grid
python scripts/v2/prep_subject_features.py --dataset chalfont
python scripts/v2/analyze_v2.py --dataset chalfont               # regional z_LI, smoothed z-mass, leave-one-out calibration
python scripts/v2/multimetric_comparison.py --dataset chalfont   # all metrics, Pareto front
python scripts/v2/advanced_lateralization.py --dataset chalfont  # exploratory covariate model + antisymmetric classifier
```

The regional statistic is the mean, over ten temporal pairs, of the z-score of the laterality index against the controls of the same cohort; negative means the left side is lower. A call is made only when the statistic exceeds the 95th percentile of the leave-one-out controls, so the false-positive rate is known before any patient is read.

## Methods audit

`docs/audit/METHODS_AUDIT_2026-10-20.md` records, with evidence, why the voxel-wise method cannot lateralize (noise budget, global-offset direction class, tier definition, no cluster null, no regional control reference), two independent code reviews, the enhanced estimators on both datasets, the multi-metric view and the recommended path (regional inference, same-scanner controls, calibrated thresholds, acquisition SNR, multimodal data, outcome-defined reference sides).

## Memory-safe execution

Steps 01–03 can use 5–23 GB per subject job on the 0.8 mm grid. Launch long runs through `scripts/run_capped.sh <name> <MemoryMax> <log> -- <cmd>`: the job runs in its own systemd scope with a hard memory cap, no swap, pinned BLAS threads and detached from the terminal, so an out-of-memory condition kills one job, never the desktop session. Never run two pipelines at once; `CHALFONT_JOBS=5` is a safe default on a 64 GB machine.

## Data layout

```
AsymmetryAnalysis/
├── Dataset/                    # 17 patients (P013–P030, P014 excluded)
│   └── P013/
│       ├── aparc+aseg.nii.gz
│       ├── T1w_acpc_dc_restore.nii.gz
│       └── P013_perfusion_calib_resampled_to_T1w.nii.gz   # MNI space despite the name
├── DatasetControls/            # 3 sex-pooled age bands, n = 60 each (HCP Aging/Development)
│   ├── FM_20_39/<ctrl_id>/{*_MNISpace_perfusion_calib_upsampled.nii.gz, aparc+aseg.nii.gz}
│   ├── FM_40_59/...
│   └── FM_60_79/...
├── symreg/                     # symmetric template, registration scripts, registered volumes
├── clinical_spreadsheet.xlsx   # ID, AGE, SEX
└── ez_ground_truth.py          # MDT labels (L, R, B, U) and certainty grades
```

Patient and control data are private and excluded from the repository (see `.gitignore`).

## Results layout

```
results_zscore/
├── groups/FM_<band>/           # perfusion mean/SD, consensus parcellation
├── patients/<PID>/             # perfusion z-score
├── asymmetry/
│   ├── groups/FM_<band>/       # AI mean/SD, brain mask
│   ├── patients/<PID>/         # zAI map, cluster report with direction columns
│   └── summary/                # thresholding comparison
└── clinical/<PID>/             # gray-matter maps and patient reports
results_analysis/               # per-patient regional asymmetry tables (Pipeline A input)
ez_analysis_mdt/                # regional ROC, LOOCV, stratified voxel-wise analyses
ez_analysis_mdt/v2/<dataset>/   # enhanced estimators: summary, strata, multimetric_comparison
v2_features/<dataset>/          # ROI means and smoothed maps (rebuild with scripts/v2/prep_subject_features.py)
external_chalfont/              # external cohort: metadata, atlas, sym_perf, results_zscore (all gitignored)
visual_analysis/                # figures and per-patient overlays
```

## Asymmetry indices (15)

Implemented in `src/calculator.py`, each with a primary statistical source.

| Category | Indices |
|---|---|
| Ratio-based | Laterality index (LI), absolute asymmetry index, percent difference, log ratio, coefficient of asymmetry |
| Baselines | Simple difference, ratio |
| Effect sizes | Cohen's *d*, Hedges' *g*, Glass's Δ |
| Within-patient distribution | z-score of LI, robust z-score of LI |
| Other | Coefficient-of-variation ratio, hyperperfusion ratio, normalized difference |

Several indices are monotone transforms of one another; seven signed indices give identical AUC.

## Thresholding methods (13)

Implemented in `src/thresholding.py`. The default is M11 (quality mask + TFCE). The comparison reports coverage and cluster counts only; it is not an accuracy ranking, because the MDT label gives a side and not a voxel-wise extent.

| ID | Method | Notes |
|---|---|---|
| M1 | Fixed z + cluster | \|z\| ≥ 1.64, ≥ 50 voxels |
| M2 | Quality + percentile | 90th percentile within the bilateral quality mask |
| M3 | Benjamini–Hochberg FDR | q = 0.05 |
| M4 | Bonferroni | α = 0.05 |
| M5 | TFCE | Smith and Nichols 2009 |
| M6 | Gaussian random field | Worsley et al. 1996 |
| M7 | Permutation | spatial permutation of \|z\| within the quality mask, cluster-extent null, 200 permutations |
| M8 | Gaussian mixture model | |
| M9 | Otsu | |
| M10 | Random baseline | 3% of voxels, null reference |
| M11 | **Quality + TFCE** | Default |
| M12 | Quality + GMM | |
| M13 | Quality + Otsu | |

## Reporting tiers and the 37-region clinical scope

| Tier | \|zAI\| | Cluster size | Volume (native 2.5-mm ASL voxels) |
|---|---|---|---|
| Research | ≥ 3 | ≥ 50 voxels | 26 mm³ (1.7) |
| Clinical (default) | ≥ 4 | ≥ 100 voxels | 51 mm³ (3.3) |
| Surgical | ≥ 5 | ≥ 200 voxels | 102 mm³ (6.5) |
| Native resolution | ≥ 4 | ≥ 610 voxels | 312 mm³ (20) |

The clinical tier matches the z > 4 operating point reported for symmetrized FDG-PET asymmetry maps (Aslam et al. 2022). As implemented, clusters are formed at |zAI| ≥ 1.96 with ≥ 50 voxels and a tier keeps a cluster whose peak reaches the tier value and whose extent at 1.96 reaches the tier size; the audit shows that at the native ASL noise level the clinical tier is reached only at tissue boundaries, coverage edges and cavities. The clinical scope is 37 region pairs: thalamus, hippocampus, amygdala and the 34 Desikan–Killiany cortical regions. Caudate, putamen and pallidum are excluded. zAI maps are stored unmasked; the scope is applied when reports are generated.

## Interactive viewer

`python interactive_viewer.py` opens a menu-driven FSLeyes interface (options 0–16). It currently overlays maps on the original-space anatomy; use the symmetric template in `symreg/templates/` as the underlay when viewing symmetric-space zAI maps.

## Limitations

- 17 patients (13 unilateral). Every AUC interval includes chance.
- The reference is the MDT hypothesis; only 5 patients have stereo-EEG or histopathological support, and follow-up is short.
- Controls are HCP Aging/Development data acquired at other sites with the same protocol. The 20–39 band has no controls aged 22–35 (30 aged ≤ 21, 30 aged ≥ 36).
- One patient (P024) was imaged after a temporal resection.
- The interval from the last seizure to imaging was not recorded, although it governs the direction of perfusion change.
- The symmetric template is study specific.
- The external cohort has no postoperative outcome; 52 of 99 reference sides are clinical hypotheses, and sex and age are unrecorded for part of it.
- At 0.8 mm the voxel-wise asymmetry test cannot detect asymmetries of the size reported in TLE; regional or smoothed inference is required (see the audit).

## How to cite

```bibtex
@unpublished{jalali2026asl,
  title   = {Normative voxel-wise arterial spin labeling asymmetry for lateralizing
             the epileptogenic zone in drug-resistant focal epilepsy},
  author  = {Jalali, Amin and Rai, Aikam and Suppa Modrusan, Matthew and Mullett, Ada and
             Levy, Ron and Boisse Lomax, Lysa and Shukla, Garima and Winston, Gavin P.},
  note    = {Manuscript in preparation},
  year    = {2026},
  url     = {https://github.com/aminjalali-research/AsymmetryAnalysis}
}
```

## References

1. Aslam S, et al. Asymmetry index in anatomically symmetrized FDG-PET for improved epileptogenic focus detection in pharmacoresistant epilepsy. *J Neurosurg* 2022;138:828–836. doi:10.3171/2022.6.JNS22717
2. Boscolo Galazzo I, et al. Cerebral metabolism and perfusion in MR-negative individuals with refractory focal epilepsy assessed by simultaneous acquisition of 18F-FDG PET and arterial spin labeling. *NeuroImage Clin* 2016;11:648–657. doi:10.1016/j.nicl.2016.04.005
3. Didelot A, et al. Voxel-based analysis of asymmetry index maps increases the specificity of 18F-MPPF PET abnormalities for localizing the epileptogenic zone in temporal lobe epilepsies. *J Nucl Med* 2010;51:1732–1739. doi:10.2967/jnumed.109.070938
4. Gaxiola-Valdez I, et al. Seizure onset zone localization using postictal hypoperfusion detected by arterial spin labelling MRI. *Brain* 2017;140:2895–2911. doi:10.1093/brain/awx241
5. Gennari AG, et al. Lesion volume and spike frequency on EEG impact perfusion values in focal cortical dysplasia: a pediatric arterial spin labeling study. *Sci Rep* 2024;14:7601. doi:10.1038/s41598-024-58352-9
6. Han J, et al. Clinical utility of Z-score distribution mapping in epileptogenic zone localizations. *CNS Neurosci Ther* 2026;32:e70876. doi:10.1002/cns.70876
7. Kirk TF, et al. Arterial spin labelling perfusion MRI analysis for the Human Connectome Project Lifespan Ageing and Development studies. *Imaging Neurosci* 2025;3. doi:10.1162/imag_a_00444
8. Mattioli P, et al. Head to head comparison between arterial spin labelling MRI and [18F]FDG-PET in presurgical evaluation of epilepsy in children: the role of voxel-based asymmetry index analysis. *Neurol Sci* 2026. doi:10.1007/s10072-026-08891-y
9. Smith SM, Nichols TE. Threshold-free cluster enhancement. *NeuroImage* 2009;44:83–98. doi:10.1016/j.neuroimage.2008.03.061
10. Alsop DC, et al. Recommended implementation of arterial spin-labeled perfusion MRI for clinical applications. *Magn Reson Med* 2015;73:102–116. doi:10.1002/mrm.25197

## License

MIT, see [`LICENSE`](LICENSE).

---

**Funding.** Southeastern Ontario Academic Medical Organization (SEAMO) Innovation Fund (SEA-24-011).

**Acknowledgments.** We thank the patients, the epilepsy multidisciplinary team of Kingston Health Sciences Centre, and the Physimals group for the open-source [HCP-ASL](https://github.com/physimals/hcp-asl) pipeline.
