# AsymmetryAnalysis — Project Guide for Claude Code

## What This Project Does

This is a **clinical neuroimaging research project** for **epilepsy presurgical planning**. It analyzes ASL (Arterial Spin Labeling) perfusion MRI scans to find brain regions with abnormal blood flow that may indicate epileptogenic zones.

**The core question:** "Where in this patient's brain is blood flow abnormal compared to healthy people, and is the abnormality lateralized (left vs right)?"

## ✅ CURRENT STATE — 2026-06-23 (READ FIRST; supersedes older sections below)

The 2026-06-22 upgrade is **complete**. Full detail in
`docs/superpowers/2026-06-22-manuscript-consolidation-spec.md`, the audit ledger
`docs/superpowers/audits/2026-06-22-contradiction-and-improvement-audit.md`, and the verified
Pipeline B numbers in `symreg/PIPELINE_B_FACTS.md`.

- **Controls: 30 → 180.** `DatasetControls/FM_20_39 | FM_40_59 | FM_60_79` (60 each, sex-pooled,
  thesis design). Old `20_29F/30_39F` (and all F_*/M_*) dirs deleted. GROUP_DEFINITIONS in `01_/02_`
  use FM_* bands; `03_clinical_maps.py` resolves each patient's band per-patient.
- **Patients: 15 → 17.** `ez_ground_truth.py` from thesis (Rai 2026) Table 11: **P014 EXCLUDED**;
  **P028/P029/P030 ADDED**. Labels **L=8, R=5, B=2, U=2** (B-L/B-R scheme retired). Unilateral n=13.
- **✅ zAI over-dispersion FIXED via left–right symmetric-template registration** (FSL flirt+fnirt,
  no ANTs). See `symreg/` (`build_sym_template.py`, `reg_subject.sh`, `run_sym_pipeline.sh`) and the
  `--sym` flag added to `01_/02_/03_` + `scripts/rerun_thresholding_methods.py`. All 197 subjects
  (180 controls + 17 patients) registered onto a flip-symmetric T1 template; warp applied to
  perfusion (spline) + aparc (nn) → `symreg/sym_perf/{ID}_{perf,aparc}_sym.nii.gz`. **Result:** cohort
  mean %|zAI|≥1.96 in GM dropped **31% → 12.9%** (control LOO 6.9%), within the calibrated 5–15% band.
  Original-space `results_zscore/{groups,asymmetry,patients,clinical,summary}` archived to
  `_archived_origspace_zai_20260622/`; **canonical `results_zscore/` is now symmetric-space.**
  Reproduce with `symreg/run_sym_pipeline.sh`. See memory `project_zai_overdispersion`.
- **Pipeline B consolidated (n=17):** direction-aware MDT concordance **7/13 unilateral**
  (`results_zscore/clinical/cohort_concordance.csv`); P015 hyperperfusion phenotype 229 hyper/14 hypo;
  cohort cluster directions 7785 subtle / 1406 hypo / 1180 hyper / 3 mixed; thresholding table n=17
  (`results_zscore/asymmetry/summary/`).
- **Manuscript DONE:** Pipeline A honest reframe (best `temporal_signed_li`/`dominant_side` AUC 0.600,
  `cv_ratio` 0.650; Weighted AAI collapsed 0.771→0.525), all tables rewritten (L/R/B/U scheme),
  Pipeline B Methods (symmetric registration) + Results (n=17) + abstract updated, citations resolve,
  LaTeX balanced. Native-vs-MNI subsections + cohort statements already applied.

### Follow-up accomplishments (2026-06-23/30)
- **M3/M4/M7 thresholding FIXED** (zero-contamination bug, see Known Issue #5): tab:thresholding_compare
  now M3 10.4% / M4 1.0% / M7 0.5%. `M7_N_PERMUTATIONS` 1000→200.
- **Figures regenerated in symmetric space:** `04_publication_figures.py --sym` (63 PNGs) +
  `clean_overlay.py --all --sym` (17 overlays; fixed a slice-index off-by-one via clamp in `_ortho`).
  Added `--sym` flag to both. P015 overlay wired into the manuscript
  (`manuscript/figures/p015_clinical_zai.png`, was a placeholder).
- **Deep-research methodology audit applied** (5 findings, all addressed in manuscript): (1) AI
  half-sum vs full-sum denominator — added explicit convention + scale-invariance note (zAI unaffected);
  (2) cited PASCOM + ICBM152-2009c for symmetric registration + study-specific-template limitation;
  (3) cited PASCOM "Z4C" validating the z≥4/≥100-vox tier + added mm³ (26/51/102 mm³); (4) softened
  FCD-hyperperfusion claim (6/16≈37%, correlational, pediatric-scoped); (5) **added bootstrap 95% CIs**:
  temporal_signed_li 0.600 [0.25,0.91], cv_ratio 0.650 [0.24,1.00] — both span chance. references.bib now 63 keys.

## Key Directories

| Directory | Contents |
|-----------|----------|
| `Dataset/` | **Patient data** — 15 epilepsy patients (P013-P027). Each has: `aparc+aseg.nii.gz`, `T1w_acpc_dc_restore.nii.gz`, `{ID}_perfusion_calib_resampled_to_T1w.nii.gz` (actually MNI space despite filename) |
| `DatasetControls/` | **Healthy control data** — organized by age/sex (e.g., `20_29F/`, `30_39F/`). Each subject has: `*_MNISpace_perfusion_calib_upsampled.nii.gz`, `aparc+aseg.nii.gz` |
| `src/` | Core modules — `calculator.py` (11 asymmetry methods) |
| `results_zscore/` | Z-score pipeline outputs: `groups/` (mean/SD/parcellation), `patients/` (z-score maps), `clinical/` (gray-matter-only maps), `asymmetry/` (asymmetry of z-scores) |
| `results_voxel/` | Within-subject asymmetry results (no control comparison) |
| `results/` | ROI-based asymmetry results |
| `visual_analysis/` | Publication-quality PNG figures |

## The Full Pipeline

```
1. CONTROLS → GROUP STATS
   01_build_control_normative.py
   Loads 30 control scans → Welford's algorithm → mean + SD perfusion maps
   Output: results_zscore/groups/F_20_39/{mean,sd}_perfusion.nii.gz

2. PATIENT → Z-SCORE MAP
   01_build_control_normative.py
   z = (patient_perfusion - mean_controls) / sd_controls  per voxel
   Output: results_zscore/patients/{pid}/{pid}_vs_F_20_39_zscore.nii.gz

3. ASYMMETRY → CONTROL-REFERENCED zAI (Pipeline B canonical)
   02_compute_zai.py
   AI = 100 * (L - R) / ((L+R)/2) per voxel; then z = (AI - mean_AI) / sd_AI
   ALSO: per-cluster direction-of-abnormality inference using
   mean_perfusion.nii.gz / sd_perfusion.nii.gz — replaces polarity-flip
   rule with per-side perfusion z classification.
   New CSV columns: cluster_hemi, z_perf_cluster_side, z_perf_mirror_side,
   direction_class (hyper/hypo/mixed/subtle), ez_side_pred.
   Output: results_zscore/asymmetry/patients/{pid}/{pid}_asymmetry_zscore.nii.gz
           + 16-column cluster_report.csv with direction columns

4. CLINICAL MAPS (gray matter only, strict threshold)
   03_clinical_maps.py
   Mask zAI to GM, |zAI|>=3.0, cluster>=50 voxels, lateralized maps
   Output: results_zscore/clinical/{pid}/{pid}_clinical_*.nii.gz

5. PUBLICATION FIGURES
   04_publication_figures.py
   Figures 10-16 (zAI panels, distributions, cluster overlays)
   Output: visual_analysis/

6. ROI MDT DISCRIMINATION (Pipeline A canonical)
   05_roi_discrimination.py + ez_ground_truth.py
   ROI-based, 15 indices x 4 aggregation strategies, ROC + LOOCV
   Output: ez_analysis_mdt/

7. CLINICAL INTERPRETATION (surgeon-facing report)
   06_clinical_interpretation.py
   Surgical thresholds (|zAI|>=4, >=100 vox) + 37-ROI scope + lobe aggregation
   + lateralization confidence (Bartolomei 2008 framework) + MDT concordance
   Side prediction uses direction-of-abnormality inference (ez_side_pred
   from cluster CSV); legacy polarity-flip is a graceful fallback only
   for older 11-column CSVs.
   Output: results_zscore/clinical/<PID>/<PID>_surgeon_report.{txt,json}
           results_zscore/clinical/cohort_concordance.csv (cohort summary)

   Full-cohort concordance (n=17, 2026-06-23 symmetric-space zAI;
   results_zscore/clinical/cohort_concordance.csv):
     Unilateral (L/R, n=13): 7 AGREE / 6 DISAGREE.
       agree:    P013, P016, P022, P024, P025, P026, P029
       disagree: P015, P017, P023, P027, P028, P030
     Bilateral (P019, P020): partial.  Unclear (P018, P021): disagree.
     P015 disagree = hyperperfusion phenotype (229 hyper/14 hypo) — informative, not error.
     P013 now AGREE (L) — symmetric registration recovered correct lateralization
       (was discordant under the over-dispersed zAI); dominant clusters still frontal
       (mesial-temporal localization gap persists).
     (Old n=5 F_20_39 "3/5" figure is historical/superseded.)

     MDT confidence rule (tightened 2026-05-20, per ILAE Neuroimaging Task Force
     framework): High confidence = >=4/5 sources concordant AND surgical-planning
     recommendation aligned; Moderate = surgical recommendation + >=2 others;
     Low = surgical recommendation alone, or active discordance, or sEEG
     planned. P015's MDT label was Moderate confidence — its Pipeline-B
     disagreement is therefore best read as surfacing real multimodal
     ambiguity rather than as framework failure.
```

## Surgical Threshold Tiers (2026-05-05 policy)

| Use case | zAI threshold | Cluster size | Source |
|----------|---------------|--------------|--------|
| Research / sensitivity | \|zAI\|>=3 | >=50 voxels | Shang 2021 |
| Standard clinical view | \|zAI\|>=4 | >=100 voxels | Boscolo Galazzo 2016 |
| Surgical confidence | \|zAI\|>=5 | >=200 voxels | Bartolomei 2008 |
| Concordance-required | \|zAI\|>=4 + within 2cm of MRI lesion | + EEG/neuropsych | Gennari 2025 |

`06_clinical_interpretation.py` defaults to **standard clinical view**; pass `--peak-threshold` and `--size-threshold` to override.

## Critical Technical Facts

- **All data is in MNI152 space** at 227x272x227 voxels, 0.8mm isotropic
- Patient files named `*_resampled_to_T1w.nii.gz` are actually in MNI space (confirmed by affine matrix)
- Control files: `*_MNISpace_perfusion_calib_upsampled.nii.gz`
- **No registration needed** — all volumes share identical geometry
- **Processing pipeline:** HCP-ASL (https://github.com/physimals/hcp-asl) — the Oxford Physimals group's HCP-Lifespan pipeline. Built on BASIL/`oxford_asl` with structural pre-processing, EPI distortion correction, M0 calibration, partial-volume correction, registration to T1w/MNI152.
- **Scanner:** 3T Siemens MAGNETOM Prisma (Siemens Healthineers, Erlangen, Germany), 32-channel head coil
- Group stats use **Welford's online algorithm** (numerically stable, memory-efficient)
- Consensus parcellation: majority-vote across 30 control aparc+aseg files
- Patient demographics: `clinical_spreadsheet.xlsx` (cols: ID, AGE, SEX)
- **Direction-of-abnormality inference:** uses `mean_perfusion.nii.gz` + `sd_perfusion.nii.gz` (per-voxel control perfusion mean/SD) in addition to the AI normative database. Per-side perfusion z threshold `z_thr=2.0`.
- **SD floor + bilateral pooling (2026-05-20 fix in `02_compute_zai.py`):** the control AI SD map is now (a) elementwise-max-pooled with its left-right mirror so anatomical mirror voxels share the same denominator, and (b) floored at 5% in percent-AI units. The previous `sd > 0.01` gate admitted near-zero-SD voxels and produced spurious clip-to-20 zAI at the dorsal convexity (precentral, sup.frontal, sup.parietal). Post-fix peak |zAI| no longer saturates the ±20 clip; convexity clusters that previously dominated are substantially reduced. Rebuilt F_20_39 control DB + regenerated all 5 patient zAI maps on 2026-05-20.

## Key Recent Literature (for citations and framing)

- **Han et al. 2026** (CNS Neurosci Ther, doi:10.1002/cns.70876): SEEG-validated PET Z-map in n=120 drug-resistant focal epilepsy. Specificity 0.94, NPV 0.91, sensitivity 0.62. Regional heterogeneity: frontal/parietal κ≈0.65 vs. temporal/insular κ≈0.45 — surface methods miss mesial structures. Anchors **exclusion-utility framing**.
- **Gennari et al. 2026** (Neurol Sci) and **2025** (Epilepsia, doi:10.1111/epi.18375): head-to-head ASL voxel-AI vs FDG-PET in pediatric drug-resistant epilepsy. ASL-AI concordance rises to PET-comparable levels.
- **Ferrari et al. 2024** (Sci Rep, doi:10.1038/s41598-024-58352-9): ~40% of FCD lesions hyperperfused (vs hypo), correlated with EEG spike rate. Empirical justification for direction-of-abnormality inference.
- **Sierra-Marcos et al. 2017** (Brain, doi:10.1093/brain/awx241): postictal hypoperfusion ASL — time-since-seizure framework.
- **Biagioli et al. 2025** (Epileptic Disorders): ILAE Neuroimaging Task Force on multimodal presurgical workup.

## Control Groups (3 sex-pooled age bands — canonical, thesis design)

The canonical normative cohort is **3 sex-pooled bands of n=60** (180 controls total). The
per-sex F_*/M_* groups in `GROUP_DEFINITIONS` are retained in code but their decade subdirs
(`20_29F/` etc.) do not exist in the dataset, so they are reported "unavailable" (a sex-specific
sensitivity analysis would need per-control sex labels to split the FM_* dirs).

| Group | Subdirectory | Status |
|-------|---------------|--------|
| FM_20_39 | `DatasetControls/FM_20_39/` | ✅ Built (n=60), symmetric-space |
| FM_40_59 | `DatasetControls/FM_40_59/` | ✅ Built (n=60), symmetric-space |
| FM_60_79 | `DatasetControls/FM_60_79/` | ✅ Built (n=60), symmetric-space |
| F_*/M_* (6 groups) | per-sex decade dirs (absent) | Unavailable (sex-pooled storage) |

## 15 Asymmetry Indices (src/calculator.py — literature-cited, 2026-05-04 formula-fix)

**Ratio-based:** laterality_index (LI), absolute_asymmetry_index (AAI), log_ratio, simple_difference, ratio
**Effect sizes:** cohen_d_asymmetry (Cohen 1988), hedges_g_asymmetry (Hedges 1981, small-sample-corrected), glass_delta_asymmetry (Glass 1976)
**Distribution-based:** zscore_asymmetry, robust_zscore_asymmetry (Huber 1981 / Leys 2013, MAD-based)
**Advanced:** percent_difference, cv_ratio (Galaburda 1978), hyperperfusion_ratio (max/min form, Van Bogaert 2000), normalized_difference (Toga & Thompson 2003), coefficient_of_asymmetry

All formulas literature-cited. Audit doc: `docs/superpowers/audits/2026-05-04-formula-fix.md`.

## 13 Thresholding/Clustering Methods (src/thresholding.py)

M1: Fixed Z + cluster, M2: Quality + percentile, M3: FDR (BH), M4: Bonferroni, M5: TFCE, M6: GRF, M7: Permutation, M8: GMM, M9: Otsu, M10: Random baseline, M11: Quality+TFCE (recommended), M12: Quality+GMM, M13: Quality+Otsu

Module: `src/thresholding.py` (extracted from deleted `control_zscore_asymmetry_clustering.py` on 2026-05-04, 18/18 smoke tests passing).

## Clinical Scope — 37 ROI Pairs (POLICY, 2026-05-05)

**Per user policy, the canonical clinical scope for surgeon-facing Pipeline B output is the 37 ROI pairs:**

- **3 subcortical pairs** (FreeSurfer labels): Thalamus (10/49), Hippocampus (17/53), Amygdala (18/54)
- **34 cortical Desikan-Killiany regions**: Left labels 1001-1035, Right labels 2001-2035

Total: 76 FreeSurfer labels = 37 paired regions. This matches Pipeline A's ROI scope and the manuscript's reported analysis space.

**Implication for Pipeline B (voxel zAI):**

- Voxel-level zAI maps still cover all gray matter (no information loss in raw output).
- **All clinical/surgeon-facing visualizations and cluster reports MUST be masked to the 37 ROIs by default** (e.g., `interactive_viewer.py` option 11, `03_clinical_maps.py` clinical output).
- Voxels outside the 37 ROIs (e.g., paracentral lobule, isthmus cingulate, occipital pole, basal ganglia other than thalamus) are excluded from the "where is the EZ" decision.

**Excluded basal ganglia structures** (intentionally not in the 37): Caudate (11/50), Putamen (12/51), Pallidum (13/52). These are not part of the surgeon-facing EZ region set per the manuscript.

**Code reference:** see `interactive_viewer.py::_clinical_roi_labels()` and the policy follow-up F-18 in `.claude/NEXT.md` (refactor `03_clinical_maps.py` and `04_publication_figures.py` to default to this mask).

## Code Conventions

- Standalone Python scripts at project root
- Class-based analyzers (e.g., `VoxelBasedAsymmetryAnalyzer`, `ControlGroupBuilder`)
- NIfTI I/O via `nibabel`, analysis via `numpy`/`scipy`/`pandas`
- Visualizations: `matplotlib` with `seaborn-v0_8-darkgrid` style, dpi=300
- Output figures numbered sequentially in `visual_analysis/` (01_, 02_, etc.)
- FSLeyes commands printed for interactive 3D viewing
- Interactive viewer: `interactive_viewer.py` (menu-driven, options 1-20)

## How to Run

```bash
# ── Pipeline B is now SYMMETRIC-SPACE (zAI over-dispersion fix). ──
# One-time: register all 197 subjects to the flip-symmetric T1 template, then
# rebuild the whole symmetric-space pipeline (archives original-space outputs):
python symreg/build_sym_template.py                     # build T1_sym (once)
#   (then symreg/make_jobs.py full | xargs ... symreg/reg_subject.sh  — see run_sym_pipeline.sh)
bash symreg/run_sym_pipeline.sh                         # 01 --sym → 02 --sym → 03 --sym → 06 → thresholding --sym
python symreg/check_full_calibration.py                 # GM %|zAI|>=1.96 calibration check (target 5-15%)

# Individual steps (add --sym to consume symreg/sym_perf/ symmetric-space inputs):
# Step 1: Build control normative DB (mean/SD perfusion + per-patient z-score)
python 01_build_control_normative.py --sym --rebuild-group

# Step 2: Compute control-referenced zAI per patient (per band)
python 02_compute_zai.py --sym --rebuild --group FM_20_39

# Step 3: Clinical-grade gray-matter zAI maps
python 03_clinical_maps.py --sym

# Step 4: Publication figures
python 04_publication_figures.py

# Step 5: ROI-based MDT discrimination (Pipeline A)
python 05_roi_discrimination.py

# Step 6: Clinical interpretation report (surgeon-facing) — NEW 2026-05-05
python 06_clinical_interpretation.py --patient P015     # one patient
python 06_clinical_interpretation.py --all              # cohort summary

# Diagnostic / methodology aids
python analyze_zai_distributions.py     # AI distribution histograms

# Interactive 3D viewing (option 15: Clinical Report; option 10: Top-N clusters; option 11: 37-ROI mask)
python interactive_viewer.py
```



## Helper Scripts

`scripts/` directory:
- `scripts/resample_parcellation_to_mni.sh` — bash helper to resample a parcellation NIfTI to MNI space (uses FSL `flirt`)
- `scripts/per_index_discrimination.py` — magnitude-weighted aggregation of each of the 15 individual asymmetry indices across the 37 ROI pairs, then ROC analysis against MDT labels. Writes `ez_analysis_mdt/per_index_performance_{exclude,dominant}.csv`. Used by manuscript Table 3 (per-index discrimination).
- `scripts/localization_metrics.py` — cluster-to-target-lobe overlap on the F_20_39 patients from `results_zscore/clinical/<PID>/<PID>_clinical_zai_cluster_report.csv`. Writes `ez_analysis_mdt/localization_performance.csv`. (Manuscript no longer reports localization; this script is retained for future outcome-anchored validation.)
- `scripts/rerun_thresholding_methods.py` — runs all 13 thresholding methods on the patient zAI maps. Overwrites `results_zscore/asymmetry/summary/all_patients_method_summary.csv`. Used by manuscript Table 6.
- `scripts/post_sd_fix_rerun.sh` — full downstream re-run driver after a `02_compute_zai.py --rebuild` (chains `03_clinical_maps.py` + per-index + localization + thresholding).

## Known Issues & Solutions

1. **Patient coverage mismatch**: Patients have fewer non-zero voxels than controls. Solution: only compute z-scores where patient has data (`patient_data > 0`).
2. **Extreme z-scores**: Very small SD at some voxels. Solution: clip z-scores to [-20, 20] AND apply 5% SD floor + bilateral SD pooling in `02_compute_zai.py` (2026-05-20). Without the SD floor, near-zero control-SD voxels at dorsal convexity get spuriously inflated zAI clipped at ±20 and dominate cluster reports.
3. **White matter noise**: 70% of "significant" voxels were in white matter. Solution: gray-matter-only clinical maps (`03_clinical_maps.py`).
4. **Everything highlighted at z=1.96**: Too liberal for 2.6M voxels. Solution: use stricter threshold (|z|>=3.0) and larger clusters (50+ voxels), or use TFCE/FDR methods.
5. **Multiple-comparisons methods returning zero — FIXED 2026-06-23**: M3 (FDR-BH), M4 (Bonferroni), M7 (permutation) previously returned 0 for all/most patients. ROOT CAUSE was NOT a tuning issue: `m3_fdr`/`m4_bonferroni`/`m7_permutation` in `src/thresholding.py` estimated the half-normal noise sigma over the bilateral_quality_mask, which is far larger than the non-zero zAI region — the zero (no-data) voxels dominated → median|zAI|=0 → sigma_noise=0 → early `if sigma_noise<=0: return zeros`. FIX: restrict each to `bilateral_quality_mask() & (abs_ai>0)`. Post-fix (n=17): M3 ~10.4%, M4 ~1.0%, M7 ~0.5% (1/17 still zero — expected conservatism). Also reduced `M7_N_PERMUTATIONS` 1000→200 (documented; 1000×17×full-volume labelling = hours). Default reporting method remains M11 (Quality+TFCE).
