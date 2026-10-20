# AsymmetryAnalysis — Project Guide for Claude Code

## What this project does

Clinical neuroimaging research for **epilepsy presurgical evaluation**. It quantifies left-right
perfusion asymmetry on ASL (arterial spin labeling) MRI, compares it with healthy controls, and asks
whether it lateralizes the epileptogenic zone (EZ) and whether each abnormality is hypo- or hyperperfused.

## Current state (2026-10-12, authoritative)

- **Cohort:** 17 patients (P013–P030, **P014 excluded**), labels from Rai 2026 MSc thesis Table 11:
  L=8, R=5, B=2 (P019, P020), U=2 (P018, P021); 13 unilateral. P028/P029/P030 = thesis #15/16/17.
  **P024 was imaged after a partial right temporal resection** (report results with and without it).
  Invasively supported side: P016 (stereo-EEG), P017, P023, P025 (resection + histology), P026 (biopsy).
- **Controls:** 180 healthy adults (90 F / 90 M per thesis) from HCP Aging/Development, 3 sex-pooled bands
  of 60: `FM_20_39`, `FM_40_59`, `FM_60_79`. FM_20_39 = 30 HCD (≤21 y) + 30 HCA (≥36 y): **no control aged
  22–35**. Controls were scanned at HCP sites, not locally. Old per-sex dirs (`20_29F`, `F_*`, `M_*`) do not exist.
- **Pipeline B runs in symmetric space** (`--sym`): every subject registered (FSL flirt + fnirt) to a
  flip-symmetric T1 template. Calibration: GM %|zAI|≥1.96 = **12.6%** (was 31.5% in plain MNI; 12.9% before the 2026-10-20 validity-rule rerun); controls
  leave-one-out 6.9%. Original-space outputs archived in `_archived_origspace_zai_20260622/`.
- **zAI maps are antisymmetric → every cluster is a mirrored pair.** Count and vote with one member per
  pair (`cluster_hemi == ez_side_pred`). Pair counts: cohort 698 hypo / 556 hyper / 3915 subtle / 1 mixed;
  P015 106 hyper / 6 hypo; P027 151 hypo / 2 hyper. Older numbers (229/14, 7785/1406/1180/3) are double counts.
- **Headline results (honest, after the 2026-10-20 validity-rule rerun):**
  - Regional (Pipeline A) best AUC 0.60 (95% CI 0.28–0.90); leave-one-out 8/13 = always-left baseline.
  - Voxel-wise primary rule (clinical tier): 7/13 agree (P013, P016, P017, P023, P024, P026, P029), balanced
    accuracy 0.55; no rule >8/13 or balanced accuracy >0.61; always-left 8/13, 0.50. Peak zAI 5.9–9.2 (no clipping).
  - Direction class tracks the global CBF offset vs the HCP band (ρ≈0.87): hyper-dominant P015/P029/P030 = artefact.
  - Chalfont (93 unilateral, 66 same-scanner controls): primary rule 58/93 (0.63) but 56 % of LOO controls get a
    strong/moderate call; regional normative temporal z 61/93 (0.65, AUC 0.69), calibrated calls 12/13 at 6 % FP.
  - Voxel noise budget: control SD of AI 55–70 % at 0.8 mm → |zAI|=4 needs >200 % asymmetry (docs/audit/).
- **Manuscript (private, `manuscript/`, gitignored):** targets *Clinical Neurophysiology* (prototype:
  Scarabello et al. 2026, Clin Neurophysiol 191:2112368). Title: "Normative voxel-wise arterial spin
  labeling asymmetry for lateralizing the epileptogenic zone in drug-resistant focal epilepsy". Synced to
  Overleaf project 67201298ee1769ba89e90bad (upload via the file-tree Upload dialog; check the editor
  content hash after upload). Build: `conda run -n tex tectonic manuscript/Manuscript_Complete.tex`
  (tables on float pages at the end). Supplement: `manuscript/Supplementary_Material.tex`
  (+ `references_supp.bib`). Bibliography is PubMed-verified (`scripts/build_references_bib.py`, private);
  never add a reference without a PMID/DOI check. Old NeuroImage:Clinical version in `manuscript/_archive/`.
  Ethics: Queen's HSREB DMED-2472-21. Funding: SEAMO SEA-24-011.
  Open (authors): co-author approval of CRediT roles; name of the person who assigned MDT certainty grades.
- **Never push** `manuscript/`, `docs/manuscript/`, `docs/superpowers/` (private).

## Key directories

| Directory | Contents |
|---|---|
| `Dataset/` | 17 patients: `aparc+aseg.nii.gz`, `T1w_acpc_dc_restore.nii.gz`, `{ID}_perfusion_calib_resampled_to_T1w.nii.gz` (MNI space despite the name) |
| `DatasetControls/FM_*/` | 3 bands × 60 controls: `*_MNISpace_perfusion_calib_upsampled.nii.gz`, `aparc+aseg.nii.gz` |
| `symreg/` | symmetric template, registration scripts, `sym_perf/{ID}_{perf,aparc}_sym.nii.gz`, `PIPELINE_B_FACTS.md` |
| `src/` | `calculator.py` (15 indices), `thresholding.py` (13 methods), `freesurfer_parser.py` |
| `results_zscore/` | `groups/FM_*`, `patients/`, `asymmetry/{groups,patients,summary}`, `clinical/` |
| `results_analysis/` | per-patient regional asymmetry CSVs (Pipeline A input, `scripts/build_comprehensive_asymmetry.py`) |
| `ez_analysis_mdt/` | regional ROC/LOOCV/separability, stratified voxel-wise analyses (`clinical_*.csv`) |
| `visual_analysis/` | figures and per-patient overlays |

## Pipeline

```
1. 01_build_control_normative.py --sym       control perfusion mean/SD per band (Welford), patient perfusion z
2. 02_compute_zai.py --sym --group FM_<band>  AI = 100 (L-R)/((L+R)/2); control AI mean/SD (mirror-pooled SD,
                                             5% floor); zAI clipped ±20; clusters + direction columns
                                             (cluster_hemi, z_perf_cluster_side, z_perf_mirror_side,
                                             direction_class, ez_side_pred). One run per band.
3. 03_clinical_maps.py --sym                 GM-only zAI, 37-ROI scope, |zAI|>=3, >=50 vox
4. 04_publication_figures.py --sym           exploratory figures 10-16
5. 05_roi_discrimination.py                  Pipeline A: 15 indices x 9 aggregations, ROC vs MDT (n=13)
6. 06_clinical_interpretation.py --all       patient reports + results_zscore/clinical/cohort_concordance.csv
```

Direction rule: per-side perfusion z vs controls (threshold 2) → hypo / hyper / mixed / subtle; predicted
side = deviating hemisphere. The legacy polarity-flip fallback in `06_` (old CSVs only) keeps one pair member.
`top_lobe` in `cohort_concordance.csv` = side–lobe with the largest summed assigned volume (not largest cluster).

## Reporting tiers (06_clinical_interpretation.py defaults to clinical)

| Tier | \|zAI\| | Cluster | Volume (native 2.5-mm voxels) |
|---|---|---|---|
| Research | ≥3 | ≥50 vox | 26 mm³ (1.6) |
| Clinical (default) | ≥4 | ≥100 vox | 51 mm³ (3.3); matches PASCOM Z4C (Aslam 2022) |
| Surgical | ≥5 | ≥200 vox | 102 mm³ (6.6) |
| Native | ≥4 | ≥610 vox | 312 mm³ (20) |

**As implemented (audit 2026-10-20):** `02_` forms clusters at |zAI| ≥ 1.96 with ≥ 50 voxels; the tiers in `06_` and
`scripts/clinical_stratified_analysis.py` keep a 1.96-cluster when its *peak* |zAI| reaches the tier value and its extent
*at 1.96* reaches the voxel count. `03_` maps use a voxel threshold of 3 / ≥ 50. See `docs/audit/METHODS_AUDIT_2026-10-20.md`.

Primary rule: priority to temporal/hippocampal/amygdalar clusters: strong ≥3 clusters, ≥3000 vox, ≥3×
contralateral; moderate unilateral or ratio ≥2; weak smaller tilt or extratemporal only; else bilateral.

## Clinical scope — 37 ROI pairs (policy)

3 subcortical pairs (thalamus 10/49, hippocampus 17/53, amygdala 18/54) + 34 Desikan–Killiany cortical
pairs (1001–1035 / 2001–2035). Caudate, putamen, pallidum excluded. zAI maps are stored unmasked; the scope
is applied in clinical outputs (`03_`, `06_`, `interactive_viewer.py::_clinical_roi_labels()`).

## Critical technical facts

- All data on the MNI152 0.8 mm grid (227×272×227). Patient files named `*_resampled_to_T1w` are in MNI.
- HCP-ASL processing (Kirk et al. 2025, physimals/hcp-asl); 3T Siemens Prisma, 32-ch coil; multi-PLD pCASL
  2.5 mm iso (native voxel 15.6 mm³), 5 PLDs 0.2–2.2 s, label 1.5 s.
- Consensus parcellation: majority vote across the band's 60 control aparc+aseg files.
- SD floor 5% (percent-AI units) + mirror max-pooling in `02_` prevent clip-to-20 artefacts at the convexity.
- `interactive_viewer.py` still overlays on original-space anatomy; use `symreg/templates/T1_sym.nii.gz`
  as underlay for symmetric-space maps.

## Pipeline A details (`05_roi_discrimination.py`)

9 aggregation scores: signed_weighted_li (ΣA|A|/Σ|A|, all pairs), temporal_signed_li (same, temporal pairs),
mesial_temporal_li, mean_li, weighted_aai (direction-blind |LI|), anatomically_weighted (ΣwA/Σw; w=3 mesial,
2 other temporal, 0.5 elsewhere), temporal_focused_score (0.6 mesial mean + 0.4 temporal mean),
dominant_side, cohen_d. Orientation chosen empirically (AUC 0.5 is a floor). 15 indices in `src/calculator.py`;
seven signed indices are first-order equivalent (identical AUC 0.500).

## 13 thresholding methods (`src/thresholding.py`)

M1 fixed |z|≥1.64, M2 quality+percentile, M3 FDR, M4 Bonferroni, M5 TFCE (95th percentile of positive TFCE,
no permutation), M6 GRF, M7 permutation (200), M8 GMM, M9 Otsu, M10 random 3%, **M11 quality+TFCE (default)**,
M12 quality+GMM, M13 quality+Otsu. Mean coverage of the non-zero zAI region (n=17, after the 2026-10-20 validity-rule rerun): M3 32.6%, M4 3.1%, M7 1.6% (now 2–13 clusters in every patient), M11 4.7%; M8 13.5% (no longer degenerate in P025).
Comparison is coverage only, not accuracy (no voxel-wise ground truth).

## How to run

**Memory rule (2026-10-20):** launch every long run through `scripts/run_capped.sh <name> <MemoryMax> <log> -- <cmd>`
(own systemd scope, hard memory cap, no swap, detached, BLAS threads pinned) and never run two pipelines at once.
Measured per-subject peaks on the 0.8-mm grid: 01 = 1.0 GB (was 14.5 GB before the flat-index cluster fix),
02 = 4.9 GB, 03 = 1.8 GB → `CHALFONT_JOBS=5` ≈ 25 GB. Running 10–12 jobs plus the in-house rerun exhausted
61 GB; the OOM killer/systemd-oomd then killed the desktop session four times.

```bash
python symreg/build_sym_template.py
python symreg/make_jobs.py full | tr '\t' ' ' | xargs -P 8 -L 1 bash symreg/reg_subject.sh
bash symreg/run_sym_pipeline.sh            # 01 -> 02 (3 bands) -> 03 -> 06 -> thresholding
python symreg/check_full_calibration.py    # GM %|zAI|>=1.96 per patient
python scripts/build_comprehensive_asymmetry.py && python 05_roi_discrimination.py
python scripts/clinical_stratified_analysis.py   # rules x scope x tier, balanced accuracy, strata
python scripts/regional_loocv.py                 # LOOCV of regional scores
python scripts/aggregation_separability.py       # paired bootstrap ΔAUC
python scripts/make_manuscript_figures.py        # manuscript Fig 1-3 (writes to manuscript/figures)
python -m pytest -q tests
```

## Enhanced estimators (v2, post hoc, 2026-10-20)

`scripts/v2/prep_subject_features.py --dataset inhouse|chalfont` (ROI means + 8-mm smoothed maps on a mirror-symmetric
2.4-mm grid → `v2_features/`), then `scripts/v2/analyze_v2.py --dataset …` (regional normative LI z, smoothed voxel
hypoperfused mass, LOO-control calibration at 5 % FP → `ez_analysis_mdt/v2/<dataset>/`). Chalfont: 61/93 by sign,
balanced accuracy 0.65 (0.56–0.75), calibrated calls 12/13 correct; in-house 7–9/13. Audit:
`docs/audit/METHODS_AUDIT_2026-10-20.md` (noise budget, global-offset direction artefact, tier definition).

## Helper scripts (`scripts/`)

`build_comprehensive_asymmetry.py` (regional tables), `per_index_discrimination.py`,
`aggregation_separability.py`, `regional_loocv.py`, `clinical_stratified_analysis.py`,
`rerun_thresholding_methods.py` (`--sym`), `localization_metrics.py` (cluster-to-lobe overlap; not reported),
`make_manuscript_figures.py`, `build_references_bib.py` (private), `resample_parcellation_to_mni.sh`.

## Code conventions

Standalone scripts at the project root; class-based analyzers; nibabel/numpy/scipy/pandas; matplotlib
dpi=300. Always report balanced accuracy and the majority-class baseline with lateralization counts.

## Key literature (verified)

- Han et al. 2026, CNS Neurosci Ther 32:e70876 — PET Z-map vs SEEG (n=120): specificity 0.94, temporal sensitivity 0.41.
- Aslam et al. 2022, J Neurosurg 138:828 — PASCOM symmetrized FDG-PET asymmetry; Z4C optimal.
- Gennari et al. 2024, Sci Rep 14:7601 — pediatric FCD: 6/16 hyperperfused, linked to spike frequency.
- Gaxiola-Valdez et al. 2017, Brain 140:2895 — postictal ASL hypoperfusion localizes SOZ.
- Mattioli et al. 2026, Neurol Sci — pediatric ASL voxel-AI raises concordance to PET level.
- Rentzeperis et al. 2023, Epilepsy Res 193:107163 — asymmetry significant only in lesional TLE.
- Biagioli et al. 2025, Epileptic Disord 27:439 — ILAE multimodal neuroimaging highlight.
- Scarabello et al. 2026, Clin Neurophysiol 191:2112368 — style/structure prototype; EZ reference tiers.

## Known issues (fixed; kept for context)

1. Patient coverage < controls → z only where patient data > 0.
2. Near-zero control SD → SD floor 5% + mirror pooling (2026-05-20).
3. White-matter noise → GM-only clinical maps.
4. zAI over-dispersion in plain MNI → symmetric registration (2026-06-23).
5. M3/M4/M7 returned zero (zero-voxel-contaminated sigma) → fixed 2026-06-23.
6. Mirrored-pair double counting in sign-based rules and cluster counts → fixed 2026-10-12 (commit 5d78234).
7. `ez_ground_truth.py` P027 evidence was a copy of P030's → fixed 2026-10-12.
8. `01_` kept a full 14-M-voxel mask per cluster (14.5 GB for ~1000 clusters) → flat voxel indices, outputs byte-identical (2026-10-20).
