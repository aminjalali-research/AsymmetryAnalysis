# Pipeline B verified facts (updated 2026-10-12)

Cluster counts below count each mirrored pair once (rows with `cluster_hemi == ez_side_pred`),
because the zAI map is antisymmetric and every finding appears in both hemispheres.

## Method
- Left-right symmetric registration with FSL flirt (12 dof) + fnirt; no ANTs.
- Template: T1_sym = (avg + flip0(avg)) / 2, built from 30 control T1w_restore_brain images on the
  HCP-MNI 0.8 mm grid. Each subject: flirt + fnirt T1 -> T1_sym; warp applied to perfusion (spline)
  and aparc+aseg (nearest neighbour). 197 subjects (180 controls + 17 patients).
- Registration: `python symreg/make_jobs.py full | tr '\t' ' ' | xargs -P 8 -L 1 bash symreg/reg_subject.sh`,
  then `bash symreg/run_sym_pipeline.sh`.

## Calibration (gray matter of the consensus parcellation)
- Cohort mean %|zAI| >= 1.96: 12.9% (n=17), versus 31.5% without symmetric registration (range 16.6-49.4%).
- Control leave-one-out: 6.9% (target about 5%).
- Per patient: P013 18.5, P015 8.5, P016 9.8, P017 16.6, P018 9.3, P019 17.7, P020 11.1, P021 4.1,
  P022 7.9, P023 19.7, P024 20.3, P025 12.9, P026 7.0, P027 18.3, P028 15.0, P029 13.1, P030 10.1.

## Controls
- 3 sex-pooled bands of 60 (FM_20_39, FM_40_59, FM_60_79); 180 total, 90 F / 90 M; HCP Aging/Development.
- FM_20_39 = 30 HCP-Development (age <= 21) + 30 HCP-Aging (age >= 36); no control aged 22-35.

## Direction of abnormality (|z_perf| >= 2 per side; research tier |zAI| >= 3, >= 50 voxels)
- Cohort pairs: subtle 3915, hypo 698, hyper 556, mixed 1.
- Hyperperfusion-dominant by volume: P015 (106 hyper / 6 hypo pairs; right sided), P029 (50/5; left),
  P030 (64/12; right). Hypo-dominant example: P027 (151 hypo / 2 hyper; left sided).

## Lateralization (13 unilateral patients: 8 L, 5 R)
- Primary rule (06_clinical_interpretation.py, clinical tier |zAI| >= 4, >= 100 voxels):
  7/13 agree (P013, P016, P022, P024, P025, P026, P029); balanced accuracy 0.51.
- Best rule tested (direction-aware volume majority, clinical or native tier): 9/13 (7/8 left, 2/5 right), balanced accuracy 0.64.
- Call strength: strong 2/2 agree (P024 post-resection, P026), moderate 3/6, weak 2/5.
- Always predicting left: 8/13, balanced accuracy 0.50.
- P024 was imaged after a partial right temporal resection; without it, the volume-index AUC is 0.50.
- Source: `scripts/clinical_stratified_analysis.py` -> `ez_analysis_mdt/clinical_*.csv`.

## Thresholding (results_zscore/asymmetry/summary/all_patients_method_summary.csv, n=17, mean coverage %)
- Coverage is relative to the brain-mask voxels with non-zero zAI (mean 1.41 M voxels).
- M01 8.4, M02 9.1, M03 FDR 32.4, M04 Bonferroni 3.1, M05 TFCE 4.6, M06 GRF 13.1,
  M07 permutation 1.6 (P021 with no surviving cluster), M08 GMM 12.1 (degenerate in P025), M09 Otsu 31.8,
  M10 random 3.0, M11 quality+TFCE 4.6, M12 quality+GMM 12.6, M13 quality+Otsu 30.1.
- M3/M4/M7 noise scale is estimated over the quality mask intersected with |zAI| > 0 (fixed 2026-06-23;
  before the fix, zero voxels collapsed it and all three returned nothing). M7 uses 200 permutations.
- Cluster counts in that summary include both members of each mirrored pair.
