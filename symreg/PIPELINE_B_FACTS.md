# Pipeline B verified facts (2026-06-23, symmetric-registration zAI fix)

## Method
- zAI over-dispersion FIXED by left-right symmetric-template registration (FSL flirt+fnirt, no ANTs).
- Symmetric T1 template T1_sym=(avg+flip0(avg))/2 from 30 control T1w_restore_brain on the
  HCP-MNI 0.8mm grid; per-subject flirt(12dof)+fnirt T1->T1_sym; warp applied (spline) to perfusion
  and (nn) to aparc+aseg. 197 subjects (180 controls + 17 patients).

## Calibration (GM = consensus parcellation gray matter)
- Cohort mean %|zAI|>=1.96 in GM: NEW 12.9% (n=17) vs OLD ~31% (range 16.6-49.4%).
- Validation subset control leave-one-out: 6.9% (target ~5%); patient subset 25.7%->11.7%.
- Per-patient NEW GM %: P013 18.5, P015 8.5, P016 9.8, P017 16.6, P019 17.7, P020 11.1, P026 7.0,
  P028 15.0, P029 13.1, P030 10.1, P018 9.3, P021 4.1, P022 7.9, P023 19.7, P024 20.3, P027 18.3, P025 12.9.

## Control bands: 3 sex-pooled bands n=60 each (FM_20_39/40_59/60_79); 180 total.

## Direction-of-abnormality (cluster_report direction_class, |z_perf|>=2.0 per side)
- Cohort cluster totals: subtle 7785, hypo 1406, hyper 1180, mixed 3.
- P015 hyper-dominant: 229 hyper / 14 hypo (hyperperfusion phenotype, key finding).
- P027 hypo-dominant: 299 hypo / 4 hyper.
- P013 mixed: 113 hyper / 91 hypo / 499 subtle.

## Algorithmic concordance (results_zscore/clinical/cohort_concordance.csv, n=17)
- Unilateral (L/R, n=13): 7 agree / 6 disagree.
  agree:  P013,P016,P022,P024,P025,P026,P029
  disagree: P015,P017,P023,P027,P028,P030
- Bilateral (B): P019,P020 = partial. Unclear (U): P018,P021 = disagree.
- P015 disagree = hyperperfusion phenotype (predicted R, MDT L) -- informative, not error.

## Thresholding (results_zscore/asymmetry/summary/all_patients_method_summary.csv, n=17, mean coverage%)
- M01 10.51, M02 7.96, M03_FDR 0.00 (0/17), M04_Bonf 0.00 (0/17), M05_TFCE 1.50,
  M06_GRF 12.01, M07_Perm 0.61 (16/17 zero), M08_GMM 8.15, M09_Otsu 15.25, M10_Random 3.00,
  M11_Quality_TFCE 1.48, M12_Quality_GMM 7.70, M13_Quality_Otsu 14.71.
- M03/M04/M07 reject everything at n=17 (parametric multiple-comparison too strict on near-Gaussian
  corrected tail) -- methodology note, re-tuning needed.
