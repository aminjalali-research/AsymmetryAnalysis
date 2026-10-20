# Pipeline B verified facts (updated 2026-10-20, after the validity-rule rerun)

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
- Cohort mean %|zAI| >= 1.96: 12.6% (n=17; 12.9% before the 2026-10-20 validity rule), versus 31.5% without
  symmetric registration (range 16.6-49.4%).
- Control leave-one-out: 6.9% (target about 5%).
- Per patient: P013 18.3, P015 8.4, P016 9.6, P017 16.4, P018 9.3, P019 17.4, P020 11.0, P021 4.1,
  P022 7.7, P023 19.2, P024 17.4, P025 12.4, P026 6.9, P027 18.0, P028 14.9, P029 12.9, P030 9.9.
- Validity rule (2026-10-20): AI is defined only where both mirror voxels have positive perfusion; the +-20 clip
  is no longer reached anywhere (peak |zAI| per patient 5.9-9.2; 99.9th percentile 4.4 on average).
- Noise: control SD of AI is 58% (GM), 55% (hippocampus), 70% (temporal cortex) at 0.8 mm; 17% after 8-mm
  smoothing; 9-10% (AI units) for region means. |zAI| = 4 is unreachable at a typical gray-matter voxel.
- The mirror max-pooling of the SD in 02 is a no-op (AI antisymmetric, validity mask symmetric); only the 5% floor acts.

## Controls
- 3 sex-pooled bands of 60 (FM_20_39, FM_40_59, FM_60_79); 180 total, 90 F / 90 M; HCP Aging/Development.
- FM_20_39 = 30 HCP-Development (age <= 21) + 30 HCP-Aging (age >= 36); no control aged 22-35.

## Direction of abnormality (|z_perf| >= 2 per side; research tier |zAI| >= 3, >= 50 voxels)
- Cohort pairs: subtle 3883, hypo 698, hyper 562, mixed 2 (median 313 per patient, range 128-399).
- Hyperperfusion-dominant by volume: P015 (107 hyper / 6 hypo pairs; right sided), P029 (51/5; left),
  P030 (66/12; right). Hypo-dominant example: P027 (151 hypo / 2 hyper; left sided).
- The class tracks the patient's whole-brain CBF offset from the HCP band (Spearman rho = 0.87, p = 5e-6):
  all three hyper-dominant patients are young-band patients with the largest positive offsets. Treat the
  direction class as a scanner/cohort artefact unless same-scanner controls or global normalization are used.

## Lateralization (13 unilateral patients: 8 L, 5 R)
- Tiers as implemented: clusters are formed at |zAI| >= 1.96, >= 50 voxels; a tier keeps a cluster whose peak
  |zAI| reaches the tier value and whose extent at 1.96 reaches the tier size.
- Primary rule (06_clinical_interpretation.py, clinical tier): 7/13 agree (P013, P016, P017, P023, P024, P026, P029);
  4/8 left, 3/5 right; balanced accuracy 0.55; P028 bilateral call.
- No rule exceeds 8/13 or balanced accuracy 0.61 (direction-aware majority at the surgical tier; temporal
  polarity flip at the clinical tier). Always predicting left: 8/13, 0.50.
- Call strength: strong 2/2 agree (P024 post-resection, P029), moderate 2/5, weak 3/5.
- Volume index AUC 0.55 (95% CI 0.09-1.00); 0.44 without P024.
- The in-house best rule did not replicate externally (Chalfont 50/93, 0.54).
- Source: `scripts/clinical_stratified_analysis.py` -> `ez_analysis_mdt/clinical_*.csv`.

## Thresholding (results_zscore/asymmetry/summary/all_patients_method_summary.csv, n=17, mean coverage %)
- Coverage is relative to the brain-mask voxels with non-zero zAI (mean 1.41 M voxels).
- M01 8.6, M02 9.2, M03 FDR 32.6, M04 Bonferroni 3.1, M05 TFCE 4.6, M06 GRF 13.1,
  M07 permutation 1.6 (2-13 clusters in every patient after the validity rule), M08 GMM 13.5 (no longer
  degenerate in P025), M09 Otsu 31.3, M10 random 3.0, M11 quality+TFCE 4.7, M12 quality+GMM 12.7,
  M13 quality+Otsu 30.4.
- M3/M4/M7 noise scale is estimated over the quality mask intersected with |zAI| > 0 (fixed 2026-06-23;
  before the fix, zero voxels collapsed it and all three returned nothing). M7 uses 200 permutations.
- Cluster counts in that summary include both members of each mirrored pair.

## External validation and enhanced estimators (2026-10-20)
- Chalfont (3 T GE eASL, 66 same-scanner controls, 99 patients, 93 unilateral 43 L / 50 R): primary rule 58/93
  (balanced accuracy 0.63, majority 0.54); strong calls 16/20; leave-one-out controls receive a strong/moderate
  call in 37/66 (56%), any call in 63/66; calibration 6.1% (controls) vs 9.6% (patients).
- Regional normative temporal statistic (scripts/v2): 61/93 by sign (0.65, AUC 0.69); calibrated calls 12/13 at
  6% control FP. Smoothed voxel z-mass: 59/93, 8/8 calibrated calls. Mesial-only statistic dominated on every
  metric. Multivariate antisymmetric classifier: 55-59/93, no gain.
- Source: external_chalfont/results_zscore/lateralization/eval_*.csv, ez_analysis_mdt/v2/<dataset>/.
