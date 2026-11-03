# Methods audit: why the asymmetry methods do not lateralize, and what to change

Date: 2026-10-20. Scope: Pipeline A (regional) and Pipeline B (voxel-wise zAI), in-house cohort (17 patients, 180 HCP controls), with the Chalfont external run as it completes. All numbers below were recomputed today from the current outputs unless a file is cited.

## 1. Verdict

The pipelines are not broken by a coding slip; they are **under-powered by design**. The voxel-wise normative test at 0.8 mm has a detection floor of roughly a 200 % left–right difference, far above the 10–20 % interictal hypoperfusion that ASL studies report in temporal lobe epilepsy. What survives the thresholds is edge, partial-volume, cavity and noise voxels in both hemispheres, so every rule built on top of those clusters reduces to a coin toss on a huge bilateral background. Two further confounds make the results look structured when they are not: a global CBF offset between patients and their control band drives the hypo/hyper "phenotype", and the regional pipeline never had a control reference at all.

## 2. Evidence that the methods fail (recap)

| Result | Value | Source |
|---|---|---|
| Voxel-wise primary rule, clinical tier | 7/13 agree, balanced accuracy 0.51; always-left 8/13, 0.50 | `ez_analysis_mdt/clinical_stratified_summary.csv` |
| Best of 40 rule × tier × scope combinations | 9/13, 0.64 (post hoc) | same |
| Regional best AUC | 0.60 (95 % CI 0.28–0.90); LOOCV 8/13 = baseline | `ez_analysis_mdt/regional_loocv.csv` |
| Cluster pairs per patient at the clinical tier | 129–400 (median 313); L and R volumes both in the tens of cm³ in every patient | `clinical_direction_profile.csv`, `clinical_production_rule_by_tier.csv` |
| Peak \|zAI\| | clipped at 20 in 15/17 patients | `results_zscore/clinical/cohort_concordance.csv` |
| Lobe with the largest assigned volume | frontal in 9/13 (largest lobe, not the hypothesis) | same |
| Calibration | 12.9 % of GM voxels at \|zAI\| ≥ 1.96 in patients vs 6.9 % in leave-one-out controls (nominal 5 %) | `symreg/check_full_calibration.py` |

## 3. Root causes, ranked

### 3.1 The voxel-wise noise budget makes realistic pathology undetectable (critical)

Control SD of the voxel-wise AI (percent units, band FM_40_59, 60 controls, symmetric space, `sd_asymmetry.nii.gz`):

| Region | SD of AI, median (p25–p75) | \|AI\| needed for z = 4 | control mean \|AI\| |
|---|---|---|---|
| Gray matter, 37-ROI scope | 58 % (49–68) | 231 % | 8 % |
| Hippocampus | 55 % (45–67) | 220 % | 7 % |
| Amygdala | 63 % (56–71) | 253 % | 7 % |
| Temporal cortex | 70 % (59–86) | 281 % | 10 % |

The 5 % SD floor binds in 0.1 % of GM voxels; it is irrelevant. An AI of 200 % means one side is near zero: only CSF/GM borders, coverage gaps, resection cavities (P024) and noise outliers reach the clinical tier. This is why the two "strong" calls are a post-resection cavity and a gross temporal-pole lesion, why 200–400 pairs survive per patient on both sides, and why zAI saturates at the ±20 clip.

The noise is intrinsic to voxel-level ASL (voxel CV ≈ 40 % gives SD(AI) ≈ 100·√2·CV ≈ 57 %); upsampling 2.5 mm data to 0.8 mm adds no information. Recomputing the control SD after smoothing the perfusion maps with a 6 mm FWHM Gaussian (20 controls) gives 22 % in GM, 20 % in the hippocampus and 29 % in temporal cortex, i.e. still an 80–115 % detection floor at z = 4 and ~40 % at z = 2. Voxel-wise z-scoring of ASL asymmetry cannot reach the 10–20 % effect size without far heavier aggregation (ROI level or ≥ 10 mm smoothing with a lower z and cluster-level inference).

With the 8 mm kernel used by the v2 estimator (`python scripts/audit/voxel_noise_budget.py 8`): GM 17 % (IQR 13–21), hippocampus 16 %, temporal cortex 23 %, i.e. a 67 % asymmetry for z = 4 and 34 % for z = 2.

Script: `scripts/audit/voxel_noise_budget.py [fwhm]`.

### 3.2 The direction rule measures the control band, not the patient (critical)

Per-patient mean perfusion z against the band (`results_zscore/patients/*/summary.json`):

| Band | Patients | mean z (range) | voxels with z > 2 | voxels with z < −2 |
|---|---|---|---|---|
| FM_20_39 | P013, P015, P016, P017, P019, P020, P026, P028, P029, P030 | +0.09 to +1.07 | 4–22 % | 0.4–2.6 % |
| FM_40_59 | P018, P021, P022, P023, P024, P027 | −0.56 to +0.34 | 2–7 % | 0.1–3.7 % |
| FM_60_79 | P025 | −0.08 | 5 % | 0.6 % |

The three hyperperfusion-dominant patients (P015, P029, P030) are young-band patients with mean z of +1.07, +0.66 and +0.80; the two most hypoperfusion-dominant (P023, P027) are mid-band patients with the most negative mean z (−0.32, −0.56). The per-side perfusion z (threshold 2) that classifies clusters as hypo or hyper is therefore driven by a whole-brain offset between the local scanner/cohort and the HCP band (site, calibration, the 22–35-year gap, antiseizure medication), not by focal pathology. Any absolute-CBF comparison needs a global normalization (e.g. CBF divided by the subject's GM mean) and, preferably, same-scanner controls.

### 3.2b The reported "tiers" are not the tiers the documents describe (critical, verified)

`02_compute_zai.py` forms clusters once, at |zAI| ≥ 1.96 with ≥ 50 voxels (module defaults; `symreg/run_sym_pipeline.sh` passes no threshold; every `*_asymmetry_summary.json` records `threshold: 1.96`). `06_clinical_interpretation.py::_filter` and `scripts/clinical_stratified_analysis.py::predict` then keep a 1.96-blob when its **peak** |zAI| ≥ 4 and its extent **at 1.96** ≥ 100 voxels. A "clinical-tier cluster" is therefore a blob of mostly 2–3 z voxels containing at least one voxel ≥ 4 (median |mean z| of clinical-tier clusters is 2.7). CLAUDE.md, the manuscript Methods and `scripts/chalfont/CHALFONT_PLAN.md` describe ≥ 100 contiguous voxels all ≥ 4, and `03_clinical_maps.py` draws the surgeon-facing maps at a voxel threshold of 3 with ≥ 50 voxels, so the maps and the reports use different cluster definitions. The Chalfont primary endpoint inherits the implemented (peak-based) definition; its description must be corrected before results are written up, and the thresholds re-derived on the control null.

### 3.3 No error control for cluster inference on an antisymmetric map (major)

Extent thresholds of 50/100/200 voxels at 0.8 mm are 1.6/3.3/6.6 native voxels, i.e. below the acquired resolution; no family-wise or false-discovery control is applied; and every cluster has an exact mirror, so the counts describe the noise field twice. With 6.9 % of control GM voxels beyond \|z\| ≥ 1.96 and ~1.3 M GM voxels, hundreds of supra-threshold clusters per control brain are expected, matching the 129–400 pairs per patient. A tier can only be called "clinical" once its false-positive rate in controls is known (leave-one-out pseudo-patients; implemented for Chalfont, pending for in-house).

### 3.4 The regional pipeline has no normative reference (major)

`05_roi_discrimination.py` fits index orientation and threshold on the 13 patients themselves (so an AUC of 0.5 is a floor) and never compares a patient's regional LI with the control distribution. Regional control LI statistics are cheap to compute from the same symmetric-space data (prototype in Section 5).

### 3.5 Residual systematic asymmetry after symmetric registration (minor)

Control mean \|AI\| is 7–10 % in GM even in symmetric space. The normative mean absorbs it, but registration error inflates the SD and the GM/CSF borders dominate the surviving clusters. Any voxel-wise variant needs GM-probability weighting or surface-based sampling rather than a hard parcellation mask.

### 3.6 Reference standard and sample (major, not fixable by code)

Thirteen unilateral patients with an 8:5 split give a 95 % interval of roughly ±0.27 on balanced accuracy; only 5 sides are invasively or histologically supported; 5 patients are MRI-negative, the group in which Rentzeperis et al. (2023) found no asymmetry; P024 is post-resection; seizure timing and medication were not recorded.

### 3.7 Implementation defects found today

- Steps 01–03 held a full 14-M-voxel mask per cluster (14.5 GB per step-01 job, 23 GB per step-03 job), which caused the out-of-memory session kills; step 01 is fixed with byte-identical outputs, patches for 02 and 03 are staged.
- Independent code review findings: see Section 6 (pending).

## 4. What the literature says works (PubMed-verified)

- Guo et al. 2015, J Magn Reson Imaging 42:1386 (doi:10.1002/jmri.24920): ROI AIs of hippocampal head, amygdala and temporal lobe on 3D pCASL; hippocampal AI differed from controls in TLE with and without MTS; combining MRI with CBF AIs at optimal cutoffs lateralized 96 %.
- Wang et al. 2018, Neuroimage Clin 19:824 (doi:10.1016/j.nicl.2018.06.008): simultaneous ASL/FDG-PET, SEEG-defined MTLE (n = 12); normalized hippocampal ASL 0.84 ipsilateral vs 0.97 contralateral (≈ 14 % difference); ASL hypoperfusion matched SEEG, ASL–PET AI correlation r = 0.74.
- Rentzeperis et al. 2023, Epilepsy Res 193:107163: ipsilateral temporal hypoperfusion in both MRI+ and MRI− TLE, but asymmetry significantly increased only in MRI+; "lack of asymmetry in the MRI− group may negatively impact the utility of interictal ASL for lateralization".
- Tortora et al. 2022, Diagnostics 12:811 (doi:10.3390/diagnostics12040811): pediatric focal epilepsy (n = 65); SOZ detection 27 % visual, 73 % voxel z-score AI, 77 % voxel z-score vs age-matched normative CBF in MRI+; 18/46/64 % in MRI−; threshold z ≤ −1.64 on smoothed, normalized maps.
- Lam et al. 2020, J Neurosurg Pediatr 27:243 (doi:10.3171/2020.7.PEDS20397): quantitative z-score clusters raised sensitivity to 100 % but specificity fell to 23 %; visual reading 91 %/50 %.
- Sone et al. 2019, Front Neurol 10:318 (doi:10.3389/fneur.2019.00318): VOI-level ASL vs FDG-PET in 27 TLE; ASL discriminative value highest in temporal neocortex and amygdala VOIs; ASL not yet comparable to PET.
- Abdennadher et al. 2026, J Neuroradiol 53:101548 (doi:10.1016/j.neurad.2026.101548): multi-PLD ASL shows interictal hypo- and hyperperfusion extending beyond the focus; single-PLD accuracy depends on regional arterial transit time.
- Han et al. 2026 (CNS Neurosci Ther 32:e70876) and Aslam et al. 2022 (J Neurosurg 138:828): the PET Z-map/PASCOM designs the clinical tier was borrowed from use smoothed PET with far lower voxel noise; the z = 4 operating point does not transfer to unsmoothed ASL.

Common pattern of the methods that work: regional or smoothed quantities, relative (globally normalized) CBF, same-scanner controls, mesial-temporal ROIs, and lesional cohorts.

## 5. Enhanced estimators (v2) on both datasets

Code: `scripts/v2/prep_subject_features.py` (ROI means with the subject's own parcellation or the fixed atlas; 8-mm-smoothed CBF on a mirror-symmetric 2.4-mm grid) and `scripts/v2/analyze_v2.py` (regional normative LI z per ROI; smoothed voxel-wise normative AI with hypoperfused-mass lateralization; every threshold calibrated on leave-one-out controls at a 5 % false-positive call rate). Outputs: `ez_analysis_mdt/v2/<dataset>/`.

**Noise budget at the regional level.** Control SD of LI per ROI is 4–5 % (hippocampus 4.4–5.0 % in-house, 4.7 % Chalfont; median over 37 ROIs 3.9–5.5 %), i.e. ≈ 8–11 % in AI units, ten times smaller than the voxel-wise SD of Section 3.1. A 10–20 % asymmetry is a 1–2.5 SD event at the regional level and invisible at the voxel level.

**In-house (13 unilateral, 8 L / 5 R, HCP controls):** sign rules agree in 7–9 of 13 (balanced accuracy 0.55–0.68; always-left 0.62); only 1–3 patients exceed the control 95th percentile. Smoothed voxel calibration: LOO controls 4.1 % of GM beyond |z| ≥ 1.96 (nominal 5 %), patients 2–35 %. Nothing is distinguishable from chance at this n (95 % interval ≈ ± 0.27).

**Chalfont (93 unilateral, 43 L / 50 R, 66 same-scanner controls):**

| Estimator | Sign rule | Balanced acc. (Wilson 95 %) | Calibrated calls (control FP 6 %) |
|---|---|---|---|
| Regional temporal z | 61/93 | 0.65 (0.56–0.75) | 13 called, 12 correct |
| Regional all-37 z | 60/93 | 0.64 (0.54–0.74) | 14 called, 13 correct |
| Regional mesial z | 60/93 | 0.64 (0.54–0.74) | 12 called, 7 correct |
| Smoothed voxel, 37-ROI scope, t = 2 | 59/93 | 0.63 (0.53–0.73) | 8 called, 8 correct |
| Smoothed voxel, temporal scope | 53/93 | 0.56 | threshold at the ±1 boundary: not a 5 % calibration (control FP 14 %) |

Majority baseline 0.54. The regional temporal sign rule beats the majority baseline (binomial p ≈ 0.02) and a calibrated call is correct in 12 of 13 patients. Calibration of the smoothed voxel model on Chalfont: LOO controls 3.8 % of GM beyond |z| ≥ 1.96, patients 7.0 %.

**Strata (sign rules, Chalfont).** Temporal lobe epilepsy 34/49 (0.70, temporal statistic); lesional TLE 25/34 (0.74, mesial statistic); resection-defined side 9/12 (0.75); frontal 12/21 (0.53, chance); MRI-negative 32/47 (0.65 temporal). Calibrated temporal calls: 13 patients, 12 correct (6 lesional, 6 MRI-negative); the miss is D019, a left FCD with left *hyper*perfusion (z = +4.8), i.e. the one case where a direction-aware rule on relative CBF would have been right. Offset-free direction (GM-normalized ROI CBF) splits the cohort 47 hyper / 46 hypo, and the deviating side equals the reference in 57 %, so direction remains unhelpful for lateralization even without the global offset.

**Reading.** The large same-scanner dataset shows a modest, reproducible lateralizing signal (≈ 0.65 balanced accuracy, ≈ 92 % precision when a calibrated call is made) that the 13-patient in-house sample cannot resolve. The failure in the paper is therefore both: the voxel-wise method as implemented has no sensitivity to realistic effect sizes and no calibration, and the in-house sample is too small to measure a 0.65-vs-0.54 difference. See Section 7 for the as-implemented method on Chalfont.

## 5b. Multi-metric view (clinician's lens)

`scripts/v2/multimetric_comparison.py` → `ez_analysis_mdt/v2/<dataset>/multimetric_comparison.csv`. For every method: call rate, PPV of calls, sensitivity per side (no-call = miss), balanced accuracy, AUC with bootstrap CI, LR+, κ, control FP rate, strata. No single metric ranks methods; the call-rate × PPV Pareto front is reported.

Chalfont Pareto front: regional temporal z̄ always calling (call 1.00, PPV 0.66, AUC 0.69, LR+ 1.9, MRI-negative 32/47) → voxel rule strong+moderate (0.71, 0.68) → voxel rule strong only (0.22, 0.80, LR+ 4) → regional all-37 calibrated (0.16, 0.87, LR+ 6.5) → regional temporal calibrated (0.14, 0.92, LR+ 12, κ 0.85) → smoothed voxel calibrated (0.09, 1.00).

What one metric hides: "always majority side" reaches raw agreement 0.54 with sensitivity 1.00/0.00; the mesial statistic matches the temporal one on balanced accuracy (0.64 vs 0.65) but has the lowest AUC (0.63) and the worst calibrated calls (7/12, PPV 0.58), so it is dominated at every operating point; the voxel rule's moderate calls add coverage (0.71) without precision (0.68) and a worse right-sided sensitivity (0.42). In-house (n = 13) every always-calling method gives 7–9/13 and calibrated calls are 0–2, so the in-house cohort cannot rank methods at all.

**Recommendation.** Report the regional temporal statistic at two operating points: the continuous value with the control range (coverage) and the calibrated call (precision), never the mesial-only statistic alone, and always with the control FP rate and per-side sensitivity.

## 6. Independent code review (fresh agent, read-only; items marked ✔ were re-verified in the code by me)

Critical / major:
1. ✔ Direction inference runs on absolute CBF against HCP controls; whole-brain offset vs. fraction of hyper clusters has Spearman ρ = 0.87 (p = 5e-6) across the 17 patients; 72 % of clinical-tier pair members are "subtle", and for those the predicted side is simply the side farther from the HCP mean, i.e. the global offset. Same conclusion as Section 3.2.
2. ✔ Cluster definition mismatch (Section 3.2b).
3. ✔ No null for cluster inference; zAI tails are far heavier than Gaussian (Bonferroni-level |z| ≳ 5.3 retains 3.1 % of voxels in every patient, nominal 1e-7); one native voxel becomes ~30 grid voxels after spline resampling, so the 50-voxel extent is satisfied by a single extreme native voxel. Peak zAI is clipped at 20 in 16/17 patients, at brain-edge, susceptibility and partial-volume sites.
4. ✔ Rule selection is post hoc over 40 configurations; exact hypergeometric p for the best rule (9/13 with 10 L predictions) is 0.31, for the production rule 0.68. No CI on balanced accuracy.
5. ✔ The GM / 37-ROI mask is not mirror-symmetric (`valid = brain_mask & gm_mask & patient_valid & sd_ai > 0`), so mirrored clusters differ in size by a median 5–7 %, 2–20 pairs per patient disagree on the predicted side between their members, 15–25 % have no clean mirror, and summed left volume exceeds right by 1–3 % in 12/17 patients, a bias toward the 8:5 left majority. Fix: `gm &= np.flip(gm, 0)` once, and evaluate per-side z on the union of mask and flip(mask).
6. ✔ `03_clinical_maps.py` README/FSLeyes labels describe the superseded polarity-flip convention ("left-dominant usually points away from the EZ"); the step-06 report uses the opposite convention.
7. "Moderate" confidence is reachable with a single 51 mm³ temporal cluster on one side; with 55–190 clinical-tier clusters per patient this happens by chance. The literature attributions for the thresholds (Bartolomei 2008; Boscolo Galazzo 2016; Aslam 2022) do not support a 51 mm³ extent at ASL resolution.

Minor:
8. ✔ Mirror max-pooling of the AI SD is a no-op: AI is exactly antisymmetric and the validity mask is symmetric, so SD(v) = SD(mirror v) identically. The 2026-05-20 fix acted through the 5 % floor only; comment and manuscript Methods should be corrected.
9. The 5 % floor makes z non-comparable across regions (it penalizes low-variance deep GM relative to convexity cortex, which is where clusters land); clipping at ±20 destroys ordering among the extreme population, so ranking by peak is arbitrary.
10. Validity `(L > 0) & (R > 0)` selects on noise sign and admits pairs such as (0.5, 15) with |AI| ≈ 190, the clipped population.
11. Pipeline A: orientation chosen in-sample and the bootstrap CI does not re-select it; the 0/13 LOOCV of two scores is the anti-learning artefact of a null signal; `cohen_d_directional` standardizes by between-ROI variance (not an asymmetry effect size); `build_comprehensive_asymmetry.py` includes zero-perfusion voxels in regional means and uses original-space data with the patient's own parcellation.
12. Clusters are labelled with the control consensus parcellation while step 03 masks with the patient's own parcellation (P024's cavity is labelled "R-temporal" by control anatomy).

## 7. As-implemented method on Chalfont (pre-specified primary endpoint)

All stages completed 2026-10-20 13:00 (`external_chalfont/results_zscore/lateralization/eval_*.csv`).

| Quantity | Value |
|---|---|
| Calibration, GM % with \|zAI\| ≥ 1.96 | LOO controls 6.1 % (median 4.5 %); patients 9.6 % (median 7.5 %) |
| Primary rule, clinical tier, 93 unilateral (43 L / 50 R) | 58/93 agree (62 %), balanced accuracy 0.63, majority 0.54, binomial p = 0.06 |
| Calls | 48 L, 40 R, 5 bilateral; strong 16/20, moderate 29/46, weak 13/22 |
| Strata | lesional 32/46 (0.69); MRI-negative 26/47 (0.55); temporal 33/49 (0.67); frontal 10/21 (0.47); resected 8/12 (0.67) |
| Control false-positive call rate (LOO, clinical tier) | any call 63/66 (95 %); strong or moderate 37/66 (56 %, CI 44–67 %); strong 6/66 (9 %); specificity of strong/moderate 0.44 |
| Secondary endpoint 1 (in-house best rule, direction-aware majority) | 50/93 (0.54): did not replicate |
| Volume index AUC | 0.55 (0.42–0.67) |

**Reading.** On same-scanner data with n = 93 the as-implemented rule shows a weak signal (0.63) and the enhanced regional statistic a slightly stronger one (0.65) with far better precision when calibrated (12/13 vs strong calls 16/20). The in-house negative result was therefore mostly the sample and the off-site controls; the method's defects (Sections 3.1–3.3) cap what any voxel-wise cluster rule can deliver and explain why its calls cannot be trusted individually.

## 8. Recommended redesign, in order of expected gain

1. **Change the unit of inference from the 0.8-mm voxel to regions (and, for maps, to ≥ 8–10 mm smoothed, 2-mm data).** Compute AI/LI on ROI mean CBF (hippocampus, amygdala, parahippocampal, entorhinal, temporal pole, fusiform, inferior/middle/superior temporal, insula) and on heavily smoothed maps; z-score against controls; report per-ROI z with the control false-positive rate.
2. **Normalize globally before any absolute comparison.** Divide CBF by the subject's GM mean (or use the AI alone, which is global-free) and drop the absolute per-side z direction rule; infer direction from the normalized ROI CBF of each side.
3. **Calibrate every threshold on controls run as pseudo-patients (leave-one-out).** Choose the operating point by the control false-positive rate (e.g. ≤ 5 % of controls receive a lateralizing call), not by PET literature values.
3b. **Define clusters at the tier threshold** (voxel threshold = tier, extent in native voxels after smoothing), symmetrize the analysis mask, label with the patient's own anatomy, and make the map (step 03) and the report (step 06) use one definition.
4. **Replace cluster counting with a single calibrated statistic per hemisphere** (e.g. mean z over mesial-temporal ROIs, or smoothed-map cluster mass with a permutation/empirical-null extent threshold). Count each mirrored pair once by construction (use the positive-AI member only).
5. **Use same-scanner controls** (the Chalfont design) and age-matched bands without gaps; if HCP controls are kept, harmonize (ComBat-style) or restrict to the AI.
6. **Acquisition and QC:** record time since last seizure and medication; use the multi-PLD ATT map to flag vascular/transit artifacts that masquerade as asymmetry; analyze P024 separately.
7. **Clinical output:** a one-page report with (a) per-ROI z table with control reference ranges, (b) one smoothed asymmetry map at the calibrated threshold, (c) the subject's global CBF and calibration (fraction of GM beyond the control 95 % range), and (d) an explicit "no lateralizing abnormality" call when nothing passes the calibrated threshold.
8. **Validate on the pre-specified Chalfont endpoint** before any claim; report balanced accuracy with the majority baseline and the control false-positive rate every time.

## 8b. Independent code review of v2 (2026-10-20, second pass)

Fixed after review: rounded threshold in `multimetric_comparison.py` (all-37 calibrated calls 15/13 → 14/13, PPV 0.93); LR+ now averaged over sides (a constant call gives 1.0); Pareto flag and composite rank on unrounded values with NaN kept out; exact-zero statistic = no call; V variants whose LOO 95th percentile sits at |LI_M| = 1 are flagged `threshold_at_boundary` (only V_all_t2 is a true 5 % calibration); in-house stratum renamed 'invasively supported'. Headline numbers unchanged.

## 9. Advanced-method spike (2026-10-20)

`scripts/v2/advanced_lateralization.py`: covariate-adjusted regional normative model (age, sex, global GM CBF), antisymmetric L2 logistic regression over 37 LI z and 37 per-side relative-CBF z with mirror augmentation and no intercept (nested LOO), zero-shot transfer to the in-house scanner, and lobe-level localization. Result: 55–59/93 (balanced accuracy 0.59–0.63) versus 60/93 for the untrained temporal mean z; calibrated calls degenerate (control score range wider than patients); transfer 3–6/13; lobe localization 25 % vs 68 % majority. Conclusion: the statistical model is not the bottleneck; ASL signal, reference-standard quality and modality are. Full reasoning and the recommended path: `docs/superpowers/specs/2026-10-20-advanced-lateralization-design.md` (private).


## 10. Protocol compliance, analysis 3 on its own, and lobe-level localization (2026-11-03)

The study's analysis plan (Winston's protocol note) lists four analyses on one registration chain (ASL → T1 → symmetric
template): (1) region LI on the FreeSurfer parcellation, (2) voxel-wise AI in symmetric space, (3) voxel-wise z of CBF
against controls, (4) z of the voxel-wise AI. Status: all four exist. Deviations: Chalfont has no FreeSurfer parcellation
(fixed template atlas); the symmetric template is study specific, not a published symmetric MNI atlas. Analysis 3 had
only fed the direction class until `scripts/audit/perfusion_z_lateralization.py --dataset inhouse|chalfont`
(→ `ez_analysis_mdt/perfz/<dataset>/{summary,localization,localization_per_patient,localization_lobe_fraction}.csv`).

| Rule (analysis 3, 37-ROI gray matter) | Chalfont (93 unilateral; 66 LOO controls) | In-house (13; HCP controls) |
|---|---|---|
| side with more hypoperfused voxels (z ≤ −2) | 59/93, bal. 0.63, AUC 0.64 (0.53–0.76), binomial p = 0.04 | 6/13 (0.45) |
| same, temporal lobe only | 58/93 (0.62) | 8/13 |
| sign of mean-z difference, temporal | 58/93 (0.62) | 10/13 (0.78; Wilson 0.50–0.92, in-sample) |
| calibrated call (95th pct of \|d\| in LOO controls; 6 % FP) | 13 called, 11 correct (LR+ 5.1, κ 0.70) | 4 called, 3 correct |
| offset-corrected (z − median GM z), hypo volume | 54/93 (0.58) | 6/13 |
| reference: analysis 4 primary rule / regional z̄_temporal / majority | 58/93 / 61/93 (calibrated 12/13) / 50/93 | 7/13 / 8/13 / 8/13 |

Lobe-level localization (patients with one clinical lobe; lobe with the most abnormal voxels on the clinical side):

| Method | Chalfont (72: 49 temporal, 21 frontal, 1 parietal, 1 occipital) | In-house (13) |
|---|---|---|
| analysis 4, \|zAI\| ≥ 1.96 | 28/72 (frontal named in 57; temporal patients 9/49, frontal patients 19/21) | 2/13 (P024, P029) |
| analysis 3, z ≤ −2 (raw / offset-corrected) | 35/72 / 34/72 (frontal named in 45) | 1/13 |
| regional z̄, most negative lobe on the predicted side | 18/72 | — |
| counts divided by lobe size (analysis 3) | 23/72 | 2/13 |
| always "temporal" | 49/72 | 12/13 |

Reading: analysis 3 lateralizes like analysis 4 and below the regional statistic; in-house it is confounded by the global
offset (P027, P023: largest negative offsets, > 29 k hypoperfused voxels per hemisphere, wrong side). No map localizes:
antisymmetry (a cluster carries no side), boundary/noise voxels collected by the largest lobe, and the offset in analysis 3.
Localize only after a calibrated side call, at the regional scale, with per-region control false-positive rates.

Manuscript (2026-11-03): analysis-plan paragraph, `sec:perfz`, `sec:perfz_results`, rows in `tab:enhanced`,
`tab:method_comparison`, `tab:reading`, Algorithm 6, file-table row. Cluster-pair counts in the Results were relabelled:
698/562/2/3883 (median 313 per patient, 128–399) are the counts of all 1.96 clusters, not of the research tier
(688/442/2/3421; median 275, 90–373); the per-patient volumes in the table are 1.96-cluster volumes.

## 11. Thesis compliance, and why sensitivity and specificity cannot both be reasonable (2026-11-03)

**Did we follow the thesis (Rai 2026, `manuscript/Thesis.pdf`, Methods pp. 55–58)?** The thesis specifies (a) a ROI
analysis: AI = 100 (L − R)/(L + R) of mean CBF in 36 FreeSurfer regions (ASL registered to T1, 0.8 mm), |AI| ≥ 10 read as
significant (threshold taken from Hauf 2009 / Lim 2008), compared with the MDT lateralization; and (b) a voxel-wise
analysis: z = (patient CBF − control mean)/control SD against 60 age-matched HCP controls (3 sex-pooled bands built from
12 age × sex strata of 15), |z| ≥ 1.96 with an extent threshold K_E = 100 voxels, clusters read visually on the T1 with the
aparc+aseg overlay. The thesis reports reader-judged usefulness (voxel-wise "useful" in 7/13 unilateral patients, 6 hypo-
and 1 hyperperfused) and no sensitivity or specificity of its own; its discussion asks for subject-specific thresholds
and notes outliers in hyperperfused clusters and z-scores "beyond physiologically plausible ranges".

| Thesis step | Our implementation | Followed? |
|---|---|---|
| ROI AI, 36 regions, |AI| ≥ 10 | same AI on 37 pairs (thesis 36 + thalamus); the fixed |AI| ≥ 10 rule was never scored until today (`scripts/audit/thesis_roi_rule.py`); Pipeline A used ROC-fitted aggregations instead | partly |
| voxel z of CBF vs controls, |z| ≥ 1.96, K_E 100, visual reading | computed in step 01 (analysis 3) but used only for the direction class until 2026-11-03; our primary voxel analysis is the asymmetry z (analysis 4, Winston note) | partly (deviation justified by the note) |
| controls: 3 bands × 60, sex-pooled | same bands (`FM_20_39/40_59/60_79`) | yes |
| MNI space, 0.8 mm upsampling | MNI 0.8 mm plus flip-symmetric template registration (calibration 31 % → 13 %) | deviation, justified |
| reader judgement of usefulness | replaced by explicit rules, balanced accuracy, control false-positive rates | deviation, intended |

**The thesis ROI rule scored explicitly** (side with more hypoperfused regions at |AI| ≥ 10; `ez_analysis_mdt/audit/thesis_roi_rule/`):

| Scope | Chalfont (93 pat., 66 ctl) | In-house (13 pat., 180 HCP ctl) |
|---|---|---|
| temporal (10 pairs) | called 36/93, correct 26 (side acc. 0.72), balanced 0.28; **35 % of controls have a flagged temporal region** | called 9/13, correct 5 (0.56); **73 % of controls flagged** |
| all 37 pairs | called 64/93, correct 39 (0.61), balanced 0.42; 59 % of controls flagged | called 12/13, correct 7 (0.58); 91 % of controls flagged |

A 10 % asymmetry is about one control SD of a regional LI (SD 0.045–0.05 → AI SD ≈ 9–10 %), so with 10 temporal regions
most healthy subjects exceed it somewhere; the threshold is a PET/SPECT convention that does not transfer to ASL region means.

**Sensitivity and specificity over the whole threshold range** (`scripts/audit/sens_spec_operating_points.py` →
`ez_analysis_mdt/audit/sens_spec/`; sensitivity = patients called on the correct side, specificity = leave-one-out
controls not called, detection AUC = |statistic| of patients vs controls):

| Statistic | Chalfont detection AUC (95 % CI) | always call: sens / side acc. | at spec 0.80: sens (side acc.) | at spec 0.90: sens (side acc.) | In-house detection AUC |
|---|---|---|---|---|---|
| regional temporal z̄ | 0.53 (0.44–0.62) | 0.66 / 0.66 | 0.20 (0.73) | 0.17 (0.80) | 0.53 |
| regional all-37 z̄ | 0.55 (0.45–0.64) | 0.65 / 0.65 | 0.25 (0.82) | 0.17 (0.89) | 0.60 |
| regional mesial z̄ | 0.48 (0.39–0.56) | 0.65 / 0.65 | 0.16 (0.65) | 0.09 (0.57) | 0.49 |
| smoothed z-mass LI (all GM) | 0.45 (0.36–0.54) | 0.63 / 0.63 | 0.19 (0.90) | 0.11 (1.00) | 0.51 |
| smoothed z-mass LI (temporal) | 0.50 (0.41–0.59) | 0.57 / 0.57 | 0.16 (0.68) | 0.00 | 0.60 |
| analysis 3, temporal mean-z difference | 0.51 (0.42–0.60) | 0.62 / 0.62 | 0.19 (0.78) | 0.16 (0.83) | 0.62 |
| analysis 3, hypoperfused-volume LI | 0.45 (0.36–0.54) | 0.63 / 0.63 | 0.16 (0.94) | 0.11 (1.00) | 0.49 |

**Reading.** No statistic has an operating point with sensitivity and specificity both above 0.5. The reason is not the
rule but the signal: the *magnitude* of interictal asymmetry in patients is indistinguishable from the healthy range
(detection AUC ≈ 0.5 in 93 patients with same-scanner controls), while the *sign* carries only a weak side signal
(66 % correct, balanced accuracy 0.65). Every threshold therefore trades one error for the other: always calling gives
sensitivity ≈ 0.65 at specificity 0; a 5 % false-positive threshold gives specificity 0.94 at sensitivity 0.13–0.17 (the
"12 of 13 calibrated calls" of the manuscript). Methods tuning cannot manufacture a magnitude signal that is absent at
this noise level (regional LI SD ≈ 0.05; interictal TLE hypoperfusion 10–20 %, i.e. 1–2 SD at best, 0.5–1 SD in most
patients), against a clinical-hypothesis reference. The best available lateralizer remains the regional normative z̄
(temporal or all-37) read by sign, with the calibrated call as a high-precision, low-coverage adjunct; the thesis rule
and the voxel-wise cluster rules are dominated at every operating point.

**What would raise both numbers** (evidence-based, not yet done): acquire closer to seizures or postictally
(Gaxiola-Valdez 2017: postictal hypoperfusion localizes the SOZ), partial-volume and transit-time (multi-PLD ATT) corrected
GM CBF, more averages or background suppression for SNR, outcome-defined (resection + Engel I) reference sides, and
combination with structural asymmetry (hippocampal volume/FLAIR) or FDG-PET where available; see Section 12 for the
2023–2026 literature comparison.

## 12. What the 2023–2026 literature does differently (2026-11-03)

Full survey with 70 PubMed-verified records (PMID + DOI), per-component comparison and a ranked list of strategies:
`docs/audit/LITERATURE_ASYMMETRY_METHODS_2023_2026.md`. Headlines:

- **No 2023–2026 ASL study reports both sensitivity and specificity for single-subject lateralization with a control
  false-positive check.** The only ASL voxel-AI study (Mattioli 2026, children, n = 28) uses no controls and no numeric
  threshold; its 92.9 % "concordance" is a reader's call, not a tested rule. Interictal perfusion gives a confident
  lateralization rarely wherever it is measured (interictal SPECT 4.5 % high-confidence calls, Baradaran-Salimi 2024;
  PET surface z-map temporal sensitivity 0.41 at specificity 0.96 against SEEG, Han 2026), so our 14 % calibrated call
  rate at 6 % FP is in line with the field.
- **The methods that do reach sensitivity ≈ 0.87 / specificity ≈ 0.91** (Zhang 2025, PET, MRI-negative, resection +
  outcome reference) share features we do not use: cortical-surface sampling at 50 % depth, 20-mm FWHM surface
  smoothing, global-mean normalisation, same-scanner controls and a rank threshold (lowest 0.5 % of vertices) instead
  of a fixed |z|. They are PET, not ASL.
- **Structural asymmetry outperforms perfusion asymmetry by a wide margin:** hippocampal volume AI alone lateralizes
  histological hippocampal sclerosis with AUC 0.95–0.96 at fixed cut-offs (Ndagijimana 2026, n = 442); asymmetry
  features z-scored against controls with a logistic model reach 90 % sensitivity / 94 % specificity with external
  validation (AID-HS, Ripart 2025). Our `aparc+aseg` files are in MNI space (227 × 272 × 227, 0.8 mm) for patients and
  controls alike, so a native-space volume AI still needs the FreeSurfer `aseg.stats`; not done today.
- **Global-level normalisation before any one-sided comparison** is standard (Rentzeperis 2023 GM-median; Han/Zhang
  global mean) and Abdennadher 2026 states that asymmetry indices cannot separate a global shift from a focal one —
  the mechanism behind our direction-class artefact (ρ = 0.87). Our offset-corrected analysis-3 variant is the additive
  version and gave no lateralization gain (54/93).
- **Harmonised or held-out controls** (Jin 2023: FCD detection 50 % → 71 % after resolution/global harmonisation and age
  pruning; Qian 2026: 0 FP in 10 held-out controls) and **per-side ROC cut-offs** (Kershaw 2025: left sens 0.90 / spec
  0.80, right 0.88 / 0.76 for a PET z-score asymmetry index) are the two transferable steps; both tested in
  `scripts/audit/strategies_2023_2026.py` (results in Section 13).
- **Not evidence of a better ASL method:** ML lateralization accuracies of 0.90–1.00 come from structural/diffusion
  features in lesional TLE with small or split-sample validation; the Chen 2026 meta-analysis grades them at very high
  risk of bias. MAP z-maps in truly MRI-negative patients reach balanced accuracy 51–60 % with 3–6× more false than
  true positives (Fearns 2023) — the same profile as our voxel-wise maps.
