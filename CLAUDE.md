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
  pair (`cluster_hemi == ez_side_pred`). Pair counts (clusters at |zAI| ≥ 1.96, ≥ 50 vox; verified 2026-11-03): cohort
  698 hypo / 562 hyper / 3883 subtle / 2 mixed, median 313 pairs per patient (128–399); research tier (peak ≥ 3): 688 / 442 /
  3421 / 2, median 275 (90–373). P015 107 hyper / 6 hypo; P027 151 hypo / 2 hyper. The per-patient volumes in the manuscript
  (tab:perpatient, P029 10.4 vs 2.9 cm³) are 1.96-cluster volumes, not research-tier ones. Older numbers (229/14,
  7785/1406/1180/3, 698/556/3915/1) are double counts or stale.
- **Headline results (honest, after the 2026-10-20 validity-rule rerun):**
  - Regional (Pipeline A) best AUC 0.60 (95% CI 0.28–0.90); leave-one-out 8/13 = always-left baseline.
  - Voxel-wise primary rule (clinical tier): 7/13 agree (P013, P016, P017, P023, P024, P026, P029), balanced
    accuracy 0.55; no rule >8/13 or balanced accuracy >0.61; always-left 8/13, 0.50. Peak zAI 5.9–9.2 (no clipping).
  - Direction class tracks the global CBF offset vs the HCP band (ρ≈0.87): hyper-dominant P015/P029/P030 = artefact.
  - Chalfont (93 unilateral, 66 same-scanner controls): primary rule 58/93 (0.63) but 56 % of LOO controls get a
    strong/moderate call; regional normative temporal z 61/93 (0.65, AUC 0.69), calibrated calls 12/13 at 6 % FP.
  - Voxel noise budget: control SD of AI 55–70 % at 0.8 mm → |zAI|=4 needs >200 % asymmetry (docs/audit/).
  - Protocol check (2026-11-03, Winston note: #1 region LI, #2 voxel AI, #3 CBF z vs controls, #4 z of AI): all four exist; #3 was only
    used for the direction class until `scripts/audit/perfusion_z_lateralization.py` (→ `ez_analysis_mdt/perfz/`). #3 hypo-volume side:
    Chalfont 59/93 (bal. 0.63), calibrated 11/13 at 6 % control FP; in-house 6–10/13 (in-sample sign rules, CI 0.50–0.92). Lobe-level
    localization on the true side is below the always-temporal baseline for every method (Chalfont: #4 28/72, #3 35/72, regional 18/72 vs 49/72;
    in-house 1–2/13): the largest abnormality lands frontal. Deviations from the note: Chalfont uses a fixed template atlas (no FreeSurfer);
    the symmetric template is study-specific, not MNI-sym.
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
- **Manuscript state (2026-11-04, journal limits met):** 4,987 words of main-text prose (`ms/wc.py` counter), 5 tables + 5 figures, 50 references
  (glass1976primary dropped, patenaude2011first added, PMID 21352927), 36 pages, no undefined refs. The supplement lives INSIDE
  `manuscript/Manuscript_Complete.tex` after the reference list (there is no Supplementary_Material.tex). Added 2026-11-04: Methods (iii)
  structural prior + pointer sentences; Results paragraph on the uncorrected-flow map, FIRST volume prior, sequential rule 68/93, cluster-size
  null 17/20, nested LOO, PVC, index forms; six new rows in `tab:enhanced` (Pareto marks moved to the sequential rule and the calibrated
  post hoc calls); Discussion/Limitations/Conclusions sentences; new supplementary sections `sec:provenance`, `sec:registration_supp`,
  `sec:indexform` (tab:indexform), `sec:flow_prior` (tab:flowprior, tab:improvements, tab:league) placed before the workflow section;
  the 2023–2026 literature paragraph, the thesis-rule sentence, the localization details and the reader's LI scale (now in the
  tab:reading caption) moved out of the main text to stay under 5,000 words. Build: `conda run -n tex tectonic manuscript/Manuscript_Complete.tex`.
- **Manuscript state (2026-11-03, superseded):** 4,994 words of main-text prose (counter `ms/wc.py` in the
  2026-11-03 session scratchpad: floats/captions excluded, `\%` handled), 5 tables + 5 figures in the article, 50 references
  (all cited), reference list at the end of the article, supplement after it (S1–S14). Fig. 5 is now the sensitivity/specificity/
  localization figure (`fig5_sens_spec`, `scripts/make_manuscript_figures.py::fig5_sens_spec`); the external-cohort schematic
  moved to the supplement (`fig5_external_cohort` → Fig. S4). Content added 2026-11-03: analysis-plan paragraph (#1–#4, two
  deviations), `sec:perfz` / `sec:perfz_results` (analysis 3, lobe localization), analysis-3 rows in `tab:enhanced`, detection-AUC
  sentence + thesis-rule result in Results, 2023–2026 literature paragraph in Discussion (5 new verified refs: zhang2025cbm,
  kershaw2025zai, jin2023harmonization, ndagijimana2026hippocampal, ripart2025aidhs; dropped pustina2015, marquand2016,
  huber1981, leys2013, worsley1996 + 9 earlier), supplement sections S8 alternative rules, S9 pipelines vs visual, S10 multivariate
  + methods table, S11 per-patient table, S12 strategies tested, S13 workflow/pseudocode (Algorithm S6), S14 case montages.
  Abstract 250 words incl. labels, 3 highlights ≤ 130 chars. Build: `conda run -n tex tectonic manuscript/Manuscript_Complete.tex`
  → 32 pages, no undefined refs. **Overleaf:** the user copies files there (own instruction 2026-11-03); the Overleaf copy had
  drifted (see memory), so copy local → Overleaf only.
- **Audit 2026-11-03 (thesis, sensitivity/specificity, literature):** `docs/audit/METHODS_AUDIT_2026-10-20.md` §10–12 +
  `docs/audit/LITERATURE_ASYMMETRY_METHODS_2023_2026.md` (70 PubMed-verified records). Scripts: `scripts/audit/thesis_roi_rule.py`
  (|AI| ≥ 10 rule: flags 35 % of Chalfont and 73 % of HCP controls), `scripts/audit/sens_spec_operating_points.py` (detection AUC
  0.45–0.55 on Chalfont for every statistic → no threshold with sens and spec both > 0.5; at spec 0.80 sens 0.19–0.25),
  `scripts/audit/strategies_2023_2026.py` (per-side Youden cut-offs collapse to always calling; PASCOM-style peak-cluster rule calls
  88/93 patients and 92 % of controls), `scripts/audit/best_operating_points.py` (combined/consensus statistics and the MNI-space
  hippocampal/amygdalar volume AI: best balanced point max_t min(sens, spec) 0.40–0.54 everywhere; **no method on today's
  images reaches sens and spec both ≥ 0.6**; §13 lists what would: peri-ictal timing, PVC/ATT-corrected GM CBF, native-space
  volumes, outcome labels, surface z-maps). Native-space structural prior on Chalfont (`scripts/audit/first_volume_ai.sh` = FSL FIRST
  on 170 staged T1s, 84 s each, `scripts/audit/volume_ai_chalfont.py` → `ez_analysis_mdt/audit/first_chalfont/`): hippocampal volume
  AI detection AUC 0.70, sign 62/93 (HS 16/18, MRI-negative 24/42), calibrated 24/31 at 6 % FP, Ndagijimana cut-offs 32/41; adding
  perfusion (sequential / two-feature LOO logistic) gives 64/93, HS unchanged; best balanced point 0.52 → requirement still unmet.
  In-house anatomy is MNI-space, so `first_volume_ai_inhouse.sh` must not be run (volumes normalized). Outputs under `ez_analysis_mdt/audit/`. Deck slides 31–34 carry these (43 slides).
- **Misdetected cases (2026-11-09, `interactive_viewer.py --misses`, options 26-27):** every method's call per patient
  against the reference, with the label file, each clinical source's side and the reasons. Tables
  `ez_analysis_mdt/misdetected/<dataset>/`, figures + case panels + `REPORT.md` in `visual_analysis/misdetected/<dataset>/`
  (gitignored); hand-coded in-house source sides and the thesis's own readings (thesis Tables 11-14, manuscript Table 1)
  in private `Dataset/reference_sources.csv`. Reproduces every published count. Chalfont: wrong calls inside the
  leave-one-out control range (|stat| < control 95th pct.) = rule 30/30, regional z 31/32, Flow 26/28, voxel mass
  34/34, #3 34/34, hippocampal volume 25/31; calls beyond it: regional z 12/13 right, Flow 14/16, voxel mass 8/8
  (inside: 60-66 %); rule Moderate/Strong in 56 % of LOO controls; sequential rule 12/12 resected vs 31/47
  clinical-hypothesis labels; PET agrees with the methods against the label in N019 and N034. In-house rule misses
  (P015, P022, P025, P027, P028, P030) are all carried by subtle-class clusters; groups: the rule's own vote while
  others are right (P022, P025 parietal, P028 tie, P030), perfusion against the label (P027, all methods + thesis
  reader left, unchanged without the offset), global offset (P015). Offset test as the field normalizes (Boscolo
  Galazzo 2015 divides CBF by the GM mean; here the median GM #3 z is removed from both sides of every cluster and the
  rule re-run): in-house 8/13 vs 7 (P015, P025 reverse to correct, P028 tie -> R, P023 breaks), Chalfont 56/93 vs 58
  (8 misses fixed, 10 hits broken) -- explains single misses, not a fix (Borghammer 2008 normalization bias).
  Extreme-only test (normative models count only |z| > 2 deviations, Rutherford 2022 eLife): rule with the subtle
  class dropped = 4/13 in-house (7 as run), 48/93 Chalfont (58): hits rest on sub-threshold clusters as much as misses.
  Reason flags are threshold-free tests (offset removed / extreme-only call changes; hyper vs hypo volume).
  Source coding verified: 149/149 codes backed by a verbatim quote found in the patient's row of thesis Tables 11-14
  (private `Dataset/reference_sources_quotes.csv`, `verify_inhouse_sources()`, test), 17/17 tiers match manuscript
  Table 1; user decision 2026-11-09: tentative findings ('?') count as weak support for that side. The thesis's readings were graded beside the presumed EZ (blinding not stated): visual 9/13, ROI
  9/13, voxel 7/13 vs blind 5-9/13; they miss P017 and P027 like our regional methods. P023's regional LI score equals
  the fitted 05_ cut (called R only by strict inequality). Deck slides 49-56 (62 slides; 56 = bilateral / unclear patients); supplement S14
  (`sec:misdetected`, tab:misdetected_inhouse / _external, figS_misdetected_*), 39 pages.
  **Bilateral / unclear patients, kept apart (2026-11-18, user goal):** `MissAnalysis.bilateral_unclear()` ->
  `bilateral_unclear.csv` / `.png`, REPORT section 9 (never scored; manuscript untouched by design). In-house P018 /
  P019 / P020 / P021: blind methods all R in P019 (stereo-EEG 19 R vs 15 L), split in the other three; 11/12
  statistics with controls inside the control range (unilateral 35/39), calibrated regional z calls a side only in
  P020 -- the cancellation a bilateral EZ predicts. Chalfont `R_patients.csv` also carries B-labelled patients
  (N028, N059, ...), reported the same way. Test `test_bilateral_unclear_kept_apart`.
- **Chalfont step 1 (ASL -> T1) audit and fix (2026-11-18, user goal "images not aligned"):** `bet -f 0.3` on the PD
  reference kept the whole head (bright scalp fat), so the rigid fit was head-to-brain normmi. Audit of 170 subjects
  (header geometry vs saved matrix; CBF-cortex correlation and ventricle/cortex CBF in T1 space): saved fit better
  than headers in 162/169, worse in none, but BBR on the GIF white matter improved every weak subject (D042 8.7 deg
  off). `reg_chalfont_subject.sh` step a = whole-head normmi init (`_asl2t1_init.mat`) + `flirt -cost bbr` (labels
  81-94, global_abs), accepted only within 15 deg / 8 mm of the init and with a CBF-cortex correlation not lower
  (`chalfont_io.py step1pick`; N052's BBR slid 14 mm and halved the correlation -> rejected); `<ID> step1` mode and
  `run_registration.sh --redo-step1` redo a) + d) only. Re-estimated 2026-11-18 for all 170: correlation median
  0.41 -> 0.45, better (> 0.02) in 86 of 169, worse in none, median move 1.3 deg / 1.2 mm, largest C061 9.7 deg, D042 8.7 deg (`visual_analysis/registration_audit/step1_population.png`).
  Weighted-normmi search is unstable (D042 60 deg off): do not use. `make_qc.py` adds `asl_t1_method/gm_corr/vent_ratio` (+ flag `low_asl_t1_fit` < 0.20) and shows masked CBF.
  The dragged rim at the brain edge is trilinear resampling of 4-mm slices 15-40 deg oblique to the T1 (maps are zero
  outside the head natively; synthetic perfect alignment gives vent/cortex ~0.2 vs 0.35 measured); step 4 composition
  verified (rigid-then-warp r = 0.995). Viewer option 6 now shows spline CBF masked to the T1 brain + 2 mm, with native
  voxels, PD and mask as hidden layers, cached per matrix mtime. Old matrices: `external_chalfont/qc/asl2t1_v1/`;
  comparison `qc/asl2t1_v1_vs_v2.csv`; deck slides 11-12 (62 slides). **Downstream Chalfont outputs (results_zscore*, v2, audits, numbers in the
  manuscript/deck) are from the old matrices until the pipeline is re-run.**
- **Table S2 ranking (2026-11-18, user goal):** `scripts/per_index_discrimination.py` ranks the 15 indices by
  min(sensitivity, specificity) (sens = left called left /8, spec = right called right /5), ties by balanced
  accuracy then AUC; columns `min_sens_spec`, `balanced_accuracy`, `left_correct`, `right_correct`. Glass's delta /
  simple difference / raw ratio lead (6/8, 3/5, 0.60); CV ratio (highest AUC 0.65) is one-sided (8/8, 2/5, 0.40).
  Table S1 is definitions (no ranking); main-text Table 2 and S9/S10 still AUC / balanced-accuracy ordered. Test
  `test_per_index_table_ranked_by_the_weaker_side`.
- **Audit 2026-11-04 (registration chain, index form, plain Flow, label provenance, improvement list):** audit doc §14.
  Labels: in-house = thesis Table 11 → `ez_ground_truth.py`; Chalfont = `build_subjects.py` priority (lateralisation column 47,
  laterality code 13, side column 4, regex on recruitment-log text 29 of the 93; tiers 12/34/47). AI = 200 (L−R)/(L+R): the 200 is
  cosmetic (z identical to 1e-13); six index forms give 58–61/93, none better (`scripts/audit/ai_definitions.py` →
  `ez_analysis_mdt/audit/ai_definitions/`). Plain Flow (the protocol's eASL_Flow) ≥ TC-Flow on every statistic: regional temporal
  65/93 (cal. 14/16) vs 61/93 (`ez_analysis_mdt/audit/flow_variant/`, features `v2_features/chalfont_flow/`); sequential FIRST
  volume-AI call → sign of Flow regional z̄ = **68/93 (0.73)**, best always-call result (`first_chalfont/results_flow.csv`). Rules call
  both sides (in-house L 4/8, R 3/5; Chalfont L 29/43, R 29/50): "always left" is the baseline, not the rule. Improvement list landed:
  `make_qc.py` mirror Dice + median−3·MAD rule, `prep_subject_features.py --tissue` + `analyze_v2.py --pvc`, empirical t*/Student-t
  null, cluster-size null, threshold families, nested LOO, global offset (results pending); `interactive_viewer.py --dataset chalfont`;
  `docs/pipeline/08-interactive-viewer.md` lists every map/table to open. Deck 46 slides: 4 = label provenance, 8–9 = registration
  chain + index definition, per-side bullet on the in-house slide, analysis-3 status cell corrected. **Deck conflict risk:** the
  project deck was overwritten at 14:21 and ~14:35 from scratchpad builds; the user's open PowerPoint copy may hold unsaved edits
  (the slide scripts anchor on titles and can be re-run on any version).
  Improvement list implemented the same evening (`analyze_v2.py` extensions, `prep_subject_features.py --tissue`, `make_qc.py` mirror Dice; audit §14 table): mirror-Dice QC flags 6 (no accuracy change); t* 1.80–1.85; temporal cluster-null call 20/17 at 6 % FP (TC) vs 13/12 regional; offset–direction ρ −0.43 to −0.54 persists after GM normalisation; nested LOO 61/93 TC, 58/93 Flow; PVC worse (57–58/93); FDR/Bonferroni 0 % control coverage but FP 0.12–0.20 at the statistic level. Deck: 52 slides (slide 37 = #3 vs #4 tie: McNemar 18 vs 17, p = 1.00, and the scanner-offset argument; league table after the strategies slide, Done lines on the improvement slide; 2026-11-09: four registration slides around the Chalfont chain slide — template build, three-dataset chain, one subject per dataset, calibration + mirror-Dice QC; Methods `sec:symreg` paragraph expanded to both cohorts, main text ≈ 4,996 words).
- **Manuscript state (2026-10-20):** 33 pages; Figs 1–5 + Appendix B (workflow figure, file table, Algorithms 1–5) + Appendix C
  (z-map montages S1a/S1b/S2, `case_montage` in `scripts/make_manuscript_figures.py`, 3× upsampled, PDF includes); notation
  section (`sec:notation`); multi-metric table (`tab:enhanced`), method comparison (`tab:method_comparison`), external rule
  grid (`tab:rules_external`), noise budget, demographics; references pruned to cited PMID/DOI-verified entries. Deck:
  `AsymmetryAnalysis.pptx` (43 slides, 2026-11-03: protocol-compliance table, method #3 vs #4 maps, #3 lateralization table, localization audit, thesis compliance, sensitivity/specificity figure, 2023–2026 literature and tested strategies added; 2026-10-23: 14 visual-explainer slides added — voxel/mirror walkthrough, control normative,
  distributions, calibration %, experiment distributions, noise budget, clusters/tiers, 13 thresholding methods on P016 (3 slides, rule +
  cutoff histogram + improvement each), Pipeline A, direction rule, v2 estimator, improvements map; neurologist questions 36–38; build = `build_deck.js` + `add_questions.py` in the session scratchpad). Single file,
  no backup copies (user rule).
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

**As implemented (audit 2026-10-20):** `02_` forms 18-connected clusters (`CLUSTER_CONNECTIVITY = 2`; the manuscript said 26 until 2026-11-09) at |zAI| ≥ 1.96 with ≥ 50 voxels; the tiers in `06_` and
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
- `interactive_viewer.py` (menu audited 2026-11-09, both cohorts): every symmetric-space view sits on
  `symreg/templates/T1_sym.nii.gz`; stated thresholds are the on-screen thresholds; focal views open on the strongest
  clinical-tier cluster (06_ report); region outlines = consensus parcellation (patients' own warped aparcs spill
  outside the average template, up to 7.8 % of cortex labels); #3, its clusters and CBF are shown inside the
  consensus brain only (they cover the scanner FOV; masked copies in /tmp/asym_view/); guide in
  `docs/pipeline/08-interactive-viewer.md`. **Menu = Methods order (2026-11-09):** 1-2 inputs, 3 #1 regional LI
  (Pipeline A + 05_ scores), 4-10 registration chain (affine/warped T1 recomputed with FSL, cached in
  /tmp/asym_registration/<ID>/), 11 #2 voxel AI, 12 control normative, 13 #4 zAI + calibration (11-13 open on one
  voxel and print AI -> mean/SD -> zAI), 14 clusters at 1.96, 15 clinical maps, 16 #3 (direction input), 17 direction
  classes, 18-21 decision (report, top-3, 37 regions, combined), 22 regional normative z_LI, 23 #3 on its own, 24-25
  all layers, 26-27 misdetected cases (cohort / patient; also `--misses`), 28 island cleaning, 29-30 utilities. Numbers come from `MENU`/`OPTION`; never hard-code "option N"
  (test_menu_follows_methods_order). Deck: every procedure slide carries a `code:` file:line reference and a `view:`
  option.

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
python interactive_viewer.py --subject <ID> [--png --open | --print]   # all layers (CBF, #3, #4, clinical) in FSLeyes or as PNG; menu options 24-25; --dataset chalfont --all = every patient
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
