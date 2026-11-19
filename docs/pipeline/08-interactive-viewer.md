# Interactive viewer (`interactive_viewer.py`)

> One entry point for looking at every map of any patient, in-house or Chalfont: a menu that opens FSLeyes with
> the right files, colours and thresholds, in the order of the manuscript Methods, plus a command-line mode that
> opens one patient directly or writes PNGs for the IDE. It reads pipeline outputs and does not change them, with
> three exceptions:
> - option 18 rewrites the patient's `06_` report files (`<ID>_surgeon_report.txt/.json`), as `06_` itself does;
> - option 28 writes a cleaned map and montage;
> - options 26-27 (and `--misses`) write the misdetection tables to `ez_analysis_mdt/misdetected/` and figures, case
>   panels and a report to `visual_analysis/misdetected/` (both gitignored);
> - options 3, 11, 17, 19, 20 and 22 write display maps (regional values painted on the parcellation, the voxel AI,
>   direction-class masks, temporary cluster masks) to `/tmp/asym_view/` or `/tmp`.

All commands below were run on 2026-11-09 and work as written. Use the `Asym` environment. The script changes to the
project root itself, so it can be started from any directory. Printed FSLeyes commands use paths relative to the
project root.

## Quick start

```bash
conda activate Asym
cd ~/AsymmetryAnalysis

python interactive_viewer.py                               # menu, in-house cohort (17 patients)
python interactive_viewer.py --dataset chalfont            # menu, Chalfont cohort (99 patients, D* and N*)

python interactive_viewer.py --subject P016                # one patient, every layer, in FSLeyes
python interactive_viewer.py --subject D002 --png --open   # the same as a PNG, opened in VS Code
python interactive_viewer.py --subject D002 --registration --open   # the registration chain as one PNG
```

At the patient prompt, type the list number or the ID itself (`p016`, `D002`; case does not matter). Then pick an
option. `S` switches patient and `0` exits.

## Menu order = Methods order

The menu lists the views in the order of the manuscript Methods (Section 2), so walking it from 1 to 23 presents
the methods in the order the paper and the deck describe them. Each menu heading names its Methods subsection.
The numbers are positions in `MENU` (`interactive_viewer.py`); every "option N" the viewer prints is taken from
there, so a reorder renumbers everything at once (test `test_menu_follows_methods_order`).

| Options | Menu section | Manuscript (Methods) | Deck slides |
|---|---|---|---|
| 1–2 | Inputs (original space, in-house) | MRI acquisition and processing | 3 Two cohorts · 5 Framework |
| 3 | Analysis #1, regional LI (Pipeline A) | Regional asymmetry analysis | 27 Pipeline A, visually |
| 4–10 | Left-right symmetric registration | Left–right symmetric registration | 8 template · 9 three datasets · 10 Chalfont chain · 11 one subject per dataset · 12 calibration and QC |
| 11–13 | Analyses #2 and #4: voxel AI, control normative, zAI | Asymmetry index and normative z-score | 13 index form · 15 one voxel, its mirror · 16 control normative · 18–19 calibration · 20/22 noise budget |
| 14–15 | Thresholding: clusters at 1.96 (`02_`), clinical maps (`03_`) | Thresholding | 23 z map → clusters → tiers · 24–26 thresholding methods |
| 16–17 | Direction: #3 as the rule's input, direction classes | Direction-of-abnormality inference | 29 class tracks the offset · 30 direction rule step by step |
| 18–21 | Lateralization decision: report, top-3, 37 regions, combined | Lateralization decision and clinical scope | 31 representative cases · 46 what the clinician should read |
| 22–23 | Post hoc: regional normative z_LI, #3 on its own | Enhanced estimators; Direct perfusion z-score | 34–35 regional normative estimator · 36–37 method #3 |
| 24–25 | All layers of one subject | (summary view) | 14 #3 next to #4, same slice |
| 26–27 | Misdetected cases: every method against the reference, and why | Supplementary S14 (Results) | 50–57 misdetected cases |
| 28–30 | Display aid and utilities | — | — |

Where the deck differs: it shows Pipeline A (slide 27) after thresholding and the #3/#4 side-by-side (slide 14)
before the index; the viewer keeps the manuscript order, in which the regional analysis needs only the inputs
(in-house it runs on the patient's own FreeSurfer parcellation in MNI space, before any symmetric registration).

Options 11, 12 and 13 open on the same voxel (the strongest clinical-tier cluster) and print its arithmetic, so one
voxel can be followed from AI to zAI:

```
11  At the opened voxel: AI -140.2 %.
12  At the opened voxel: control mean -12.1 %, SD 61.5 % AI.
13  At the opened voxel: zAI = (AI -140.2 - mean -12.1) / SD 61.5 = -2.08 (map value -2.08; clipped at +-20).
```

(P016; the AI of option 11 reproduces `02_`'s zAI to 5 × 10⁻⁷ over every brain voxel.)

## Guided examples

### Example 1 · In-house P016 (MRI-negative right TLE, stereo-EEG confirmed)

```bash
python interactive_viewer.py
# patient: P016
```

The header tells you what you are looking at:

```
INTERACTIVE OVERLAY VIEWER — inhouse cohort (results_zscore/)
Current patient: P016   control band: FM_20_39
MDT EZ: R (Moderate confidence)
Available data: [+] orig anatomy  [+] sym-space underlay  [+] raw z-score  [+] zAI  [+] clinical
Underlay: symmetric space (symreg/templates/T1_sym.nii.gz)
```

To present the methods, walk the menu in order: 1–2 (inputs), 3 (regional LI and the nine aggregation scores),
4–9 (registration, or 10 as one PNG), 11 → 12 → 13 (one voxel from AI to zAI, with the calibration), 14 → 15 (clusters,
clinical maps), 16 → 17 (the rule's input and the class of each pair), 18 (the call), 21, then 22–23 (post hoc).

For a quick clinical read of one patient:

1. **18 · Patient report.** This is the rule's answer in text: clinical-tier clusters (|zAI| ≥ 4, ≥ 100 voxels, 37
   region pairs), predicted side per cluster, lobe totals. Answer `y` at the end to jump to the top-3 cluster view.
2. **21 · Combined presurgical view.** It opens on the strongest clinical cluster. The zAI map (faded) and the
   clinical clusters are visible, with region outlines; #3, the control mean CBF and the side masks are hidden layers.
3. **17 · Direction classes.** Which pairs the rule calls hypo- or hyperperfused, and on which side, with the global
   #3 offset printed: for P016, 22 L / 25 R hypo pairs, 11 L / 8 R hyper, 244 subtle.
4. **24 · All layers.** CBF, #3, #4 and the clinical clusters in one window, opened on the strongest clinical cluster.
   Use **25** for the same view as a PNG.

### Example 2 · Chalfont D002 (left frontal FCD, histology)

```bash
python interactive_viewer.py --dataset chalfont
# patient: D002
```

```
Current patient: D002   control band: CH_ALL
Reference side: L (discrete; frontal; FCD [histology])
Available data: [-] orig anatomy  [+] sym-space underlay  [+] raw z-score  [+] zAI  [+] clinical
```

Options 3–28 work as for in-house patients; option 3 has no FreeSurfer parcellation here and paints the regional
means of the fixed template atlas (`v2_features/chalfont/<ID>_roi.json`) in symmetric space, as the Methods state.
Options 1–2 are marked `(in-house only: no native anatomy)`, because Chalfont subjects have no `Dataset/<ID>/`.
Option 18 prints the rule's report for this patient, which calls **bilateral**: 5 clinical-tier clusters, the
strongest a left precentral hypoperfusion cluster at zAI −4.33. Options 11–13, 15, 17, 21 and 24 open on that cluster
and say so:

```
Opened at voxel (176, 159, 149): strongest clinical-tier cluster 119: L-precentral, hypo, peak zAI -4.33, this cluster's side L (centroid)
```

The clinical layer shows both members of each pair: red is the member above its mirror, blue the member below. The
crosshair sits on the blue left member, where the rule puts the EZ for a hypoperfused cluster.

### Example 3 · Straight to one patient, no menu

```bash
python interactive_viewer.py --subject D002            # FSLeyes, every layer, opened on the strongest clinical cluster
python interactive_viewer.py --subject D002 --print    # only print the FSLeyes command (paste it elsewhere)
python interactive_viewer.py --subject P016 --at 60 150 90   # open at a voxel of your choice
```

The dataset is taken from the ID: `P…` is in-house; `C`, `D` and `N…` are Chalfont. The first line printed
summarises the patient:

```
D002 [chalfont, CH_ALL]  reference L  rule B (Bilateral)  top L-parietal  peak |zAI| 4.33
```

### Example 4 · PNGs for the IDE (no display needed)

```bash
python interactive_viewer.py --subject D002 --png --open    # one PNG, opened in VS Code (`code`) or the default viewer
python interactive_viewer.py --dataset chalfont --all       # all 99 Chalfont patients, ~1.5 s each
python interactive_viewer.py --dataset inhouse --all        # all 17 in-house patients
```

PNGs go to `visual_analysis/subject_views/<dataset>/<ID>.png`. That folder is gitignored, so regenerate it on each
machine. Each PNG has one row per layer (CBF, #3, #4, clinical clusters) and three views (sagittal, coronal, axial)
through the strongest clinical cluster, marked by a green crosshair; the title says which voxel and why. In VS Code, click a PNG in the Explorer, then use the arrow
keys to step through the patients.

### Example 5 · Copy-paste commands for a patient (option 30)

Option 30 prints the FSLeyes commands that options 13, 14, 15, 16, 21 and 24 would run for the current patient, each
labelled with its option, plus the PNG command. Paste them on any machine that has FSLeyes and the project files.

### Example 6 · The registration chain of one patient (options 4–10)

Options 4–10 show how a patient got from the scanner to the symmetric template. They follow the same steps as the
deck slide "Three datasets, one registration target".

```bash
python interactive_viewer.py --dataset chalfont   # patient D002, then 4 ... 10
python interactive_viewer.py --subject D002 --registration --open   # options 5-9 in one PNG, no menu
```

| # | Step | What you see | What to check |
|---|---|---|---|
| 4 | template | `T_sym` with the voxels where the plain mean of 30 HCP T1s differed from its mirror by ≥ 10 % (hidden layer: the plain mean) | `T_sym` is identical to its mirror; the hot voxels are what averaging removed |
| 5 | 0 · inputs | the input T1 on its own grid, with the CBF where the scanner coordinates put it; hidden: the ASL reference and native labels | Chalfont: CBF and T1 are visibly offset before registration |
| 6 | 1 · ASL → T1 | the CBF resampled into the T1 with the saved rigid matrix (`<ID>_asl2t1.mat`) as the pipeline does it (spline), inside the T1 brain + 2 mm; hidden: the same CBF at its native voxels (nearest neighbour, unmasked), the PD reference, the T1 brain mask; the method and the ASL → T1 QC numbers in the terminal | perfusion fills the cortex and spares the ventricles; the smear along the oblique slice direction in the smooth layer is interpolation of 4-mm slices (compare the native layer). In-house: a note only, because HCP-ASL did this step |
| 7 | 2 · T1 → T_sym, affine | the T1 after the affine on `T_sym`, with the template's brain and white-matter outlines | brain edge and ventricles near the outlines; folds still individual |
| 8 | 3 · T1 → T_sym, non-linear | the T1 after fnirt with the same outlines; hidden: the T1 mirrored left-right (mirror check); QC numbers in the terminal | cortex and ventricles on the outlines; the mirrored copy should overlap |
| 9 | 4 · CBF and labels on T_sym | the CBF after one resampling, with the consensus region outlines | the input of `01_`, `02_` and `03_` (and of options 11–13) |
| 10 | the chain as a PNG | columns: input T1, step 1, step 2, step 3, step 4; QC in the title; `visual_analysis/subject_views/<dataset>/<ID>_registration.png` | everything above at a glance |

The registration scripts deleted the affine matrix and the warped T1. Steps 2–3 recompute them with the same FSL
commands (`flirt` and `applywarp`; FSL must be on `PATH`) and cache them in `/tmp/asym_registration/<ID>/`. The first
call per patient takes 25–45 s, later calls are instant. QC for D002: NCC with `T_sym` 0.79 and mirror Dice 0.956 on
the 0.8-mm grid; the 2-mm QC file gives 0.819 and 0.960.

#### Why step 1 is brain to brain, and what the rim at the brain edge is (audit 2026-11-18)

Until 2026-11-18 step 1 ran `bet -f 0.3` on the PD reference and registered the result to the T1 brain with
normalized mutual information. On a PD image the scalp fat is brighter than brain, so bet kept the whole head in every
subject (the 20 000-voxel fallback never fired): the fit was head-to-brain. An audit of the 170 registered subjects
(scanner-header geometry versus the saved matrix, scored in T1 space by the correlation of the CBF with the GIF cortex
and by the ventricle / cortex CBF ratio) found the fit still better than the headers in 162 of 169 and worse in none
(correlation 0.28 → 0.40, ventricle / cortex 0.56 → 0.36), but boundary-based registration on the GIF white matter
(`flirt -cost bbr`, FSL's ASL → T1 method) improved every weak subject, by 8.7° in D042. Step 1 now keeps the whole-head
normmi fit as the initialisation (`<ID>_asl2t1_init.mat`) and refines it with BBR; the refinement is accepted only when
it stays within 15° and 8 mm of the initialisation and its CBF–cortex correlation is not lower (`chalfont_io.py
step1pick`); the genuine refinements moved ≤ 9.7° / 4.8 mm, the one divergence (N052) 14 mm with the correlation
halved, and it is now rejected. Over the 169 subjects with GIF, the correlation rose from 0.41 to 0.45 (median), by
more than 0.02 in 86 and lower in none, and the median matrix moved 1.3° / 1.2 mm (`visual_analysis/registration_audit/step1_population.png`; deck slides 11–13). Weighting the
normmi search by the projected T1 brain mask was tried and is unstable (D042 converged 60° off). `registration_qc.csv`
carries `asl_t1_method`, `asl_t1_gm_corr` (flag `low_asl_t1_fit` below 0.20) and `asl_t1_vent_ratio`; the per-subject QC
image shows the brain-masked CBF in the T1 instead of a PD contour. The pre-2026-11-18 matrices are kept in
`external_chalfont/qc/asl2t1_v1/` and compared with the new ones in `qc/asl2t1_v1_vs_v2.csv`; re-run with
`bash scripts/chalfont/run_registration.sh --redo-step1 -P 4` (ASL → T1 and the resampling only; the T1 warps stay).

What looks like misalignment at the brain edge, the vertex and the cerebellum is mostly not registration. In their own
space the eASL maps are zero outside the head and 7–20 ml/100 g/min outside the brain, but their 1.9 × 1.9 × 4 mm voxels
lie 15–40° oblique to the T1, so trilinear resampling onto the 1-mm grid smears every slice across the brain edge and the
ventricles; a perfectly aligned synthetic map resampled the same way gives a ventricle / cortex ratio of about 0.2, the
measured maps 0.35 (median). The slab of 36–40 slices ends in a straight edge below the cerebellum and, in some subjects,
at the vertex. The CBF on `T_sym` (step 4) is the rigid matrix and the T1 warp composed in one spline resampling;
recomputing it as rigid-then-warp reproduces it to r = 0.995 (median difference 1 ml/100 g/min, interpolation only), and
CBF ≥ 10 covers 95–98 % of both the warped subject brain and the template brain.


**Re-run on the new matrices (2026-11-18 evening).** Pipeline B for both CBF maps, the v2 estimators (features rebuilt: `prep_subject_features.py` skips subjects whose `.npz` exists, so `v2_features/chalfont*` was deleted and the base pass run before `--tissue`), analysis #3, the FIRST prior, the operating-point audits and the misdetected report were re-run. The pre-specified voxel rule fell from 58/93 to 49/93 (balanced accuracy 0.53, below the 0.54 baseline) with 44 of 66 leave-one-out controls receiving a Strong or Moderate call (37 before); 19 of 93 calls flipped, most of them near-ties in cluster volume, and the median hemispheric cluster-volume asymmetry fell from 0.68 to 0.55 log units. Region-mean statistics barely moved (regional temporal z 60/93 vs 61/93, calibrated 12/14; Flow 64/93, 14/16; #3 temporal sign 58/93 unchanged; hypo-voxel #3 55/93 vs 59/93; sequential FIRST -> Flow 67/93 vs 68/93). Deck slide 13 carries the before/after table, and the manuscript and the deck's Chalfont slides were updated to the re-run numbers the same night (Methods step (a) now describes the BBR refinement, Greve and Fischl 2009).
#### Why step 2 differs between the cohorts

In-house T1s run `flirt 12 dof -nosearch`; Chalfont T1s run `flirt 6 dof` (full search) and then `flirt 12 dof -init
-nosearch`. The difference is the starting point. The matrices recomputed for P016 and D002 (FSL `avscale`) show it:

| | In-house P016 | Chalfont D002 |
|---|---|---|
| input T1 | HCP `MNINonLinear` output, already on the template's 0.8-mm MNI grid (227 × 272 × 227) | 1-mm MPRAGE in scanner space (256 × 220 × 256) |
| brain centre vs template | within 1–2 mm | 26 mm higher |
| rotation the affine must find | ~1° | ~30° (head pitch) |
| translation | < 2 mm | 40–70 mm |
| scaling | 2–12 % | 11–14 % |

A 12-dof fit started from the identity (`-nosearch`) only polishes a transform that is already close. That is the
in-house case, and a search could only jump to a wrong optimum. A 30° rotation is too far for a local optimiser, so
for Chalfont a rigid 6-dof fit with flirt's rotation search first finds the head's pose, and the 12-dof fit refines it
from there. Both chains end in a 12-dof affine, feed the identical fnirt (step 3) and are checked by the same QC.

## Menu reference

Every FSLeyes view uses the same background: the symmetric template T1, with the patient's control-band mean CBF as
a hidden layer. Signed maps are drawn red-yellow above zero and blue-lightblue below zero, starting at the stated
threshold; anything weaker is transparent. Views marked "peak" open on the voxel with the largest |value| of their
main layer. Options 11–13, 15, 17, 21, 24 and 25 open instead on the strongest clinical-tier cluster: the first row
of the `06_clinical_interpretation` report, by |peak|, at its centroid on the side predicted for that cluster. The
map peak is the fallback, and the terminal says which one was used. The patient-level call (option 18) weighs all
clusters and can differ. For P016 the strongest cluster is left medial orbitofrontal (its side L), while the call is
R (moderate).

| # | View | Main layer and threshold on screen | Opens at | Chalfont |
|---|---|---|---|---|
| **Inputs** | | | | |
| 1 | T1 + FreeSurfer parcellation, original space | `Dataset/<ID>/T1w_acpc_dc_restore` + `aparc+aseg` outlines | — | no (no native anatomy) |
| 2 | CBF on the T1, original space | `<ID>_perfusion_calib_resampled_to_T1w`, hot 0–80 | — | no |
| **Analysis #1** | | | | |
| 3 | regional LI of the 37 pairs (Pipeline A) | LI = (L − R)/(L + R) painted on the parcellation, the right member with the opposite sign (blue = lower than its homologue), \|LI\| ≥ 0.01, saturating at 0.15; prints the ten largest pairs and the aggregation scores of `05_` (signed-weighted, temporal, mesial, mean, anatomically weighted, temporal-focused, dominant side). In-house: `results_analysis/<ID>/<ID>_comprehensive_asymmetry_analysis.csv` on `Dataset/<ID>/aparc+aseg`, original space | — | yes: template-atlas means, symmetric space |
| **Registration** | | | | |
| 4–10 | registration chain | template build, inputs, ASL → T1, affine, fnirt + mirror check + QC, CBF on `T_sym`, PNG (Example 6) | — | yes (6: Chalfont only) |
| **Analyses #2 and #4** | | | | |
| 11 | #2 voxel AI, patient alone | AI = 200 (v − v̄)/(v + v̄) from `02_compute_zai.compute_asymmetry_map` on the CBF on `T_sym`, inside a mirror-symmetric brain mask, \|AI\| ≥ 10 % (the thesis ROI cut-off), saturating at 100 %; prints the median \|AI\| in 37-pair gray matter and the AI at the opened voxel. Cached in `/tmp/asym_view/<ID>_<dataset>_voxel_ai.nii.gz` | strongest clinical cluster | yes |
| 12 | control normative | the band's per-voxel SD of AI (`<results>/asymmetry/groups/<band>/sd_asymmetry`, hot 20–100 %); hidden: the mean AI and the SD of CBF (#3's normative); prints the median SD, the \|AI\| needed for \|zAI\| = 1.96 and the mean and SD at the opened voxel | strongest clinical cluster | yes |
| 13 | #4 zAI map | `<ID>_asymmetry_zscore`, \|zAI\| ≥ 1.96; every finding appears twice (pair); prints the gray-matter calibration (% \|zAI\| ≥ 1.96) and (AI − mean)/SD at the opened voxel | strongest clinical cluster | yes |
| **Thresholding** | | | | |
| 14 | clusters at 1.96 (`02_`) | `<ID>_asymmetry_significant`, \|zAI\| ≥ 1.96, ≥ 50 voxels (18-connected), both pair members; hidden: the sign masks `_left_dominant` (voxel > mirror) / `_right_dominant`, not hemisphere-specific | peak | yes |
| 15 | clinical maps (`03_`) | `<ID>_clinical_zai_significant_clusters`, gray matter, 37 pairs, \|zAI\| ≥ 3, ≥ 50 voxels, both members of each pair; hidden: the higher-member-only map and L>R / R>L masks | strongest clinical cluster | yes |
| **Direction** | | | | |
| 16 | #3 perfusion z vs the control band | `<ID>_vs_<band>_zscore`, \|z\| ≥ 1.96: the input of the direction rule | — | yes |
| 17 | direction classes | each `02_` cluster pair drawn once, on the member in the predicted hemisphere (`cluster_hemi == ez_side_pred`): hypo blue, hyper red, mixed green, subtle yellow; hidden: the mirror members and #3 at \|z\| ≥ 2; prints pairs per class and side and the global #3 offset (median in 37-pair gray matter) | strongest clinical cluster | yes |
| **Lateralization decision** | | | | |
| 18 | patient report | `06_clinical_interpretation`: clinical tier (peak \|zAI\| ≥ 4, ≥ 100 voxels), predicted side, lobe totals | — | yes |
| 19 | top-3 clusters (table + overlay) | clusters with peak \|zAI\| ≥ 4 from the clinical cluster report, yellow mask | peak of the 3 | yes |
| 20 | 37-region view (table + overlay) | zAI inside the 37 region pairs, \|zAI\| ≥ 4; top-5 regions printed | peak | yes |
| 21 | combined presurgical view (recommended) | zAI ≥ 1.96 (faded) + clinical zAI ≥ 3 (both members) + outlines; hidden: #3, mean CBF, side masks | strongest clinical cluster | yes |
| **Post hoc** | | | | |
| 22 | regional normative z_LI (enhanced estimator i) | z_LI = (LI − control mean)/control SD per pair (`scripts/v2/analyze_v2.py`), painted like option 3, \|z\| ≥ 1, saturating at 3; prints the mesial, temporal and all-37 means (equal to `ez_analysis_mdt/v2/<dataset>/R_patients.csv`) and the calibrated temporal call | — | yes |
| 23 | #3 on its own: hypo / hyper clusters | `_hypo_clusters` (blue), `_hyper_clusters` (red), z beyond ±1.96, ≥ 10 voxels, region outlines on; prints the #3 rule (hypoperfused 37-pair gray-matter voxels at z ≤ −2, L vs R) from `ez_analysis_mdt/perfz/<dataset>/subjects.csv` | — | yes |
| **Summary** | | | | |
| 24 | all layers in FSLeyes | template, atlas, CBF, #3, #4 and clinical zAI; only the template and clinical layers on | strongest clinical cluster | yes |
| 25 | all layers as a PNG | as 24; `visual_analysis/subject_views/<dataset>/<ID>.png` | strongest clinical cluster | yes |
| **Misdetected cases (Results)** | | | | |
| 26 | every method's misses in the cohort | prints each method's right / wrong / no call and writes the tables, figures, case panels and `REPORT.md` (section below) | — | yes |
| 27 | this patient | the reference and the file it came from, each clinical source, each method's call (and whether its statistic is inside the controls' range), the primary rule's vote by direction class and the reasons it differs; opens the case panel PNG | — | yes |
| **Display aid and utilities** | | | | |
| 28 | island-cleaned zAI | `clean_overlay.py`: \|zAI\| ≥ 3, ≥ 50 voxels, 37 pairs, opening + closing; writes `<ID>_zai_cleaned.nii.gz` and a montage; says so when nothing survives (P016, D002). A display aid, not an analysis step | peak | yes |
| 29 | file list | every file the views use, with sizes | — | yes |
| 30 | FSLeyes commands | the commands of 13, 14, 15, 16, 21 and 24 for copy and paste | — | yes |

Three thresholds appear, and they mean different things:
- **1.96:** where the pipeline forms clusters (`02_`).
- **3:** the voxel threshold of the clinical maps (`03_`).
- **4:** the clinical tier, applied to a cluster's peak (`06_`, options 18 and 19).

The menu labels say which one each view uses.

### Layers of options 24 and 25

| Layer (bottom → top) | File | FSLeyes start state | PNG row |
|---|---|---|---|
| Template | `symreg/templates/T1_sym.nii.gz` | on | underlay |
| Region outlines | consensus parcellation: in-house `results_zscore/groups/<band>/consensus_parcellation.nii.gz`; Chalfont `external_chalfont/atlas/consensus_parcellation_sym.nii.gz` | off | — |
| CBF | `symreg/sym_perf/<ID>_perf_sym.nii.gz` · `external_chalfont/sym_perf/<ID>_perf_sym.nii.gz` (TC-Flow); FSLeyes hot 10–80 ml/100 g/min (below 10 hidden), PNG per-subject 2nd–98th percentile (in the row label) | off | 1 |
| #3 perfusion z | `<results>/patients/<ID>/<ID>_vs_<band>_zscore.nii.gz`, \|z\| ≥ 1.96 | off | 2 |
| #4 zAI | `<results>/asymmetry/patients/<ID>/<ID>_asymmetry_zscore.nii.gz`, \|zAI\| ≥ 1.96 | off | 3 |
| Clinical zAI | `<results>/clinical/<ID>/<ID>_clinical_zai_significant_clusters.nii.gz`, \|zAI\| ≥ 3, both members of each pair | **on** | 4 |

`<results>` is `results_zscore` in-house and `external_chalfont/results_zscore` for Chalfont. Missing layers are
skipped and listed. For example, Chalfont controls (`C…`) have no z-maps, so `--subject C001` shows CBF only.

## Reading the views

- **Colours:**
  - #3: red-yellow means CBF above the controls (hyper); blue means below (hypo).
  - #4 (zAI): red-yellow means the voxel is above its mirror voxel; blue means below.
- **Mirrored pairs:** zAI is antisymmetric, so every zAI finding appears twice: red on the higher side, blue on the
  mirror. The views keep both members visible. The crosshair of options 15, 21 and 24 marks the member the rule assigns
  to the EZ side: the blue member of a hypoperfused cluster, the red member of a hyperperfused one.
- **Higher side is not the predicted EZ side:** the rule decides the EZ side from which hemisphere deviates from the
  controls (`direction_class`, `ez_side_pred`). Option 18 prints it; option 17 colours every pair by its class.
- **Inside the brain only:** the #3 perfusion z map, its hypo/hyper cluster maps and the CBF are computed over the
  scanner field of view, which includes the scalp. The viewer shows them inside the brain only (the consensus
  parcellation), using masked copies cached in `/tmp/asym_view/`; the files on disk are unchanged. zAI and the
  clinical maps lie inside the brain already.
- **Hidden layers:** in FSLeyes, tick the eye icon next to a layer in the Overlay list (bottom left) to show it.
  Raise or lower a layer's display range to change its threshold.
- **Orientation:**
  - FSLeyes is radiological by default (R on the left) and labels the sides.
  - The PNGs are neurological (L on the left) and labelled.
  - Index 0 of the x axis is the right hemisphere; FSLeyes reads this from the affine.

## Command-line options

| Flag | Effect |
|---|---|
| `--dataset inhouse\|chalfont` | which cohort the menu (or `--all`) uses; default in-house |
| `--orig` | in-house menu: original-space T1 underlay instead of the template (not aligned with the maps; debugging only) |
| `--subject ID` | skip the menu: all layers of one patient (dataset taken from the ID) |
| `--png` | write a PNG instead of opening FSLeyes |
| `--open` | open the PNG (VS Code `code` if on PATH, else `xdg-open`) |
| `--print` | print the FSLeyes command only |
| `--at I J K` | open at this voxel instead of the peak |
| `--all` | PNG for every patient of `--dataset` |
| `--out DIR` | PNG folder (default `visual_analysis/subject_views`) |
| `--registration` | with `--subject`: PNG of the registration chain (option 10); with `--all`: for every patient of `--dataset` |
| `--misses` | misdetected cases: tables, figures and report for `--dataset` (both cohorts when not given); with `--subject`: that patient's summary and case panel |

## Without the viewer: files and tables

All symmetric-space maps share one grid (227 × 272 × 227 at 0.8 mm). Any of them can be loaded on
`symreg/templates/T1_sym.nii.gz` in FSLeyes, with `-cm red-yellow -nc blue-lightblue -un -dr <threshold> <max>`
for signed maps. A diverging map such as `brain_colours_diverging_bwr` paints every zero voxel white and hides the
anatomy.

| What | In-house | Chalfont |
|---|---|---|
| #4 zAI, continuous | `results_zscore/asymmetry/patients/<ID>/<ID>_asymmetry_zscore.nii.gz` | `external_chalfont/results_zscore/asymmetry/patients/<ID>/…` |
| clinical maps, cluster table, montage | `results_zscore/clinical/<ID>/<ID>_clinical_zai_*.nii.gz`, `…_cluster_report.csv`, `…_montage.png` | same under `external_chalfont/results_zscore/clinical/<ID>/` |
| #3 perfusion z, hypo / hyper clusters | `results_zscore/patients/<ID>/<ID>_vs_<band>_zscore.nii.gz`, `…_hypo_clusters.nii.gz`, `…_zscore_montage.png` | `…/patients/<ID>/<ID>_vs_CH_ALL_*` |
| registration QC | — | `external_chalfont/qc/<ID>.png`, `external_chalfont/qc/registration_qc.csv` |

| Question | Table |
|---|---|
| which side each method called, per patient (Chalfont, as-implemented rule) | `external_chalfont/results_zscore/lateralization/eval_production_rule_per_patient.csv`, `eval_per_patient.csv` |
| per-stratum accuracy, control false-positive calls | `external_chalfont/results_zscore/lateralization/eval_rule_tier_strata.csv`, `eval_control_false_positives.csv` |
| regional normative statistic and smoothed voxel mass (v2) | `ez_analysis_mdt/v2/<dataset>/{R_patients,V_patients,R_loo_controls,V_loo_controls,summary,strata}.csv` |
| #3 lateralization and lobe localization | `ez_analysis_mdt/perfz/<dataset>/{summary,localization,localization_per_patient,localization_lobe_fraction}.csv` |
| sensitivity / specificity, best operating points | `ez_analysis_mdt/audit/sens_spec/`, `ez_analysis_mdt/audit/best_operating_points/summary.csv` |
| structural prior (FSL FIRST volume AI) and its combination with perfusion | `ez_analysis_mdt/audit/first_chalfont/{volume_ai,results,results_flow}.csv` |
| index definition (ratio, log ratio, difference, z-difference) | `ez_analysis_mdt/audit/ai_definitions/{summary,per_subject_chalfont,per_subject_inhouse}.csv` |
| plain Flow instead of transit-corrected CBF (Chalfont) | `ez_analysis_mdt/audit/flow_variant/{summary,R_patients,V_patients}.csv` |
| in-house concordance (primary rule, top lobe) | `results_zscore/clinical/cohort_concordance.csv`, `ez_analysis_mdt/clinical_*.csv` |

Figures regenerated from these tables:
- `python scripts/make_manuscript_figures.py`: Figs 1–5 and the supplementary case montages.
- `python 04_publication_figures.py --sym`: exploratory figures 10–16 in `visual_analysis/`.

## Troubleshooting

| Message | Meaning and fix |
|---|---|
| `fsleyes is not on PATH` | FSL is not set up in this shell: `source $FSLDIR/etc/fslconf/fsl.sh`, or paste the printed command where FSLeyes exists. Use `--png` meanwhile. |
| `FSLeyes exited with code N` + log lines | FSLeyes refused the command, for example because there is no display; the full log is `/tmp/fsleyes_<ID>.log`. Over SSH without X11 forwarding, use `--png`. |
| `missing, layer skipped: <file>` | that output was never produced for this patient; the view opens without it. Re-run the pipeline step that writes it (`01_` for #3, `02_` for zAI, `03_` for clinical maps; add `--sym`, and `--dataset chalfont` for Chalfont). |
| `(in-house only: no native anatomy)` on 1–2 | Chalfont has no `Dataset/<ID>/`; use 9 or 24 for the patient's CBF on the template. |
| patient not in the list | the menu lists patients with a zAI map; Chalfont controls (`C…`) have none, so use `--subject C001`. |
| stale or odd overlay after re-running the pipeline | masked copies refresh when the source file is newer; to force it, delete `/tmp/asym_view/`. |
| `flirt (FSL) is not on PATH` (options 6–10) | steps 1–3 recompute intermediates with FSL: `source $FSLDIR/etc/fslconf/fsl.sh`. Cached results stay in `/tmp/asym_registration/<ID>/`; delete that folder to recompute. |

## Misdetected cases (options 26–27, `--misses`)

```bash
python interactive_viewer.py --misses                        # both cohorts: tables, figures, case panels, REPORT.md
python interactive_viewer.py --misses --dataset inhouse      # one cohort (~10 s; Chalfont ~35 s)
python interactive_viewer.py --misses --subject P027 --open  # one patient: summary in the terminal + case panel
```

For every patient: the reference side, the file it came from and how it was read; the side each clinical source
pointed to; the call of every method; and, for each miss, why.

- **Miss:** a method calls the other side (wrong) or no side (no call). Bilateral / unclear references are not scored
  and are shown below the line; they get their own table, figure and report section (`bilateral_unclear.csv` / `.png`,
  report section 9): every method's call, whether the blind methods agree, which sources point each way, and how large
  each statistic is against the control range (a bilateral abnormality cancels in a left-minus-right statistic).
- **Methods:** the blind ones (primary rule, regional LI, regional normative z and its calibrated call, the plain-Flow
  variant, voxel mass, #3, thesis rule, hippocampal volume and its calibrated call, the sequential rule) and, in-house,
  the thesis's own visual, ROI and voxel-wise readings. The thesis readings were reported beside the presumed EZ, so
  they are shown for comparison and not counted among the methods.
- **Control range:** for each method with leave-one-out controls, the 95th percentile of the controls' |statistic|
  (the calibrated 5 % FP cut). A wrong call inside it is no stronger than what healthy controls produce: no signal.
- **Primary-rule reasons:** the in-scope temporal + hippocampal + amygdalar volume assigned to each side by direction
  class, the global CBF offset, the largest abnormal lobe, the EZ lobe, label strength, whether a clinical source or
  most other methods point the other way.
- **Label sources:**
  - in-house: `ez_ground_truth.py` (thesis Table 11) plus `Dataset/reference_sources.csv`, which codes by hand the side
    of each source (thesis Tables 11 and 14, manuscript Table 1) and the thesis readings (Tables 12-14). That file is
    private: it holds invasive and follow-up findings not yet published.
  - Chalfont: `external_chalfont/metadata/subjects.csv` (method and tier) and the sheet(s) behind each label, parsed from
    `build_subjects.py`; source sides come from the anonymised MRI-negative-study sheet (semiology is not coded).

| Output | What it shows |
|---|---|
| `matrix.png` | every method's call per patient (green right, red wrong, grey no call), hardest patients first |
| `control_range.png` | each statistic / control 95th percentile, signed toward the reference side; grey band = controls |
| `rule_vote.png` | what the primary rule voted on, by direction class |
| `offset.png` | hyper share of cluster pairs against the global CBF offset |
| `sources.png` | each clinical source, the reference, each method (and the thesis readings in-house) |
| `strata.png` | share right by label strength, label reading, lesion and EZ lobe, with Wilson intervals |
| `agreement.png` | how often two methods call the same side; how many methods get each patient right |
| `bilateral_unclear.png` | the patients without a reference side: sources, reference and every method's call; each statistic against the control range beside the unilateral patients |
| `cases/<ID>.png` | one patient: clusters by class through the deciding cluster, the vote, sources, methods, reasons |
| `REPORT.md` | all of the above with the tables, one line per primary-rule miss and one paragraph per bilateral / unclear patient |
| `ez_analysis_mdt/misdetected/<dataset>/*.csv` | `cases`, `method_summary`, `rule_miss_reasons`, `strata`, `method_agreement`, `label_sources`, `bilateral_unclear` |

## Reorder 2026-11-09: the menu follows the Methods

The menu used to run anatomy → #3 → #4 → utilities → all layers → registration. Registration came last although
every symmetric-space map depends on it, #3 came before the zAI it serves, and analyses #1 and #2 had no view. Now
the order is that of the Methods (table above), the numbers come from `MENU`, and four views were added or merged:

| Old | New | Change |
|---|---|---|
| 1, 2 | 1, 2 | — |
| — | 3 | new: analysis #1, regional LI of the 37 pairs + the `05_` aggregation scores |
| 19–25 | 4–10 | moved up: registration precedes every symmetric-space map |
| — | 11 | new: analysis #2, the voxel AI of the patient alone (`02_` computes it but does not save it) |
| — | 12 | new: the control normative (mean and SD of AI) behind zAI |
| 6 | 13 | + calibration and the one-voxel arithmetic; opens on the clinical cluster |
| 8 (+ 7) | 14 | the sign masks of old 7 are hidden layers here |
| 9 | 15 | — |
| 3 | 16 | moved to the direction step it feeds |
| — | 17 | new: clusters coloured by direction class, one member per pair |
| 15, 10, 11, 12 | 18, 19, 20, 21 | — |
| — | 22 | new: regional normative z_LI (post hoc enhanced estimator i) |
| 4 (+ 5) | 23 | region outlines on (old 5 was old 4 with outlines); prints the post hoc #3 rule |
| 17, 18 | 24, 25 | — |
| — | 26, 27 | new (later the same day): misdetected cases, below |
| 16, 13, 14 | 28, 29, 30 | island cleaning is a display aid, not a Methods step |

## Audit 2026-11-09: what changed and why

Option numbers in this section are those of the menu at the time of the audit (before the reorder above).

Every option was run for P016, P016 with `--orig`, D002 and N015. A logging stand-in for `fsleyes` captured each
command, every file it named was checked, and each view was rendered offscreen with the real FSLeyes. Findings and
fixes:

- **Thresholds on screen did not match the labels.**
  - Options 8, 9, 11 and 12 said "|zAI| ≥ 4", but `-dr -4 4` only sets the colour range, so every non-zero voxel
    was drawn.
  - Option 8 said "TFCE", but its file holds the `02_` clusters at 1.96.
  - Option 9's file starts at 3.
  - Now the lower display bound is the threshold, and each label states the real one.
- **Whole-brain colour washes.** Options 3, 6, 10 and 11 coloured nearly every voxel, because they drew the continuous
  map from zero. They now start at the stated threshold.
- **Focal views opened anywhere.** Small clusters (options 9 and 16) were usually outside the default slices. Options
  8, 10, 11 and 16 now open on their peak voxel. Options 9, 12, 17 and 18 open on the strongest clinical-tier
  cluster, on its predicted side. The peak of the clinical map alone would often be the mirror member: for D002 it is
  in the right hemisphere, while that cluster is left precentral.
- **Inconsistent background.** Most views used the blurry group mean CBF. All symmetric-space views now use the
  template T1, with the mean CBF as a hidden layer. Parcellations draw as thin FreeSurfer-coloured outlines instead of
  an opaque random colour wash.
- **Silent failures.** FSLeyes errors went to `/dev/null`, so the menu said "launched" when nothing opened. One
  launcher (`_launch`) now builds every command, reports skipped layers, prints a copy-paste command and shows
  start-up errors.
- **Chalfont:**
  - Option 15 looked for in-house files ("no cluster report"); it now applies the dataset switch.
  - Option 16 could not find the control band; `clean_overlay.set_dataset()` now points it at the Chalfont results
    and the fixed atlas.
  - Option 14 printed `results_zscore/` paths; it now prints the exact commands of the views.
  - The header said "(None confidence)"; it now shows side, group, lobe and pathology.
  - Options 1–2 are marked in-house only.
- **In-house option 16** built its 37-region mask from the MNI-space `Dataset/<ID>/aparc+aseg`, while the zAI maps are
  in symmetric space. It now uses `symreg/sym_perf/<ID>_aparc_sym`. With the correct mask nothing survives cleaning
  for P016; the old 108-voxel `P016_zai_cleaned.nii.gz` came from the misplaced mask and has been rewritten.
- **Clinical layer showed one member of each pair.** Options 9, 12 and 17 drew only the higher-perfused member, so for
  a hypoperfused cluster the crosshair (on the rule's side) landed on an uncoloured spot. They now draw both members
  (`<ID>_clinical_zai_significant_clusters`); the higher-member-only map is a hidden layer.
- **Options 3–5** looked for clinical z files that `03_` no longer writes, and printed a misleading "run 03_" note.
  They now show the #3 files that exist.
- **Region outlines spilled outside the brain.** The outlines were each patient's own warped parcellation
  (`symreg/sym_perf/<ID>_aparc_sym`). That is individual anatomy, which a 10-mm warp cannot fit to the average,
  smooth-edged template: up to 7.8 % of its cortex labels (P013) fall outside the template brain. All symmetric-space
  views now draw the consensus parcellation (majority vote of the controls on the template; 0 % outside), which is
  also the gray matter and 37-pair scope the analysis used. Options 11 and 16 build their masks from it too. The maps
  themselves were never outside: 0.0 % of every zAI map, |zAI| ≥ 1.96 voxel and clinical cluster lies outside the
  template brain, in all 116 patients. The template is displayed at 0–0.95, which makes the cortical ribbon at the
  edge visible, and CBF below 10 ml/100 g/min is hidden.
- **Salt-and-pepper islands around the skull.** #3 (`01_`) and CBF cover the scanner field of view: P016's #3 map has
  4.3 M non-zero voxels against a 3.5 M-voxel template brain. Outside the brain these are noise, and they showed as red
  and blue specks around the skull. Clipping by the brain in FSLeyes (`-cl`) also disables the transparent threshold.
  So the viewer now writes brain-masked copies (`brain_masked()`, cached in `/tmp/asym_view/`, keyed by source and
  mask) and the PNGs mask in memory. Options 3, 4, 5, 12, 17, 18 and 24 are affected. The maps on disk are unchanged.
- **Working directory:** the menu only worked when started from the project root. It now changes there itself.
- **Patient prompt:** it now accepts IDs as well as list numbers.

Tests: sections f–i of `tests/test_regressions.py` (all-layer view, launcher commands, thresholds, reference line, registration views, menu order, voxel AI, regional painting, direction classes, regional normative z).

## Limitations

- FSLeyes needs a desktop session (X11 or Wayland). `--png` and options 10 and 25 need no display.
- Colormap names (`red-yellow`, `blue-lightblue`, `hot`, `freesurfercolorlut`) are FSLeyes-specific.
- Options 1–2 and `--orig` exist for the in-house cohort only.
- Chalfont region outlines come from one fixed template atlas, not individual anatomy.

## Cross-references

- Outputs shown: `results_zscore/{patients,asymmetry/patients,clinical}/` and the same under `external_chalfont/results_zscore/`.
- Pipeline steps: `05_roi_discrimination.py` and `scripts/build_comprehensive_asymmetry.py` (#1, option 3), `01_build_control_normative.py` (#3), `02_compute_zai.py` (#2 and #4: `compute_asymmetry_map`, the control normative, clusters, direction classes), `symreg/check_full_calibration.py` (calibration, option 13), `03_clinical_maps.py` (clinical maps), `06_clinical_interpretation.py` (report), `scripts/v2/analyze_v2.py` (option 22), `scripts/audit/perfusion_z_lateralization.py` (#3 rule, option 23), `clean_overlay.py` (option 28).
- Dataset switch: `dataset_config.py`. Install notes for FSLeyes: [`install.md`](install.md).
