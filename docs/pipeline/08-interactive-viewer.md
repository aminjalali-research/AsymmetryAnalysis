# Interactive viewer (`interactive_viewer.py`)

> One entry point for looking at every Pipeline-B map of any patient, in-house or Chalfont: a menu that opens
> FSLeyes with the right files, colours and thresholds, plus a command-line mode that opens one patient directly
> or writes PNGs for the IDE. It reads pipeline outputs and does not change them, with three exceptions:
> - option 15 rewrites the patient's `06_` report files (`<ID>_surgeon_report.txt/.json`), as `06_` itself does;
> - option 16 writes a cleaned map and montage;
> - options 10–11 write temporary masks to `/tmp`.

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

A good order for one patient:

1. **15 · Patient report.** This is the rule's answer in text: clinical-tier clusters (|zAI| ≥ 4, ≥ 100 voxels, 37
   region pairs), predicted side per cluster, lobe totals. Answer `y` at the end to jump to the top-3 cluster view.
2. **12 · Combined presurgical view.** It opens on the strongest clinical cluster. The zAI map (faded) and the
   clinical clusters are visible, with region outlines; #3, the control mean CBF and the side masks are hidden layers.
3. **3 · #3 perfusion z.** This shows the same patient against the controls, one-sided. For in-house patients the
   controls are HCP subjects from other sites, so a global CBF offset colours the whole brain one way. The deck slide
   "#3 vs #4" explains why this is not a better lateralizer.
4. **17 · All layers.** CBF, #3, #4 and the clinical clusters in one window, opened on the strongest clinical cluster.
   Use **18** for the same view as a PNG.

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

Options 3–18 work exactly as for in-house patients. Options 1–2 are marked `(in-house only: no native anatomy)`,
because Chalfont subjects have no `Dataset/<ID>/`. Option 15 prints the rule's report for this patient, which calls
**bilateral**: 5 clinical-tier clusters, the strongest a left precentral hypoperfusion cluster at zAI −4.33. Options
9, 12 and 17 open on that cluster and say so:

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

### Example 5 · Copy-paste commands for a patient (option 14)

Option 14 prints the FSLeyes commands that options 3, 6, 8, 9, 12 and 17 would run for the current patient, plus the
PNG command. Paste them on any machine that has FSLeyes and the project files.

### Example 6 · The registration chain of one patient (options 19–25)

Options 19–25 show how a patient got from the scanner to the symmetric template. They follow the same steps as the
deck slide "Three datasets, one registration target".

```bash
python interactive_viewer.py --dataset chalfont   # patient D002, then 19 ... 25
python interactive_viewer.py --subject D002 --registration --open   # options 20-24 in one PNG, no menu
```

| # | Step | What you see | What to check |
|---|---|---|---|
| 19 | template | `T_sym` with the voxels where the plain mean of 30 HCP T1s differed from its mirror by ≥ 10 % (hidden layer: the plain mean) | `T_sym` is identical to its mirror; the hot voxels are what averaging removed |
| 20 | 0 · inputs | the input T1 on its own grid, with the CBF where the scanner coordinates put it; hidden: the ASL reference and native labels | Chalfont: CBF and T1 are visibly offset before registration |
| 21 | 1 · ASL → T1 | CBF and the PD reference resampled into the T1 with the saved rigid matrix (`<ID>_asl2t1.mat`) | perfusion fills the cortex and spares the ventricles. In-house: a note only, because HCP-ASL did this step |
| 22 | 2 · T1 → T_sym, affine | the T1 after the affine on `T_sym`, with the template's brain and white-matter outlines | brain edge and ventricles near the outlines; folds still individual |
| 23 | 3 · T1 → T_sym, non-linear | the T1 after fnirt with the same outlines; hidden: the T1 mirrored left-right (mirror check); QC numbers in the terminal | cortex and ventricles on the outlines; the mirrored copy should overlap |
| 24 | 4 · CBF and labels on T_sym | the CBF after one resampling, with the consensus region outlines | the input of `01_`, `02_` and `03_` |
| 25 | the chain as a PNG | columns: input T1, step 1, step 2, step 3, step 4; QC in the title; `visual_analysis/subject_views/<dataset>/<ID>_registration.png` | everything above at a glance |

The registration scripts deleted the affine matrix and the warped T1. Steps 2–3 recompute them with the same FSL
commands (`flirt` and `applywarp`; FSL must be on `PATH`) and cache them in `/tmp/asym_registration/<ID>/`. The first
call per patient takes 25–45 s, later calls are instant. QC for D002: NCC with `T_sym` 0.79 and mirror Dice 0.956 on
the 0.8-mm grid; the 2-mm QC file gives 0.819 and 0.960.

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
main layer. Options 9, 12, 17 and 18 open instead on the strongest clinical-tier cluster: the first row of the
`06_clinical_interpretation` report, by |peak|, at its centroid on the side predicted for that cluster. The map peak
is the fallback, and the terminal says which one was used. The patient-level call (option 15) weighs all clusters
and can differ. For P016 the strongest cluster is left medial orbitofrontal (its side L), while the call is R
(moderate).

| # | View | Main layer and threshold on screen | Opens at | Chalfont |
|---|---|---|---|---|
| 1 | T1 + FreeSurfer parcellation, original space | `Dataset/<ID>/T1w_acpc_dc_restore` + `aparc+aseg` outlines | — | no (no native anatomy) |
| 2 | CBF on the T1, original space | `<ID>_perfusion_calib_resampled_to_T1w`, hot 0–80 | — | no |
| 3 | #3 perfusion z vs the control band | `<ID>_vs_<band>_zscore`, \|z\| ≥ 1.96 | — | yes |
| 4 | #3 hypo / hyper clusters | `_hypo_clusters` (blue), `_hyper_clusters` (red); z beyond ±1.96, ≥ 10 voxels | — | yes |
| 5 | option 4 + region outlines | same | — | yes |
| 6 | #4 zAI map | `<ID>_asymmetry_zscore`, \|zAI\| ≥ 1.96; every finding appears twice (pair) | — | yes |
| 7 | zAI sign masks | `_left_dominant` (red = voxel > mirror), `_right_dominant` (blue); not hemisphere-specific | — | yes |
| 8 | significant zAI clusters | `<ID>_asymmetry_significant`, \|zAI\| ≥ 1.96, ≥ 50 voxels, both pair members | peak | yes |
| 9 | clinical zAI | `<ID>_clinical_zai_significant_clusters`, gray matter, 37 pairs, \|zAI\| ≥ 3, ≥ 50 voxels, both members of each pair; hidden: the higher-member-only map and L>R / R>L masks | strongest clinical cluster | yes |
| 10 | top-3 clusters (table + overlay) | clusters with peak \|zAI\| ≥ 4 from the clinical cluster report, yellow mask | peak of the 3 | yes |
| 11 | 37-region view (table + overlay) | zAI inside the 37 region pairs, \|zAI\| ≥ 4; top-5 regions printed | peak | yes |
| 12 | combined presurgical view (recommended) | zAI ≥ 1.96 (faded) + clinical zAI ≥ 3 (both members) + outlines; hidden: #3, mean CBF, side masks | strongest clinical cluster | yes |
| 13 | file list | every file the views use, with sizes | — | yes |
| 14 | FSLeyes commands | the commands of 3, 6, 8, 9, 12 and 17 for copy and paste | — | yes |
| 15 | patient report | `06_clinical_interpretation`: clinical tier (peak \|zAI\| ≥ 4, ≥ 100 voxels), predicted side, lobe totals | — | yes |
| 16 | island-cleaned zAI | `clean_overlay.py`: \|zAI\| ≥ 3, ≥ 50 voxels, 37 pairs, opening + closing; writes `<ID>_zai_cleaned.nii.gz` and a montage; says so when nothing survives (P016, D002) | peak | yes |
| 17 | all layers in FSLeyes | template, atlas, CBF, #3, #4 and clinical zAI; only the template and clinical layers on | strongest clinical cluster | yes |
| 18 | all layers as a PNG | as 17; `visual_analysis/subject_views/<dataset>/<ID>.png` | strongest clinical cluster | yes |
| 19–25 | registration chain | template build, inputs, ASL → T1, affine, fnirt + mirror check + QC, CBF on `T_sym`, PNG (Example 6) | — | yes (21: Chalfont only) |

Three thresholds appear, and they mean different things:
- **1.96:** where the pipeline forms clusters (`02_`).
- **3:** the voxel threshold of the clinical maps (`03_`).
- **4:** the clinical tier, applied to a cluster's peak (`06_`, options 10 and 15).

The menu labels say which one each view uses.

### Layers of options 17 and 18

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
  mirror. The views keep both members visible. The crosshair of options 9, 12 and 17 marks the member the rule assigns
  to the EZ side: the blue member of a hypoperfused cluster, the red member of a hyperperfused one.
- **Higher side is not the predicted EZ side:** the rule decides the EZ side from which hemisphere deviates from the
  controls (`direction_class`, `ez_side_pred`). Option 15 prints it.
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
| `--registration` | with `--subject`: PNG of the registration chain (option 25); with `--all`: for every patient of `--dataset` |

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
| `(in-house only: no native anatomy)` on 1–2 | Chalfont has no `Dataset/<ID>/`; use 17 for the patient's CBF on the template. |
| patient not in the list | the menu lists patients with a zAI map; Chalfont controls (`C…`) have none, so use `--subject C001`. |
| stale or odd overlay after re-running the pipeline | masked copies refresh when the source file is newer; to force it, delete `/tmp/asym_view/`. |
| `flirt (FSL) is not on PATH` (options 21–25) | steps 1–3 recompute intermediates with FSL: `source $FSLDIR/etc/fslconf/fsl.sh`. Cached results stay in `/tmp/asym_registration/<ID>/`; delete that folder to recompute. |

## Audit 2026-11-09: what changed and why

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

Tests: sections f and g of `tests/test_regressions.py` (all-layer view, launcher commands, thresholds, reference line).

## Limitations

- FSLeyes needs a desktop session (X11 or Wayland). `--png` and option 18 need no display.
- Colormap names (`red-yellow`, `blue-lightblue`, `hot`, `freesurfercolorlut`) are FSLeyes-specific.
- Options 1–2 and `--orig` exist for the in-house cohort only.
- Chalfont region outlines come from one fixed template atlas, not individual anatomy.

## Cross-references

- Outputs shown: `results_zscore/{patients,asymmetry/patients,clinical}/` and the same under `external_chalfont/results_zscore/`.
- Pipeline steps: `01_build_control_normative.py` (#3), `02_compute_zai.py` (#4), `03_clinical_maps.py` (clinical maps), `06_clinical_interpretation.py` (report), `clean_overlay.py` (option 16).
- Dataset switch: `dataset_config.py`. Install notes for FSLeyes: [`install.md`](install.md).
