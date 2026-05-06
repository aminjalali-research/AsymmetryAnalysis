# Step 5: ROI-Based MDT Discrimination (Pipeline A)

> The canonical script for Pipeline A — ROI-based EZ-side classification. Independent of Pipeline B (voxel-wise zAI); uses ROI mean perfusion summarized by laterality scores.

## Purpose

For every patient, summarize ROI (Desikan–Killiany region) perfusion asymmetry into a small set of per-patient laterality scores, and evaluate how well each score classifies the EZ side against the MDT-reviewed ground truth (`ez_ground_truth.py`). The script reports ROC AUC and LOOCV (leave-one-out cross-validation) accuracy under four aggregation strategies and across 15 candidate asymmetry indices.

## Scripts

- `05_roi_discrimination.py` (canonical, at repo root) — entry point
- `ez_ground_truth.py` (canonical, at repo root) — MDT-reviewed labels (imported)

`ez_ground_truth.py` keeps its original name to preserve the `from ez_ground_truth import …` statement at the top of `05_roi_discrimination.py`.

## Inputs

- `verified_perfusion_all_patients.csv` (or upstream ROI perfusion table) — left-/right-hemisphere mean ASL perfusion per Desikan–Killiany region per patient
- `ez_ground_truth.py` — patient → EZ-side label dictionary (`'L'`, `'R'`, `'B'`, `'B-L'`, `'B-R'`, `'Unknown'`, `'Parietal'`)

## Outputs

In `ez_analysis_mdt/`:

- `analysis_report.txt` — top-line summary (best methods, AUC, accuracy)
- `method_comparison.png` — AUC bar chart across the 15 indices
- `method_performance_dominant.csv` — performance using bilateral patients with their dominant side
- `method_performance_exclude.csv` — performance excluding ambiguous bilateral cases

## Usage

```bash
python 05_roi_discrimination.py
```

No required arguments. Edit constants near the top of the script to change patient inclusion (unilateral-only vs. dominant-side bilateral) or aggregation strategy.

Typical runtime: <1 minute.

## What this step produces (relative to the manuscript)

- Pipeline A's headline AUC and LOOCV results in §3 Pipeline A
- The 16-indices reconciliation table referenced in `docs/superpowers/audits/2026-05-04-16-indices-reconciliation.md`
- The cross-strategy AUC deltas in `docs/superpowers/audits/2026-05-04-auc-deltas.md`

## Sign convention (important!)

Standard laterality index `LI = (L − R) / (L + R)`:

- Positive LI → left has higher perfusion
- Negative LI → right has higher perfusion

Because the EZ characteristically shows **interictal hypoperfusion**:

- Left EZ → left hypoperfused → R > L → **negative LI**
- Right EZ → right hypoperfused → L > R → **positive LI**

The `evaluate_methods` routine handles this by trying both `+score` and `−score` and reporting whichever yields the higher AUC, flagging `flipped=True` for the inverted direction.

## Aggregation strategies

The four aggregation strategies that turn 40 ROI laterality measurements into a single per-patient score:

1. `signed_weighted_li` — weighted average of signed LIs across regions
2. `anatomically_weighted` — region-specific weights based on temporal-lobe anatomy
3. `temporal_signed_li` / `temporal_focused_score` — restrict to temporal-lobe ROIs
4. `dominant_side_score` — pick the most extreme region per patient

## Dependencies

- Python: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`
- Data: ROI-level perfusion table for all 15 patients
- Other scripts: `ez_ground_truth.py` (imported, must remain at repo root with this exact name)

## Cross-references

- Manuscript section: §2.9 "Pipeline A: ROI-based MDT discrimination"; §3 Pipeline A results
- Audits:
  - `docs/superpowers/audits/2026-05-04-16-indices-reconciliation.md`
  - `docs/superpowers/audits/2026-05-04-auc-deltas.md`
  - `docs/superpowers/audits/2026-05-04-ez-discrimination-consolidation.md`
  - `docs/superpowers/audits/2026-05-04-cohen_d-and-auc-refresh.md`
  - `docs/superpowers/audits/2026-05-04-weighted_aai-formula.md`
- Related canonical scripts: independent of the voxel pipeline (steps 1–4)

---

## Quick reference

### From QUICK_LI_REFERENCE.md (consolidated 2026-05-05)

Cheat sheet for the legacy ROI-based LI maps in `laterality_maps/{patient}_*.nii.gz`. Most users now invoke these views via `interactive_viewer.py` options 11–17, but the raw FSLeyes commands below are still useful for ad-hoc inspection or remote-machine sessions.

#### Essential FSLeyes commands

```bash
# Basic LI overlay
cd /home/amin/AsymmetryAnalysis
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
        laterality_maps/P013_laterality_index_map.nii.gz \
        -cm red-yellow -dr -0.4 0.4 -a 70 &

# Significant asymmetries only (|LI| > 0.1 binary mask)
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
        laterality_maps/P013_significant_asymmetry_mask.nii.gz \
        -cm red -dr 0 1 -a 80 &

# Left-dominant vs right-dominant overlay
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
        laterality_maps/P013_left_dominant_regions.nii.gz -cm red -a 70 \
        laterality_maps/P013_right_dominant_regions.nii.gz -cm blue -a 70 &
```

#### Color interpretation

| Color | Meaning | Clinical significance |
|---|---|---|
| Red | Right > left | Rightward asymmetry |
| Yellow | Left > right | Leftward asymmetry |
| Dark | Balanced | No asymmetry |
| Brighter | Stronger asymmetry | Higher clinical priority |

#### Clinical thresholds (mirrors the manuscript's Table-2 threshold ladder)

- |LI| > 0.4 — strong (investigate artefacts).
- |LI| > 0.2 — moderate (clinically significant).
- |LI| > 0.1 — mild (statistical threshold).
- |LI| ≤ 0.1 — balanced / normal.

#### Files generated per patient (legacy laterality_maps/)

- `{patient}_laterality_index_map.nii.gz` — full LI map.
- `{patient}_significant_asymmetry_mask.nii.gz` — binary |LI| > 0.1 mask.
- `{patient}_left_dominant_regions.nii.gz` — LI > 0.1 only.
- `{patient}_right_dominant_regions.nii.gz` — LI < −0.1 only.
- `{patient}_LI_histogram.png` — LI distribution PNG.

#### Common quick fixes

- Map looks dark → adjust `-dr` (try `-0.2 0.2`).
- Wrong colours → switch colormap (`-cm red-yellow` or `-cm blue-red`).
- No overlay visible → check transparency (`-a 70`).
- Misalignment → confirm both files share the same image space (`fslinfo`).

### From VOXEL_ROI_QUICK_REFERENCE.md (consolidated 2026-05-05)

Cheat sheet for the voxel-ROI hybrid extraction (`voxel_roi_extraction.py`, exposed as Option 9 in the legacy interactive viewer). Combines voxel-level resolution with anatomical ROI grouping.

#### Setup — create MNI parcellation (REQUIRED)

The voxel-ROI extraction needs the FreeSurfer parcellation in MNI space at `Dataset MNI/{patient}/aparc+aseg_mni.nii.gz`. See [docs/data/layout.md → MNI transform notes](../data/layout.md) for the full ANTs / FSL transform recipe. One-liner with ANTs:

```bash
antsApplyTransforms -d 3 \
  -i Dataset/P022/aparc+aseg.nii.gz \
  -r "Dataset MNI/P022/T1w_restore.nii.gz" \
  -o "Dataset MNI/P022/aparc+aseg_mni.nii.gz" \
  -n NearestNeighbor \
  -t path/to/transform.mat
```

Use **NearestNeighbor** (ANTs `-n NearestNeighbor` or FSL `--interp=nn`) to preserve integer label values.

#### Run the extraction and the viewer

```bash
python voxel_roi_extraction.py
python interactive_viewer.py    # then select patient, choose option 9
```

Outputs:

- `results_voxel_roi/{patient}/{patient}_voxel_roi_statistics.csv` (per-patient).
- `results_voxel_roi/all_patients_voxel_roi_statistics.csv` (cohort).

#### CSV columns

```
patient_id, left_segid, right_segid, left_name, right_name,
left_nvoxels, right_nvoxels, total_nvoxels,
left_ai_mean, left_ai_median, left_ai_std,
right_ai_mean, right_ai_median, right_ai_std,
bilateral_ai_mean, bilateral_ai_median, bilateral_ai_std,
left_zscore_mean, right_zscore_mean, bilateral_zscore_mean,
left_sig_pct, right_sig_pct, bilateral_sig_pct
```

#### Per-region statistics produced

- Left region: mean / median / std AI, voxel count.
- Right region: mean / median / std AI, voxel count.
- Bilateral aggregate: mean AI across both hemispheres.
- z-score: standardised asymmetry.
- Significance: % of voxels with |z| ≥ 1.64.

The viewer display additionally shows the top 10 regions by asymmetry magnitude, statistical summary (mean / std / max), hemisphere-dominance distribution, and an FSLeyes overlay.

#### Comparison: ROI vs voxel vs voxel-ROI

| Feature | ROI-based | Voxel-based | **Voxel-ROI** |
|---|---|---|---|
| Spatial detail | One value per region | ~140 K voxels | Voxel stats per region |
| Interpretability | Easy (37 regions) | Hard (too many voxels) | **Easy (37 regions)** |
| Heterogeneity | Not captured | Captured but hard to summarize | **Captured + summarized** |
| Cross-patient comparability | Easy | Hard | **Easy** |

#### Voxel-based gradient visualization (Option 7 background)

The original hemisphere maps showed every voxel with any asymmetry, producing a noisy "salt-and-pepper" pattern because the AI formula `AI = 100·(L−R)/((L+R)/2)` saturates at ±200 wherever one hemisphere is near zero. The 3-layer fix (now implemented in `show_voxel_results.py`):

| Variable | Default | Purpose |
|---|---|---|
| `PERCENTILE_THRESHOLD` | 90 | Keep top 10% most asymmetric quality-masked voxels (AI cutoff ≈125–145) |
| `DISPLAY_MIN_CLUSTER` | 10 | Drop connected components < 10 voxels |
| `CLUSTER_CONNECTIVITY` | 2 | 18-connected neighbourhood |

Plus a bilateral signal-quality mask requiring both the original and mirrored hemisphere to have CBF > 10% of brain median (removes ~25–35% of boundary / ventricle / midline artefacts).

Result: ~1,000–1,700 voxels per hemisphere (~2–3% of quality voxels), spatially coherent hot-spots with the original AI magnitude preserved for gradient display. Outputs:

```
results_voxel/{patient}/
├── {patient}_voxel_left_hemi_thresholded.nii.gz   # RED (L > R)
├── {patient}_voxel_right_hemi_thresholded.nii.gz  # BLUE (R > L)
```

Regenerate with custom settings:

```bash
# Edit PERCENTILE_THRESHOLD / DISPLAY_MIN_CLUSTER at the top of show_voxel_results.py, then:
python -c "from show_voxel_results import run_all_thresholded_generation; run_all_thresholded_generation()"
```

Or use the percentile slider in the interactive viewer (option 7).

#### Cluster-thresholded hyper/hypoperfusion (Section 4 of show_voxel_results.py)

Configurable variables:

| Variable | Default | Purpose |
|---|---|---|
| `MIN_CLUSTER_SIZE` | 50 | Min contiguous voxels for a valid z-score cluster |
| `ZSCORE_THRESHOLD` | 1.64 | \|z\| ≥ 1.64 → p < 0.05 one-tailed |
| `CLUSTER_CONNECTIVITY` | 2 | 18-connected neighbourhood |

Pipeline: load the z-score map, split into directional maps (left hyper if z > +threshold, right hyper if z < −threshold), label connected components, drop sub-threshold clusters, report per-cluster statistics. 6 of 15 patients had significant clusters (P015, P016, P018, P021, P022, P026); the other 9 had no voxels exceeding the threshold. Output files:

```
results_voxel/{patient}/
├── {patient}_cluster_left_hyperperfusion.nii.gz    # binary L > R
├── {patient}_cluster_right_hyperperfusion.nii.gz   # binary R > L
├── {patient}_cluster_perfusion_laterality.nii.gz   # combined: +1 = L hyper, −1 = R hyper
```

Run selectively:

```bash
python -c "from show_voxel_results import run_all_cluster_analysis; run_all_cluster_analysis()"
```

#### Color legend (voxel-ROI viewer)

| Color | Meaning | Clinical interpretation |
|---|---|---|
| Red | Left > Right perfusion (L>R) | Left hyperperfusion |
| Blue | Right > Left perfusion (R>L) | Right hyperperfusion |
| Brighter | Larger asymmetry magnitude | Stronger lateralisation |
| No color | Below threshold or balanced | Normal / symmetric |

In epilepsy, hyperperfusion may indicate seizure focus / ictal-postictal change; hypoperfusion may indicate interictal EZ state, atrophy, or compensatory change. AI is relative — left hyperperfusion is mathematically equivalent to right hypoperfusion.

#### `show_voxel_results.py` section overview

| Section | Purpose |
|---|---|
| 1. Pure voxel analysis | Full-brain voxel asymmetry without ROI grouping (~140 K voxels/patient) |
| 2. Voxel-ROI analysis | Voxel statistics grouped by 37 anatomical regions |
| 3. Comparative analysis | Pure voxel vs voxel-ROI cross-method comparison |
| 4. Cluster-thresholded hyper/hypoperfusion | z + cluster-size filtering (binary masks) |
| 5. Percentile-thresholded gradient maps | Quality-masked + percentile + cluster-filtered intensity maps for FSLeyes |

#### Troubleshooting

- "MNI parcellation not found" → create `Dataset MNI/{patient}/aparc+aseg_mni.nii.gz`. See [docs/data/layout.md → MNI transform notes](../data/layout.md).
- "Labels are blurred" → use `-n NearestNeighbor` (ANTs) or `--interp=nn` (FSL).
- "No results displayed" → run `voxel_roi_extraction.py` first.
- "Thresholded maps are empty" → lower `PERCENTILE_THRESHOLD` to 85 or set `DISPLAY_MIN_CLUSTER = 0`.
- "Salt-and-pepper / noisy Option 7 view" → you are viewing the raw hemisphere maps; regenerate the thresholded ones.
