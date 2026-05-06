# Step 4: Publication Figures

> Part of the canonical voxel-based zAI pipeline (Pipeline B). Step 4 of 4. Generates publication-quality figures from the zAI pipeline outputs.

## Purpose

Render the seven publication figures (10–16) that document the Pipeline B (voxel-wise zAI) results. Output PNG files are placed alongside the older Pipeline A figures (00–09) in `visual_analysis/`, sharing a consistent matplotlib `seaborn-v0_8-darkgrid` style at 300 dpi for journal print.

## Script

`04_publication_figures.py` (canonical, at repo root)

## Inputs

- `results_zscore/asymmetry/patients/<pid>/<pid>_asymmetry_zscore.nii.gz` (from step 2)
- `results_zscore/asymmetry/patients/<pid>/<pid>_asymmetry_cluster_report.csv` (from step 2)
- `results_zscore/asymmetry/groups/<group>/brain_mask.nii.gz` (from step 1)
- `results_zscore/groups/<group>/mean_perfusion.nii.gz` (anatomical background)
- `results_zscore/groups/<group>/consensus_parcellation.nii.gz`

## Outputs

In `visual_analysis/`:

| File | What it shows |
|------|--------------|
| `10_zai_control_overview.png` | Cohort overview — significant-voxel counts per patient |
| `11_zai_regional_heatmap.png` | Region × patient heatmap of zAI region means |
| `12_zai_subcortical_focus.png` | Subcortical structures (thalamus, hippocampus, amygdala, basal ganglia) |
| `13_zai_top_clusters.png` | Top clusters per patient (axial slice + cluster table) |
| `14_zai_multipatient_axial.png` | Multi-patient axial montages |
| `15_zai_regional_consistency.png` | Consistency of regional zAI signal across patients |
| `16_zai_patient_clustering.png` | Patient-similarity clustering on zAI region profiles |

Identical-numbered `*_zscore_*.png` files (e.g., `10_zscore_control_overview.png`) exist from earlier raw-z-score iterations and are retained for archival comparison.

## Usage

```bash
python 04_publication_figures.py
```

No arguments. The script discovers patients with available zAI maps under `results_zscore/asymmetry/patients/`.

Typical runtime: ~3 minutes for the full cohort.

## What this step produces (relative to the manuscript)

- Figures 10–16 in the published manuscript
- The zAI panels referenced from §3 Pipeline B results

## Dependencies

- Python: `numpy`, `pandas`, `nibabel`, `matplotlib`, `seaborn`
- Data: outputs of steps 1–2
- Other scripts: `01_build_control_normative.py`, `02_compute_zai.py` must have been run

## Style notes

- **Matplotlib backend:** `Agg` (non-interactive) — figures save without a display
- **Style:** `seaborn-v0_8-darkgrid`
- **DPI:** 300 (print quality)
- **Color maps:** `RdBu_r` for diverging zAI; `viridis` / categorical seaborn palettes for region/patient identity

## Cross-references

- Manuscript section: §3 Pipeline B results; figure captions for figs. 10–16
- Related canonical scripts: `02_compute_zai.py`, `03_clinical_maps.py`
- Display tool: `interactive_viewer.py` for FSLeyes-based interactive 3D viewing

---

## From VISUAL_ANALYSIS_GUIDE.md (consolidated 2026-05-05)

Historical guide for the Pipeline A-era set of 8 figures (`01_..._png` through `08_..._png`) produced by the now-canonical `04_publication_figures.py` (formerly `visual_comparative_analysis.py`). Retained for figure-by-figure interpretation.

### What each Pipeline A figure contains

#### 1. Patient Overview Dashboard — `01_patient_overview.png`

Four panels:
- Top-left: significant asymmetries per patient (bar chart). Red bars indicate >5 regions; horizontal line is the cohort mean.
- Top-right: mean laterality bias per patient. Red = left-dominant, blue = right-dominant.
- Bottom-left: hemispheric dominance distribution — left>right vs right>left regions per patient.
- Bottom-right: variability vs maximum asymmetry, bubble size = number of significant asymmetries.

P013, P020, P024 show the highest asymmetry; P021 shows near-perfect symmetry.

#### 2. Regional Heatmap — `02_regional_heatmap.png`

Full 13 × 37 patient × region matrix, colour-scaled −0.4 to +0.4 LI on `RdBu_r`. Vertical patterns = regions consistently asymmetric across patients; horizontal patterns = an individual patient's profile. Entorhinal cortex is frequently right-asymmetric; frontal pole consistently right-asymmetric.

#### 3. Clustering Analysis — `03_clustering_analysis.png`

Left panel: hierarchical dendrogram (Ward linkage) on patient asymmetry vectors. Right panel: clustered heatmap with patients reordered by similarity. P021/P016/P018 cluster together (low-asymmetry); P013 diverges from the main cluster.

#### 4. PCA Analysis — `04_pca_analysis.png`

Left: scree plot with cumulative-variance line and 80%-threshold marker. Right: PC1 vs PC2 scatter, bubble size = significant-asymmetry count, colour intensity also = asymmetry count. First two PCs explain ≈40% of variance.

#### 5. Regional Consistency — `05_regional_consistency.png`

Four panels: top-15 regions by mean asymmetry; most-consistently-asymmetric regions (count of patients with |LI| > 0.1); mean-vs-variability scatter; histogram of regional means. Most regions show small mean asymmetry (<0.05).

#### 6. Subcortical vs Cortical — `06_subcortical_vs_cortical.png`

Mean LI, variability, and significant-count comparisons subcortical vs cortical, plus a per-patient subcortical-vs-cortical scatter against an identity line. Cortical regions show more asymmetry than subcortical. P013 has high cortical but low subcortical asymmetry.

#### 7. Correlation Matrix — `07_correlation_matrix.png`

13 × 13 patient similarity matrix (correlation of region-level LI vectors), with values overlaid. Most pairs show moderate positive correlation (0.3–0.6); P021 correlates weakly with others.

#### 8. Distribution Analysis — `08_distribution_analysis.png`

Four panels: pooled histogram with mean and ±0.1 threshold lines; per-patient box plot; per-patient violin plot; per-patient CDF (first five patients shown).

### Headline Pipeline A statistics (n = 13)

- Mean LI: −0.0064 (slight right-hemisphere bias).
- Standard deviation: 0.0709.
- Median: −0.0102.
- Range: −0.398 to 0.421.
- Significant asymmetries: 10.8% of all regions (left-biased 4.4%, right-biased 6.5%).
- Most consistently asymmetric regions: entorhinal cortex (8/12 patients), frontal pole (7/13, right-dominant), temporal pole (5/13, left-dominant), inferior temporal (5/13), pars orbitalis (4/13, left-dominant).
- Most/least asymmetric patients: P013 (9 sig. regions), P020 (8), P024 (8) vs P021 (0), P016 (1), P018 (1).

### Figure-selection guidance for manuscripts

- Methodology papers: figs. 02 (heatmap), 05 (regional consistency).
- Clinical papers: fig. 01 (overview), fig. 06 (subcortical vs cortical).
- Neuroscience papers: fig. 03 (clustering), fig. 04 (PCA).

All figures are 300 dpi PNG with colourblind-friendly palettes; for vector output change the `plt.savefig(..., format='svg')` line in `04_publication_figures.py`.

### Customization knobs in `04_publication_figures.py`

- Significance threshold (currently |LI| > 0.1) — search for `(df['laterality_index'].abs() > 0.1)`.
- Heatmap colormap — `RdBu_r` (alternatives: `viridis`, `plasma`, `coolwarm`).
- Default figure size — `plt.rcParams['figure.figsize']`.
- DPI — `plt.savefig(..., dpi=300)` (use 600 for journal print).

### Troubleshooting

- Figure not opening — try `eog`, `display`, or `feh` instead of `xdg-open`.
- Missing dependencies — `conda install matplotlib seaborn scikit-learn scipy pandas` inside the `Asym` environment.
- Low resolution — bump `dpi` in the relevant `plt.savefig` line.

---

## From VISUALIZATION_GUIDE.md (consolidated 2026-05-05)

Historical interactive-visualization reference covering the within-patient (Pipeline A and pre-zAI voxel) overlay options. The viewer has since been renamed to `interactive_viewer.py` and re-numbered (1–20); see [`08-interactive-viewer.md`](08-interactive-viewer.md) for the current option list. The following content is preserved for interpretation of legacy outputs and the thresholded-gradient colour conventions.

### Per-patient files generated by the legacy viewer / pipelines

Each patient folder in `results/` contains:

- `{patient}_asymmetry_analysis.csv` — all 11 asymmetry indices for 37 regions.
- ROI-based NIfTI maps (4 files): `{patient}_laterality_index_map.nii.gz`, `{patient}_left_hemisphere_asymmetry.nii.gz`, `{patient}_right_hemisphere_asymmetry.nii.gz`, `{patient}_significant_asymmetry_mask.nii.gz`.

Each `results_voxel/{patient}/` folder contains:

- `{patient}_voxel_ai_map.nii.gz` — full AI map (−200 to +200).
- `{patient}_voxel_zscore_map.nii.gz` — z-scored asymmetry.
- `{patient}_voxel_significance_map.nii.gz` — binary significance (|z| ≥ 1.64).
- `{patient}_voxel_brain_mask.nii.gz` — valid-voxel mask.
- `{patient}_voxel_left_hemisphere_asymmetry.nii.gz` / `..._right_hemisphere_asymmetry.nii.gz` — raw all-voxel hemisphere maps.
- `{patient}_voxel_left_hemi_thresholded.nii.gz` / `..._right_hemi_thresholded.nii.gz` — clean thresholded gradient maps (see below).
- `{patient}_cluster_left_hyperperfusion.nii.gz`, `{patient}_cluster_right_hyperperfusion.nii.gz`, `{patient}_cluster_perfusion_laterality.nii.gz` — binary cluster masks (combined map encodes +1 = L hyper, −1 = R hyper).

### Thresholded gradient maps (3-layer pipeline)

The original raw hemisphere maps showed a noisy "salt-and-pepper" pattern because the AI formula `AI = 100·(L−R)/((L+R)/2)` saturates at ±200 wherever one hemisphere is near zero (≈25% of brain voxels — boundaries, ventricles, midline). With FSLeyes display range `1–100` essentially the entire brain lit up.

The fix is a 3-layer pipeline implemented in `show_voxel_results.py`:

| Layer | Effect | Variable (default) |
|---|---|---|
| 1. Bilateral signal-quality mask | Require both hemispheres to have CBF > 10% of brain median; removes ~25–35% boundary artefacts. | `MIN_PERFUSION_FRACTION = 0.10` |
| 2. Percentile threshold on \|AI\| | Keep only the top (100−P)% most asymmetric quality-masked voxels. AI cutoff ≈125–145 per patient at default. | `PERCENTILE_THRESHOLD = 90` |
| 3. Cluster-size filter | Drop connected components smaller than N contiguous voxels. | `DISPLAY_MIN_CLUSTER = 10` (`CLUSTER_CONNECTIVITY = 2` → 18-connected) |

Result: ~1,000–1,700 voxels per hemisphere (~2–3% of quality voxels), spatially coherent hot-spots with original AI magnitude preserved for gradient display.

Regenerate for all patients:

```bash
python -c "from show_voxel_results import run_all_thresholded_generation; run_all_thresholded_generation()"
```

The interactive viewer also exposes a percentile slider — when prompted, type a percentile (85, 90, 95, 99) to regenerate on the fly for the current patient.

### Cluster-thresholded hyper/hypoperfusion (separate from gradient maps)

`show_voxel_results.py` Section 4 generates **binary** cluster masks via:

- `MIN_CLUSTER_SIZE = 50` (contiguous voxels)
- `ZSCORE_THRESHOLD = 1.64` (p < 0.05 one-tailed)
- `CLUSTER_CONNECTIVITY = 2` (18-connected)

6 of 15 patients had significant clusters (P015, P016, P018, P021, P022, P026). All patients show perfectly symmetric L/R cluster counts — this is inherent to the midline-flip method, not a bug.

### Manual FSLeyes commands (legacy, P013 example)

```bash
# Basic ROI laterality view
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
        results/P013/P013_laterality_index_map.nii.gz \
        -cm red-yellow -dr -0.4 0.4 -a 70 &

# ROI gradient (Red = L>R, Blue = R>L)
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
        results/P013/P013_left_hemisphere_asymmetry.nii.gz \
        -cm red -dr 0.001 0.4 -a 75 \
        results/P013/P013_right_hemisphere_asymmetry.nii.gz \
        -cm blue -dr 0.001 0.4 -a 75 &

# Voxel-based thresholded gradient (MNI space)
fsleyes "Dataset MNI/P013/T1w_restore.nii.gz" \
        "results_voxel/P013/P013_voxel_left_hemi_thresholded.nii.gz" \
        -cm red -dr 144 200 -a 80 -n "Left Hyperperfusion L>R" \
        "results_voxel/P013/P013_voxel_right_hemi_thresholded.nii.gz" \
        -cm blue -dr 144 200 -a 80 -n "Right Hyperperfusion R>L" &

# Cluster-filtered binary significance
fsleyes "Dataset MNI/P022/T1w_restore.nii.gz" \
        "results_voxel/P022/P022_cluster_left_hyperperfusion.nii.gz" \
        -cm red -dr 0.5 1 -a 80 -n "L > R Clusters" \
        "results_voxel/P022/P022_cluster_right_hyperperfusion.nii.gz" \
        -cm blue -dr 0.5 1 -a 80 -n "R > L Clusters" &
```

### Color-scheme conventions

- Full LI map (`red-yellow` colormap): yellow = strong left dominance, dark = balanced, purple/blue = right dominance.
- Voxel gradient asymmetry (`red`/`blue`): red = L > R hyperperfusion, blue = R > L hyperperfusion, intensity ∝ |AI|.
- Quality regions are filtered (`min_volume = 100 mm³`, `min_mean_perfusion = 10`); some patients lose 1–2 regions to QC (e.g., P024 has 36 regions instead of 37).

### Troubleshooting (legacy viewer)

- Salt-and-pepper voxel map → you are viewing the raw hemisphere maps; regenerate thresholded versions (see above).
- Empty thresholded maps → default `PERCENTILE_THRESHOLD = 90` is too strict; try `85` or set `DISPLAY_MIN_CLUSTER = 0`.
- FSLeyes doesn't open → check FSLeyes is on `PATH`; for remote sessions confirm X11 forwarding.
