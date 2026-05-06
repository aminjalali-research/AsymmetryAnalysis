# Visualize Zscore Results

**Status:** canonical
**Last touched:** 2026-03-24
**Owner script(s):** visualize_zscore_results.py
**Result directory:** visual_analysis/
**Promoted to pipeline?:** no

## Purpose

Visualize Z-Score Control Comparison Results

## Inputs

(to be filled during pruning conversation)

## Outputs

(to be filled during pruning conversation)

## Related variants

Family: `zscore_pipeline`

- [control_zscore_analysis](./control_zscore_analysis.md)
- [control_zscore_asymmetry_clustering](./control_zscore_asymmetry_clustering.md)
- [control_asymmetry_zscore](./control_asymmetry_zscore.md)
- [generate_clinical_zscore_maps](./generate_clinical_zscore_maps.md)
- [compare_thresholding_methods](./compare_thresholding_methods.md)

## Verdict

**canonical** (2026-05-04, zscore_pipeline pruning) — **with adaptation needed**

Generates publication-quality figures from `results_zscore/` → `visual_analysis/`. The figure types (regional heatmaps, cluster overlays, patient overviews, etc.) are needed regardless of which lateralization pipeline produces the input maps.

**Adaptation required:** currently visualizes the raw z-score / Approach A asymmetry outputs. Post-pivot, must consume the zAI maps and clinical-grade maps from `control_asymmetry_zscore.py` (#3) and `generate_clinical_zscore_maps.py` (#4). The plotting code is reusable; data-loading paths and figure captions need updating.

**Proposed rename (post-pruning cleanup):** `04_publication_figures.py`
