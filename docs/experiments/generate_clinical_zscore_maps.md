# Generate Clinical Zscore Maps

**Status:** canonical
**Last touched:** 2026-03-24
**Owner script(s):** generate_clinical_zscore_maps.py
**Result directory:** results_zscore/
**Promoted to pipeline?:** no

## Purpose

Generate Clinical-Grade Z-Score Maps for Presurgical Planning

## Inputs

(to be filled during pruning conversation)

## Outputs

(to be filled during pruning conversation)

## Related variants

Family: `zscore_pipeline`

- [control_zscore_analysis](./control_zscore_analysis.md)
- [control_zscore_asymmetry_clustering](./control_zscore_asymmetry_clustering.md)
- [control_asymmetry_zscore](./control_asymmetry_zscore.md)
- [compare_thresholding_methods](./compare_thresholding_methods.md)
- [visualize_zscore_results](./visualize_zscore_results.md)

## Verdict

**canonical** (2026-05-04, zscore_pipeline pruning) — **with adaptation needed**

Provides the gray-matter masking + strict clinical thresholding step. Without this, ~35% of brain voxels survive |z|≥1.96 (white matter, ventricles, unlabeled regions) — not clinically actionable.

**Adaptation required:** currently consumes the raw z-score maps from `control_zscore_analysis.py` (#1). Post-pivot, must consume the zAI maps from `control_asymmetry_zscore.py` (#3) instead. The masking + thresholding logic itself is reusable; only the input file path / variable names need updating.

**Proposed rename (post-pruning cleanup):** `03_clinical_maps.py`
