# Control Zscore Analysis

**Status:** canonical
**Last touched:** 2026-03-23
**Owner script(s):** control_zscore_analysis.py
**Result directory:** results_zscore/
**Promoted to pipeline?:** no

## Purpose

Voxel-Wise Z-Score Control Comparison Pipeline

## Inputs

(to be filled during pruning conversation)

## Outputs

(to be filled during pruning conversation)

## Related variants

Family: `zscore_pipeline`

- [control_zscore_asymmetry_clustering](./control_zscore_asymmetry_clustering.md)
- [control_asymmetry_zscore](./control_asymmetry_zscore.md)
- [generate_clinical_zscore_maps](./generate_clinical_zscore_maps.md)
- [compare_thresholding_methods](./compare_thresholding_methods.md)
- [visualize_zscore_results](./visualize_zscore_results.md)

## Verdict

**canonical** (2026-05-04, zscore_pipeline pruning)

Provides the control-group infrastructure: demographic matching, file scanning, Welford aggregation, group caching. Even though the literature-standard pipeline computes zAI (asymmetry-then-z-score, see `control_asymmetry_zscore.md`) rather than raw voxel z-scores, this script's group-building logic is reused/paralleled by #3, and the raw z-score map is useful as a supplementary/debugging view.

**Proposed rename (post-pruning cleanup):** `01_build_control_normative.py` (or extract group-building helpers into `src/controls.py` for shared use).

**Follow-up:** factor the control-group-building infrastructure into `src/controls.py` so `control_asymmetry_zscore.py` (#3) can import it.
