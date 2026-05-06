# Experiments — Inventory

One row per `.py` script at the repo root. **All 34 scripts have been assigned a final status** (canonical / archive / delete) as of 2026-05-04. See `.claude/DECISIONS.md` for the per-family rationale.

## Status legend

- `canonical` — keep, will be promoted to `docs/pipeline/` with full polish
- `archive` — keep but move to `archive/<name>.py` (and doc to `docs/experiments/archive/`)
- `delete` — remove from repo (git history preserves it)

## Inventory

| Family | Script | Last touched | Result dir | Status | Doc |
| --- | --- | --- | --- | --- | --- |
| (other) | `ai_distribution_analysis.py` | 2026-02-23 | — | **canonical** | [doc](./ai_distribution_analysis.md) |
| (other) | `extract_perfusion_from_nifti.py` | 2026-02-02 | nifti_extraction_results/ | **archive** | [doc](./extract_perfusion_from_nifti.md) |
| (other) | `interactive_overlay_viewer.py` | 2026-03-26 | — | **canonical** | [doc](./interactive_overlay_viewer.md) |
| (other) | `run_advanced_analysis.py` | 2025-08-25 | — | **delete** | [doc](./run_advanced_analysis.md) |
| (other) | `setup.py` | 2025-08-11 | — | **archive** | [doc](./setup.md) |
| (other) | `targeted_p013_analysis.py` | 2025-11-07 | P013_Analysis_Results/ | **archive** | [doc](./targeted_p013_analysis.md) |
| batch | `batch_analyze_all_patients.py` | 2026-02-02 | — | **delete** | [doc](./batch_analyze_all_patients.md) |
| batch | `batch_process_all_patients.py` | 2026-02-02 | — | **delete** | [doc](./batch_process_all_patients.md) |
| comparative | `comparative_patient_analysis.py` | 2025-12-03 | (see comparative_*.csv at root) | **archive** | [doc](./comparative_patient_analysis.md) |
| comparative | `visual_comparative_analysis.py` | 2025-12-09 | (see comparative_*.csv at root) | **archive** | [doc](./visual_comparative_analysis.md) |
| ez_discrimination | `advanced_ez_discrimination.py` | 2026-02-02 | advanced_ez_analysis/ | **delete** | [doc](./advanced_ez_discrimination.md) |
| ez_discrimination | `directional_ez_discrimination.py` | 2026-02-02 | directional_ez_analysis/ | **delete** | [doc](./directional_ez_discrimination.md) |
| ez_discrimination | `epileptogenic_zone_discrimination.py` | 2026-01-27 | epileptogenic_zone_analysis/ | **delete** | [doc](./epileptogenic_zone_discrimination.md) |
| ez_discrimination | `ez_discrimination_mdt.py` | 2026-02-02 | ez_analysis_mdt/ | **canonical** | [doc](./ez_discrimination_mdt.md) |
| ez_discrimination | `integrated_ez_discrimination.py` | 2026-02-02 | integrated_ez_analysis/ | **delete** | [doc](./integrated_ez_discrimination.md) |
| ez_support | `advanced_directional_analysis.py` | 2026-02-02 | advanced_directional_analysis/ | **delete** | [doc](./advanced_directional_analysis.md) |
| ez_support | `ez_ground_truth.py` | 2026-02-02 | ez_analysis_mdt/ | **canonical** | [doc](./ez_ground_truth.md) |
| laterality_viz | `create_laterality_map.py` | 2025-10-01 | laterality_maps/ | **delete** | [doc](./create_laterality_map.md) |
| laterality_viz | `create_onesided_maps.py` | 2025-12-02 | laterality_maps/ | **delete** | [doc](./create_onesided_maps.md) |
| laterality_viz | `demo_li_visualization.py` | 2025-08-18 | laterality_maps/ | **delete** | [doc](./demo_li_visualization.md) |
| laterality_viz | `visualize_laterality_maps.py` | 2025-08-18 | laterality_maps/ | **delete** | [doc](./visualize_laterality_maps.md) |
| verify_utilities | `check_li_values.py` | 2025-12-02 | — | **delete** | [doc](./check_li_values.md) |
| verify_utilities | `verify_perfusion_sources.py` | 2026-02-02 | — | **delete** | [doc](./verify_perfusion_sources.md) |
| view_utilities | `quick_overlay_demo.py` | 2025-10-01 | — | **delete** | [doc](./quick_overlay_demo.md) |
| view_utilities | `show_voxel_results.py` | 2026-02-23 | visual_analysis/ | **delete** | [doc](./show_voxel_results.md) |
| view_utilities | `view_visualizations.py` | 2025-12-03 | visual_analysis/ | **delete** | [doc](./view_visualizations.md) |
| voxel | `voxel_based.py` | 2026-01-08 | results_voxel/ | **delete** | [doc](./voxel_based.md) |
| voxel | `voxel_roi_extraction.py` | 2025-12-08 | results_voxel_roi/ | **delete** | [doc](./voxel_roi_extraction.md) |
| zscore_pipeline | `compare_thresholding_methods.py` | 2026-02-19 | results_zscore/ | **delete** | [doc](./compare_thresholding_methods.md) |
| zscore_pipeline | `control_asymmetry_zscore.py` | 2026-03-25 | results_zscore/ | **canonical** | [doc](./control_asymmetry_zscore.md) |
| zscore_pipeline | `control_zscore_analysis.py` | 2026-03-23 | results_zscore/ | **canonical** | [doc](./control_zscore_analysis.md) |
| zscore_pipeline | `control_zscore_asymmetry_clustering.py` | 2026-03-25 | results_zscore/ | **delete** | [doc](./control_zscore_asymmetry_clustering.md) |
| zscore_pipeline | `generate_clinical_zscore_maps.py` | 2026-03-24 | results_zscore/ | **canonical** | [doc](./generate_clinical_zscore_maps.md) |
| zscore_pipeline | `visualize_zscore_results.py` | 2026-03-24 | visual_analysis/ | **canonical** | [doc](./visualize_zscore_results.md) |

## Pruning progress

✅ **All 10 families pruned (2026-05-04).**

| Family | Result | Date |
| --- | --- | --- |
| zscore_pipeline | 4 canonical, 2 delete | 2026-05-04 |
| ez_discrimination | 1 canonical, 4 delete | 2026-05-04 |
| voxel | 2 delete | 2026-05-04 |
| laterality_viz | 4 delete | 2026-05-04 |
| ez_support | 1 canonical, 1 delete | 2026-05-04 |
| view_utilities | 3 delete | 2026-05-04 |
| verify_utilities | 2 delete | 2026-05-04 |
| batch | 2 delete | 2026-05-04 |
| comparative | 2 archive | 2026-05-04 |
| (other) | 2 canonical, 3 archive, 1 delete | 2026-05-04 |

## Summary

| Outcome | Count |
| --- | --- |
| **canonical** | 8 |
| **archive** | 5 |
| **delete** | 21 |
| **Total** | 34 |

The next phase is **post-pruning cleanup** — see `.claude/NEXT.md` for the follow-up tasks (method extraction, manuscript pivot, file renames, archive moves, etc.).
