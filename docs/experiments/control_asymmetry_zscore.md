# Control Asymmetry Zscore

**Status:** canonical
**Last touched:** 2026-03-25
**Owner script(s):** control_asymmetry_zscore.py
**Result directory:** results_zscore/
**Promoted to pipeline?:** no

## Purpose

Control-Referenced Asymmetry Z-Score Analysis

## Inputs

(to be filled during pruning conversation)

## Outputs

(to be filled during pruning conversation)

## Related variants

Family: `zscore_pipeline`

- [control_zscore_analysis](./control_zscore_analysis.md)
- [control_zscore_asymmetry_clustering](./control_zscore_asymmetry_clustering.md)
- [generate_clinical_zscore_maps](./generate_clinical_zscore_maps.md)
- [compare_thresholding_methods](./compare_thresholding_methods.md)
- [visualize_zscore_results](./visualize_zscore_results.md)

## Verdict

**canonical** (2026-05-04, zscore_pipeline pruning) — **THE primary lateralization pipeline**

Implements **Approach B** — the literature-standard "zAI" pipeline:

1. Compute L-R asymmetry index `AI = 2(L−R)/(L+R)` per subject (patient + each control)
2. Build control AI normative database: per-voxel mean/SD across controls
3. `zAI = (AI_patient − μ_AI_controls) / SD_AI_controls`

This matches the methodology in:
- Shang et al. 2021, *Sci Rep* (PMC8149682) — exact ZAI definition
- Boscolo Galazzo et al. 2016, *NeuroImage:Clin* (PMC4872676) — simultaneous PET/ASL
- Kim et al. 2012; Didelot et al. 2022 — z-mapping the AI is more sensitive for EZ lateralization than direct z-score of raw signal

**Proposed rename (post-pruning cleanup):** `02_compute_zai.py`

**Follow-ups required (will be added to `.claude/NEXT.md`):**
1. **Audit:** confirm this script actually builds a control-AI normative DB (mean/SD of AI across all controls), not z-scoring against perfusion mean/SD by accident.
2. **Add M1–M13 thresholding:** extract from `control_zscore_asymmetry_clustering.py` into `src/thresholding.py` and import here.
3. **Cite recent high-impact paper** in the manuscript revision (search for 2024–2025 *Brain* / *NeuroImage* / *Epilepsia* zAI papers before submission).
