# Ez Discrimination Mdt

**Status:** canonical
**Last touched:** 2026-02-02
**Owner script(s):** ez_discrimination_mdt.py
**Result directory:** ez_analysis_mdt/
**Promoted to pipeline?:** no

## Purpose

EZ Discrimination Analysis with MDT-Reviewed Ground Truth Labels

## Inputs

(to be filled during pruning conversation)

## Outputs

(to be filled during pruning conversation)

## Related variants

Family: `ez_discrimination`

- [epileptogenic_zone_discrimination](./epileptogenic_zone_discrimination.md)
- [advanced_ez_discrimination](./advanced_ez_discrimination.md)
- [directional_ez_discrimination](./directional_ez_discrimination.md)
- [integrated_ez_discrimination](./integrated_ez_discrimination.md)

## Verdict

**canonical** (2026-05-04, ez_discrimination pruning — Path R consolidation) — **THE ROI-level discrimination pipeline**

Smallest of the 5 ez_discrimination scripts (546 lines), uses MDT-reviewed ground truth (imports from `ez_ground_truth.py`), evaluates 7 aggregation methods including `signed_weighted_li`, `temporal_signed_li`, `mesial_temporal_li`, `dominant_side_score`, `cohen_d_directional`. This is the script the manuscript's MDT-aligned discrimination analysis is built on.

**Path R commitment:** this script will absorb the methods from #1, #2, #3, #4 (after extraction) and become the **sole** ROI-level discrimination script. The other 4 will be deleted once that consolidation is verified.

**Methods to absorb from siblings (see their stub docs for details):**
- From `advanced_ez_discrimination.py` (#2): **`anatomically_weighted`** aggregation — required, named in manuscript abstract
- From `advanced_ez_discrimination.py` (#2): **`temporal_focus`** — verify whether duplicate of existing `temporal_signed_li` or genuinely different
- From `integrated_ez_discrimination.py` (#4): verify `weighted_aai` formula matches manuscript's `S_weighted = Σ(A_i · |A_i|) / Σ|A_i|`
- From `directional_ez_discrimination.py` (#3): port the **sign-convention** documentation block (hypoperfusion at EZ → opposite-direction LI) into the docstring

**Manuscript alignment:** the abstract claims **16 indices** but the user-selected manuscript text only enumerates 11 (3 ratio-based + 3 effect sizes + 2 distribution-based + 3 advanced). Either:
- 5 additional indices are listed elsewhere in the manuscript and need to also be reproducible by #5, OR
- The "16" count itself needs revision in the manuscript.
This is added as follow-up F-13 in `.claude/NEXT.md`.

**Proposed rename (post-pruning cleanup):** `roi_discrimination.py` (paired with `voxel_zai.py` for the dual-pipeline release). Or numbered: `05_roi_discrimination.py` after the 4 zscore_pipeline canonical scripts.

**Output:** `ez_analysis_mdt/` (76K, smallest result dir — already concise).
