# Ez Ground Truth

**Status:** canonical
**Last touched:** 2026-02-02
**Owner script(s):** ez_ground_truth.py
**Result directory:** ez_analysis_mdt/
**Promoted to pipeline?:** no

## Purpose

Ground Truth EZ Lateralization Labels

## Inputs

(to be filled during pruning conversation)

## Outputs

(to be filled during pruning conversation)

## Related variants

Family: `ez_support`

- [advanced_directional_analysis](./advanced_directional_analysis.md)

## Verdict

**canonical** (2026-05-04, ez_support pruning) — **HARD DEPENDENCY of the canonical pipeline**

Imported directly by `ez_discrimination_mdt.py` (lines 28–29: `from ez_ground_truth import EZ_GROUND_TRUTH, get_ez_label, get_unilateral_patients, get_bilateral_patients, get_left_ez_patients, get_right_ez_patients`).

Contains the only authoritative MDT-reviewed lateralization labels for all 15 patients plus accessor helpers. This data exists nowhere else in the repo.

**Proposed rename (post-pruning cleanup):** keep as `ez_ground_truth.py` — name is already clear and self-documenting.
