# Interactive Overlay Viewer

**Status:** canonical
**Last touched:** 2026-03-26
**Owner script(s):** interactive_overlay_viewer.py
**Result directory:** none
**Promoted to pipeline?:** no

## Purpose

Interactive Overlay Viewer for FreeSurfer and Laterality Maps

## Inputs

(to be filled during pruning conversation)

## Outputs

(to be filled during pruning conversation)

## Related variants

(no variants in this repo)

## Verdict

**canonical** (2026-05-04, (other) pruning) — **THE QC tool**

76 KB; the most recent file in the entire repo (2026-03-26); explicitly documented in `CLAUDE.md` as the project's interactive 3D inspection tool with menu-driven options 1–20. References all major result directories (`results/`, `results_voxel/`, `results_zscore/`, `results_voxel_roi/`) and both Dataset trees.

User-confirmed canonical (2026-05-04 pruning conversation). Stays at the repo root. README.md will list **FSLeyes as a soft dependency** (the script launches FSLeyes subprocess for 3D viewing).

**Proposed rename (post-pruning cleanup):** `interactive_viewer.py` (drops redundant "overlay").
