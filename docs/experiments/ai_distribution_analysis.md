# Ai Distribution Analysis

**Status:** canonical
**Last touched:** 2026-02-23
**Owner script(s):** ai_distribution_analysis.py
**Result directory:** none
**Promoted to pipeline?:** no

## Purpose

|AI| Distribution Analysis for Statistical Threshold Selection

## Inputs

(to be filled during pruning conversation)

## Outputs

(to be filled during pruning conversation)

## Related variants

(no variants in this repo)

## Verdict

**canonical** (2026-05-04, (other) pruning) — **methodology validation tool**

Produces 4 publication-quality figures (per-patient |AI| histograms, group CDF overlays, threshold sensitivity curves, noise decomposition) into `results_voxel/distribution_analysis/`. Directly supports threshold selection justification for the manuscript.

**Adaptation needed:** currently reads from `results_voxel/`. Post-pivot, must consume `results_zscore/asymmetry/` (zAI distributions instead of within-subject AI distributions). The plotting + analysis logic is reusable.

**Proposed rename (post-pruning cleanup):** `analyze_zai_distributions.py`.
