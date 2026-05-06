# Methodology Aid: zAI Distribution Analysis

> Methodology validation script — not in the main pipeline data flow, but cited in the manuscript to justify threshold choices.

## Purpose

Generate publication-quality distribution charts of the absolute Asymmetry Index (`|AI|`) — both at the patient level and pooled across the cohort — and visualize where the various candidate thresholds fall on those distributions. This provides empirical justification for the strict `|zAI| ≥ 3.0` cutoff used in `03_clinical_maps.py` and for any percentile / TFCE / GMM thresholds compared in supplementary methodology figures.

The script is a methodology aid: it does not feed into downstream pipeline scripts. Its outputs back the threshold-justification subsection of the manuscript and the threshold-comparison rows of the supplementary table.

## Script

`analyze_zai_distributions.py` (canonical, at repo root)

## Inputs

- `results_voxel/<pid>/` — voxel-wise asymmetry maps and quality masks
- (Optional) per-patient TFCE-effective threshold values, if precomputed

## Outputs

In `results_voxel/distribution_analysis/`:

- `<pid>_ai_distribution.png` — per-patient `|AI|` histogram with quality-mask subset, half-normal noise fit, and annotated threshold lines (percentile + TFCE effective cutoff)
- `group_cdf_overlay.png` — group-level overlay of all 15 patients' `|AI|` cumulative distributions
- `threshold_sensitivity.png` — fraction of voxels retained as a function of `|AI|` cutoff
- `noise_vs_signal_decomposition.png` — half-normal (noise) + empirical tail (signal) overlay

## Usage

```bash
python analyze_zai_distributions.py                     # all patients
python analyze_zai_distributions.py --patients P013 P020
```

Typical runtime: ~30 seconds for all patients.

## What this step produces (relative to the manuscript)

- Supports the threshold-choice subsection of §2.11 / §2.12 (thresholding methodology)
- Backs the supplementary methodology figures (S2-style threshold comparison)
- Used by `interactive_viewer.py` option for the per-patient distribution chart (the viewer regenerates the chart on demand if missing)

## Dependencies

- Python: `numpy`, `pandas`, `scipy.stats` (for half-normal fit), `matplotlib`, `seaborn`
- Data: voxel-wise results (`results_voxel/`)
- Other scripts: voxel pipeline must have produced AI maps

## Cross-references

- Manuscript section: §2.11–2.12 "Thresholding methodology"; supplementary figures
- Consumer: `interactive_viewer.py` regenerates per-patient histograms when launched and a chart is missing
- Related canonical scripts: methodology-only — does not feed steps 1–5
