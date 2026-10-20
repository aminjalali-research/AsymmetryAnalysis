# Quickstart — Reproduce the Canonical Results (2026-10-20)

> Run these commands in order to reproduce the manuscript's results from the released codebase. Assumes you have followed [`install.md`](install.md) and that FSL is on `PATH`.

## Prerequisites

1. Cloned this repository; Python ≥ 3.9 environment with the requirements installed
2. `Dataset/` (17 patients) and `DatasetControls/FM_*/` (3 bands × 60 controls) populated with HCP-ASL outputs and FreeSurfer `aparc+aseg`
3. `clinical_spreadsheet.xlsx` and `ez_ground_truth.py` at the repo root
4. FSL (flirt, fnirt) for the symmetric registration; FSLeyes only for `interactive_viewer.py`

## Memory rule

Every long run goes through `scripts/run_capped.sh <name> <MemoryMax> <log> -- <cmd>` (own systemd scope, hard memory cap, no swap, detached). Never run two pipelines at once; per-subject jobs of steps 01–03 peak at 1–9 GB (step 03 up to 23 GB on cluster-rich subjects), so `CHALFONT_JOBS=5` and a 34–40 GB cap are safe on a 64 GB machine.

## Pipeline B (voxel-wise, symmetric space)

```bash
python symreg/build_sym_template.py                                   # once: flip-symmetric T1 template
python symreg/make_jobs.py full | tr '\t' ' ' | xargs -P 8 -L 1 bash symreg/reg_subject.sh   # 197 subjects
scripts/run_capped.sh sym 34G symreg/logs/run.out -- bash symreg/run_sym_pipeline.sh         # 01 -> 02 (3 bands) -> 03 -> 06 -> thresholding
python symreg/check_full_calibration.py                               # GM % |zAI| >= 1.96 per patient (12.6 %; controls 6.9 %)
python scripts/clinical_stratified_analysis.py                        # rule x tier x scope grid, balanced accuracy
```

## Pipeline A (regional)

```bash
python scripts/build_comprehensive_asymmetry.py && python 05_roi_discrimination.py
python scripts/regional_loocv.py && python scripts/aggregation_separability.py
```

## External cohort (Chalfont) and enhanced estimators

```bash
python scripts/chalfont/stage_chalfont.py && python scripts/chalfont/asl_quality.py && python scripts/chalfont/build_template_atlas.py
bash scripts/chalfont/run_registration.sh
scripts/run_capped.sh chalfont 34G external_chalfont/logs/run.out -- env CHALFONT_JOBS=5 bash scripts/chalfont/run_chalfont_pipeline.sh tc all
for d in inhouse chalfont; do
  python scripts/v2/prep_subject_features.py --dataset $d
  python scripts/v2/analyze_v2.py --dataset $d
  python scripts/v2/multimetric_comparison.py --dataset $d
done
```

## Figures and tests

```bash
python scripts/make_manuscript_figures.py      # Figs 1-5, S1-S3 (writes to manuscript/figures/, private)
python -m pytest -q tests
```

## Expected runtime

| Stage | Time |
|-------|------|
| Registration (197 subjects, 8 parallel) | hours |
| Steps 01–06 in-house | ~40 min |
| Thresholding methods M1–M13 (permutation) | ~5.5 h, single process |
| Chalfont pipeline (99 patients, 5 jobs) + pseudo-controls | ~1.5 h |
| v2 features + analysis, both cohorts | ~10 min |

## What to check

- `results_zscore/clinical/cohort_concordance.csv`: 7/13 agree at the clinical tier (balanced accuracy 0.55).
- `ez_analysis_mdt/v2/chalfont/summary.csv`: `R_temporal_sign` 61/93; `R_temporal` calibrated 13 called, 12 correct.
- `external_chalfont/results_zscore/lateralization/eval_primary_endpoint.txt`: 58/93, balanced accuracy 0.63; controls strong/moderate 37/66.

## Read next

- [`01-overview.md`](01-overview.md), [`03-compute-zai.md`](03-compute-zai.md), [`09-clinical-interpretation.md`](09-clinical-interpretation.md)
- `docs/audit/METHODS_AUDIT_2026-10-20.md` for why the voxel-wise method fails and what the v2 estimators change
