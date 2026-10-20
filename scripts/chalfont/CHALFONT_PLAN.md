# Chalfont external validation — analysis plan

**Status: pre-specified. Written 2026-10-20 before any Chalfont lateralization output had been read.**
The first complete pipeline run with the corrected validity rule (both mirror members must carry positive
CBF, `02_compute_zai.py::compute_asymmetry_map`, 2026-10-19) started 2026-10-20 10:15; the run that
preceded it was killed by a machine reset during step 2 and produced no patient or evaluation output.
This file is version-controlled so that the commit date documents the order of events.

## Question

Does the normative voxel-wise ASL asymmetry method, developed on 17 in-house patients and 180 HCP
controls, lateralize the clinically determined side in an independent cohort scanned on a different
vendor's sequence (GE eASL, Chalfont), with controls from the same scanner?

## Primary endpoint (one number)

| Element | Choice |
|---|---|
| Rule | production rule of `06_clinical_interpretation.py` (mesial-temporal priority, call strength) |
| Tier | clinical: \|zAI\| >= 4, >= 100 voxels. **Clarification 2026-10-20 (audit, before any output was read):** as implemented, clusters are formed at \|zAI\| >= 1.96 with >= 50 voxels in `02_compute_zai.py`; the tier keeps a cluster whose peak \|zAI\| >= 4 and whose extent at 1.96 is >= 100 voxels (`06_clinical_interpretation.py::_filter`). The endpoint is unchanged; its description is corrected. |
| Scope | 37 ROI pairs (3 subcortical + 34 Desikan–Killiany) |
| Perfusion map | transit-time-corrected CBF (`tc`) |
| Normative reference | pooled CH_ALL controls (`band_policy=all`) |
| Cohort | primary: QC technical failures out (`qc_exclusions.json`), flagged subjects kept |
| Patients | clinical lateralization L or R (B / unknown excluded from accuracy, kept in maps) |
| Metric | balanced accuracy (mean of L and R sensitivities); raw agreement with Wilson 95% CI; compared with the majority-side baseline; Bilateral / Unclear calls count as disagreement |

Code: `scripts/chalfont/evaluate_lateralization.py::PRIMARY_ENDPOINT`; written to
`external_chalfont/results_zscore/lateralization/eval_primary_endpoint.txt` and as the first block of
`eval_summary.txt`.

## Secondary endpoints (confirmatory, one each)

1. Direction-aware volume majority rule, clinical tier, 37-ROI scope: the rule selected post hoc on the
   in-house cohort (9/13). Same metric as the primary.
2. Control false-positive rate: fraction of healthy controls, scored leave-one-out as pseudo-patients,
   that receive a Strong or Moderate lateralizing call at the clinical tier, with Wilson 95% CI.

## Everything else is sensitivity analysis

Rule x tier x scope grid (5 rules x 4 tiers x 2 scopes), plain-flow CBF (`flow`), age-band references
(`specific`), group (lesional vs MRI-negative), lobe, QC-flagged sensitivity cohorts, post-surgical
exclusion, direction phenotype. Reported in a supplement; the word "accuracy" in the abstract refers to
the primary endpoint only.

## Wording commitments

- The reference standard is the clinical lateralization (resection side, lesion side or presurgical
  hypothesis); no post-surgical outcome is available, so the endpoint is **concordance**, not EZ accuracy.
  Reference-standard tiers (audit finding 4) to be added as a stratum before the results are written up.
- Results are reported with the majority baseline and lateralization counts (L:R) every time.
- Mirrored cluster pairs are counted once (`cluster_hemi == ez_side_pred`).

## Audit 2026-10-19 — status of findings

| # | Finding | Status |
|---|---|---|
| 1 | one-sided non-positive CBF saturates zAI | fixed in `02` (both members > 0), regression test added; in-house rerun started 2026-10-20 (`symreg/run_sym_pipeline.sh`) |
| 2 | no pre-specified endpoint | this file + `PRIMARY_ENDPOINT` block (2026-10-20) |
| 3 | fixed atlas vs individual GIF anatomy | open: planned as a sensitivity analysis (atlas ∩ warped GIF GM) |
| 4 | reference standard presumed | open: `reference_tier` column + stratum |
| 5 | QC exclusions not enforced in 01/02 discovery | fixed 2026-10-20: `dataset_config.configure()` prunes `qc_exclusions.json` IDs from every band before 01/02 see the table (`load_chalfont_groups(excluded=...)`, test in `tests/test_dataset_config.py`) |
| 6 | direction inference depends on global CBF | open: add GM-CBF z per subject; normalised variant |
| 7 | `asl_quality.py` hemisphere label inverted | fixed 2026-10-20: left mask from the affine (world x < 0); `lr_gm_ratio` is now L/R; metrics regenerated; exclusions unchanged (symmetric bounds) |
| 8 | stale metadata (n=69 vs 66, prescan wording) | fixed 2026-10-20 (`groups.json` age_summary, `dataset_config.py` comment, `CITATION_MAP.md` section 1) |
| 9–12 | minor | open |
