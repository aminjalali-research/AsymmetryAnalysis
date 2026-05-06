# Ground Truth: EZ Lateralization

This document explains how the epileptogenic zone (EZ) lateralization label was assigned for each of the 15 patients used to evaluate the asymmetry pipelines, and how that label is encoded in the codebase.

---

## MDT review process

Each patient's lateralization label was determined by a **multidisciplinary team (MDT)** review that integrates evidence from independent modalities. No single modality was used in isolation.

The MDT considered:

1. **Video-EEG monitoring (EMU).** Ictal onset patterns, interictal epileptiform discharges, semiology, and propagation. Lateralized rhythmic onset and concordant interictal spikes carry the strongest weight.
2. **Structural and research MRI.** Hippocampal sclerosis, focal cortical dysplasia, tumors, vascular malformations; volumetric asymmetries; FLAIR hyperintensities; automated lesion-detection outputs (MELDgraph for FCD, AIDHS for hippocampal sclerosis); language fMRI.
3. **Neuropsychological assessment.** Memory lateralization and language dominance consistent with lateralized dysfunction.

Based on the integrated evidence, the MDT assigned each patient one of the following labels:

| Code | Meaning |
|---|---|
| `L` | Unilateral left EZ |
| `R` | Unilateral right EZ |
| `B-L` | Bilateral pathology with **dominant left** side |
| `B-R` | Bilateral pathology with **dominant right** side |
| `B` | Bilateral pathology, no clear dominant side |

A confidence rating (`High`, `Moderate`, `Low`) was assigned per patient based on inter-modality concordance.

For the 15-patient cohort the final distribution is: 7 unilateral L, 5 unilateral R, 1 B-L, 1 B-R, 1 B. The discrimination analysis (`05_roi_discrimination.py`) reports two configurations: **unilateral-only** (n = 12) and **including bilateral-dominant** (n = 14, treating B-L as L and B-R as R).

---

## `EZ_GROUND_TRUTH` data structure

The labels live in `ez_ground_truth.py` at the repository root. The module's name is preserved verbatim — it is the canonical source imported by other scripts.

### Schema

`EZ_GROUND_TRUTH` is a `dict[str, dict]` keyed by patient ID. Each value has the following fields:

| Key | Required | Type | Notes |
|---|---|---|---|
| `ez` | yes | str | One of `L`, `R`, `B-L`, `B-R`, `B`. |
| `confidence` | yes | str | `High`, `Moderate`, or `Low`. |
| `evidence` | yes | str | One-line MDT evidence summary used for the manuscript Table 1. |
| `note` | no | str | Free-form notes (e.g., P025 carries `"PARIETAL - not temporal lobe epilepsy"`). |

### Helper functions (in `ez_ground_truth.py`)

| Function | Returns |
|---|---|
| `get_ez_label(patient_id, include_bilateral_as=None)` | The lateralization label. With `include_bilateral_as="dominant"`, B-L collapses to L and B-R to R. |
| `get_unilateral_patients()` | List of patient IDs with `ez in {"L", "R"}`. |
| `get_bilateral_patients()` | List of patient IDs whose label starts with `B`. |
| `get_left_ez_patients(include_bilateral_dominant=False)` | Left-EZ patient list. |
| `get_right_ez_patients(include_bilateral_dominant=False)` | Right-EZ patient list. |

### Imported by

- `05_roi_discrimination.py` — primary consumer; uses helpers to build the binary classification truth vector for ROC analysis.

---

## `clinical_spreadsheet.xlsx` and the ground-truth label

The spreadsheet (see [layout.md](layout.md#clinical_spreadsheetxlsx)) contains the **raw clinical evidence** that the MDT reviewed:

- `EEG/EMU FINDINGS`
- `CLINICAL MRI`
- `NEUROPSYCHOLOGY`
- `RESEARCH MRI`
- `SURGICAL PLAN AFTER MDT`
- `HISTOLOGY` (where surgery was performed)
- `POST-SURGICAL OUTCOME`

In earlier iterations of this codebase, ad-hoc EZ columns parsed directly from these free-text fields were consumed by the now-deleted `epileptogenic_zone_discrimination.py`. That approach was superseded for the manuscript by the curated, MDT-reviewed labels in `ez_ground_truth.py`. The spreadsheet remains the authoritative source for **demographics** (`AGE (SCAN)`, `SEX`) and for the per-patient evidence narrative reproduced in Table 1, but the **label vector** the manuscript reports against is `EZ_GROUND_TRUTH`.

---

## Sign convention

The manuscript adopts the **interictal ASL hypoperfusion** model: the epileptogenic tissue exhibits *lower* perfusion between seizures relative to the contralateral homolog. The Pipeline A laterality index is defined as

$$\mathrm{LI} = \frac{\mathrm{CBF}_L - \mathrm{CBF}_R}{\mathrm{CBF}_L + \mathrm{CBF}_R}$$

Combining the formula with the hypoperfusion model gives the convention used by `05_roi_discrimination.py`:

| EZ side | Hypoperfusion side | CBF inequality | Sign of LI |
|---|---|---|---|
| **Left EZ** | Left | $\mathrm{CBF}_R > \mathrm{CBF}_L$ | **Negative LI** |
| **Right EZ** | Right | $\mathrm{CBF}_L > \mathrm{CBF}_R$ | **Positive LI** |

A score function therefore predicts left EZ for sufficiently negative scores and right EZ for sufficiently positive scores. The optimal Youden threshold is determined empirically by the ROC analysis.

This convention was made explicit during the methodology pivot of 2026-05-04 (see [`docs/manuscript/notes.md`](../manuscript/notes.md)) and is reproduced verbatim in the class docstring of `EZDiscriminatorMDT` inside `05_roi_discrimination.py`.

For Pipeline B (voxel-wise zAI), the equivalent convention applies pixel-wise: a voxel with $z_{\mathrm{AI}} < 0$ is **right-dominant in the patient relative to controls** (consistent with left hypoperfusion and therefore left EZ), and $z_{\mathrm{AI}} > 0$ is **left-dominant** (right EZ).
