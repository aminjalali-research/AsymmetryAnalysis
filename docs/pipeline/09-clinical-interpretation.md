# Step 6: Clinical Interpretation (Surgeon-Facing Report)

> The canonical script for the deterministic per-patient surgeon report
> derived from Pipeline B Step 3 cluster outputs. This is what an
> epileptologist or surgeon should read BEFORE picking up FSLeyes — the
> one-screen answer from the ASL-perfusion arm of the MDT review.

## Purpose

Pipeline B Step 3 (`02_compute_zai.py`) produces a per-patient cluster
report CSV with hundreds of rows (433 for P015, 476 for P020). A
surgeon cannot interpret a 433-row spreadsheet; they need a deterministic
prediction with a confidence label and an explicit agree/disagree with
the MDT ground truth. This step does exactly that: it filters the cluster
report to the 37-ROI clinical scope, applies surgical thresholds, computes
a lateralization prediction with a Bartolomei-style confidence band, and
emits a one-screen surgeon-facing report (txt + json) plus a cohort
concordance CSV.

The output is the **deterministic answer** — not a substitute for
multi-modal MDT review, but the headline result the surgeon reads first.

## Script

`06_clinical_interpretation.py` (canonical, at repo root)

The script is pure stdlib + `pandas` + `ez_ground_truth.py`. No nibabel /
numpy / matplotlib dependencies in this step (it operates entirely on the
cluster_report CSV).

## Inputs

- `results_zscore/asymmetry/patients/<PID>/<PID>_asymmetry_cluster_report.csv`
  — the per-patient cluster table from Pipeline B Step 3
- `ez_ground_truth.py` — MDT-reviewed EZ side labels (imported)

## Outputs

Per-patient (`results_zscore/clinical/<PID>/`):

- `<PID>_surgeon_report.txt` — the human-readable one-screen report
  (≤ 80 lines, ≤ 80 columns wide). Sections: thresholds, MDT ground
  truth, lobe-level aggregate table, top-10 surviving clusters,
  algorithmic prediction with confidence + rationale, concordance with
  MDT, multi-modal concordance reminder.
- `<PID>_surgeon_report.json` — the same data structured for downstream
  programmatic use (cohort statistics, manuscript figures, etc.).

Cohort-level (`results_zscore/clinical/cohort_concordance.csv`, written
only when `--all` is passed):

| Column | Meaning |
|---|---|
| `patient_id` | Patient ID |
| `mdt_ez` | MDT-noted EZ side from `ez_ground_truth.py` |
| `mdt_confidence` | MDT confidence (`High` / `Moderate` / `Low`) |
| `algo_prediction` | Algorithm's predicted EZ (`L` / `R` / `B` / `Unclear`) |
| `algo_confidence` | Algorithm confidence (`Strong-{L,R}` / `Moderate-{L,R}` / `Weak-{L,R}` / `Bilateral` / `Unclear`) |
| `concordance` | `agree` / `disagree` / `partial` / `indeterminate` / `no-mdt-label` |
| `top_lobe` | The (side, lobe) bucket with the most surviving voxels |
| `top_voxels` | Voxel count of the top lobe |
| `peak_zai` | Maximum \|peak\_z\| among surviving in-scope clusters |

## Default thresholds

| Threshold | Default | Source / rationale |
|---|---|---|
| `--peak-threshold` | **\|zAI\| ≥ 4** | Boscolo Galazzo 2016 (focal hyper-/hypoperfusion in pre-surgical ASL); Gennari 2025 (surgical-grade ASL-PASL asymmetry) |
| `--size-threshold` | **≥ 100 voxels** at MNI 0.8 mm iso | ≈ 51 mm³ minimum focal cluster — slightly above the FreeSurfer cortical-thickness vertex sampling, suppresses subvoxel noise |
| 37-ROI clinical scope | 3 subcortical pairs (Thal/Hipp/Amyg) + 34 Desikan-Killiany cortical pairs = 76 FreeSurfer labels | Project policy (see `CLAUDE.md` "34 Cortical + 6 Subcortical Paired Regions"); excludes basal-ganglia + cerebellum + ventricles + white matter that are not surgical targets |

Both thresholds are tunable via CLI; report headers show the exact values
that produced the result.

## Usage

```bash
# Single patient (default thresholds: |zAI|>=4, size>=100)
python 06_clinical_interpretation.py --patient P015

# Cohort summary across every patient with zAI data
python 06_clinical_interpretation.py --all

# Stricter thresholds (suppress weaker / smaller clusters)
python 06_clinical_interpretation.py --patient P015 --peak-threshold 5.0 --size-threshold 200

# Output to a different location
python 06_clinical_interpretation.py --patient P015 --output-dir /tmp/myreports

# Suppress stdout (still writes the .txt and .json)
python 06_clinical_interpretation.py --patient P015 --quiet
```

Typical runtime: < 1 second per patient (pure CSV processing).

## Confidence levels (Bartolomei 2008 framework)

The algorithm classifies the result into one of seven confidence bands
based on multi-modal-concordance principles applied to the surviving
in-scope clusters. The "side" referred to here is the **predicted EZ
side** — not the side of the asymmetry signal (see "Polarity flip"
below).

| Band | Criteria |
|---|---|
| `Strong-L` / `Strong-R` | ≥ 3 mesial-temporal-axis (temporal + hippocampus + amygdala) clusters on one side, total ≥ 3000 voxels, contralateral MTL voxel total < 1/3 of dominant |
| `Moderate-L` / `Moderate-R` | 1–2 MTL clusters on one side and none on the other; OR bilateral MTL with one side ≥ 2× the other |
| `Weak-L` / `Weak-R` | Bilateral MTL with one side 1.05–2× the other; OR extra-temporal-only signals concentrated on one side |
| `Bilateral` | MTL involvement bilaterally with no clear voxel-count majority (ratio < 1.05); or extra-temporal voxels balanced |
| `Unclear` | No clusters survive the surgical thresholds within clinical scope |

All bands include the cluster count and voxel total in the report's
rationale line so the surgeon can see how the algorithm got there.

## Polarity flip (zAI sign → EZ side)

The `direction` field in the cluster report records the sign of the zAI:
- `direction = left-dominant` (positive zAI) — the LEFT hemisphere is
  abnormally MORE perfused than the RIGHT. In an interictal ASL study
  this corresponds to **right-hemisphere hypoperfusion**, so the
  predicted EZ side is **R**.
- `direction = right-dominant` (negative zAI) — the RIGHT is abnormally
  more perfused than the LEFT, so the predicted EZ side is **L**.

This flip is applied per-cluster before the lobar aggregation. The
report's lobe-level table is keyed by **predicted EZ side** (post-flip),
not by the cluster's anatomical hemisphere.

## How to interpret concordance against MDT

The `concordance` value in the JSON / cohort CSV summarizes the relation
between the algorithm's prediction and the MDT ground truth from
`ez_ground_truth.py`:

| Value | Meaning |
|---|---|
| `agree` | Algorithm's L/R/B prediction matches MDT exactly (or matches the dominant side of B-L/B-R) |
| `disagree` | Different sides predicted (e.g. algo=R, MDT=L) |
| `partial` | Bilateral case where dominance differs (e.g. algo=L, MDT=B-R) |
| `indeterminate` | Algorithm returned `Unclear` (no surviving clusters) |
| `no-mdt-label` | Patient has no MDT label in `ez_ground_truth.py` |

A `disagree` is not necessarily a failure of the algorithm — single
ASL-perfusion modality is one input to the MDT decision, and ASL is
known to disagree with EEG/MRI in roughly 1 in 4 patients in the
literature. The Bartolomei framework deliberately requires multi-modal
concordance rather than single-modality dominance.

## Cohort-level expectation

For the 5 patients with zAI data on disk at the time of canonical
write-up (P013, P014, P015, P020, P026):

- 4/5 algorithm-MDT concordance at default thresholds
- 1/5 disagreement (P013) — MTL voxel count is essentially balanced
  (1.09× ratio favouring R-side aggregation, MDT label = L). This is
  the kind of "single-modality-only-just" case the multi-modal reminder
  is designed for. Stricter thresholds (e.g. `--peak-threshold 5.0`)
  push P013 into the `Bilateral` / `Unclear` band rather than into a
  false R prediction.

## Cross-references

- Manuscript section: §3 Pipeline B clinical interpretation
- Upstream step that produces the cluster report: [03-compute-zai.md](03-compute-zai.md)
- Sister surgeon-facing visualization: [08-interactive-viewer.md](08-interactive-viewer.md)
  (option 15 wraps this script)
- Audit doc: `docs/superpowers/audits/2026-05-05-clinical-interpretation-tool.md`
- 37-ROI scope policy: `CLAUDE.md` ("34 Cortical + 6 Subcortical Paired
  Regions") and `docs/superpowers/audits/2026-05-05-viewer-clinical-fixes.md`

## References (literature, full citations in `references.bib`)

- Bartolomei et al. 2008 — multi-modal concordance framework for SEEG
  presurgical evaluation
- Boscolo Galazzo et al. 2016 — focal hyper-/hypoperfusion thresholds
  for surgical-grade ASL in TLE
- Gennari et al. 2025 — ASL-PASL asymmetry index thresholds for
  presurgical use in drug-resistant focal epilepsy
