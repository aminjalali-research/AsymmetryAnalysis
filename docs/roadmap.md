# Roadmap

This document outlines the planned trajectory of AsymmetryAnalysis beyond the current manuscript submission. It is intentionally written for three audiences:

- **Open-source visitors** evaluating whether to depend on the toolkit.
- **Future co-authors** considering extensions.
- **Reviewers** assessing whether the framework has a credible path to broader clinical impact.

Items are grouped by horizon. Within each section, items are unordered — priorities will shift with data availability and collaborator interest.

---

## Short-term (next 1–3 months)

These items are scoped, well-understood, and gated mostly on data and routine engineering effort. They directly close gaps in the current manuscript.

- **Acquire and process the remaining eight control demographic groups.** The current Pipeline B Results section is restricted to the five patients matched to `F_20_39` (n = 30 controls). To analyze all 15 patients under Pipeline B requires `F_40_59`, `F_60_79`, the three male bands, and the three sex-pooled bands. The pipeline scales to additional groups without code changes — see [docs/data/layout.md](data/layout.md).
- **Pipeline A vs Pipeline B concordance analysis.** Once all 15 patients have a Pipeline B prediction, compute patient-level concordance with the Pipeline A prediction (ROI weighted AAI). The preliminary n = 5 observation (5/5 concordant) is consistent with concordant lateralization signals across pipelines but is not statistically informative; the n = 15 analysis will be reported as a primary result. This work is currently deferred from the manuscript per §3.7.4.
- **Generate Pipeline B figures.** Produce per-patient zAI cluster overlay panels (axial / coronal / sagittal montage with M11 Quality+TFCE significant clusters annotated and labeled by region). The single figure currently referenced as `fig:zai_p013` is not yet rendered (see manuscript follow-up F-21).
- **Resolve manuscript Table 2 follow-up F-17.** Either implement per-region effect-size aggregation in `05_roi_discrimination.py` so that the three orphan rows (Weighted Cohen's d, Mean Hedges' g, Glass's Δ) are reproducible, or delete those rows and update the surrounding narrative. See [docs/manuscript/notes.md §4](manuscript/notes.md#4-open-follow-ups-manuscript-specific).
- **End-to-end manuscript compile verification.** Run `pdflatex` (twice + bibtex + twice) on `manuscript/Manuscript_Complete.tex` and confirm no undefined references, no overfull boxes on the longtables, and that every `\cite` resolves against `references.bib`.

---

## Mid-term (3–12 months)

These items extend the methodology, broaden validation, and begin clinical integration. Each requires either external collaboration, additional data acquisition, or non-trivial new development.

- **Multi-site validation.** Apply the dual-pipeline framework to an external cohort scanned at a different institution with a different (but harmonized) pCASL protocol. The current single-site, single-scanner design cannot establish generalization. Harmonization will require explicit protocol comparison, scanner-specific bias estimation, and potentially per-site normative control databases.
- **Head-to-head comparison against FDG-PET.** Several FDG-PET asymmetry biomarkers exist for EZ lateralization (cf. *Neurological Sciences* 2026, DOI 10.1007/s10072-026-08891-y; Boscolo Galazzo 2016 for an explicit ASL/PET methodological template). A subset of patients with both PCASL and FDG-PET would allow a direct sensitivity/specificity comparison and a pooled multimodal index.
- **Surgical outcome prediction.** Following Gennari et al. 2025 (pediatric ASL → seizure freedom), test whether the spatial overlap between Pipeline B's significant zAI cluster and the actual surgical resection cavity (post-operative MRI) predicts Engel-class outcome. This requires post-surgical imaging in operated patients and a longer follow-up window than the current cohort.
- **Surgical workflow integration.** Make the framework usable by surgical-planning teams without command-line proficiency:
  - DICOM export of clinical zAI maps with appropriate study/series tagging so that PACS-integrated workstations display them alongside the pre-operative anatomical study.
  - A 3D Slicer extension wrapping `02_compute_zai.py` and `03_clinical_maps.py` so that a clinician can drop a patient perfusion volume on the canvas and obtain a thresholded overlay.
- **Scanner / protocol robustness study.** Quantify how stable each of the 15 indices and the zAI map are under variation of pCASL post-labeling delay, label duration, and field strength. The current 3T-only protocol is one point in a multi-dimensional acquisition space.

---

## Aspirational (> 12 months)

These items represent the long-term vision. They will require additional funding, sustained collaboration, or technological maturation beyond the scope of any single project.

- **Generalization to other neurological conditions.** Hemispheric perfusion asymmetry is informative beyond epilepsy: in **stroke** (penumbra mapping, contralateral diaschisis), in **dementia** (asymmetric posterior parietal hypoperfusion in Alzheimer's, asymmetric frontotemporal patterns in FTD), and in **traumatic brain injury** (focal contusion-related hypoperfusion). The dual-pipeline framework is not specific to epilepsy; the per-condition challenge is curating an appropriate normative control database and ground-truth labeling protocol.
- **Real-time intraoperative integration.** ASL acquired intraoperatively (with adapted sequences and motion mitigation) could provide near-real-time perfusion feedback during resection. This is beyond current scanner workflow capabilities but is an active research direction.
- **Federated multi-site validation.** A privacy-preserving framework where each site computes per-voxel control statistics locally, shares only the sufficient statistics, and a central server combines them into a global normative database. This would let the framework benefit from cohort scale without ever moving raw imaging data, addressing the data-sharing barriers currently faced by multi-site neuroimaging studies.
- **Pre-trained zAI normative atlas as a community resource.** Release a curated, age- and sex-stratified normative AI database (the contents of `results_zscore/asymmetry/groups/*/`) as a downloadable resource so that other groups can compute zAI on their patients without needing to acquire their own healthy control cohort. This depends on multi-site validation establishing that the normative database transfers across scanners and protocols.
- **Integration with deep-learning lesion detectors.** The current cluster-to-region mapping (`02_compute_zai.py`) operates on anatomical parcellation. A learned mapping that fuses zAI with structural FLAIR/MELDgraph features and outputs patient-specific lesion probability maps could outperform either modality alone, especially in MRI-negative cases (currently the hardest subgroup; see manuscript §3.6 misclassification analysis).

---

## How to contribute

If you are working on an extension that fits one of the items above, open an issue on the GitHub repository (link in `README.md`) describing the planned scope and data sources. The codebase is organized so that new control demographic groups, new asymmetry indices, and new thresholding methods can each be added without touching the canonical pipeline scripts — see `src/calculator.py` and `src/thresholding.py`.

---

## From ZSCORE_ANALYSIS_PLAN.md (consolidated 2026-05-05)

The original z-score-pipeline plan listed implementation tasks that are either now part of the canonical Pipeline B or remain open as roadmap items. Items still open are absorbed into the short-term and mid-term sections above. The remainder are noted here for traceability.

### Originally planned, now delivered

- 9-group control demographic database (`F_20_39` available; remaining 8 groups noted as **In processing** — see short-term roadmap).
- Welford-streamed per-voxel mean/SD per group → `01_build_control_normative.py`.
- Per-patient z-score map → `02_compute_zai.py` (now AI-then-z, not z-then-AI; see manuscript notes §1).
- Cluster-to-region mapping using consensus parcellation → `02_compute_zai.py` outputs `cluster_report.csv` and `region_report.csv`.
- 13 thresholding/clustering methods (M1–M13) — implemented in `src/thresholding.py`, with **M11 (Quality+TFCE)** as the empirical winner (score 0.778, coverage ~3.2%).
- Gray-matter-only clinical maps (resolves the "everything highlighted" issue) → `03_clinical_maps.py`.
- Per-patient axial montages and 7 publication figures → `04_publication_figures.py`.
- FSLeyes-based interactive viewer with z-score-pipeline options → `interactive_viewer.py` options 18–20.

### Still open (folded into the short- and mid-term sections above)

- **Cross-patient summary CSV** for Pipeline B: not yet written. Aggregating the per-patient cluster reports into a cohort-level table is part of the Pipeline-A-vs-Pipeline-B concordance analysis (short-term).
- **Per-patient figure renders** for the manuscript Pipeline B section: see short-term item "Generate Pipeline B figures" and manuscript follow-up F-21 (`fig:zai_p013`).
- **Demographic generalization**: the framework's age/sex stratification mirrors the data layout in [docs/data/layout.md](data/layout.md). Mid-term roadmap items "Multi-site validation" and "Scanner / protocol robustness study" address this beyond the current single-site cohort.
