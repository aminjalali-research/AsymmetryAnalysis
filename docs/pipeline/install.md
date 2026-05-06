# Installation & Environment Setup

> Get a working Python environment for the AsymmetryAnalysis pipeline. Skip the FSL/FSLeyes section unless you plan to use the interactive viewer.

## Python

Python ≥ **3.9** is required (the codebase uses `pathlib`, f-strings, and dataclasses-style patterns). Tested on 3.10 / 3.11.

```bash
# create an isolated environment (preferred)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## Python packages

The canonical pipeline depends on:

```
numpy
scipy
pandas
nibabel
matplotlib
seaborn
scikit-learn      # ROC + LOOCV in 05_roi_discrimination.py
openpyxl          # reading clinical_spreadsheet.xlsx
```

If a `requirements.txt` or `pyproject.toml` is present at repo root, prefer it:

```bash
pip install -r requirements.txt          # if present
# or
pip install -e .                         # if pyproject.toml is present
```

Otherwise install the list above directly:

```bash
pip install numpy scipy pandas nibabel matplotlib seaborn scikit-learn openpyxl
```

## Data layout (must be in place before running step 1)

```
AsymmetryAnalysis/
├── Dataset/                      # 15 epilepsy patients (P013-P027)
│   └── <pid>/
│       ├── aparc+aseg.nii.gz
│       ├── T1w_acpc_dc_restore.nii.gz
│       └── <pid>_perfusion_calib_resampled_to_T1w.nii.gz   # actually MNI space
├── DatasetControls/              # healthy controls organised by age/sex
│   └── 20_29F/, 30_39F/, ...
│       └── <subj>/
│           ├── *_MNISpace_perfusion_calib_upsampled.nii.gz
│           └── aparc+aseg.nii.gz
└── clinical_spreadsheet.xlsx     # patient demographics (cols: ID, AGE, SEX)
```

All NIfTIs share MNI152 0.8 mm isotropic geometry (227×272×227). No registration is performed inside the released codebase — registration / preprocessing are upstream of this repo.

## FreeSurfer (upstream — NOT invoked by this codebase)

The released codebase **consumes** FreeSurfer outputs (`aparc+aseg.nii.gz`); it does not run FreeSurfer itself. If you need to regenerate `aparc+aseg.nii.gz` for new subjects, install FreeSurfer (https://surfer.nmr.mgh.harvard.edu/) separately. For reproducing the manuscript with the existing 15 patients and 30 controls, FreeSurfer is **not** needed — the precomputed `aparc+aseg.nii.gz` files are part of the dataset.

## FSL & FSLeyes (only for `interactive_viewer.py`)

The interactive viewer (`interactive_viewer.py`, options 18–20) shells out to `fsleyes`. None of the other canonical scripts depend on FSL.

```bash
# Linux example via the official FSL installer
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
# Then ensure fsleyes is on PATH:
which fsleyes
```

If FSLeyes is unavailable, `interactive_viewer.py` will still print the exact command it *would* have run, so you can paste it into a machine that has it.

## Verify your environment

```bash
python3 -c "import numpy, scipy, pandas, nibabel, matplotlib, sklearn, seaborn; print('OK')"
python3 -c "import ast; ast.parse(open('01_build_control_normative.py').read()); print('canonical scripts parse OK')"
```

If both print, you are ready to follow [`quickstart.md`](quickstart.md).

## Troubleshooting

- **`nibabel` fails to load NIfTIs:** confirm `pip install nibabel` finished without errors and that the files are not corrupted (`nib-ls Dataset/P013/T1w_acpc_dc_restore.nii.gz`).
- **`seaborn-v0_8-darkgrid` style not found:** seaborn renamed builtin styles in v0.12; `pip install 'seaborn>=0.12'`.
- **`openpyxl` ImportError when reading `clinical_spreadsheet.xlsx`:** `pip install openpyxl` (not bundled with pandas).
- **`fsleyes` not found:** only matters for `interactive_viewer.py`; install FSL (above) or skip that script.

## Read next

- [`quickstart.md`](quickstart.md) for the minimal command sequence
- [`01-overview.md`](01-overview.md) for the full pipeline architecture
