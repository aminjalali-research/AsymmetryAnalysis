#!/usr/bin/env python3
"""Preprocess newly-downloaded HCP-Aging control subjects into the layout the
existing normative pipeline (``01_build_control_normative.py``) expects.

WHAT IT DOES (per subject under DatasetControls/FM_20_39, FM_40_59, FM_60_79):
  1. Load  MNINonLinear/ASL/perfusion_calib.nii.gz          (2.5 mm, ~(72,87,72))
  2. Resample it onto the EXACT 0.8 mm MNI grid (227,272,227) using the
     subject's own MNINonLinear/aparc+aseg.nii.gz as the target-grid reference.
     -> write  <subjroot>/<ID>_MNISpace_perfusion_calib_upsampled.nii.gz
  3. Copy    MNINonLinear/aparc+aseg.nii.gz  ->  <subjroot>/aparc+aseg.nii.gz

INTERPOLATION CHOICE (methodological)
-------------------------------------
Perfusion / CBF is a CONTINUOUS physical quantity (mL/100 g/min), so it is
resampled with CONTINUOUS (TRILINEAR, order=1) interpolation -- NOT
nearest-neighbour. We deliberately do NOT use higher-order spline
(order>=2): cubic/quintic splines overshoot at sharp edges (GM/CSF
boundaries, brain/background) and can manufacture NEGATIVE CBF values, which
are physiologically impossible and would bias the control-group mean/SD. With
non-negative input, trilinear interpolation is bounded by the local data and
therefore cannot introduce negatives. This matches the patients' grid: their
``*_perfusion_calib_resampled_to_T1w.nii.gz`` (already (227,272,227)) contains
zero negative voxels and a CBF distribution consistent with trilinear
resampling, so controls and patients share the same interpolation philosophy.

IMPLEMENTATION NOTE -- why scipy, not nilearn:
nilearn.image.resample_to_img(interpolation="continuous") does NOT perform
pure trilinear: it dispatches to scipy.ndimage with cubic-spline prefiltering
(order=3), which overshoots at the brain/background boundary and produced
~4.7M negative voxels (down to -147 mL/100g/min) on the test subject. We
therefore resample directly with scipy.ndimage.affine_transform using
order=1 and prefilter=False (true bounded trilinear). On the test subject
this yields exactly 0 negatives, a max bounded below the source max, and a
nonzero-GM mean (~37 mL/100g/min) matching the old controls (~38).

The aparc+aseg parcellation is a LABEL image -> it must NEVER be linearly
interpolated. Here it is already on the 0.8 mm grid, so we simply COPY it
(no resampling, label integrity preserved exactly).

GRID FIDELITY
-------------
The subject's own MNINonLinear/aparc+aseg.nii.gz is verified (at runtime) to
be voxel-identical in shape AND affine to the existing old upsampled controls
in DatasetControls/20_29F/. Using it as ``target_img`` guarantees every new
upsampled perfusion map lands on the byte-identical grid the old controls use,
so downstream Welford summation aligns voxel-for-voxel.

USAGE
-----
  # Test ONE subject (default behaviour shown in --help):
  python preprocess_new_controls.py --subject HCA6018857
  python preprocess_new_controls.py --subject HCA6018857 --dry-run

  # Process ALL 180 new subjects across the three FM_* groups:
  python preprocess_new_controls.py --all

  # Idempotent: existing outputs are skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CONTROLS_ROOT = PROJECT_ROOT / "DatasetControls"
NEW_GROUP_DIRS = ["FM_20_39", "FM_40_59", "FM_60_79"]

# Reference old-control upsampled grid (for affine/shape verification only).
REFERENCE_OLD_GROUP = "20_29F"

EXPECTED_SHAPE = (227, 272, 227)
EXPECTED_ZOOMS = (0.8, 0.8, 0.8)

SRC_PERF_REL = Path("MNINonLinear") / "ASL" / "perfusion_calib.nii.gz"
SRC_PARC_REL = Path("MNINonLinear") / "aparc+aseg.nii.gz"


# --------------------------------------------------------------------------
def _find_reference_grid() -> nib.Nifti1Image:
    """Return any existing old upsampled control image as the canonical grid."""
    ref_dir = CONTROLS_ROOT / REFERENCE_OLD_GROUP
    cands = sorted(ref_dir.glob("*/*_MNISpace_perfusion_calib_upsampled.nii.gz"))
    if not cands:
        raise FileNotFoundError(
            f"No old upsampled control found under {ref_dir} for grid verification."
        )
    return nib.load(str(cands[0]))


def _affines_match(a: np.ndarray, b: np.ndarray, atol: float = 1e-4) -> bool:
    return np.allclose(a, b, atol=atol)


def discover_subjects(only: str | None = None) -> list[Path]:
    """Return subject directories under the three new FM_* group dirs."""
    subjects: list[Path] = []
    for grp in NEW_GROUP_DIRS:
        gdir = CONTROLS_ROOT / grp
        if not gdir.is_dir():
            print(f"  WARN: group dir not found (skipping): {gdir}")
            continue
        for entry in sorted(gdir.iterdir()):
            if entry.is_dir() and (entry / SRC_PERF_REL).exists():
                if only is None or entry.name == only:
                    subjects.append(entry)
    return subjects


def process_subject(
    subj_dir: Path,
    ref_grid: nib.Nifti1Image,
    dry_run: bool = False,
    force: bool = False,
) -> str:
    """Process one subject. Returns a status string: done|skip|dry|error."""
    sid = subj_dir.name
    src_perf = subj_dir / SRC_PERF_REL
    src_parc = subj_dir / SRC_PARC_REL
    out_perf = subj_dir / f"{sid}_MNISpace_perfusion_calib_upsampled.nii.gz"
    out_parc = subj_dir / "aparc+aseg.nii.gz"

    if not src_perf.exists():
        print(f"  [{sid}] ERROR missing source perfusion: {src_perf}")
        return "error"
    if not src_parc.exists():
        print(f"  [{sid}] ERROR missing source aparc+aseg: {src_parc}")
        return "error"

    # Idempotency: skip if both outputs already present.
    if out_perf.exists() and out_parc.exists() and not force:
        print(f"  [{sid}] skip (outputs already exist)")
        return "skip"

    # The subject's own aparc+aseg is the same-space 0.8mm grid reference.
    parc_img = nib.load(str(src_parc))
    if parc_img.shape != EXPECTED_SHAPE:
        print(f"  [{sid}] ERROR aparc+aseg shape {parc_img.shape} != {EXPECTED_SHAPE}")
        return "error"
    if not _affines_match(parc_img.affine, ref_grid.affine):
        print(
            f"  [{sid}] ERROR aparc+aseg affine does not match old-control grid.\n"
            f"        subject:\n{parc_img.affine}\n        reference:\n{ref_grid.affine}"
        )
        return "error"

    if dry_run:
        print(f"  [{sid}] DRY-RUN would:")
        print(f"          resample {src_perf}")
        print(f"            -> {out_perf}  (trilinear, target=subject aparc+aseg grid)")
        print(f"          copy {src_parc}")
        print(f"            -> {out_parc}")
        return "dry"

    # --- Step 2: resample perfusion (true bounded TRILINEAR, order=1) ---
    perf_img = nib.load(str(src_perf))
    src_data = perf_img.get_fdata(dtype=np.float32)

    # Build the target-voxel -> source-voxel index mapping:
    #   src_vox = inv(src.affine) @ tgt.affine @ tgt_vox
    mapping = np.linalg.inv(perf_img.affine) @ parc_img.affine
    matrix = mapping[:3, :3]
    offset = mapping[:3, 3]
    resampled = affine_transform(
        src_data,
        matrix,
        offset=offset,
        output_shape=parc_img.shape,
        order=1,            # trilinear -- bounded, no overshoot, no negatives
        mode="constant",
        cval=0.0,
        prefilter=False,    # MUST be False: prefilter implies spline order>1 fit
    )

    # Pin the output geometry to the canonical old-control grid (byte-identical
    # affine/header) so downstream Welford summation aligns voxel-for-voxel.
    out_img = nib.Nifti1Image(
        np.asarray(resampled, dtype=np.float32),
        ref_grid.affine,
        header=ref_grid.header,
    )
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, str(out_perf))

    # --- Step 3: copy aparc+aseg to subject root (labels: copy, never interp) ---
    shutil.copyfile(src_parc, out_parc)

    # Post-write sanity check
    d = out_img.get_fdata()
    nz = d[d != 0]
    negs = int((d < 0).sum())
    print(
        f"  [{sid}] done  perf shape={out_img.shape} "
        f"nz={nz.size} mean={nz.mean():.2f} max={d.max():.1f} negs={negs}"
        + ("  <-- WARNING negatives!" if negs else "")
    )
    return "done"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subject", help="Process ONE subject by ID (e.g. HCA6018857).")
    g.add_argument("--all", action="store_true",
                   help="Process ALL new subjects across the three FM_* groups.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report planned actions without writing anything.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing outputs (disable idempotent skip).")
    args = ap.parse_args(argv)

    ref_grid = _find_reference_grid()
    print(f"Reference grid: shape={ref_grid.shape}, zooms="
          f"{tuple(round(z,3) for z in ref_grid.header.get_zooms()[:3])}")

    subjects = discover_subjects(only=args.subject)
    if not subjects:
        target = args.subject if args.subject else "<all FM_* dirs>"
        print(f"No subjects found for: {target}")
        return 1

    print(f"Subjects to process: {len(subjects)}\n")
    counts = {"done": 0, "skip": 0, "dry": 0, "error": 0}
    for subj in subjects:
        status = process_subject(subj, ref_grid, dry_run=args.dry_run, force=args.force)
        counts[status] = counts.get(status, 0) + 1

    print("\n" + "=" * 60)
    print(f"Summary: done={counts['done']} skip={counts['skip']} "
          f"dry={counts['dry']} error={counts['error']} "
          f"(total {len(subjects)})")
    print("=" * 60)
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
