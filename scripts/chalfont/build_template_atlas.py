#!/usr/bin/env python3
"""Build the fixed template-space Desikan-Killiany atlas for the Chalfont run.

Chalfont subjects have no FreeSurfer aparc+aseg, but every subject is warped to
the same left-right symmetric template as the 180 HCP controls. The voxel-wise
mode over the 180 symmetric-space control parcellations
(``symreg/sym_perf/<ID>_aparc_sym.nii.gz``; patients ``P0*`` excluded) is
therefore used as ONE fixed atlas for gray-matter masking, the 37-ROI clinical
scope and region naming.

The mode is the same majority vote as
``ControlGroupBuilder.build_consensus_parcellation`` (scipy.stats.mode: ties go
to the smallest label), computed memory-safely with a per-label vote-count
accumulator instead of stacking all 180 volumes.

Usage:
    python scripts/chalfont/build_template_atlas.py
    python scripts/chalfont/build_template_atlas.py --sym-dir symreg/sym_perf \
        --out external_chalfont/atlas/consensus_parcellation_sym.nii.gz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CORTICAL_LABELS = set(range(1001, 1036)) | set(range(2001, 2036))
SUBCORTICAL_LABELS = {10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60}
GRAY_MATTER_LABELS = CORTICAL_LABELS | SUBCORTICAL_LABELS
CLINICAL_ROI_LABELS = {10, 49, 17, 53, 18, 54} | CORTICAL_LABELS


def control_parcellations(sym_dir: Path, exclude_prefixes=("P0",)) -> list[Path]:
    files = sorted(sym_dir.glob("*_aparc_sym.nii.gz"))
    return [f for f in files if not f.name.startswith(exclude_prefixes)]


def build_mode_atlas(files: list[Path], log=print):
    """Voxel-wise mode over integer label volumes (ties -> smallest label)."""
    # Pass 1: label set (so the vote-count table can be allocated once).
    labels = set()
    shape = None
    ref = None
    for i, f in enumerate(files):
        img = nib.load(str(f))
        arr = np.asarray(img.dataobj)
        if shape is None:
            shape, ref = arr.shape, img
        elif arr.shape != shape:
            raise ValueError(f"{f.name}: shape {arr.shape} != {shape}")
        labels.update(np.unique(arr).tolist())
        if (i + 1) % 30 == 0:
            log(f"  pass 1 [{i + 1}/{len(files)}] labels so far: {len(labels)}")
    labels = np.array(sorted(int(l) for l in labels), dtype=np.int32)
    lut = np.full(int(labels.max()) + 1, -1, dtype=np.int32)
    lut[labels] = np.arange(len(labels), dtype=np.int32)
    n_vox = int(np.prod(shape))
    n_files = len(files)
    if n_files > 255:
        raise ValueError("uint8 vote counter supports at most 255 subjects")

    # Pass 2: per-label vote counts (n_labels x n_vox, uint8).
    counts = np.zeros((len(labels), n_vox), dtype=np.uint8)
    flat_idx = np.arange(n_vox)
    for i, f in enumerate(files):
        arr = np.asarray(nib.load(str(f)).dataobj).astype(np.int32).ravel()
        idx = lut[arr]
        if (idx < 0).any():
            raise RuntimeError(f"{f.name}: unseen label after pass 1")
        counts[idx, flat_idx] += 1
        if (i + 1) % 30 == 0:
            log(f"  pass 2 [{i + 1}/{len(files)}]")
    # argmax returns the first (= smallest) label among ties, like stats.mode
    mode_idx = np.argmax(counts, axis=0)
    mode_votes = counts[mode_idx, flat_idx]
    consensus = labels[mode_idx].reshape(shape).astype(np.int32)
    agreement = (mode_votes.astype(np.float32) / n_files).reshape(shape)
    return consensus, agreement, labels, ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sym-dir", type=Path, default=ROOT / "symreg" / "sym_perf")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "external_chalfont" / "atlas" / "consensus_parcellation_sym.nii.gz")
    ap.add_argument("--max-files", type=int, default=None, help="debug: use only the first N files")
    args = ap.parse_args(argv)

    files = control_parcellations(args.sym_dir)
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        print(f"No *_aparc_sym.nii.gz control files in {args.sym_dir}", file=sys.stderr)
        return 1
    print(f"Building template atlas from {len(files)} control parcellations in {args.sym_dir}")
    t0 = time.time()
    consensus, agreement, labels, ref = build_mode_atlas(files)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(consensus, ref.affine, ref.header)
    out_img.set_data_dtype(np.int32)
    out_img.header["descrip"] = f"Template DK atlas: mode of {len(files)} HCP aparc_sym"
    nib.save(out_img, str(args.out))
    agree_path = args.out.with_name(args.out.name.replace(".nii.gz", "_agreement.nii.gz"))
    nib.save(nib.Nifti1Image(agreement.astype(np.float32), ref.affine), str(agree_path))

    nonzero = consensus > 0
    gm = np.isin(consensus, list(GRAY_MATTER_LABELS))
    clin = np.isin(consensus, list(CLINICAL_ROI_LABELS))
    present = sorted(int(l) for l in np.unique(consensus))
    # Left/right symmetry QC on paired labels (1001-1035 <-> 2001-2035, subcortical pairs)
    pairs = [(l, l + 1000) for l in range(1001, 1036)] + [(10, 49), (17, 53), (18, 54)]
    pair_rows = []
    for l, r in pairs:
        nl, nr = int((consensus == l).sum()), int((consensus == r).sum())
        pair_rows.append({"left": l, "right": r, "left_vox": nl, "right_vox": nr,
                          "ratio_L_over_R": round(nl / nr, 3) if nr else None})
    missing = sorted(int(l) for l in (GRAY_MATTER_LABELS | {16}) if l not in present)
    meta = {
        "built_at": datetime.now().isoformat(),
        "n_subjects": len(files),
        "subject_files": [f.name for f in files],
        "sym_dir": str(args.sym_dir),
        "shape": list(consensus.shape),
        "voxel_mm": [float(abs(ref.affine[i, i])) for i in range(3)],
        "labels_seen_in_any_subject": [int(l) for l in labels],
        "labels_in_consensus": present,
        "n_labels_in_consensus": len(present),
        "labelled_voxels": int(nonzero.sum()),
        "gray_matter_voxels": int(gm.sum()),
        "gray_matter_mm3": float(gm.sum() * np.prod([abs(ref.affine[i, i]) for i in range(3)])),
        "clinical_37roi_voxels": int(clin.sum()),
        "mean_agreement_in_labelled": float(agreement[nonzero].mean()),
        "pct_labelled_voxels_agreement_ge_0_5": float(100 * (agreement[nonzero] >= 0.5).mean()),
        "gray_matter_labels_missing": missing,
        "paired_label_volumes": pair_rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.out.with_suffix("").with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  consensus labels: {len(present)}  labelled voxels: {nonzero.sum():,}")
    print(f"  gray-matter voxels: {gm.sum():,}  ({meta['gray_matter_mm3'] / 1000:.0f} cm3)")
    print(f"  37-ROI clinical voxels: {clin.sum():,}")
    print(f"  mean vote agreement (labelled): {meta['mean_agreement_in_labelled']:.3f}")
    print(f"  saved: {args.out}  (+ agreement map, JSON)  in {meta['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
