#!/usr/bin/env python3
"""Regenerate per-patient comprehensive ROI asymmetry CSVs (Pipeline A input).

For a given patient, extract per-region mean/std/nvoxels/volume perfusion from
that patient's aparc+aseg.nii.gz + perfusion volume over the canonical 37
bilateral region pairs (3 subcortical + 34 cortical Desikan-Killiany), then
materialise all 15 asymmetry indices via src.calculator.AsymmetryCalculator.

Writes results_analysis/<PID>/<PID>_comprehensive_asymmetry_analysis.csv with
the exact column schema consumed by 05_roi_discrimination.py and
scripts/per_index_discrimination.py.

This reconstructs the (deleted) original batch ROI-extraction step. Validated to
reproduce the existing P013 CSV to within float tolerance.

Usage:
    python3 scripts/build_comprehensive_asymmetry.py P028 P029 P030
    python3 scripts/build_comprehensive_asymmetry.py --validate P013
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.calculator import AsymmetryCalculator  # noqa: E402

VOXEL_VOLUME_MM3 = 0.8 ** 3  # 0.512

# 37 canonical region pairs: (region, region_type, left_segid, right_segid,
# left_struct_name, right_struct_name)
SUBCORTICAL = [
    ("Thalamus-Proper", 10, 49, "Left-Thalamus-Proper", "Right-Thalamus-Proper"),
    ("Hippocampus", 17, 53, "Left-Hippocampus", "Right-Hippocampus"),
    ("Amygdala", 18, 54, "Left-Amygdala", "Right-Amygdala"),
]

# Desikan-Killiany cortical region names indexed by their (left) aparc label.
# Left labels 1001..1035 minus 1004 (corpuscallosum, not a DK gyrus).
DK_NAMES = {
    1001: "bankssts", 1002: "caudalanteriorcingulate", 1003: "caudalmiddlefrontal",
    1005: "cuneus", 1006: "entorhinal", 1007: "fusiform", 1008: "inferiorparietal",
    1009: "inferiortemporal", 1010: "isthmuscingulate", 1011: "lateraloccipital",
    1012: "lateralorbitofrontal", 1013: "lingual", 1014: "medialorbitofrontal",
    1015: "middletemporal", 1016: "parahippocampal", 1017: "paracentral",
    1018: "parsopercularis", 1019: "parsorbitalis", 1020: "parstriangularis",
    1021: "pericalcarine", 1022: "postcentral", 1023: "posteriorcingulate",
    1024: "precentral", 1025: "precuneus", 1026: "rostralanteriorcingulate",
    1027: "rostralmiddlefrontal", 1028: "superiorfrontal", 1029: "superiorparietal",
    1030: "superiortemporal", 1031: "supramarginal", 1032: "frontalpole",
    1033: "temporalpole", 1034: "transversetemporal", 1035: "insula",
}


def build_pair_specs():
    specs = []
    for name, lseg, rseg, lstruct, rstruct in SUBCORTICAL:
        specs.append((name, "subcortical", lseg, rseg, lstruct, rstruct))
    for lseg in sorted(DK_NAMES):
        name = DK_NAMES[lseg]
        rseg = lseg + 1000
        specs.append((name, "cortical", lseg, rseg,
                      f"ctx-lh-{name}", f"ctx-rh-{name}"))
    return specs


def region_stats(perf, aparc, segid):
    """Mean/std/nvoxels/volume over all voxels with label==segid.

    Matches the original (deleted) batch extractor, which did NOT apply a
    perfusion>0 mask: every labelled voxel — including any perfusion==0
    voxels — contributes to the region mean/std/count. Verified to reproduce
    the existing P013 CSV exactly.
    """
    mask = aparc == segid
    vals = perf[mask]
    n = int(vals.size)
    if n == 0:
        return np.nan, np.nan, 0, 0.0
    return float(vals.mean()), float(vals.std()), n, n * VOXEL_VOLUME_MM3


def build_patient(pid: str) -> pd.DataFrame:
    pdir = ROOT / "Dataset" / pid
    perf = nib.load(pdir / f"{pid}_perfusion_calib_resampled_to_T1w.nii.gz").get_fdata()
    aparc = nib.load(pdir / "aparc+aseg.nii.gz").get_fdata().astype(np.int32)

    rows = []
    for name, rtype, lseg, rseg, lstruct, rstruct in build_pair_specs():
        lm, ls, ln, lv = region_stats(perf, aparc, lseg)
        rm, rs, rn, rv = region_stats(perf, aparc, rseg)
        rows.append({
            "region": name,
            "region_type": rtype,
            "left_segid": lseg,
            "right_segid": rseg,
            "left_index": lseg,
            "right_index": rseg,
            "left_struct_name": lstruct,
            "right_struct_name": rstruct,
            "display_name": f"[{lseg}/{rseg}] {name}",
            "left_mean_perfusion": lm,
            "right_mean_perfusion": rm,
            "left_volume": lv,
            "right_volume": rv,
            "left_std": ls,
            "right_std": rs,
            "left_nvoxels": ln,
            "right_nvoxels": rn,
            "segid_order": lseg,
        })
    df = pd.DataFrame(rows)
    calc = AsymmetryCalculator()
    df = calc.calculate_all_indices(df)
    return df


def validate(pid: str) -> None:
    new = build_patient(pid)
    ref_path = ROOT / "results_analysis" / pid / f"{pid}_comprehensive_asymmetry_analysis.csv"
    ref = pd.read_csv(ref_path)
    cols = ["left_mean_perfusion", "right_mean_perfusion", "laterality_index",
            "absolute_asymmetry_index", "cohen_d_asymmetry", "left_nvoxels"]
    print(f"Validation for {pid} (max abs diff vs existing CSV):")
    new_i = new.set_index("region")
    ref_i = ref.set_index("region")
    for c in cols:
        d = (new_i[c] - ref_i[c]).abs().max()
        print(f"  {c:28s} max|Δ| = {d:.3e}")


def main(argv):
    if argv and argv[0] == "--validate":
        for pid in argv[1:]:
            validate(pid)
        return
    for pid in argv:
        df = build_patient(pid)
        out_dir = ROOT / "results_analysis" / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{pid}_comprehensive_asymmetry_analysis.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {out}  ({len(df)} region pairs)")
        # quick summary
        li = df.set_index("region")["laterality_index"]
        print(f"  LI range [{li.min():+.3f}, {li.max():+.3f}], "
              f"signed_weighted_li = "
              f"{(li.values*np.abs(li.values)).sum()/np.abs(li.values).sum():+.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
