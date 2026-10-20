#!/usr/bin/env python3
"""Create a small synthetic Chalfont-layout root for smoke-testing the pipeline.

Writes <root>/sym_perf/<ID>_perf_sym.nii.gz (+ _perfflow_sym), metadata/subjects.csv,
metadata/groups.json and atlas/consensus_parcellation_sym.nii.gz on a small
flip-symmetric grid, with planted unilateral abnormalities in the patients:
  D001 L temporal hypoperfusion, D002 R temporal hyperperfusion,
  N001 (ez NA) nothing, N002 (B) bilateral, N003 (R, frontal) R frontal hypo.

    python scripts/chalfont/make_synthetic_chalfont.py <root> [--n-controls 8] [--seed 0]
    CHALFONT_ROOT=<root> bash scripts/chalfont/run_chalfont_pipeline.sh tc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

SHAPE = (40, 48, 40)


def affine_for(shape, vox=0.8):
    nx, ny, nz = shape
    aff = np.diag([-vox, vox, vox, 1.0])
    aff[0, 3] = vox * (nx - 1) / 2.0     # x = 0 between the two central columns -> flip0 symmetric
    aff[1, 3] = -vox * (ny - 1) / 2.0
    aff[2, 3] = -vox * (nz - 1) / 2.0
    return aff


def make_atlas(shape):
    nx, ny, nz = shape
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    cx, cy, cz = (nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2
    brain = ((i - cx) / (nx * 0.45)) ** 2 + ((j - cy) / (ny * 0.45)) ** 2 + ((k - cz) / (nz * 0.45)) ** 2 <= 1
    core = ((i - cx) / (nx * 0.2)) ** 2 + ((j - cy) / (ny * 0.2)) ** 2 + ((k - cz) / (nz * 0.2)) ** 2 <= 1
    atlas = np.zeros(shape, np.int32)
    right = i < cx          # x > 0 (RAS) with the negative x-step affine
    left = ~right
    temporal = brain & (k < cz - 4)
    frontal = brain & (k >= cz - 4) & (j >= cy + 4)
    parietal = brain & (k >= cz - 4) & (j < cy + 4)
    atlas[temporal & left] = 1030; atlas[temporal & right] = 2030
    atlas[frontal & left] = 1028; atlas[frontal & right] = 2028
    atlas[parietal & left] = 1029; atlas[parietal & right] = 2029
    atlas[core & left] = 17; atlas[core & right] = 53            # hippocampus
    wm = brain & (((i - cx) / (nx * 0.3)) ** 2 + ((j - cy) / (ny * 0.3)) ** 2 + ((k - cz) / (nz * 0.3)) ** 2 <= 1) & ~core
    atlas[wm & left] = 2; atlas[wm & right] = 41
    # enforce exact flip symmetry of the atlas (as a template atlas would be)
    mirror = np.flip(atlas, axis=0)
    lr = {1030: 2030, 2030: 1030, 1028: 2028, 2028: 1028, 1029: 2029, 2029: 1029, 17: 53, 53: 17, 2: 41, 41: 2}
    sym = atlas.copy()
    for a_lab, b_lab in lr.items():
        sym[(atlas == 0) & (mirror == b_lab)] = a_lab
    return sym, brain


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--n-controls", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single-band", action="store_true", help="write only CH_ALL to groups.json")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    root = args.root
    for d in ("sym_perf", "metadata", "atlas"):
        (root / d).mkdir(parents=True, exist_ok=True)
    aff = affine_for(SHAPE)
    atlas, brain = make_atlas(SHAPE)
    nib.save(nib.Nifti1Image(atlas, aff), str(root / "atlas" / "consensus_parcellation_sym.nii.gz"))

    base = np.zeros(SHAPE, np.float32)
    base[brain] = 50.0
    base[np.isin(atlas, [2, 41])] = 25.0
    smooth_field = lambda s: rng.normal(0, s, SHAPE).astype(np.float32)  # noqa: E731

    def subject_perf(mod=None, sd_pct=6.0, flip_scale=0.85):
        # symmetric baseline + subject-specific symmetric variation + small asymmetric noise
        f = base * (1 + rng.normal(0, 0.05)) + smooth_field(base * sd_pct / 100)
        f = (f + np.flip(f, 0)) / 2 + smooth_field(base * 3 / 100)
        if mod is not None:
            f = mod(f)
        f[~brain] = 0
        return np.clip(f, 0, None).astype(np.float32)

    rows = ["id,group,sex,age_at_scan,ez_side,ez_lobe,pathology"]
    ctrl_ids = [f"C{i + 1:03d}" for i in range(args.n_controls)]
    ages = {}
    for n, cid in enumerate(ctrl_ids):
        age = (25, 45, 65)[n % 3] + 3 * (n // 3)      # deterministic: >=5 per band for n>=15
        ages[cid] = age
        perf = subject_perf()
        nib.save(nib.Nifti1Image(perf, aff), str(root / "sym_perf" / f"{cid}_perf_sym.nii.gz"))
        nib.save(nib.Nifti1Image(perf * 0.9, aff), str(root / "sym_perf" / f"{cid}_perfflow_sym.nii.gz"))
        rows.append(f"{cid},control,{'F' if n % 2 else 'M'},{age},NA,,")

    def region_mod(labels, factor):
        def _m(f):
            f = f.copy(); m = np.isin(atlas, labels); f[m] *= factor; return f
        return _m

    patients = {
        "D001": ("discrete", "F", 33, "L", "temporal", "HS", region_mod([1030, 17], 0.65)),
        "D002": ("discrete", "M", 50, "R", "temporal", "FCD", region_mod([2030], 1.5)),
        "N001": ("negative", "F", 28, "NA", "", "", None),
        "N002": ("negative", "M", 41, "B", "temporal", "", region_mod([1030, 2030], 0.7)),
        "N003": ("negative", "F", 61, "R", "frontal", "", region_mod([2028], 0.6)),
    }
    for pid, (grp, sex, age, ez, lobe, path, mod) in patients.items():
        perf = subject_perf(mod)
        nib.save(nib.Nifti1Image(perf, aff), str(root / "sym_perf" / f"{pid}_perf_sym.nii.gz"))
        nib.save(nib.Nifti1Image(perf * 0.9, aff), str(root / "sym_perf" / f"{pid}_perfflow_sym.nii.gz"))
        rows.append(f"{pid},{grp},{sex},{age},{ez},{lobe},{path}")
    (root / "metadata" / "subjects.csv").write_text("\n".join(rows) + "\n")
    groups = {"CH_ALL": ctrl_ids,
              "CH_20_39": [c for c in ctrl_ids if 20 <= ages[c] <= 39],
              "CH_40_59": [c for c in ctrl_ids if 40 <= ages[c] <= 59],
              "CH_60_79": [c for c in ctrl_ids if 60 <= ages[c] <= 79],
              "age_summary": {"n_controls": len(ctrl_ids), "mean_age": float(np.mean(list(ages.values())))}}
    groups = {k: v for k, v in groups.items() if v}
    if args.single_band:
        groups = {"CH_ALL": ctrl_ids, "age_summary": groups["age_summary"]}
    (root / "metadata" / "groups.json").write_text(json.dumps(groups, indent=1))
    print(f"synthetic Chalfont root written to {root}: {len(ctrl_ids)} controls, {len(patients)} patients; "
          f"groups {[k for k in groups if isinstance(groups[k], list)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
