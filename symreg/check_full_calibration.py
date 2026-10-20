#!/usr/bin/env python3
"""Post-pipeline GM-specific calibration check (run after run_sym_pipeline.sh).

For each patient, reports %|zAI|>=1.96 within the gray-matter consensus mask of
its age-matched band, using the freshly-built symmetric-space group stats. This
is the manuscript headline calibration number. Compares against the archived
original-space zAI maps when present (in-house only).

    python symreg/check_full_calibration.py                     # in-house (hard-coded bands)
    python symreg/check_full_calibration.py --dataset chalfont  # bands from 01 outputs;
                                                                # CSV -> <results>/calibration/
For the leave-one-out control calibration see scripts/chalfont/calibration_loo.py.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
import dataset_config  # noqa: E402

GROUPS = BASE / "results_zscore/groups"
ZAI = BASE / "results_zscore/asymmetry/patients"
ARCH = BASE / "_archived_origspace_zai_20260622/asymmetry/patients"

CORT = set(range(1001, 1036)) | set(range(2001, 2036))
SUBC = {10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60}
GM = CORT | SUBC

PAT_BAND = {'FM_20_39': ['P013','P015','P016','P017','P019','P020','P026','P028','P029','P030'],
            'FM_40_59': ['P018','P021','P022','P023','P024','P027'],
            'FM_60_79': ['P025']}


def gm_mask(band):
    parc = nib.load(str(GROUPS/band/"consensus_parcellation.nii.gz")).get_fdata().astype(np.int32)
    m = np.zeros(parc.shape, bool)
    for lab in GM:
        m |= (parc == lab)
    return m


def pct(zpath, mask):
    if not zpath.exists():
        return None
    z = nib.load(str(zpath)).get_fdata(dtype=np.float32)
    m = mask & (z != 0)
    n = int(m.sum())
    return (100.0 * np.sum(np.abs(z[m]) >= 1.96) / n, n) if n else (float('nan'), 0)


def discover_patient_bands(results_dir):
    """{band: [pid, ...]} from 01's per-patient z-maps <pid>_vs_<band>_zscore.nii.gz."""
    out = {}
    pdir = results_dir / "patients"
    if not pdir.exists():
        return out
    for d in sorted(pdir.iterdir()):
        for c in sorted(d.glob(f"{d.name}_vs_*_zscore.nii.gz")):
            band = c.name[len(f"{d.name}_vs_"):-len("_zscore.nii.gz")]
            out.setdefault(band, []).append(d.name)
            break
    return out


def main(argv=None):
    global GROUPS, ZAI
    ap = argparse.ArgumentParser(description="GM %|zAI|>=1.96 per patient (calibration check)")
    dataset_config.add_dataset_args(ap)
    args = ap.parse_args(argv)
    cfg = dataset_config.configure(args.dataset, perf_variant=args.perf_variant, band_policy=args.band_policy)
    if cfg.is_inhouse:
        pat_band = PAT_BAND
        arch = ARCH
    else:
        GROUPS = cfg.groups_dir
        ZAI = cfg.cluster_report_dir
        pat_band = discover_patient_bands(cfg.output_dir)
        arch = None

    print(f"{'patient':8} {'band':9} {'NEW %|z|>=1.96 (GM)':>22} {'OLD %':>8} {'GMvox':>9}")
    new_all = []
    rows = []
    for band, pats in pat_band.items():
        mask = gm_mask(band)
        for p in pats:
            newp = pct(ZAI/p/f"{p}_asymmetry_zscore.nii.gz", mask)
            # original-space archived map used a different (per-band) GM mask, but
            # the magnitude is comparable; report with the same sym GM mask if shapes match
            oldz = arch/p/f"{p}_asymmetry_zscore.nii.gz" if arch else None
            oldp = pct(oldz, mask) if oldz is not None and oldz.exists() else None
            if newp:
                new_all.append(newp[0])
                rows.append({"patient_id": p, "band": band, "gm_voxels": newp[1],
                             "pct_abs_zai_ge_1_96": round(newp[0], 3)})
            ns = f"{newp[0]:.1f}%" if newp else "MISSING"
            os_ = f"{oldp[0]:.1f}%" if oldp else "n/a"
            nv = newp[1] if newp else 0
            print(f"{p:8} {band:9} {ns:>22} {os_:>8} {nv:>9,}")
    print(f"\nNEW cohort mean %|zAI|>=1.96 in GM = {np.mean(new_all):.1f}% (n={len(new_all)})")
    print("Target: ~5-15% for a calibrated normative asymmetry model.")
    if not cfg.is_inhouse:
        cfg.calibration_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.calibration_dir / "patient_gm_calibration.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"CSV -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
