#!/usr/bin/env python3
"""Leave-one-out control calibration of the normative zAI model (+ patients).

For every control band: the gray-matter fraction of voxels with |zAI| >= 1.96
for each control when that control is EXCLUDED from the band's AI mean/SD
(mirror-pooled SD, 5% floor, 75% coverage mask, exactly as 02_compute_zai.py),
and, for comparison, the optimistic value with the control included. The same
metric is reported for every patient from the 02_compute_zai.py zAI maps
(band resolved from the 01_build_control_normative.py output filename).

Outputs (``<results>/calibration/``):
    control_loo_calibration.csv   id, band, gm_voxels, pct_loo, pct_full
    patient_gm_calibration.csv    patient_id, band, gm_voxels, pct_abs_zai_ge_1_96
    calibration_summary.json      per-band and cohort means

    python scripts/chalfont/calibration_loo.py --dataset chalfont [--perf-variant flow]
    python scripts/chalfont/calibration_loo.py --dataset inhouse      # reproduces the 6.9% figure
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import dataset_config  # noqa: E402

_spec = importlib.util.spec_from_file_location("compute_zai_02", ROOT / "02_compute_zai.py")
zai = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(zai)

Z_THR = 1.96
SD_FLOOR_PCT = 5.0


def gm_mask_for(groups_dir: Path, band: str) -> np.ndarray:
    parc = np.asarray(nib.load(str(groups_dir / band / "consensus_parcellation.nii.gz")).dataobj).astype(np.int32)
    return np.isin(parc, list(zai.GRAY_MATTER_LABELS))


def pct_exceed(ai, valid, mean, sd, mask):
    m = valid & mask & (sd > 0)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    z = np.clip((ai[m] - mean[m]) / sd[m], -20, 20)
    return float(100.0 * np.mean(np.abs(z) >= Z_THR)), n


def band_loo(cfg, band: str, log=print):
    ids = cfg.control_ids(band)
    gm = gm_mask_for(cfg.groups_dir, band)
    S = SS = None
    ai_cache = {}
    log(f"\n=== band {band}: {len(ids)} controls ===")
    for i, cid in enumerate(ids):
        perf = nib.load(str(cfg.perf_path(cid))).get_fdata(dtype=np.float32)
        ai, valid = zai.compute_asymmetry_map(perf)
        if S is None:
            S = np.zeros(ai.shape); SS = np.zeros(ai.shape); C = np.zeros(ai.shape, np.int32)
        S[valid] += ai[valid]; SS[valid] += ai[valid] ** 2; C[valid] += 1
        if (i + 1) % 20 == 0:
            log(f"  accumulated {i + 1}/{len(ids)}")
    n = len(ids)
    # full (optimistic) statistics, as 02 builds them
    mean_full = np.zeros_like(S); ok = C > 0; mean_full[ok] = S[ok] / C[ok]
    sd_full = np.zeros_like(S); ok2 = C >= 2
    sd_full[ok2] = np.sqrt(np.maximum(SS[ok2] - C[ok2] * mean_full[ok2] ** 2, 0) / (C[ok2] - 1))
    sd_full = np.maximum(np.maximum(sd_full, np.flip(sd_full, axis=0)), SD_FLOOR_PCT)
    mask_full = (C / n) >= zai.BRAIN_MASK_COVERAGE
    rows = []
    for i, cid in enumerate(ids):
        perf = nib.load(str(cfg.perf_path(cid))).get_fdata(dtype=np.float32)
        ai, valid = zai.compute_asymmetry_map(perf)
        m_loo, s_loo, mask_loo = zai._loo_stats(S, SS, C, ai, valid, n,
                                                sd_floor=SD_FLOOR_PCT, mirror_pool=True)
        p_loo, nvox = pct_exceed(ai, valid, m_loo, s_loo, mask_loo & gm)
        p_full, _ = pct_exceed(ai, valid, mean_full, sd_full, mask_full & gm)
        rows.append({"id": cid, "band": band, "gm_voxels": nvox,
                     "pct_loo": round(p_loo, 3), "pct_full": round(p_full, 3)})
        log(f"  {cid}: LOO {p_loo:5.1f}%   full {p_full:5.1f}%   ({nvox:,} GM voxels)")
    return rows


def patient_rows(cfg, log=print):
    from importlib import import_module  # noqa: F401
    sys.path.insert(0, str(ROOT / "symreg"))
    import check_full_calibration as cfc
    bands = cfc.discover_patient_bands(cfg.output_dir)
    rows = []
    for band, pids in bands.items():
        gm = gm_mask_for(cfg.groups_dir, band)
        for pid in pids:
            zp = cfg.cluster_report_dir / pid / f"{pid}_asymmetry_zscore.nii.gz"
            if not zp.exists():
                log(f"  {pid}: no zAI map ({zp.name} missing)")
                continue
            z = nib.load(str(zp)).get_fdata(dtype=np.float32)
            m = gm & (z != 0)
            nvox = int(m.sum())
            p = float(100.0 * np.mean(np.abs(z[m]) >= Z_THR)) if nvox else float("nan")
            rows.append({"patient_id": pid, "band": band, "gm_voxels": nvox,
                         "pct_abs_zai_ge_1_96": round(p, 3)})
            log(f"  {pid} ({band}): {p:5.1f}%")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dataset_config.add_dataset_args(ap)
    ap.add_argument("--bands", nargs="+", default=None, help="only these control bands")
    ap.add_argument("--skip-patients", action="store_true")
    args = ap.parse_args(argv)
    cfg = dataset_config.configure(args.dataset, perf_variant=args.perf_variant, band_policy=args.band_policy)
    if cfg.is_inhouse and cfg.sym_mode is None:
        cfg.sym_mode = True  # the in-house normative model is the symmetric-space one
    bands = args.bands or [b for b in cfg.available_groups()
                           if (cfg.groups_dir / b / "consensus_parcellation.nii.gz").exists()]
    if not bands:
        print(f"No built control bands under {cfg.groups_dir}; run 01 first.", file=sys.stderr)
        return 1
    cfg.calibration_dir.mkdir(parents=True, exist_ok=True)
    ctrl = []
    for b in bands:
        ctrl.extend(band_loo(cfg, b))
    ctrl_df = pd.DataFrame(ctrl)
    ctrl_df.to_csv(cfg.calibration_dir / "control_loo_calibration.csv", index=False)
    summary = {"dataset": cfg.name, "perf_variant": cfg.perf_variant, "z_threshold": Z_THR,
               "bands": {}}
    for b, g in ctrl_df.groupby("band"):
        summary["bands"][b] = {"n_controls": int(len(g)),
                               "control_loo_mean_pct": round(float(g.pct_loo.mean()), 3),
                               "control_loo_median_pct": round(float(g.pct_loo.median()), 3),
                               "control_loo_range_pct": [round(float(g.pct_loo.min()), 3), round(float(g.pct_loo.max()), 3)],
                               "control_full_mean_pct": round(float(g.pct_full.mean()), 3)}
    summary["control_loo_mean_pct"] = round(float(ctrl_df.pct_loo.mean()), 3)
    summary["control_full_mean_pct"] = round(float(ctrl_df.pct_full.mean()), 3)
    print(f"\nCONTROL leave-one-out GM %|zAI|>={Z_THR}: mean {summary['control_loo_mean_pct']:.1f}% "
          f"(full-stats {summary['control_full_mean_pct']:.1f}%; target ~5%)")
    if not args.skip_patients:
        print("\n=== patients (full control statistics) ===")
        pats = patient_rows(cfg)
        pat_df = pd.DataFrame(pats)
        pat_df.to_csv(cfg.calibration_dir / "patient_gm_calibration.csv", index=False)
        if len(pat_df):
            summary["n_patients"] = int(len(pat_df))
            summary["patient_mean_pct"] = round(float(pat_df.pct_abs_zai_ge_1_96.mean()), 3)
            summary["patient_median_pct"] = round(float(pat_df.pct_abs_zai_ge_1_96.median()), 3)
            print(f"PATIENT GM %|zAI|>={Z_THR}: mean {summary['patient_mean_pct']:.1f}% (n={len(pat_df)})")
    with open(cfg.calibration_dir / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCSVs -> {cfg.calibration_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
