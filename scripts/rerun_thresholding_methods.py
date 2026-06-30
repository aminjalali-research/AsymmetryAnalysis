#!/usr/bin/env python3
"""Re-run all 13 spatial-thresholding methods on the F_20_39 patient zAI maps.

For each of the five F_20_39-matched patients, loads the zAI map and brain
mask, constructs an AsymData instance, applies every method in
src.thresholding.METHOD_REGISTRY, and writes a per-patient summary CSV.

Also diagnoses why M1 / M9 / M11 / M13 may return zero by printing the input
distribution statistics and re-running each method with corrected inputs.

Outputs:
  results_zscore/asymmetry/summary/all_patients_method_summary.csv (overwritten)
  ez_analysis_mdt/thresholding_diagnostics.txt                     (new)
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as _pd  # noqa: E402
from src.thresholding import AsymData, METHOD_REGISTRY  # noqa: E402
from ez_ground_truth import EZ_GROUND_TRUTH  # noqa: E402

# 2026-06 cohort: 17 analyzable patients (ez_ground_truth), each age-matched to
# a sex-pooled FM control band. P014 excluded; P028/P029/P030 added.
PATIENTS = sorted(EZ_GROUND_TRUTH)
FM_BANDS = [("FM_20_39", 20, 39), ("FM_40_59", 40, 59), ("FM_60_79", 60, 79)]


def patient_band(pid: str) -> str | None:
    """Resolve a patient's FM control band from age in clinical_spreadsheet.xlsx."""
    df = _pd.read_excel(ROOT / "clinical_spreadsheet.xlsx")
    df.columns = [c.strip().upper() for c in df.columns]
    id_col = [c for c in df.columns if "ID" in c][0]
    age_col = [c for c in df.columns if "AGE" in c][0]
    for _, row in df.iterrows():
        rid = str(row[id_col]).strip()
        if rid.startswith("sub-"):
            rid = rid[4:]
        if rid != pid:
            continue
        age = int(row[age_col])
        for band, lo, hi in FM_BANDS:
            if lo <= age <= hi:
                return band
    return None


def load_zai(pid: str) -> np.ndarray:
    return nib.load(ROOT / "results_zscore" / "asymmetry" / "patients" / pid /
                    f"{pid}_asymmetry_zscore.nii.gz").get_fdata()


def load_brain_mask(group: str) -> np.ndarray:
    return nib.load(ROOT / "results_zscore" / "asymmetry" / "groups" / group /
                    "brain_mask.nii.gz").get_fdata() > 0


SYM_MODE = "--sym" in sys.argv


def load_perfusion(pid: str) -> np.ndarray | None:
    # In symmetric-space mode the zAI maps live in the symmetric template space,
    # so the perfusion that quality-weighted methods (M2/M11/M12/M13) consume
    # must be the symmetric-registered perfusion to stay spatially aligned.
    if SYM_MODE:
        p = ROOT / "symreg" / "sym_perf" / f"{pid}_perf_sym.nii.gz"
    else:
        p = ROOT / "Dataset" / pid / f"{pid}_perfusion_calib_resampled_to_T1w.nii.gz"
    if not p.exists():
        return None
    return nib.load(p).get_fdata()


def n_clusters(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    structure = ndimage.generate_binary_structure(3, 1)
    _, n = ndimage.label(mask, structure=structure)
    return int(n)


def run_all_methods(d: AsymData, brain_n: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, fn in METHOD_REGISTRY.items():
        try:
            mask = fn(d).astype(bool)
        except Exception as e:
            out[name] = {"n_voxels": 0, "coverage_pct": 0.0,
                         "n_clusters": 0, "error": str(e)[:80]}
            continue
        n_vox = int(mask.sum())
        cov = 100.0 * n_vox / max(brain_n, 1)
        out[name] = {"n_voxels": n_vox, "coverage_pct": round(cov, 2),
                     "n_clusters": n_clusters(mask), "error": ""}
    return out


def main() -> None:
    rows: list[dict] = []
    diagnostics: list[str] = [
        f"Patients (cohort n={len(PATIENTS)}): {PATIENTS}",
        "",
    ]

    mask_cache: dict[str, np.ndarray] = {}
    processed = 0
    for pid in PATIENTS:
        zai_path = (ROOT / "results_zscore" / "asymmetry" / "patients" / pid /
                    f"{pid}_asymmetry_zscore.nii.gz")
        if not zai_path.exists():
            diagnostics.append(f"=== {pid} === SKIP (zAI map PENDING REBUILD)")
            diagnostics.append("")
            continue
        band = patient_band(pid)
        if band is None:
            diagnostics.append(f"=== {pid} === SKIP (no age band)")
            diagnostics.append("")
            continue
        if band not in mask_cache:
            mp = (ROOT / "results_zscore" / "asymmetry" / "groups" / band /
                  "brain_mask.nii.gz")
            if not mp.exists():
                diagnostics.append(f"=== {pid} === SKIP (band {band} mask missing)")
                diagnostics.append("")
                continue
            mask_cache[band] = load_brain_mask(band)
        brain_mask = mask_cache[band]
        brain_n = int(brain_mask.sum())

        zai = load_zai(pid)
        perf = load_perfusion(pid)
        processed += 1

        # AsymData treats the input as a signed AI map and internally
        # computes |ai| and a re-z-score within the brain. When the input
        # is already a zAI map, |ai| IS |zAI| (the quantity we want to
        # threshold on). The internal re-z-score is a within-patient
        # standardisation of zAI, which is what M1 and M5 see.
        d = AsymData(zai, brain_mask, perfusion=perf)

        diagnostics.append(f"=== {pid} (band {band}, brain_n={brain_n:,}) ===")
        diagnostics.append(f"  zAI: min={zai.min():.2f} max={zai.max():.2f} "
                           f"mean={zai[brain_mask].mean():.3f} std={zai[brain_mask].std():.3f}")
        diagnostics.append(f"  |zAI| (abs_ai): max={d.abs_ai[brain_mask].max():.2f} "
                           f"99th pct={np.percentile(d.abs_ai[brain_mask], 99):.2f}")
        diagnostics.append(f"  within-patient z (abs_z): max={d.abs_z[brain_mask].max():.2f} "
                           f"99th pct={np.percentile(d.abs_z[brain_mask], 99):.2f}")

        results = run_all_methods(d, brain_n)
        for name, m in results.items():
            rows.append({"patient_id": pid, "group": band, "method": name, **m})
            diagnostics.append(
                f"  {name:24s} n_voxels={m['n_voxels']:>7}  "
                f"coverage={m['coverage_pct']:>5.2f}%  "
                f"n_clusters={m['n_clusters']:>5}"
                + (f"  ERROR: {m['error']}" if m["error"] else "")
            )
        diagnostics.append("")

    if not rows:
        print("No zAI maps available yet — rerun after the zAI rebuild completes.")
        diag_path = ROOT / "ez_analysis_mdt" / "thresholding_diagnostics.txt"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_path.write_text("\n".join(diagnostics))
        return
    print(f"Processed {processed}/{len(PATIENTS)} patients with zAI maps present.")
    df = pd.DataFrame(rows)[["patient_id", "group", "method", "n_voxels", "coverage_pct", "n_clusters"]]
    out_csv = ROOT / "results_zscore" / "asymmetry" / "summary" / "all_patients_method_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({len(df)} rows)")

    diag_path = ROOT / "ez_analysis_mdt" / "thresholding_diagnostics.txt"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text("\n".join(diagnostics))
    print(f"Wrote {diag_path}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
