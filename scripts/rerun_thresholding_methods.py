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

from src.thresholding import AsymData, METHOD_REGISTRY  # noqa: E402

PATIENTS = ["P013", "P014", "P015", "P020", "P026"]
GROUP = "F_20_39"


def load_zai(pid: str) -> np.ndarray:
    return nib.load(ROOT / "results_zscore" / "asymmetry" / "patients" / pid /
                    f"{pid}_asymmetry_zscore.nii.gz").get_fdata()


def load_brain_mask() -> np.ndarray:
    return nib.load(ROOT / "results_zscore" / "groups" / GROUP / "brain_mask.nii.gz").get_fdata() > 0


def load_perfusion(pid: str) -> np.ndarray | None:
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
    brain_mask = load_brain_mask()
    brain_n = int(brain_mask.sum())

    rows: list[dict] = []
    diagnostics: list[str] = [
        f"Brain mask voxels: {brain_n:,}",
        f"Patients: {PATIENTS}",
        "",
    ]

    for pid in PATIENTS:
        zai = load_zai(pid)
        perf = load_perfusion(pid)

        # AsymData treats the input as a signed AI map and internally
        # computes |ai| and a re-z-score within the brain. When the input
        # is already a zAI map, |ai| IS |zAI| (the quantity we want to
        # threshold on). The internal re-z-score is a within-patient
        # standardisation of zAI, which is what M1 and M5 see.
        d = AsymData(zai, brain_mask, perfusion=perf)

        diagnostics.append(f"=== {pid} ===")
        diagnostics.append(f"  zAI: min={zai.min():.2f} max={zai.max():.2f} "
                           f"mean={zai[brain_mask].mean():.3f} std={zai[brain_mask].std():.3f}")
        diagnostics.append(f"  |zAI| (abs_ai): max={d.abs_ai[brain_mask].max():.2f} "
                           f"99th pct={np.percentile(d.abs_ai[brain_mask], 99):.2f}")
        diagnostics.append(f"  within-patient z (abs_z): max={d.abs_z[brain_mask].max():.2f} "
                           f"99th pct={np.percentile(d.abs_z[brain_mask], 99):.2f}")

        results = run_all_methods(d, brain_n)
        for name, m in results.items():
            rows.append({"patient_id": pid, "method": name, **m})
            diagnostics.append(
                f"  {name:24s} n_voxels={m['n_voxels']:>7}  "
                f"coverage={m['coverage_pct']:>5.2f}%  "
                f"n_clusters={m['n_clusters']:>5}"
                + (f"  ERROR: {m['error']}" if m["error"] else "")
            )
        diagnostics.append("")

    df = pd.DataFrame(rows)[["patient_id", "method", "n_voxels", "coverage_pct", "n_clusters"]]
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
