#!/usr/bin/env python3
"""Compute Pipeline-B localization performance against MDT-defined target lobes.

For each unilateral-EZ patient (per ez_ground_truth.get_unilateral_patients;
13 patients in the n=17 cohort), load the 02_compute_zai.py cluster report
(results_zscore/asymmetry/patients/<PID>/<PID>_asymmetry_cluster_report.csv)
and compute, over ONE member per mirrored pair (see below):
  - N_clusters (total surviving)
  - N_target_clusters (located in target temporal lobe)
  - target_hit_rate (fraction of clusters in target lobe)
  - target_voxel_fraction (voxels in target lobe / total surviving voxels)
  - top_cluster_in_target (1 if largest cluster is in target lobe, else 0)
  - top_cluster_region (primary_region of the largest cluster)

Mirrored-pair handling: the voxel zAI map is exactly antisymmetric
(zAI(v) = -zAI(mirror v)), so every abnormality appears as two mirror clusters,
one per hemisphere. Each pair is counted once by keeping the member whose
cluster_hemi equals ez_side_pred (the side the direction-of-abnormality
inference attributes the abnormality to). Hemisphere-specific region labels
(L-/R-) are therefore meaningful: a hit means the abnormality was attributed to
the target lobe on the target side.

Target lobes: the MDT-defined EZ side's temporal lobe (L or R), for every
unilateral patient except P025 (parietal; excluded because the metric is
temporal-lobe-scoped).

Output: ez_analysis_mdt/localization_performance.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ez_ground_truth import EZ_GROUND_TRUTH, get_unilateral_patients  # noqa: E402

L_TEMPORAL = {
    "L-bankssts", "L-entorhinal", "L-fusiform", "L-inf.temporal",
    "L-mid.temporal", "L-parahippo", "L-sup.temporal", "L-temporalpole",
    "L-transv.temp", "L-Hippocampus", "L-Amygdala",
}
R_TEMPORAL = {s.replace("L-", "R-") for s in L_TEMPORAL}


def _targets() -> dict:
    """Per-patient target temporal lobe from MDT ground truth (unilateral only).

    Note: P025 is parietal (LPLE), not temporal — its temporal-lobe target is a
    known mismatch and it is excluded here (localization is temporal-lobe-scoped).
    """
    out = {}
    for pid in sorted(get_unilateral_patients()):
        ez = EZ_GROUND_TRUTH[pid]["ez"]
        if pid == "P025":  # parietal localization, not temporal
            continue
        if ez == "L":
            out[pid] = ("L-temporal", L_TEMPORAL)
        else:
            out[pid] = ("R-temporal", R_TEMPORAL)
    return out


TARGETS = _targets()


def main() -> None:
    rows: list[dict] = []
    for pid, (target_name, target_regions) in TARGETS.items():
        csv_path = (ROOT / "results_zscore" / "asymmetry" / "patients" / pid
                    / f"{pid}_asymmetry_cluster_report.csv")
        if not csv_path.exists():
            print(f"  [skip] {pid}: {csv_path.name} not present (run 02_compute_zai.py)")
            continue
        df = pd.read_csv(csv_path)
        if not {"cluster_hemi", "ez_side_pred"}.issubset(df.columns):
            print(f"  [skip] {pid}: cluster report lacks cluster_hemi/ez_side_pred "
                  f"(legacy schema; rerun 02_compute_zai.py)")
            continue
        n_raw = len(df)
        # One member per mirrored pair: keep the cluster on the attributed side.
        df = df[df["cluster_hemi"].astype(str) == df["ez_side_pred"].astype(str)]
        if df.empty:
            print(f"  [skip] {pid}: no clusters after one-per-pair filter (raw {n_raw})")
            continue
        peak_col = "peak_zai" if "peak_zai" in df.columns else "peak_z"
        n_total = len(df)
        in_target = df["primary_region"].isin(target_regions)
        n_in_target = int(in_target.sum())
        v_total = int(df["size_voxels"].sum())
        v_in_target = int(df.loc[in_target, "size_voxels"].sum())
        # largest cluster (by size_voxels)
        top = df.sort_values("size_voxels", ascending=False).iloc[0]
        top_region = top["primary_region"]
        top_in_target = top_region in target_regions

        rows.append(
            {
                "patient": pid,
                "target_lobe": target_name,
                "n_clusters_raw": n_raw,
                "n_clusters": n_total,
                "n_target_clusters": n_in_target,
                "target_hit_rate": round(n_in_target / n_total, 3) if n_total else float("nan"),
                "total_voxels": v_total,
                "target_voxels": v_in_target,
                "target_voxel_fraction": round(v_in_target / v_total, 3) if v_total else float("nan"),
                "top_cluster_region": top_region,
                "top_cluster_in_target": int(top_in_target),
                "top_cluster_voxels": int(top["size_voxels"]),
                "top_cluster_peak_z": round(float(top[peak_col]), 2),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No zAI cluster reports available - run 02_compute_zai.py first.")
        return
    out = ROOT / "ez_analysis_mdt" / "localization_performance.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.to_string(index=False))

    print()
    print(f"=== Cohort summary (n={len(df)} of {len(TARGETS)} unilateral targets) ===")
    print(f"Mean target_hit_rate       : {df['target_hit_rate'].mean():.3f}")
    print(f"Mean target_voxel_fraction : {df['target_voxel_fraction'].mean():.3f}")
    print(f"Top-cluster in target      : {df['top_cluster_in_target'].sum()}/{len(df)}")


if __name__ == "__main__":
    main()
