#!/usr/bin/env python3
"""Compute Pipeline-B localization performance against MDT-defined target lobes.

For each of the five F_20_39-matched patients (P013, P014, P015, P020, P026),
load the clinical cluster report and compute:
  - N_clusters (total surviving)
  - N_target_clusters (located in target temporal lobe)
  - target_hit_rate (fraction of clusters in target lobe)
  - target_voxel_fraction (voxels in target lobe / total surviving voxels)
  - top_cluster_in_target (1 if largest cluster is in target lobe, else 0)
  - top_cluster_region (primary_region of the largest cluster)

Target lobes (MDT-defined):
  P013, P015, P020, P026 -> Left temporal
  P014                   -> Right temporal

Output: ez_analysis_mdt/localization_performance.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

L_TEMPORAL = {
    "L-bankssts", "L-entorhinal", "L-fusiform", "L-inf.temporal",
    "L-mid.temporal", "L-parahippo", "L-sup.temporal", "L-temporalpole",
    "L-transv.temp", "L-Hippocampus", "L-Amygdala",
}
R_TEMPORAL = {s.replace("L-", "R-") for s in L_TEMPORAL}

TARGETS = {
    "P013": ("L-temporal", L_TEMPORAL),
    "P014": ("R-temporal", R_TEMPORAL),
    "P015": ("L-temporal", L_TEMPORAL),
    "P020": ("L-temporal", L_TEMPORAL),
    "P026": ("L-temporal", L_TEMPORAL),
}


def main() -> None:
    rows: list[dict] = []
    for pid, (target_name, target_regions) in TARGETS.items():
        csv_path = ROOT / "results_zscore" / "clinical" / pid / f"{pid}_clinical_cluster_report.csv"
        df = pd.read_csv(csv_path)
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
                "n_clusters": n_total,
                "n_target_clusters": n_in_target,
                "target_hit_rate": round(n_in_target / n_total, 3) if n_total else float("nan"),
                "total_voxels": v_total,
                "target_voxels": v_in_target,
                "target_voxel_fraction": round(v_in_target / v_total, 3) if v_total else float("nan"),
                "top_cluster_region": top_region,
                "top_cluster_in_target": int(top_in_target),
                "top_cluster_voxels": int(top["size_voxels"]),
                "top_cluster_peak_z": round(float(top["peak_z"]), 2),
            }
        )

    df = pd.DataFrame(rows)
    out = ROOT / "ez_analysis_mdt" / "localization_performance.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.to_string(index=False))

    print()
    print("=== Cohort summary (n=5) ===")
    print(f"Mean target_hit_rate       : {df['target_hit_rate'].mean():.3f}")
    print(f"Mean target_voxel_fraction : {df['target_voxel_fraction'].mean():.3f}")
    print(f"Top-cluster in target      : {df['top_cluster_in_target'].sum()}/{len(df)}")


if __name__ == "__main__":
    main()
