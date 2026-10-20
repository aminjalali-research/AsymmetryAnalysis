#!/usr/bin/env python3
"""Rebuild the shared summary CSVs from per-subject outputs after a parallel run.

* ``<results>/summary/all_patients_zscore_summary.csv`` from
  ``patients/<ID>/<ID>_vs_<band>_summary.json`` (01_build_control_normative.py)
* ``<results>/asymmetry/pseudo_controls/<band>_pseudo_control_summary.csv`` from
  ``pseudo_controls/<ID>/<ID>_asymmetry_summary.json`` + ``<ID>_pseudo_control_meta.json``

    python scripts/chalfont/collect_summaries.py --dataset chalfont [--perf-variant flow] [--band-policy specific]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import dataset_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dataset_config.add_dataset_args(ap)
    args = ap.parse_args(argv)
    cfg = dataset_config.configure(args.dataset, perf_variant=args.perf_variant, band_policy=args.band_policy)

    rows = []
    pdir = cfg.output_dir / "patients"
    if pdir.exists():
        for d in sorted(pdir.iterdir()):
            for js in sorted(d.glob(f"{d.name}_vs_*_summary.json")):
                rows.append(json.loads(js.read_text()))
    if rows:
        out = cfg.output_dir / "summary" / "all_patients_zscore_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{len(rows)} patient z-score summaries -> {out}")

    pc = cfg.pseudo_control_dir
    if pc.exists():
        by_band: dict[str, list] = {}
        for d in sorted(pc.iterdir()):
            if not d.is_dir():
                continue
            sj, mj = d / f"{d.name}_asymmetry_summary.json", d / f"{d.name}_pseudo_control_meta.json"
            if not (sj.exists() and mj.exists()):
                continue
            row = json.loads(sj.read_text())
            row.update(json.loads(mj.read_text()))
            by_band.setdefault(row["group"], []).append(row)
        for band, rws in by_band.items():
            out = pc / f"{band}_pseudo_control_summary.csv"
            pd.DataFrame(rws).to_csv(out, index=False)
            print(f"{len(rws)} pseudo-controls in {band} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
