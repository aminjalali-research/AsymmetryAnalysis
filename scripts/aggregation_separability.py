#!/usr/bin/env python3
"""Paired bootstrap test of whether the aggregation-method ranking is real.

Table `tab:performance_unilateral` orders the 9 aggregation strategies by AUC on the
n=13 unilateral cohort (temporal_signed_li / dominant_side 0.600 down to 0.500). This
script tests whether that ordering survives resampling.

All strategies score the *same* 13 patients, so their AUC errors are correlated and a
paired test on the difference cancels the shared patient-sampling noise. That is more
sensitive than checking whether the marginal CIs overlap, so a null result here is a
strong one.

Two conventions are inherited from 05_roi_discrimination.py so the numbers match the
manuscript:
  - Score orientation (the direct-vs-sign-flipped choice) is fixed on the full sample.
    Because that choice is made in-sample, AUC cannot fall below 0.500; 0.500 is a
    floor, not a neutral midpoint.
  - Cohort selection uses get_ez_label(..., include_bilateral_as="exclude").

Resampling is stratified on the 8 Left / 5 Right split: an unstratified bootstrap of
n=13 draws all-one-class resamples often enough to make AUC undefined.

Writes ez_analysis_mdt/aggregation_separability.csv (paired deltas vs the best method)
and ez_analysis_mdt/aggregation_marginal_ci.csv (per-method marginal CIs).

Usage:
    python scripts/aggregation_separability.py [--n-boot 10000] [--seed 20260714]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "ez_analysis_mdt"


def load_analyzer():
    """Import 05_roi_discrimination.py, whose leading digit blocks normal import."""
    sys.path.insert(0, str(PROJECT_ROOT))
    path = PROJECT_ROOT / "05_roi_discrimination.py"
    spec = importlib.util.spec_from_file_location("roi_discrimination", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_scores(module):
    """Return (y_true, {method: oriented_scores}) for the unilateral cohort."""
    analyzer = module.EZDiscriminatorMDT()
    analyzer.load_data()

    methods = {
        "temporal_signed_li": analyzer.temporal_signed_li,
        "dominant_side": analyzer.dominant_side_score,
        "cohen_d": analyzer.cohen_d_directional,
        "anatomically_weighted": analyzer.anatomically_weighted,
        "mean_li": analyzer.mean_li,
        "weighted_aai": analyzer.weighted_aai,
        "mesial_temporal_li": analyzer.mesial_temporal_li,
        "signed_weighted_li": analyzer.signed_weighted_li,
        "temporal_focused_score": analyzer.temporal_focused_score,
    }

    y_true, patient_ids = [], []
    for pid in analyzer.asymmetry_data:
        if pid not in module.EZ_GROUND_TRUTH:
            continue
        ez = module.get_ez_label(pid, include_bilateral_as="exclude")
        if ez is None:
            continue
        y_true.append(1 if ez == "L" else 0)
        patient_ids.append(pid)

    y = np.array(y_true)

    scores = {}
    for name, fn in methods.items():
        raw = np.array([fn(analyzer.asymmetry_data[p]) for p in patient_ids], float)
        # Fix orientation on the full sample, matching 05_roi_discrimination.py
        scores[name] = -raw if roc_auc_score(y, -raw) > roc_auc_score(y, raw) else raw

    return y, scores


def stratified_boot_indices(y, n_boot, rng):
    """Resample within each class so every draw keeps both L and R present."""
    idx_left = np.flatnonzero(y == 1)
    idx_right = np.flatnonzero(y == 0)
    return np.concatenate(
        [
            rng.choice(idx_left, (n_boot, len(idx_left)), replace=True),
            rng.choice(idx_right, (n_boot, len(idx_right)), replace=True),
        ],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    module = load_analyzer()
    y, scores = build_scores(module)
    print(f"\nCohort: n={len(y)}  Left={y.sum()}  Right={(1 - y).sum()}")

    rng = np.random.default_rng(args.seed)
    boot_idx = stratified_boot_indices(y, args.n_boot, rng)

    observed = {n: roc_auc_score(y, s) for n, s in scores.items()}
    boot_auc = {
        name: np.array([roc_auc_score(y[i], s[i]) for i in boot_idx])
        for name, s in scores.items()
    }

    ranked = sorted(observed, key=lambda k: -observed[k])
    best = ranked[0]

    marginal = pd.DataFrame(
        [
            {
                "method": name,
                "auc": observed[name],
                "ci_low": np.percentile(boot_auc[name], 2.5),
                "ci_high": np.percentile(boot_auc[name], 97.5),
            }
            for name in ranked
        ]
    )

    rows = []
    for name in ranked:
        if name == best:
            continue
        delta = boot_auc[best] - boot_auc[name]
        # Two-sided bootstrap p: how often the observed sign fails to hold
        p = min(2 * min((delta <= 0).mean(), (delta >= 0).mean()), 1.0)
        rows.append(
            {
                "reference": best,
                "comparator": name,
                "delta_auc": observed[best] - observed[name],
                "ci_low": np.percentile(delta, 2.5),
                "ci_high": np.percentile(delta, 97.5),
                "p_value": p,
                "separable": bool(
                    np.percentile(delta, 2.5) > 0 or np.percentile(delta, 97.5) < 0
                ),
            }
        )
    paired = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(exist_ok=True)
    marginal.to_csv(OUTPUT_DIR / "aggregation_marginal_ci.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "aggregation_separability.csv", index=False)

    print(f"\nMarginal AUC ({args.n_boot} stratified resamples)")
    for r in marginal.itertuples():
        print(f"  {r.method:24s} {r.auc:.3f}  [{r.ci_low:.2f}, {r.ci_high:.2f}]")

    print(f"\nPaired delta-AUC vs {best}")
    for r in paired.itertuples():
        print(
            f"  vs {r.comparator:24s} {r.delta_auc:+.3f}  "
            f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}]  p={r.p_value:.2f}"
        )

    n_sep = int(paired["separable"].sum())
    print(
        f"\n{n_sep}/{len(paired)} comparisons separable at 95%.\n"
        f"Wrote {OUTPUT_DIR / 'aggregation_separability.csv'}"
    )


if __name__ == "__main__":
    main()
