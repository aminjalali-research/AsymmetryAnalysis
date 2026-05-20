#!/usr/bin/env python3
"""Compute per-asymmetry-index left-vs-right EZ discrimination performance.

For each of the 15 individual asymmetry indices already computed per-region per-patient
(in results_analysis/<PID>/<PID>_comprehensive_asymmetry_analysis.csv), aggregate
across the 37 bilateral region pairs using magnitude-weighted self-weighting:

    S(patient) = sum_i (A_i * |A_i|) / sum_i |A_i|

Then run ROC analysis against MDT ground truth (Left = positive class) for both
the unilateral-only (n=12) and unilateral + bilateral-dominant (n=14) configurations.

Writes:
  ez_analysis_mdt/per_index_performance_exclude.csv   (n=12)
  ez_analysis_mdt/per_index_performance_dominant.csv  (n=14)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ez_ground_truth import EZ_GROUND_TRUTH  # noqa: E402

INDEX_COLUMNS = [
    "laterality_index",
    "absolute_asymmetry_index",
    "percent_difference",
    "log_ratio",
    "coefficient_of_asymmetry",
    "simple_difference",
    "ratio",
    "cohen_d_asymmetry",
    "hedges_g_asymmetry",
    "glass_delta_asymmetry",
    "zscore_asymmetry",
    "robust_zscore_asymmetry",
    "cv_ratio",
    "hyperperfusion_ratio",
    "normalized_difference",
]

# Indices centred at 1 rather than 0 — shift before magnitude-weighting so
# that the sign convention is consistent with the other (zero-centred) indices.
SHIFT_BY_ONE = {"ratio", "cv_ratio", "hyperperfusion_ratio"}


def magnitude_weighted(values: np.ndarray) -> float:
    abs_sum = float(np.nansum(np.abs(values)))
    if abs_sum == 0.0:
        return 0.0
    return float(np.nansum(values * np.abs(values)) / abs_sum)


def patient_scores(index_col: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for pid in sorted(EZ_GROUND_TRUTH):
        csv_path = ROOT / "results_analysis" / pid / f"{pid}_comprehensive_asymmetry_analysis.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if index_col not in df.columns:
            continue
        vals = df[index_col].to_numpy(dtype=float)
        if index_col in SHIFT_BY_ONE:
            vals = vals - 1.0
        scores[pid] = magnitude_weighted(vals)
    return scores


def confusion(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    """Return (sens, spec, ppv, npv, acc, correct, total) treating positive=Left (label 1)."""
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n_pos = tp + fn
    n_neg = tn + fp
    sens = tp / n_pos if n_pos else float("nan")
    spec = tn / n_neg if n_neg else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    correct = tp + tn
    total = correct + fp + fn
    acc = correct / total if total else float("nan")
    return sens, spec, ppv, npv, acc, correct, total


def youden_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    # sweep candidate thresholds at every observed score; return the one
    # that maximises Youden's J = sens + spec - 1.
    best_j = -np.inf
    best_thr = float(np.median(scores))
    candidates = np.unique(scores)
    for thr in candidates:
        sens, spec, *_ = confusion(y_true, scores, thr)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thr = thr
    return best_thr


def evaluate_configuration(include_bilateral: bool, output_csv: Path) -> None:
    if include_bilateral:
        # B-L counts as Left, B-R counts as Right; B (no-dominance) excluded.
        label_map = {"L": 1, "R": 0, "B-L": 1, "B-R": 0}
    else:
        label_map = {"L": 1, "R": 0}

    rows: list[dict] = []
    for col in INDEX_COLUMNS:
        scores_by_pid = patient_scores(col)
        pids: list[str] = []
        y_true_list: list[int] = []
        score_list: list[float] = []
        for pid, info in EZ_GROUND_TRUTH.items():
            ez = info["ez"]
            if ez not in label_map:
                continue
            if pid not in scores_by_pid:
                continue
            pids.append(pid)
            y_true_list.append(label_map[ez])
            score_list.append(scores_by_pid[pid])
        if len(pids) == 0:
            continue
        y_true = np.array(y_true_list)
        scores = np.array(score_list)

        # try both polarities (some indices may be inverted vs Left=positive)
        try:
            auc_native = roc_auc_score(y_true, scores)
        except ValueError:
            auc_native = float("nan")
        auc_flipped = 1.0 - auc_native if not np.isnan(auc_native) else float("nan")
        flipped = (auc_flipped > auc_native) if not np.isnan(auc_native) else False
        scores_oriented = -scores if flipped else scores
        auc = max(auc_native, auc_flipped) if not np.isnan(auc_native) else float("nan")

        thr = youden_optimal_threshold(y_true, scores_oriented)
        sens, spec, ppv, npv, acc, correct, total = confusion(y_true, scores_oriented, thr)

        rows.append(
            {
                "index": col,
                "n_patients": len(pids),
                "auc": auc,
                "sensitivity": sens,
                "specificity": spec,
                "ppv": ppv,
                "npv": npv,
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "threshold": thr,
                "flipped": flipped,
            }
        )

    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}  ({len(df)} indices)")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    out_dir = ROOT / "ez_analysis_mdt"
    evaluate_configuration(False, out_dir / "per_index_performance_exclude.csv")
    print()
    evaluate_configuration(True, out_dir / "per_index_performance_dominant.csv")


if __name__ == "__main__":
    main()
