#!/usr/bin/env python3
"""Leave-one-out cross-validated accuracy of the regional (Pipeline A) aggregation scores.

For each held-out patient, the score orientation (direct or sign-flipped) and the
Youden-optimal threshold are fitted on the remaining 12 unilateral patients and
applied to the held-out one. This removes the optimism of the in-sample AUC,
orientation choice and threshold reported in the main ROC table.
Writes ez_analysis_mdt/regional_loocv.csv.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sep", ROOT / "scripts/aggregation_separability.py")
sep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sep)


def fit(y, s):
    if roc_auc_score(y, -s) > roc_auc_score(y, s):
        s, sign = -s, -1
    else:
        sign = 1
    fpr, tpr, thr = roc_curve(y, s)
    t = thr[np.argmax(tpr - fpr)]
    return sign, t


def main():
    module = sep.load_analyzer()
    y, oriented = sep.build_scores(module)
    rows = []
    for name, s_or in oriented.items():
        # build_scores returns full-sample-oriented scores; refit orientation inside each fold
        s = s_or.copy()
        correct = 0
        for i in range(len(y)):
            m = np.arange(len(y)) != i
            sign, t = fit(y[m], s[m])
            pred = int(sign * s[i] >= t)
            correct += int(pred == y[i])
        rows.append(dict(method=name, loocv_correct=correct, n=len(y), loocv_accuracy=correct / len(y)))
    df = pd.DataFrame(rows).sort_values("loocv_correct", ascending=False)
    df.to_csv(ROOT / "ez_analysis_mdt/regional_loocv.csv", index=False)
    print(df.to_string(index=False))
    print(f"majority-class (always Left) baseline: {int(y.sum())}/{len(y)}")


if __name__ == "__main__":
    main()
