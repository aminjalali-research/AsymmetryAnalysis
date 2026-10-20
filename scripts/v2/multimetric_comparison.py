#!/usr/bin/env python3
"""Multi-metric comparison of lateralization methods, from a clinician's point of view.

For every method and dataset: how often a call is made (call rate), how often a call is
right (PPV), sensitivity for left and right separately, balanced accuracy counting no-call
as wrong, AUC of the continuous statistic (sign-free, bootstrap CI), positive likelihood
ratio of a call for the predicted side, Cohen's kappa, the control false-positive call rate,
and agreement in the strata that matter clinically (lesional, MRI-negative, temporal,
extratemporal, resection-defined). No single metric is used to rank; the table is printed
with all of them and a short composite summary (rank-sum over PPV, balanced accuracy,
control FP rate and MRI-negative agreement).

Inputs: ez_analysis_mdt/v2/<dataset>/{R_patients,V_patients,summary,R_loo_controls,V_loo_controls}.csv
        results_zscore(/external_chalfont/...)/clinical/cohort_concordance.csv (as-implemented rule)
        optional: as-implemented pseudo-control calls (chalfont: external_chalfont/results_zscore/lateralization/*)
Usage: python scripts/v2/multimetric_comparison.py --dataset inhouse|chalfont
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
import dataset_config  # noqa: E402

rng = np.random.default_rng(1)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def auc_sign_free(values, ez):
    """AUC for 'more negative = more left'; values nan-dropped; CI by bootstrap."""
    v = np.asarray(values, float); y = (np.asarray(ez) == "L").astype(int)
    ok = np.isfinite(v); v, y = v[ok], y[ok]
    if y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan, np.nan, np.nan
    def _auc(v, y):
        pos, neg = -v[y == 1], -v[y == 0]       # higher score = more left
        return (np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
    a = _auc(v, y); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(v), len(v))
        if y[i].sum() and (1 - y[i]).sum():
            bs.append(_auc(v[i], y[i]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def kappa(pred, truth):
    cats = ["L", "R"]; n = len(pred)
    po = np.mean(pred == truth)
    pe = sum(np.mean(pred == c) * np.mean(truth == c) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def metrics(df, strata, label, fp_rate=np.nan):
    """df: id, ez (L/R), call (L/R/none), value (continuous, may be nan)."""
    u = df[df.ez.isin(["L", "R"])].copy()
    n = len(u); nL = int((u.ez == "L").sum()); nR = n - nL
    called = u[u.call.isin(["L", "R"])]
    ok_all = (u.call == u.ez)
    sensL = ok_all[u.ez == "L"].mean(); sensR = ok_all[u.ez == "R"].mean()
    ppv = (called.call == called.ez).mean() if len(called) else np.nan
    # likelihood ratio of a left call for left EZ (and right call for right), pooled
    tp = ((called.call == called.ez)).sum(); fp = len(called) - tp
    sens_called = tp / n; fpr_called = fp / n
    lr_pos = (sens_called / fpr_called) if fpr_called > 0 else np.inf
    kap = kappa(called.call.values, called.ez.values) if len(called) > 1 else np.nan
    a, lo, hi = auc_sign_free(u.value, u.ez)
    row = dict(method=label, n=n, nL=nL, nR=nR,
               call_rate=round(len(called) / n, 2), ppv_of_calls=round(ppv, 2) if np.isfinite(ppv) else np.nan,
               n_calls=len(called), correct_calls=int(tp),
               sens_left=round(sensL, 2), sens_right=round(sensR, 2),
               balanced_acc=round(0.5 * (sensL + sensR), 3), agree_all=f"{int(ok_all.sum())}/{n}",
               wilson_lo=round(wilson(int(ok_all.sum()), n)[0], 2), wilson_hi=round(wilson(int(ok_all.sum()), n)[1], 2),
               majority=round(max(nL, nR) / n, 2),
               auc=round(a, 2) if np.isfinite(a) else np.nan, auc_lo=round(lo, 2) if np.isfinite(lo) else np.nan, auc_hi=round(hi, 2) if np.isfinite(hi) else np.nan,
               lr_pos=round(lr_pos, 1) if np.isfinite(lr_pos) else np.inf, kappa_calls=round(kap, 2) if np.isfinite(kap) else np.nan,
               control_fp_rate=round(fp_rate, 2) if np.isfinite(fp_rate) else np.nan)
    for sname, mask in strata.items():
        v = u[u.id.isin(mask)]
        if len(v) >= 4:
            o = (v.call == v.ez); L = v.ez == "L"; R = v.ez == "R"
            b = 0.5 * (o[L].mean() + o[R].mean()) if L.any() and R.any() else np.nan
            row[f"{sname}"] = f"{int(o.sum())}/{len(v)} ({b:.2f})" if np.isfinite(b) else f"{int(o.sum())}/{len(v)}"
    return row


def main():
    ap = argparse.ArgumentParser(); dataset_config.add_dataset_args(ap); a = ap.parse_args()
    cfg = dataset_config.configure(a.dataset, perf_variant=getattr(a, "perf_variant", "tc"), band_policy=getattr(a, "band_policy", None))
    gt = cfg.ground_truth(); v2 = BASE / "ez_analysis_mdt/v2" / cfg.name
    # strata membership
    ids = list(gt)
    if cfg.is_inhouse:
        ev = {i: str(gt[i].get("evidence", "")) for i in ids}
        lesional = [i for i in ids if "normal" not in ev[i].lower().split("|")[0]]
        negative = [i for i in ids if i not in lesional]
        temporal = [i for i in ids if "TLE" in str(gt[i].get("thesis_dx", ""))]
        extratemp = [i for i in ids if i not in temporal]
        resected = ["P016", "P017", "P023", "P025", "P026"]
    else:
        sub = pd.read_csv(cfg.base_dir / "external_chalfont/metadata/subjects.csv").set_index("id")
        lesional = [i for i in ids if sub.loc[i, "group"] == "discrete"]; negative = [i for i in ids if sub.loc[i, "group"] == "negative"]
        temporal = [i for i in ids if sub.loc[i, "ez_lobe"] == "temporal"]; extratemp = [i for i in ids if i not in temporal]
        resected = [i for i in ids if sub.loc[i, "reference_tier"] == "resected"]
    strata = {"lesional": lesional, "MRI_negative": negative, "temporal": temporal, "extratemporal": extratemp, "resection_defined": resected}
    rows = []
    # --- as-implemented voxel-wise rule (clinical tier) -----------------------------------
    cc = pd.read_csv(cfg.output_dir / "clinical" / "cohort_concordance.csv")
    cc["ez"] = cc.patient_id.map(lambda p: gt.get(p, {}).get("ez"))
    cc["call"] = cc.algo_prediction.where(cc.algo_prediction.isin(["L", "R"]), "none")
    cc["value"] = np.nan
    prt = cfg.analysis_dir / "clinical_production_rule_by_tier.csv"
    if prt.exists():   # continuous volume index of the production rule: vol_LI > 0 = more left volume = left call
        t = pd.read_csv(prt); t = t[t.tier == "clinical"].set_index("patient")
        cc["value"] = -cc.patient_id.map(t["vol_LI"])          # negative = left, as for the other statistics
    strong = cc.copy(); strong["call"] = np.where(cc.algo_confidence.str.startswith("Strong"), cc.call, "none")
    strongmod = cc.copy(); strongmod["call"] = np.where(cc.algo_confidence.str.startswith(("Strong", "Moderate")), cc.call, "none")
    fp_asis = fp_sm = fp_strong = np.nan
    pc = cfg.output_dir / "lateralization" / "eval_control_false_positives.csv"   # scripts/chalfont/evaluate_lateralization.py
    if pc.exists():
        d = pd.read_csv(pc).set_index("tier")
        if "clinical" in d.index:
            r = d.loc["clinical"]; fp_asis = float(r["fp_rate_any"]); fp_sm = float(r["fp_rate_strong_moderate"]); fp_strong = float(r["Strong"] / r["n_controls"])
    d0 = cc.rename(columns={"patient_id": "id"})[["id", "ez", "call", "value"]]
    rows.append(metrics(d0, strata, "voxel rule, clinical tier (any call)", fp_asis))
    rows.append(metrics(strongmod.rename(columns={"patient_id": "id"})[["id", "ez", "call", "value"]], strata, "voxel rule, strong+moderate calls only", fp_sm))
    rows.append(metrics(strong.rename(columns={"patient_id": "id"})[["id", "ez", "call", "value"]], strata, "voxel rule, strong calls only", fp_strong))
    # --- enhanced estimators ---------------------------------------------------------
    R = pd.read_csv(v2 / "R_patients.csv"); V = pd.read_csv(v2 / "V_patients.csv"); S = pd.read_csv(v2 / "summary.csv").set_index("statistic")
    lR = pd.read_csv(v2 / "R_loo_controls.csv"); lV = pd.read_csv(v2 / "V_loo_controls.csv")
    for src, loo, stat, label in [(R, lR, "R_temporal", "regional temporal z"), (R, lR, "R_mesial", "regional mesial z"),
                                  (R, lR, "R_all37", "regional all-37 z"), (V, lV, "V_all_t2", "smoothed voxel z-mass (t=2)")]:
        thr = float(S.loc[stat, "threshold"]); fp = float((np.abs(loo[stat]) >= thr).mean())
        d = src[["id", "ez", stat]].rename(columns={stat: "value"})
        d_sign = d.copy(); d_sign["call"] = np.where(d.value < 0, "L", "R")
        d_cal = d.copy(); d_cal["call"] = np.where(np.abs(d.value) >= thr, np.where(d.value < 0, "L", "R"), "none")
        rows.append(metrics(d_sign, strata, f"{label}, sign (always calls)", 1.0))
        rows.append(metrics(d_cal, strata, f"{label}, calibrated call (5% FP)", fp))
    # majority baseline
    base = d0.copy(); base["call"] = "L" if (base.ez == "L").sum() >= (base.ez == "R").sum() else "R"
    rows.append(metrics(base, strata, "always majority side", 1.0))
    T = pd.DataFrame(rows)
    for c in ["lesional", "MRI_negative", "temporal", "extratemporal", "resection_defined"]:
        if c not in T:
            T[c] = "n<4"
        T[c] = T[c].fillna("n<4")
    # Pareto view (clinician's trade-off): a method is Pareto-optimal if no other method has both a
    # higher call rate and a higher PPV of calls (ties allowed); the composite rank-sum is reported
    # only as a secondary, equal-weight summary over PPV, balanced accuracy, control FP, MRI-negative
    # agreement and AUC.
    cr, pv = T.call_rate.values, T.ppv_of_calls.fillna(0).values
    T["pareto_calls_vs_ppv"] = [not any((cr[j] >= cr[i]) and (pv[j] >= pv[i]) and ((cr[j] > cr[i]) or (pv[j] > pv[i])) for j in range(len(T))) for i in range(len(T))]
    def frac(sr): return sr.astype(str).str.extract(r"(\d+)/(\d+)").astype(float).pipe(lambda x: x[0] / x[1])
    comp = pd.DataFrame({"ppv": -T.ppv_of_calls.fillna(0), "bal": -T.balanced_acc, "fp": T.control_fp_rate.fillna(1.0),
                         "neg": -frac(T["MRI_negative"]).fillna(0), "auc": -T.auc.fillna(0.5)})
    T["composite_rank"] = comp.rank().sum(axis=1).rank(method="min").astype(int)
    out = v2 / "multimetric_comparison.csv"; T.to_csv(out, index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
    cols = ["method", "n", "call_rate", "n_calls", "correct_calls", "ppv_of_calls", "sens_left", "sens_right", "balanced_acc", "majority", "auc", "auc_lo", "auc_hi", "lr_pos", "kappa_calls", "control_fp_rate", "lesional", "MRI_negative", "temporal", "extratemporal", "resection_defined", "pareto_calls_vs_ppv", "composite_rank"]
    print(f"\n=== {cfg.name}: multi-metric comparison (n={T.n.iloc[0]}; strata show agree/n (balanced acc)) ===")
    print(T[cols].sort_values(["call_rate", "ppv_of_calls"], ascending=[False, False]).to_string(index=False))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
