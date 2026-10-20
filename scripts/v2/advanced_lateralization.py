#!/usr/bin/env python3
"""EXPLORATORY SPIKE (2026-10-20): covariate-adjusted regional normative model + antisymmetric
multivariate classifier + lobe-level localization.

Why: the regional estimator of analyze_v2.py scores one hand-picked region set at a time and
ignores covariates. Here
  1. Normative model per feature: linear regression on the controls of the same cohort with
     covariates (age, sex, global GM CBF when available); the patient deviation is the residual
     divided by the residual SD (z). In-house controls carry only a band label, so the model
     reduces to the band mean/SD there.
  2. Features: z of LI_i (37 pairs) and z of the per-side difference of GM-normalised CBF
     (relL_i - relR_i, 37 pairs) -> 74 antisymmetric features (mirroring a brain negates them).
  3. Classifier: L2 logistic regression WITHOUT intercept on mirror-augmented data (x, y) and
     (-x, -y): the decision function is exactly antisymmetric, so it cannot learn the majority
     side and left/right are treated identically. C chosen by inner 5-fold CV; evaluated by
     leave-one-patient-out (outer) on Chalfont (93 unilateral), and zero-shot on the in-house
     cohort (model fitted on all Chalfont patients; in-house features z-scored against HCP bands).
  4. Calibrated calls: the classifier score of each control (leave-one-out normative, model
     fitted on all patients) gives the healthy range; a call needs |score| above its 95th pct.
  5. Localization: on the predicted side, the lobe with the most negative mean z_LI
     (temporal, frontal, parietal, occipital, insula/cingulate) vs ez_lobe (Chalfont).
Outputs: ez_analysis_mdt/v2/<dataset>/advanced_*.csv and a printed summary.
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(BASE))
import dataset_config  # noqa: E402
from scripts.v2.analyze_v2 import PAIRS, TEMPORAL, load_roi, li_vector, rel_side_vector, wilson  # noqa: E402

LOBE = {"temporal": ["hippocampus", "amygdala", "ctx6", "ctx7", "ctx9", "ctx15", "ctx16", "ctx30", "ctx33", "ctx34", "ctx1"],
        "frontal": ["ctx3", "ctx12", "ctx14", "ctx17", "ctx18", "ctx19", "ctx20", "ctx24", "ctx27", "ctx28", "ctx32"],
        "parietal": ["ctx8", "ctx22", "ctx25", "ctx29", "ctx31"],
        "occipital": ["ctx5", "ctx11", "ctx13", "ctx21"],
        "insula_cingulate": ["ctx2", "ctx10", "ctx23", "ctx26", "ctx35"]}
NAMES = [n for _, _, n in PAIRS]


def feature_frame(feat, ids):
    rows = []
    for s in ids:
        r = load_roi(feat, s); li = li_vector(r); rel = rel_side_vector(r)
        row = {f"li_{n}": li[n] for n in NAMES}
        row.update({f"d_{n}": rel[f"{n}_L"] - rel[f"{n}_R"] for n in NAMES})
        row["gm_mean"] = r["gm_mean"]; rows.append(row)
    return pd.DataFrame(rows, index=ids)


def covariate_z(ctrl_X, ctrl_cov, X, cov, loo=None):
    """z of each column of X vs controls after regressing out cov (columns age, sex, gm) if given."""
    Xc = ctrl_X.drop(index=loo) if loo is not None else ctrl_X
    Cc = ctrl_cov.drop(index=loo) if loo is not None else ctrl_cov
    out = pd.DataFrame(index=X.index, columns=Xc.columns, dtype=float)
    if Cc is None or Cc.shape[1] == 0:
        return (X - Xc.mean()) / Xc.std(ddof=1)
    mu = Cc.mean(); Cc = Cc.fillna(mu); cov = cov.fillna(mu)            # missing covariates -> control mean
    sdv = Cc.std(ddof=1).replace(0, 1.0); Cc = (Cc - mu) / sdv; cov = (cov - mu) / sdv
    A = np.column_stack([np.ones(len(Cc)), Cc.values]); Ap = np.column_stack([np.ones(len(cov)), cov.values])
    for col in Xc.columns:
        y = Xc[col].values; ok = np.isfinite(y)
        if ok.sum() < 10:
            out[col] = np.nan; continue
        beta, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
        res = y[ok] - A[ok] @ beta; sd = res.std(ddof=A.shape[1])
        out[col] = (X[col].values - Ap @ beta) / sd
    return out


def fit_antisym(X, y, C):
    Xa = np.vstack([X, -X]); ya = np.concatenate([y, -y])
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=2000); m.fit(Xa, ya); return m


def choose_C(X, y, Cs=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    best, bestC = -1, Cs[0]
    for C in Cs:
        acc = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            m = fit_antisym(X[tr], y[tr], C); s = m.decision_function(X[te])
            acc.append(np.mean(np.sign(s) == y[te]))
        if np.mean(acc) > best:
            best, bestC = np.mean(acc), C
    return bestC


def main():
    ap = argparse.ArgumentParser(); dataset_config.add_dataset_args(ap)
    ap.add_argument("--features", default="both", choices=["li", "diff", "both", "litemporal"]); a = ap.parse_args()
    cfg = dataset_config.configure(a.dataset, perf_variant=getattr(a, "perf_variant", "tc"), band_policy=getattr(a, "band_policy", None))
    feat = BASE / "v2_features" / cfg.name; out = BASE / "ez_analysis_mdt/v2" / cfg.name; gt = cfg.ground_truth()
    pats = [p for p in cfg.patient_ids() if (feat / f"{p}.npz").exists()]
    if cfg.is_inhouse:
        bands = ["FM_20_39", "FM_40_59", "FM_60_79"]
        import glob, re
        def band_of(pid):
            for f in glob.glob(str(BASE / "results_zscore/patients" / pid / f"{pid}_vs_*_summary.json")):
                m = re.search(r"_vs_(FM_\d+_\d+)_summary", f)
                if m: return m.group(1)
        pband = {p: band_of(p) for p in pats}
    else:
        bands = ["CH_ALL"]; pband = {p: "CH_ALL" for p in pats}
    ctrl = {b: [c for c in cfg.control_ids(b) if (feat / f"{c}.npz").exists()] for b in bands}
    # covariates (chalfont only)
    cov = None
    if not cfg.is_inhouse:
        sub = pd.read_csv(cfg.base_dir / "external_chalfont/metadata/subjects.csv").set_index("id")
        cov = pd.DataFrame({"age": sub["age_at_scan"].astype(float), "sex": (sub["sex"] == "F").astype(float)})
    cols = [f"li_{n}" for n in TEMPORAL] if a.features == "litemporal" else [f"li_{n}" for n in NAMES] if a.features == "li" else [f"d_{n}" for n in NAMES] if a.features == "diff" else [f"li_{n}" for n in NAMES] + [f"d_{n}" for n in NAMES]
    # ---- z features for patients and LOO controls -----------------------------------
    Z = {}; Zc = {}
    for b in bands:
        cX = feature_frame(feat, ctrl[b]); ccov = None if cov is None else cov.loc[ctrl[b]].join(cX[["gm_mean"]])
        ps = [p for p in pats if pband[p] == b]
        pX = feature_frame(feat, ps); pcov = None if cov is None else cov.loc[ps].join(pX[["gm_mean"]])
        z = covariate_z(cX[cols], ccov, pX[cols], pcov)
        for p in ps: Z[p] = z.loc[p].values
        for c in ctrl[b]:
            zc = covariate_z(cX[cols], ccov, cX.loc[[c], cols], None if ccov is None else ccov.loc[[c]], loo=c)
            Zc[c] = zc.iloc[0].values
    uni = [p for p in pats if gt.get(p, {}).get("ez") in ("L", "R")]
    X = np.array([np.nan_to_num(Z[p]) for p in uni]); y = np.array([1 if gt[p]["ez"] == "R" else -1 for p in uni])   # +1 = right EZ
    Xc = np.array([np.nan_to_num(Zc[c]) for c in sum(ctrl.values(), [])])
    # ---- baseline: temporal mean z (no training) ----------------------------------
    tcols = [i for i, c in enumerate(cols) if c.startswith("li_") and c[3:] in TEMPORAL]
    base_score = X[:, tcols].mean(axis=1)            # negative = left lower -> left EZ ; sign(+) = right
    # ---- leave-one-out antisymmetric logistic regression ---------------------------
    scores = np.zeros(len(uni)); Cs = []
    for i in range(len(uni)):
        tr = np.array([j for j in range(len(uni)) if j != i]); C = choose_C(X[tr], y[tr]); Cs.append(C)
        m = fit_antisym(X[tr], y[tr], C); scores[i] = m.decision_function(X[[i]])[0]
    Cfull = choose_C(X, y); mfull = fit_antisym(X, y, Cfull); cscore = mfull.decision_function(Xc)
    thr = np.percentile(np.abs(cscore), 95); thr_base = np.percentile(np.abs(Xc[:, tcols].mean(axis=1)), 95)
    def report(name, s, t):
        pred = np.where(s > 0, 1, -1); ok = pred == y; L = y == -1; R = y == 1
        called = np.abs(s) >= t; okc = ok & called
        bal = 0.5 * (ok[L].mean() + ok[R].mean())
        return dict(method=name, n=len(y), agree=int(ok.sum()), balanced_acc=round(bal, 3), sens_L=round(ok[L].mean(), 2), sens_R=round(ok[R].mean(), 2),
                    wilson=tuple(round(v, 2) for v in wilson(int(ok.sum()), len(y))), n_called=int(called.sum()), correct_called=int(okc.sum()),
                    ppv_called=round(okc.sum() / max(called.sum(), 1), 2), threshold=round(float(t), 2))
    rows = [report("temporal mean z (no training)", base_score, thr_base),
            report(f"antisymmetric LR, LOO ({a.features}, {len(cols)} feats)", scores, thr)]
    # zero-shot transfer to the other cohort
    other = "inhouse" if not cfg.is_inhouse else "chalfont"
    ofeat = BASE / "v2_features" / other
    if ofeat.exists() and not cfg.is_inhouse:
        ocfg = dataset_config.configure("inhouse"); ogt = ocfg.ground_truth()
        import glob, re
        opats = [p for p in ocfg.patient_ids() if (ofeat / f"{p}.npz").exists() and ogt[p]["ez"] in ("L", "R")]
        def oband(pid):
            for f in glob.glob(str(BASE / "results_zscore/patients" / pid / f"{pid}_vs_*_summary.json")):
                m = re.search(r"_vs_(FM_\d+_\d+)_summary", f)
                if m: return m.group(1)
        oZ = []
        for p in opats:
            b = oband(p); cids = [c for c in ocfg.control_ids(b) if (ofeat / f"{c}.npz").exists()]
            cX = feature_frame(ofeat, cids); pX = feature_frame(ofeat, [p])
            oZ.append(np.nan_to_num(covariate_z(cX[cols], None, pX[cols], None).iloc[0].values))
        oX = np.array(oZ); oy = np.array([1 if ogt[p]["ez"] == "R" else -1 for p in opats]); os_ = mfull.decision_function(oX)
        pred = np.where(os_ > 0, 1, -1); ok = pred == oy
        rows.append(dict(method="antisymmetric LR fitted on Chalfont -> in-house zero-shot", n=len(oy), agree=int(ok.sum()),
                         balanced_acc=round(0.5 * (ok[oy == -1].mean() + ok[oy == 1].mean()), 3), sens_L=round(ok[oy == -1].mean(), 2), sens_R=round(ok[oy == 1].mean(), 2),
                         wilson=tuple(round(v, 2) for v in wilson(int(ok.sum()), len(oy))), n_called=int((np.abs(os_) >= thr).sum()),
                         correct_called=int((ok & (np.abs(os_) >= thr)).sum()), ppv_called=np.nan, threshold=round(float(thr), 2)))
    R = pd.DataFrame(rows); R.to_csv(out / f"advanced_summary_{a.features}.csv", index=False)
    pd.set_option("display.width", 250); print(f"\n=== {cfg.name} (features={a.features}) ==="); print(R.to_string(index=False))
    # coefficients of the full model (top 10 by |w|)
    w = pd.Series(mfull.coef_[0], index=cols).sort_values(key=np.abs, ascending=False); print("\ntop weights (positive = larger value favours RIGHT EZ):"); print(w.head(10).round(3).to_string())
    # ---- localization (Chalfont only): lobe with most negative mean z_LI on the predicted side
    if not cfg.is_inhouse:
        sub = pd.read_csv(cfg.base_dir / "external_chalfont/metadata/subjects.csv").set_index("id")
        li_idx = {n: cols.index(f"li_{n}") for n in NAMES if f"li_{n}" in cols}
        rec = []
        for i, p in enumerate(uni):
            z = X[i]; side = 1 if scores[i] > 0 else -1          # +1 right
            # deviation on the predicted side: for right EZ, LI positive (left higher) means right lower
            lobe_score = {lb: np.nanmean([z[li_idx[n]] for n in rois if n in li_idx]) * side for lb, rois in LOBE.items()}
            best = max(lobe_score, key=lobe_score.get)             # largest "predicted side is lower" evidence
            rec.append(dict(id=p, ez=gt[p]["ez"], pred_side="R" if side > 0 else "L", ez_lobe=sub.loc[p, "ez_lobe"], pred_lobe=best, **{f"s_{k}": round(v, 2) for k, v in lobe_score.items()}))
        Lz = pd.DataFrame(rec); Lz.to_csv(out / f"advanced_localization_{a.features}.csv", index=False)
        u = Lz[Lz.ez_lobe.isin(["temporal", "frontal", "parietal", "occipital"])]
        side_ok = (u.pred_side == u.ez); lobe_ok = (u.pred_lobe == u.ez_lobe)
        prev = u.ez_lobe.value_counts(normalize=True).max()
        print(f"\nlocalization (lobe) among {len(u)} single-lobe patients: side correct {side_ok.sum()}/{len(u)}; lobe correct {lobe_ok.sum()}/{len(u)} ({lobe_ok.mean():.2f}; majority lobe {prev:.2f}); side AND lobe correct {(side_ok & lobe_ok).sum()}/{len(u)}")
        print(pd.crosstab(u.ez_lobe, u.pred_lobe))


if __name__ == "__main__":
    main()
