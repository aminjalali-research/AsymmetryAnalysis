#!/usr/bin/env python3
"""Search for an operating point with BOTH sensitivity and specificity reasonable.

Candidates beyond the single statistics already traced (sens_spec_operating_points.py):
  1. combined statistic: mean of control-standardized regional temporal z-bar, smoothed z-mass LI and
     analysis-3 temporal difference (same sign convention, negative = left);
  2. consensus-of-k: call only when >= k of the three statistics agree in sign AND |regional temporal
     z-bar| >= t (t over its range);
  3. structural prior (literature: hippocampal volume asymmetry lateralizes hippocampal sclerosis with
     AUC 0.95): volume AI = (L - R)/(L + R) of hippocampus and amygdala from the available aparc+aseg
     files (in-house patients and HCP controls; NOTE: these segmentations are in MNI space, so volumes
     are partly normalized and the AI is a conservative version of the native-space index);
  4. two-feature logistic model (regional temporal z-bar + hippocampal volume AI), in-house only,
     leave-one-out, with the control false-positive rate from the controls' leave-one-out statistics.
For every candidate the best balanced operating point max_t min(sens, spec) is reported, with the
detection AUC (|S| patients vs controls) as the ceiling, so the reader sees what is and is not reachable.

Output: ez_analysis_mdt/audit/best_operating_points/{summary.csv, volume_ai.csv}
Usage:  python scripts/audit/best_operating_points.py
"""
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "ez_analysis_mdt/audit/best_operating_points"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)


def auc(pos, neg):
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def best_point(s_pat, ez, s_ctl, called_pat=None, called_ctl=None):
    """Max over thresholds of min(sens, spec); sens = correct side & called & |s|>=t; spec = control not called or |s|<t."""
    s_pat = np.asarray(s_pat, float); s_ctl = np.asarray(s_ctl, float); ez = np.asarray(ez)
    cp = np.ones(len(s_pat), bool) if called_pat is None else np.asarray(called_pat, bool)
    cc = np.ones(len(s_ctl), bool) if called_ctl is None else np.asarray(called_ctl, bool)
    okp = np.isfinite(s_pat); okc = np.isfinite(s_ctl)
    correct = (np.where(s_pat < 0, "L", "R") == ez) & okp
    ts = np.unique(np.concatenate([[0.0], np.abs(s_pat[okp]), np.abs(s_ctl[okc]), [np.inf]]))
    best = None
    for t in ts:
        sens = float(np.mean(correct & cp & (np.abs(np.nan_to_num(s_pat, nan=-1)) >= t)))
        spec = float(np.mean(~(cc & okc & (np.abs(np.nan_to_num(s_ctl, nan=-1)) >= t))))
        m = min(sens, spec)
        if best is None or m > best[0]:
            best = (m, t, sens, spec)
    return dict(min_sens_spec=round(best[0], 3), t=round(best[1], 3), sens=round(best[2], 3), spec=round(best[3], 3))


def load(ds):
    v2 = BASE / "ez_analysis_mdt/v2" / ds
    Rp = pd.read_csv(v2 / "R_patients.csv").set_index("id"); Rc = pd.read_csv(v2 / "R_loo_controls.csv").set_index("id")
    Vp = pd.read_csv(v2 / "V_patients.csv").set_index("id"); Vc = pd.read_csv(v2 / "V_loo_controls.csv").set_index("id")
    Z = pd.read_csv(BASE / "ez_analysis_mdt/perfz" / ds / "subjects.csv").set_index("id")
    Z["d_temporal"] = Z.p_raw_meanz_temporal_L - Z.p_raw_meanz_temporal_R
    pat = Rp[Rp.ez.isin(["L", "R"])].index; ctl = Rc.index
    P = pd.DataFrame({"ez": Rp.loc[pat, "ez"], "R": Rp.loc[pat, "R_temporal"], "V": Vp.loc[pat, "V_all_t2"], "D": Z.loc[pat, "d_temporal"]})
    C = pd.DataFrame({"R": Rc.loc[ctl, "R_temporal"], "V": Vc.loc[ctl, "V_all_t2"], "D": Z.reindex(ctl)["d_temporal"]})
    return P, C


def volume_ai_inhouse():
    import nibabel as nib
    rows = []
    files = {p.name: p / "aparc+aseg.nii.gz" for p in sorted((BASE / "Dataset").glob("P0*")) if (p / "aparc+aseg.nii.gz").exists()}
    for band in ("FM_20_39", "FM_40_59", "FM_60_79"):
        for p in sorted((BASE / "DatasetControls" / band).glob("*")):
            f = p / "aparc+aseg.nii.gz"
            if f.exists():
                files[p.name] = f
    for sid, f in files.items():
        lab = np.asanyarray(nib.load(str(f)).dataobj).astype(np.int32)
        bc = np.bincount(lab.ravel(), minlength=60)
        hL, hR, aL, aR = bc[17], bc[53], bc[18], bc[54]
        rows.append(dict(id=sid, kind="patient" if sid.startswith("P") else "control", hipp_L=int(hL), hipp_R=int(hR), amyg_L=int(aL), amyg_R=int(aR),
                         hipp_ai=(hL - hR) / (hL + hR), amyg_ai=(aL - aR) / (aL + aR)))
    T = pd.DataFrame(rows); T.to_csv(OUT / "volume_ai.csv", index=False); return T


def main():
    rows = []
    for ds in ("chalfont", "inhouse"):
        P, C = load(ds)
        # standardize by control SD, combine
        sd = C[["R", "V", "D"]].std()
        Ps = P[["R", "V", "D"]] / sd; Cs = C[["R", "V", "D"]] / sd
        P["S3"] = Ps.mean(axis=1); C["S3"] = Cs.mean(axis=1)
        for name, col in (("regional temporal z-bar", "R"), ("combined mean of 3 standardized statistics", "S3")):
            bp = best_point(P[col].values, P.ez.values, C[col].values)
            rows.append(dict(dataset=ds, candidate=name, n_pat=len(P), n_ctl=len(C), detection_auc=round(auc(np.abs(P[col].values), np.abs(C[col].values)), 3), **bp,
                             sign_agree=f"{int(((np.where(P[col] < 0, 'L', 'R')) == P.ez).sum())}/{len(P)}"))
        # consensus of k signs with magnitude gate on R
        for k in (2, 3):
            sp = np.sign(P[["R", "V", "D"]].values); sc = np.sign(C[["R", "V", "D"]].values)
            agree_p = (np.abs(sp.sum(axis=1)) >= (2 * k - 3)) if k == 2 else (np.abs(sp.sum(axis=1)) == 3)
            agree_c = (np.abs(sc.sum(axis=1)) >= (2 * k - 3)) if k == 2 else (np.abs(sc.sum(axis=1)) == 3)
            side_p = np.where(sp.sum(axis=1) < 0, "L", "R")
            # use the consensus side with |R| as strength
            s_p = np.where(side_p == "L", -np.abs(P.R.values), np.abs(P.R.values)); s_c = C.R.values
            bp = best_point(s_p, P.ez.values, s_c, called_pat=agree_p, called_ctl=agree_c)
            rows.append(dict(dataset=ds, candidate=f"consensus of {k}/3 signs, gate |regional temporal z-bar| >= t", n_pat=len(P), n_ctl=len(C), detection_auc=np.nan, **bp,
                             sign_agree=f"{int(((side_p == P.ez.values) & agree_p).sum())}/{len(P)} (called {int(agree_p.sum())}; controls agreeing {agree_c.mean():.2f})"))
    # structural prior, in-house
    T = volume_ai_inhouse(); Pin, Cin = load("inhouse")
    Tp = T[T.kind == "patient"].set_index("id").reindex(Pin.index); Tc = T[T.kind == "control"].set_index("id").reindex(Cin.index)
    for name, col in (("hippocampal volume AI (MNI-space segmentation)", "hipp_ai"), ("amygdalar volume AI (MNI-space segmentation)", "amyg_ai")):
        s_p = Tp[col].values; s_c = Tc[col].values
        bp = best_point(s_p, Pin.ez.values, s_c)
        fixed_call = np.where(s_p <= -0.071, "L", np.where(s_p >= 0.061, "R", "none")); fixed_ctl = np.mean((s_c <= -0.071) | (s_c >= 0.061))
        rows.append(dict(dataset="inhouse", candidate=name, n_pat=len(Pin), n_ctl=int(np.isfinite(s_c).sum()), detection_auc=round(auc(np.abs(s_p), np.abs(s_c[np.isfinite(s_c)])), 3), **bp,
                         sign_agree=f"{int((np.where(s_p < 0, 'L', 'R') == Pin.ez.values).sum())}/{len(Pin)}; fixed cut-offs (-0.071/+0.061): called {int((fixed_call != 'none').sum())}, correct {int((fixed_call == Pin.ez.values).sum())}, control FP {fixed_ctl:.2f}"))
    # two-feature logistic, leave-one-out (in-house)
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([Pin.R.values, Tp.hipp_ai.values]); y = (Pin.ez.values == "L").astype(int)
    Xc = np.column_stack([Cin.R.values, Tc.hipp_ai.values]); okc = np.isfinite(Xc).all(axis=1); Xc = Xc[okc]
    Xa = np.vstack([X, -X]); ya = np.concatenate([y, 1 - y])     # mirror augmentation: antisymmetric model, cannot learn the majority side
    prob = np.zeros(len(y))
    for i in range(len(y)):
        keep = np.ones(len(ya), bool); keep[i] = False; keep[len(y) + i] = False
        m = LogisticRegression(C=1.0, fit_intercept=False).fit(Xa[keep], ya[keep]); prob[i] = m.predict_proba(X[i:i + 1])[0, 1]
    m_all = LogisticRegression(C=1.0, fit_intercept=False).fit(Xa, ya); pc = m_all.predict_proba(Xc)[:, 1]
    s_p = -(prob - 0.5) * 2; s_c = -(pc - 0.5) * 2                    # negative = left, magnitude = confidence
    bp = best_point(s_p, Pin.ez.values, s_c)
    rows.append(dict(dataset="inhouse", candidate="two-feature logistic (regional temporal z-bar + hippocampal volume AI), LOO, antisymmetric", n_pat=len(Pin), n_ctl=len(s_c),
                     detection_auc=round(auc(np.abs(s_p), np.abs(s_c)), 3), **bp, sign_agree=f"{int((np.where(s_p < 0, 'L', 'R') == Pin.ez.values).sum())}/{len(Pin)} (LOO sign)"))
    S = pd.DataFrame(rows); S.to_csv(OUT / "summary.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_colwidth", 90): print(S.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
