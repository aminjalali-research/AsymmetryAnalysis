#!/usr/bin/env python3
"""Structural prior on the external cohort: hippocampal and amygdalar volume asymmetry from FSL FIRST.

Volume AI = (L - R)/(L + R) of the FIRST segmentation volumes (labels 17/53 hippocampus, 18/54 amygdala);
negative = left smaller.  Under the classical reading the atrophic side is the epileptogenic side, so a
negative AI predicts a left EZ (same sign convention as LI).  Scored exactly like the perfusion statistics:
sign agreement, calibrated call (95th percentile of |AI| in the 66 controls), fixed literature cut-offs
(Ndagijimana 2026: left HS if AI <= -0.071, right if AI >= +0.061), detection AUC (|AI| patients vs controls),
and the best balanced operating point max_t min(sensitivity, specificity).  Strata: pathology (HS, other
lesion, MRI-negative), lobe, reference tier.  Combination with perfusion: (i) volume call first, regional
temporal z-bar calibrated call second, sign of z-bar third; (ii) antisymmetric two-feature logistic model
(volume AI, regional temporal z-bar) with leave-one-out and the control false-positive rate.

Inputs: ez_analysis_mdt/audit/first_chalfont/<ID>_all_fast_firstseg.nii.gz, external_chalfont/metadata/subjects.csv,
        ez_analysis_mdt/v2/chalfont/{R_patients,R_loo_controls,summary}.csv
Output: ez_analysis_mdt/audit/first_chalfont/volume_ai.csv, results.csv (+ printout); --v2-dir/--tag for another perfusion run
"""
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib

BASE = Path(__file__).resolve().parents[2]
FD = BASE / "ez_analysis_mdt/audit/first_chalfont"
# Perfusion statistic to combine with (default: the transit-corrected v2 run). `--v2-dir <dir> --tag <suffix>`
# scores the same combinations against another v2 output, e.g. the plain-Flow run in ez_analysis_mdt/audit/flow_variant
# (2026-11-04), and writes results<suffix>.csv instead of results.csv.
import argparse as _ap
_a = _ap.ArgumentParser(); _a.add_argument("--v2-dir", default=str(BASE / "ez_analysis_mdt/v2/chalfont")); _a.add_argument("--tag", default="")
_args, _ = _a.parse_known_args()
V2 = Path(_args.v2_dir); TAG = _args.tag
rng = np.random.default_rng(1)


def auc_ci(pos, neg, n=1000):
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    def _a(p, q): return float(np.mean(p[:, None] > q[None, :]) + 0.5 * np.mean(p[:, None] == q[None, :]))
    a = _a(pos, neg); bs = [_a(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n)]
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def best_point(s_pat, ez, s_ctl):
    s_pat = np.asarray(s_pat, float); s_ctl = np.asarray(s_ctl, float); ez = np.asarray(ez)
    correct = (np.where(s_pat < 0, "L", "R") == ez) & np.isfinite(s_pat)
    ts = np.unique(np.concatenate([[0.0], np.abs(s_pat[np.isfinite(s_pat)]), np.abs(s_ctl[np.isfinite(s_ctl)]), [np.inf]]))
    best = None
    for t in ts:
        sens = float(np.mean(correct & (np.abs(np.nan_to_num(s_pat, nan=-1)) >= t))); spec = float(np.mean(~(np.abs(np.nan_to_num(s_ctl, nan=-1)) >= t)))
        if best is None or min(sens, spec) > best[0]:
            best = (min(sens, spec), t, sens, spec)
    return best


def score(name, call, ez, ctl_called, stat_p=None, stat_c=None, strata=None):
    call = np.asarray(call); ez = np.asarray(ez); called = np.isin(call, ["L", "R"]); ok = call == ez
    nL = (ez == "L").sum(); nR = (ez == "R").sum()
    row = dict(rule=name, n=len(ez), called=int(called.sum()), correct=int(ok.sum()), sens_correct_side=round(ok.mean(), 3),
               side_acc_among_called=round(ok[called].mean(), 3) if called.any() else np.nan, balanced=round(np.mean([ok[ez == "L"].mean(), ok[ez == "R"].mean()]), 3),
               specificity=round(1 - float(ctl_called), 3), wilson=f"{wilson(ok.sum(), len(ez))[0]:.2f}-{wilson(ok.sum(), len(ez))[1]:.2f}")
    if strata is not None:
        for k, m in strata.items():
            row[k] = f"{int(ok[m].sum())}/{int(m.sum())}" + (f" (called {int((called & m).sum())})" if called.sum() < len(ez) else "")
    if stat_p is not None:
        a, lo, hi = auc_ci(np.abs(stat_p), np.abs(stat_c)); bp = best_point(stat_p, ez, stat_c)
        row.update(detection_auc=f"{a:.2f} ({lo:.2f}-{hi:.2f})", best_min_sens_spec=f"{bp[0]:.2f} (t={bp[1]:.3f}: sens {bp[2]:.2f}, spec {bp[3]:.2f})")
    return row


def main():
    S = pd.read_csv(BASE / "external_chalfont/metadata/subjects.csv").set_index("id")
    rows = []
    for f in sorted(FD.glob("*_all_fast_firstseg.nii.gz")):
        sid = f.name.split("_all_")[0]; im = nib.load(str(f)); a = np.asanyarray(im.dataobj).astype(int); bc = np.bincount(a.ravel(), minlength=60)
        vv = float(np.prod(im.header.get_zooms()[:3]))
        rows.append(dict(id=sid, hipp_L=bc[17] * vv, hipp_R=bc[53] * vv, amyg_L=bc[18] * vv, amyg_R=bc[54] * vv))
    V = pd.DataFrame(rows).set_index("id")
    V["hipp_ai"] = (V.hipp_L - V.hipp_R) / (V.hipp_L + V.hipp_R); V["amyg_ai"] = (V.amyg_L - V.amyg_R) / (V.amyg_L + V.amyg_R)
    V["group"] = S.reindex(V.index)["group"]; V["ez"] = S.reindex(V.index)["ez_side"]; V["pathology"] = S.reindex(V.index)["pathology"]; V["lobe"] = S.reindex(V.index)["ez_lobe"]; V["tier"] = S.reindex(V.index)["reference_tier"]
    V.to_csv(FD / "volume_ai.csv"); print(f"segmented: {len(V)} ({(V.group == 'control').sum()} controls, {(V.group != 'control').sum()} patients)")
    ctl = V[V.group == "control"]; pat = V[(V.group != "control") & V.ez.isin(["L", "R"])]
    # exclude QC-excluded subjects if the v2 tables lack them
    R = pd.read_csv(V2 / "R_patients.csv").set_index("id"); Rc = pd.read_csv(V2 / "R_loo_controls.csv").set_index("id")
    pat = pat[pat.index.isin(R.index)]; ctl = ctl[ctl.index.isin(Rc.index)]
    print(f"evaluated: {len(pat)} unilateral patients, {len(ctl)} controls; HS {int((pat.pathology == 'HS').sum())}")
    ez = pat.ez.values
    strata = {"HS": (pat.pathology == "HS").values, "other lesion": pat.pathology.isin(["FCD", "cavernoma", "lesion (unspecified)", "DNET", "tumour", "tumor"]).values,
              "MRI-negative": (pat.pathology == "MRI-negative").values, "temporal": (pat.lobe == "temporal").values, "frontal": (pat.lobe == "frontal").values}
    out = []
    for name, col in (("hippocampal volume AI", "hipp_ai"), ("amygdalar volume AI", "amyg_ai")):
        s_p = pat[col].values; s_c = ctl[col].values
        thr = float(np.nanpercentile(np.abs(s_c), 95))
        out.append(score(f"{name}, sign", np.where(s_p < 0, "L", "R"), ez, 1.0, s_p, s_c, strata))
        out.append(score(f"{name}, calibrated 5% FP (|AI| >= {thr:.3f})", np.where(np.abs(s_p) < thr, "none", np.where(s_p < 0, "L", "R")), ez, float(np.mean(np.abs(s_c) >= thr)), None, None, strata))
        if col == "hipp_ai":
            out.append(score("hippocampal volume AI, fixed cut-offs -0.071 / +0.061 (Ndagijimana 2026)", np.where(s_p <= -0.071, "L", np.where(s_p >= 0.061, "R", "none")), ez,
                             float(np.mean((s_c <= -0.071) | (s_c >= 0.061))), None, None, strata))
    # combination with the regional perfusion statistic
    zt = R.reindex(pat.index)["R_temporal"].values; zc = Rc.reindex(ctl.index)["R_temporal"].values
    summ = pd.read_csv(V2 / "summary.csv").set_index("statistic"); tz = float(summ.loc["R_temporal", "threshold"])
    th = float(np.nanpercentile(np.abs(ctl.hipp_ai.values), 95)); hp = pat.hipp_ai.values; hc = ctl.hipp_ai.values
    def seq(h, z):
        c1 = np.where(np.abs(h) >= th, np.where(h < 0, "L", "R"), "none"); c2 = np.where(np.abs(z) >= tz, np.where(z < 0, "L", "R"), "none")
        return np.where(c1 != "none", c1, c2)
    cp = seq(hp, zt); cc = seq(hc, zc)
    out.append(score("sequential: volume AI calibrated call, else regional temporal z-bar calibrated call", cp, ez, float(np.mean(cc != "none")), None, None, strata))
    cp2 = np.where(cp != "none", cp, np.where(zt < 0, "L", "R"))
    out.append(score("sequential, then sign of regional z-bar (always calls)", cp2, ez, 1.0, None, None, strata))
    # two-feature antisymmetric logistic, leave-one-out
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([hp / np.nanstd(hc), zt / np.nanstd(zc)]); y = (ez == "L").astype(int); okx = np.isfinite(X).all(axis=1)
    Xc = np.column_stack([hc / np.nanstd(hc), zc / np.nanstd(zc)]); Xc = Xc[np.isfinite(Xc).all(axis=1)]
    prob = np.full(len(y), np.nan)
    for i in np.where(okx)[0]:
        keep = okx.copy(); keep[i] = False
        Xa = np.vstack([X[keep], -X[keep]]); ya = np.concatenate([y[keep], 1 - y[keep]])
        m = LogisticRegression(C=1.0, fit_intercept=False).fit(Xa, ya); prob[i] = m.predict_proba(X[i:i + 1])[0, 1]
    m_all = LogisticRegression(C=1.0, fit_intercept=False).fit(np.vstack([X[okx], -X[okx]]), np.concatenate([y[okx], 1 - y[okx]]))
    s_p = -(prob - 0.5) * 2; s_c = -(m_all.predict_proba(Xc)[:, 1] - 0.5) * 2
    thr = float(np.nanpercentile(np.abs(s_c), 95))
    out.append(score("two-feature logistic (volume AI + regional z-bar), LOO, sign", np.where(s_p < 0, "L", "R"), ez, 1.0, s_p, s_c, strata))
    out.append(score(f"two-feature logistic, calibrated 5% FP", np.where(np.abs(s_p) < thr, "none", np.where(s_p < 0, "L", "R")), ez, float(np.mean(np.abs(s_c) >= thr)), None, None, strata))
    out.append(score("reference: regional temporal z-bar, sign", np.where(zt < 0, "L", "R"), ez, 1.0, zt, zc, strata))
    out.append(score("reference: regional temporal z-bar, calibrated", np.where(np.abs(zt) < tz, "none", np.where(zt < 0, "L", "R")), ez, float(np.mean(np.abs(zc) >= tz)), None, None, strata))
    T = pd.DataFrame(out); T.to_csv(FD / f"results{TAG}.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_colwidth", 70): print(T.to_string(index=False))
    print("control hippocampal AI: median %.3f, SD %.3f, 95th pct |AI| %.3f" % (np.nanmedian(hc), np.nanstd(hc), th))


if __name__ == "__main__":
    main()
