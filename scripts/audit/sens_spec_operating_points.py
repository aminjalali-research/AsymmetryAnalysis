#!/usr/bin/env python3
"""Sensitivity AND specificity of every lateralizing statistic, over its whole threshold range.

A lateralizing call has two ways to be wrong: calling a healthy subject (false positive) and
calling the wrong side in a patient.  For each statistic S (sign = side, |S| = strength) and each
threshold t we therefore report
    sensitivity(t)  = fraction of unilateral patients with |S| >= t AND the correct side,
    specificity(t)  = fraction of leave-one-out controls with |S| < t (no call),
    side accuracy(t) = fraction of called patients with the correct side,
together with the detection AUC (|S| of patients vs. controls, bootstrap CI) and the operating
points a clinician would ask for: the Youden maximum, sensitivity at specificity 0.80 and 0.90,
and specificity at sensitivity 0.70.  Both cohorts, every estimator computed so far (regional
normative z, smoothed z-mass, analysis-3 perfusion z, the thesis ROI rule |AI| >= 10 when
available).  Always-calling rules sit at specificity 0 by construction.

Inputs: ez_analysis_mdt/v2/<dataset>/{R,V}_{patients,loo_controls}.csv,
        ez_analysis_mdt/perfz/<dataset>/subjects.csv, optional thesis_roi_rule/<dataset>.csv
Output: ez_analysis_mdt/audit/sens_spec/<dataset>_{curves,operating_points}.csv and a figure.
Usage:  python scripts/audit/sens_spec_operating_points.py --dataset inhouse|chalfont
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(1)


def load(dataset):
    v2 = BASE / "ez_analysis_mdt/v2" / dataset
    Rp = pd.read_csv(v2 / "R_patients.csv"); Rc = pd.read_csv(v2 / "R_loo_controls.csv")
    Vp = pd.read_csv(v2 / "V_patients.csv"); Vc = pd.read_csv(v2 / "V_loo_controls.csv")
    Z = pd.read_csv(BASE / "ez_analysis_mdt/perfz" / dataset / "subjects.csv")
    stats = {}
    for col, name in [("R_temporal", "regional temporal z-bar"), ("R_mesial", "regional mesial z-bar"), ("R_all37", "regional all-37 z-bar")]:
        stats[name] = (Rp[["id", "ez", col]].rename(columns={col: "s"}), Rc[["id", col]].rename(columns={col: "s"}))
    for col, name in [("V_all_t2", "smoothed z-mass LI (all GM)"), ("V_temporal_t2", "smoothed z-mass LI (temporal)")]:
        stats[name] = (Vp[["id", "ez", col]].rename(columns={col: "s"}), Vc[["id", col]].rename(columns={col: "s"}))
    Zp = Z[Z.kind == "patient"].copy(); Zc = Z[Z.kind == "control"].copy()
    for df in (Zp, Zc):
        df["d_temporal"] = df.p_raw_meanz_temporal_L - df.p_raw_meanz_temporal_R           # negative = left lower
        df["hypoLI"] = (df.p_raw_hypo_R - df.p_raw_hypo_L) / (df.p_raw_hypo_L + df.p_raw_hypo_R).replace(0, np.nan)  # negative = more hypo on the left
        df["d_temporal_oc"] = df.p_oc_meanz_temporal_L - df.p_oc_meanz_temporal_R
    Zp = Zp.rename(columns={"truth": "ez"})
    for col, name in [("d_temporal", "perfusion z, temporal mean difference (analysis 3)"), ("hypoLI", "perfusion z, hypoperfused-volume LI (analysis 3)"),
                      ("d_temporal_oc", "perfusion z, temporal difference, offset-corrected")]:
        stats[name] = (Zp[["id", "ez", col]].rename(columns={col: "s"}), Zc[["id", col]].rename(columns={col: "s"}))
    thesis = BASE / "ez_analysis_mdt/audit/thesis_roi_rule" / f"{dataset}.csv"
    if thesis.exists():
        T = pd.read_csv(thesis)
        for col, name in [("roi_count_LI", "thesis ROI rule: hypoperfused-ROI count LI (|AI| >= 10, temporal)"), ("roi_count_LI_all", "thesis ROI rule: hypoperfused-ROI count LI (|AI| >= 10, all 37)")]:
            if col in T:
                stats[name] = (T[(T.kind == "patient")][["id", "ez", col]].rename(columns={col: "s"}), T[T.kind == "control"][["id", col]].rename(columns={col: "s"}))
    return stats


def auc_boot(pos, neg, n=1000):
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    def _a(p, q): return np.mean(p[:, None] > q[None, :]) + 0.5 * np.mean(p[:, None] == q[None, :])
    a = _a(pos, neg); bs = []
    for _ in range(n):
        bs.append(_a(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def curves(pat, ctl):
    pat = pat[pat.ez.isin(["L", "R"]) & np.isfinite(pat.s)].copy(); ctl = ctl[np.isfinite(ctl.s)].copy()
    pat["correct"] = np.where(pat.s < 0, "L", "R") == pat.ez
    pat["a"] = pat.s.abs(); ctl["a"] = ctl.s.abs()
    ts = np.unique(np.concatenate([[0.0], pat.a.values, ctl.a.values, [np.inf]]))
    rows = []
    for t in ts:
        called = pat.a >= t
        rows.append(dict(t=t, sens=float((called & pat.correct).mean()), spec=float((ctl.a < t).mean()),
                         call_rate=float(called.mean()), side_acc=float(pat.correct[called].mean()) if called.any() else np.nan,
                         n_called=int(called.sum()), n_correct=int((called & pat.correct).sum()),
                         bal_side=float(np.mean([(called & pat.correct)[pat.ez == "L"].mean(), (called & pat.correct)[pat.ez == "R"].mean()]))))
    C = pd.DataFrame(rows)
    a, lo, hi = auc_boot(pat.a.values, ctl.a.values)
    return C, (a, lo, hi), len(pat), len(ctl)


def operating_points(C):
    out = {}
    j = (C.sens + C.spec - 1); k = int(j.idxmax()); out["youden"] = C.loc[k]
    for sp in (0.80, 0.90):
        ok = C[C.spec >= sp]; out[f"sens_at_spec{int(sp*100)}"] = ok.loc[ok.sens.idxmax()] if len(ok) else None
    ok = C[C.sens >= 0.70]; out["spec_at_sens70"] = ok.loc[ok.spec.idxmax()] if len(ok) else None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=["inhouse", "chalfont"]); a = ap.parse_args()
    out = BASE / "ez_analysis_mdt/audit/sens_spec"; out.mkdir(parents=True, exist_ok=True)
    stats = load(a.dataset); allc = []; rows = []
    for name, (pat, ctl) in stats.items():
        C, (auc, lo, hi), n_p, n_c = curves(pat, ctl); C.insert(0, "statistic", name); allc.append(C)
        op = operating_points(C)
        def fmt(r): return "-" if r is None else f"t={r.t:.3g}: sens {r.sens:.2f}, spec {r.spec:.2f}, side acc {r.side_acc:.2f} ({r.n_correct}/{r.n_called})"
        rows.append(dict(dataset=a.dataset, statistic=name, n_patients=n_p, n_controls=n_c, auc_detect=round(auc, 3), auc_lo=round(lo, 3), auc_hi=round(hi, 3),
                         always_call=fmt(C.iloc[0]), youden=fmt(op["youden"]), sens_at_spec80=fmt(op["sens_at_spec80"]), sens_at_spec90=fmt(op["sens_at_spec90"]), spec_at_sens70=fmt(op["spec_at_sens70"])))
    pd.concat(allc).to_csv(out / f"{a.dataset}_curves.csv", index=False)
    T = pd.DataFrame(rows); T.to_csv(out / f"{a.dataset}_operating_points.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 80): print(T.drop(columns=["dataset"]).to_string(index=False))
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for C in allc:
        nm = C.statistic.iloc[0]; ax[0].plot(1 - C.spec, C.sens, lw=1.4, label=nm[:52]); ax[1].plot(C.call_rate, C.side_acc, lw=1.4)
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8); ax[0].set_xlabel("1 - specificity (controls called)"); ax[0].set_ylabel("sensitivity (patients called on the correct side)")
    ax[0].set_title(f"{a.dataset}: correct-side sensitivity vs control false positives"); ax[0].legend(fontsize=6.5, loc="lower right"); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    ax[1].set_xlabel("call rate in patients"); ax[1].set_ylabel("side accuracy among called"); ax[1].set_title("precision of calls vs coverage"); ax[1].set_xlim(0, 1); ax[1].set_ylim(0.3, 1.02)
    fig.tight_layout(); fig.savefig(out / f"{a.dataset}_sens_spec.png", dpi=200); print("wrote", out)


if __name__ == "__main__":
    main()
