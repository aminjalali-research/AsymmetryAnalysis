#!/usr/bin/env python3
"""Two strategies from the 2023-2026 literature, tested on our outputs without new image processing.

1. Per-side ROC cut-offs on a normative asymmetry statistic (Kershaw 2025, FDG-PET ZAI): instead of
   one symmetric threshold, a left cut-off (left patients vs right patients + controls) and a right
   cut-off (right vs left + controls), each at the Youden point.  Fitted on Chalfont (93 patients,
   66 LOO controls), then applied unchanged to the in-house cohort (13 patients, 180 LOO controls).
2. Peak-cluster rule on the asymmetry z map (Aslam 2022 PASCOM "Z4C"): keep only the cluster pair
   with the global peak |z_AI| when the peak reaches 4 (also 3 and 5) and the extent reaches 100
   voxels; the side is the hemisphere where the pair member is negative (lower than its mirror),
   i.e. the hypoperfused member.  Uses the existing 1.96-cluster reports (extent at 1.96 is an upper
   bound of the extent at the tier value, so the rule is slightly permissive).  Chalfont patients and
   66 pseudo-controls; in-house patients (no pseudo-control reports in-house).

Output: ez_analysis_mdt/audit/strategies_2023_2026/{per_side_cutoffs,peak_cluster_rule}.csv + printout.
Usage:  python scripts/audit/strategies_2023_2026.py
"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "ez_analysis_mdt/audit/strategies_2023_2026"; OUT.mkdir(parents=True, exist_ok=True)


def youden(pos, neg, sign):
    """Cut-off on sign*value maximizing sensitivity + specificity - 1 (pos: patients of that side)."""
    v = np.concatenate([sign * pos, sign * neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    best = (-1, np.inf)
    for t in np.unique(v):
        sens = np.mean(sign * pos >= t); spec = np.mean(sign * neg < t)
        if sens + spec - 1 > best[0]:
            best = (sens + spec - 1, t)
    return best[1]


def per_side_cutoffs():
    rows = []
    for stat in ("R_temporal", "R_all37", "R_mesial"):
        fit_p = pd.read_csv(BASE / "ez_analysis_mdt/v2/chalfont/R_patients.csv"); fit_c = pd.read_csv(BASE / "ez_analysis_mdt/v2/chalfont/R_loo_controls.csv")
        fit_p = fit_p[fit_p.ez.isin(["L", "R"])]
        L = fit_p[fit_p.ez == "L"][stat].values; R = fit_p[fit_p.ez == "R"][stat].values; C = fit_c[stat].values
        tL = youden(L, np.concatenate([R, C]), -1.0)      # left call when -S >= tL  (S <= -tL)
        tR = youden(R, np.concatenate([L, C]), +1.0)      # right call when S >= tR
        for ds in ("chalfont", "inhouse"):
            P = pd.read_csv(BASE / f"ez_analysis_mdt/v2/{ds}/R_patients.csv"); Cc = pd.read_csv(BASE / f"ez_analysis_mdt/v2/{ds}/R_loo_controls.csv")
            P = P[P.ez.isin(["L", "R"])]; s = P[stat].values; ez = P.ez.values
            call = np.where(s <= -tL, "L", np.where(s >= tR, "R", "none"))
            cc = Cc[stat].values; ctl_call = np.mean((cc <= -tL) | (cc >= tR))
            ok = call == ez; called = call != "none"
            sens = ok.mean(); sL = ok[ez == "L"].mean(); sR = ok[ez == "R"].mean()
            rows.append(dict(statistic=stat, fitted_on="chalfont", applied_to=ds, cut_left=round(-tL, 3), cut_right=round(tR, 3), n=len(P),
                             called=int(called.sum()), correct=int(ok.sum()), sens_correct_side=round(sens, 3), sens_L=round(sL, 2), sens_R=round(sR, 2),
                             balanced=round((sL + sR) / 2, 3), side_acc_among_called=round(ok[called].mean(), 3) if called.any() else np.nan,
                             control_call_rate=round(ctl_call, 3), specificity=round(1 - ctl_call, 3)))
    T = pd.DataFrame(rows); T.to_csv(OUT / "per_side_cutoffs.csv", index=False)
    print("\nPer-side ROC cut-offs (Kershaw 2025 style), fitted on Chalfont:")
    print(T.to_string(index=False))


def peak_cluster_rule():
    rows = []
    sets = {"chalfont patients": (sorted(glob.glob(str(BASE / "external_chalfont/results_zscore/asymmetry/patients/*/*_asymmetry_cluster_report.csv"))), "patient"),
            "chalfont pseudo-controls": (sorted(glob.glob(str(BASE / "external_chalfont/results_zscore/asymmetry/pseudo_controls/*/*_asymmetry_cluster_report.csv"))), "control"),
            "inhouse patients": (sorted(glob.glob(str(BASE / "results_zscore/asymmetry/patients/P0*/P0*_asymmetry_cluster_report.csv"))), "patient")}
    truth = {}
    ch = pd.read_csv(BASE / "ez_analysis_mdt/perfz/chalfont/subjects.csv"); truth.update(dict(zip(ch.id, ch.truth)))
    ih = pd.read_csv(BASE / "ez_analysis_mdt/perfz/inhouse/subjects.csv"); truth.update(dict(zip(ih.id, ih.truth)))
    for name, (files, kind) in sets.items():
        for f in files:
            d = pd.read_csv(f)
            if len(d) == 0:
                continue
            sid = d.patient_id.iloc[0]
            if sid == "P014":
                continue
            d["abs_peak"] = d.peak_z.abs()
            for tier in (3, 4, 5):
                top = d.sort_values("abs_peak", ascending=False).iloc[0]
                ok = (top.abs_peak >= tier) and (top.size_voxels >= 100)
                # side = hemisphere of the hypoperfused (negative) member of the peak pair
                if ok:
                    side = top.cluster_hemi if top.mean_z < 0 else ("L" if top.cluster_hemi == "R" else "R")
                else:
                    side = "none"
                rows.append(dict(set=name, kind=kind, id=sid, tier=tier, peak=round(float(top.abs_peak), 2), extent_at_1p96=int(top.size_voxels), call=side, truth=truth.get(sid, "")))
    T = pd.DataFrame(rows); T.to_csv(OUT / "peak_cluster_rule.csv", index=False)
    print("\nPeak-cluster rule (PASCOM Z4C analogue; side = hypoperfused member of the peak pair):")
    for tier in (3, 4, 5):
        for name in sets:
            x = T[(T.set == name) & (T.tier == tier)]
            if name.endswith("controls"):
                print(f"  tier {tier}  {name:26s} n={len(x):3d}  called {int((x.call != 'none').sum())} ({(x.call != 'none').mean():.2f} control FP)")
            else:
                u = x[x.truth.isin(["L", "R"])]; called = u.call != "none"; ok = u.call == u.truth
                sL = ok[u.truth == "L"].mean(); sR = ok[u.truth == "R"].mean()
                print(f"  tier {tier}  {name:26s} n={len(u):3d}  called {int(called.sum())}, correct {int(ok.sum())} (side acc among called {ok[called].mean() if called.any() else float('nan'):.2f}); balanced {np.mean([sL, sR]):.2f}")


if __name__ == "__main__":
    per_side_cutoffs()
    peak_cluster_rule()
    print("wrote", OUT)
