#!/usr/bin/env python3
"""The MSc-thesis ROI rule (Rai 2026), scored on both cohorts with controls as the null.

Thesis protocol: AI = 100 (L - R)/(L + R) of the mean CBF of each FreeSurfer region pair;
|AI| >= 10 counts as significant (Hauf 2009; Lim 2008); findings were read against the clinical
lateralization.  Here the reading is made explicit: a region with AI <= -10 is a left-hypoperfused
region, AI >= +10 a right-hypoperfused one; the side with more hypoperfused regions is the call
(temporal set: hippocampus, amygdala, entorhinal, parahippocampal, fusiform, inferior/middle/
superior temporal, temporal pole, transverse temporal; and all 37 pairs).  The continuous
statistic LI_count = (n_R - n_L)/(n_L + n_R) is negative when the left side has more hypoperfused
regions (same sign convention as LI: negative = left lower).  Controls scored the same way give
the false-positive rate of the rule, which the thesis did not report.

Inputs: v2_features/<dataset>/<ID>_roi.json (mean CBF per label), ez_analysis_mdt/perfz/<dataset>/subjects.csv (kind, truth)
Output: ez_analysis_mdt/audit/thesis_roi_rule/<dataset>.csv + printed summary
Usage:  python scripts/audit/thesis_roi_rule.py --dataset inhouse|chalfont
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parents[2]
SUB = {10: 49, 17: 53, 18: 54}
CORT = {1000 + k: 2000 + k for k in range(1, 36) if k != 4}          # 34 Desikan-Killiany pairs (no corpus callosum)
PAIRS = {**SUB, **CORT}
TEMPORAL = {17, 18, 1006, 1007, 1009, 1015, 1016, 1030, 1033, 1034}  # v2 temporal set (10 pairs)
THR = 10.0


def rule(ai, scope):
    vals = {k: v for k, v in ai.items() if k in scope and np.isfinite(v)}
    nL = sum(v <= -THR for v in vals.values()); nR = sum(v >= THR for v in vals.values())
    li = (nR - nL) / (nL + nR) if (nL + nR) else np.nan
    return nL, nR, li


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=["inhouse", "chalfont"]); a = ap.parse_args()
    S = pd.read_csv(BASE / "ez_analysis_mdt/perfz" / a.dataset / "subjects.csv")[["id", "kind", "truth", "group"]]
    rows = []
    for _, r in S.iterrows():
        f = BASE / "v2_features" / a.dataset / f"{r.id}_roi.json"
        if not f.exists():
            continue
        m = {int(k): (float(v["mean"]) if isinstance(v, dict) and v.get("mean") is not None else np.nan) for k, v in json.load(open(f)).items() if str(k).isdigit()}
        ai = {}
        for L, R in PAIRS.items():
            l, rr = m.get(L, np.nan), m.get(R, np.nan)
            ai[L] = 100.0 * (l - rr) / (l + rr) if np.isfinite(l) and np.isfinite(rr) and (l + rr) > 0 else np.nan
        nL, nR, li = rule(ai, TEMPORAL); nLa, nRa, lia = rule(ai, set(PAIRS))
        rows.append(dict(id=r.id, kind=r.kind, ez=r.truth, group=r.group, n_hypo_L_temporal=nL, n_hypo_R_temporal=nR, roi_count_LI=li,
                         n_hypo_L_all=nLa, n_hypo_R_all=nRa, roi_count_LI_all=lia, max_abs_ai_temporal=max(abs(ai[k]) for k in TEMPORAL if np.isfinite(ai[k]))))
    T = pd.DataFrame(rows); out = BASE / "ez_analysis_mdt/audit/thesis_roi_rule"; out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / f"{a.dataset}.csv", index=False)
    pat = T[(T.kind == "patient") & T.ez.isin(["L", "R"])]; ctl = T[T.kind == "control"]
    for scope, col, nl, nr in (("temporal", "roi_count_LI", "n_hypo_L_temporal", "n_hypo_R_temporal"), ("all 37", "roi_count_LI_all", "n_hypo_L_all", "n_hypo_R_all")):
        call = np.where(pat[nl] > pat[nr], "L", np.where(pat[nr] > pat[nl], "R", "none"))
        ok = call == pat.ez.values; called = call != "none"
        sL = ok[pat.ez.values == "L"].mean(); sR = ok[pat.ez.values == "R"].mean()
        ctl_called = ((ctl[nl] > 0) | (ctl[nr] > 0)).mean(); ctl_tilt = (ctl[nl] != ctl[nr]).mean()
        print(f"{a.dataset} thesis rule ({scope}, |AI|>={THR:g}): called {called.sum()}/{len(pat)}, correct {ok.sum()}/{len(pat)} "
              f"(side acc among called {ok[called].mean():.2f}), balanced {np.mean([sL, sR]):.2f}; controls with any flagged ROI {ctl_called:.2f}, with a tilt {ctl_tilt:.2f}")
    print("wrote", out / f"{a.dataset}.csv")


if __name__ == "__main__":
    main()
