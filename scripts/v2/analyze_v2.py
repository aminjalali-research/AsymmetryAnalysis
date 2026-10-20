#!/usr/bin/env python3
"""v2 pipeline, step 2: enhanced normative asymmetry analysis, identical for both datasets.

Two estimators, both referenced to controls of the same dataset and calibrated on
leave-one-out (LOO) controls so that the false-positive call rate is known:

  R  Regional normative LI.  LI_r = (CBF_L - CBF_R) / (CBF_L + CBF_R) per region pair
     (subject's own parcellation / fixed atlas).  Controls (per band) give mean and SD;
     patients and LOO controls get z_r.  Summary statistics: mean z over the mesial
     temporal pair set (hippocampus, amygdala), over the temporal set, and over all 37.
     Negative z = left lower than its homologue = left hypoperfused -> predict L.
     Direction (offset-free): per-side CBF normalised by the subject's GM mean, z vs
     controls; the deviating side is the one with the larger |z| in the mesial set.

  V  Smoothed voxel-wise normative AI.  AI from 8-mm-smoothed CBF on the 2.4-mm
     mirror-symmetric grid; z vs controls (Welford, LOO for controls); scope = symmetrised
     37-ROI gray matter.  Hypoperfusion mass per hemisphere at |z| >= t:
     M_hypo(L) = sum_{v in L, z(v) <= -t} |z(v)| (each mirrored pair counted once by
     construction), lateralization index LI_V = (M_L - M_R)/(M_L + M_R), predict the
     side with the larger hypoperfused mass; temporal-scope variant.

Calibration: for every statistic the LOO-control |stat| 95th percentile is the call
threshold; a patient below it gets "no call" (counted as wrong in balanced accuracy,
as pre-specified for Chalfont).  Reported: agreement, balanced accuracy with the
majority baseline, Wilson CI, control false-positive call rate, strata.

Usage: python scripts/v2/analyze_v2.py --dataset inhouse|chalfont [--out ez_analysis_mdt/v2]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
import dataset_config  # noqa: E402

PAIRS = [(17, 53, "hippocampus"), (18, 54, "amygdala"), (10, 49, "thalamus")] + \
        [(1000 + i, 2000 + i, f"ctx{i}") for i in range(1, 36) if i != 4]
NAME = {n: n for _, _, n in PAIRS}
MESIAL = ["hippocampus", "amygdala"]
TEMPORAL = ["hippocampus", "amygdala", "ctx6", "ctx7", "ctx9", "ctx15", "ctx16", "ctx30", "ctx33", "ctx34"]
TEMPORAL_CTX_LABELS = [1006, 1007, 1009, 1015, 1016, 1030, 1033, 1034, 2006, 2007, 2009, 2015, 2016, 2030, 2033, 2034]
SCOPE_LABELS = [10, 49, 17, 53, 18, 54] + list(range(1001, 1036)) + list(range(2001, 2036))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ----------------------------------------------------------------------------- features
def load_roi(feat_dir, sid):
    with open(feat_dir / f"{sid}_roi.json") as f:
        return json.load(f)


def li_vector(roi):
    out = {}
    for l, r, n in PAIRS:
        a, b = roi[str(l)]["mean"], roi[str(r)]["mean"]
        out[n] = (a - b) / (a + b) if a and b and (a + b) > 0 else np.nan
    return out


def rel_side_vector(roi):
    g = roi["gm_mean"]
    out = {}
    for l, r, n in PAIRS:
        for lab, side in ((l, "L"), (r, "R")):
            m = roi[str(lab)]["mean"]
            out[f"{n}_{side}"] = m / g if (m and g) else np.nan
    return out


def zscore_frame(ctrl: pd.DataFrame, x: pd.Series, loo_index=None):
    """z of x against ctrl rows (dropping loo_index if given)."""
    c = ctrl.drop(index=loo_index) if loo_index is not None else ctrl
    return (x - c.mean()) / c.std(ddof=1)


# ----------------------------------------------------------------------------- evaluation
def evaluate(df, stat, thr, gt, label):
    """df: rows with id, stat value; gt: {id: ez}. Returns dict of metrics."""
    rows = []
    for _, r in df.iterrows():
        ez = gt.get(r["id"], {}).get("ez")
        v = r[stat]
        call = ("L" if v < 0 else "R") if (np.isfinite(v) and abs(v) >= thr) else "none"
        rows.append((r["id"], ez, v, call))
    t = pd.DataFrame(rows, columns=["id", "ez", "value", "call"])
    u = t[t.ez.isin(["L", "R"])]
    nL, nR = int((u.ez == "L").sum()), int((u.ez == "R").sum())
    agree = (u.call == u.ez)
    sensL = agree[u.ez == "L"].mean() if nL else np.nan
    sensR = agree[u.ez == "R"].mean() if nR else np.nan
    called = u[u.call != "none"]
    acc_called = (called.call == called.ez).mean() if len(called) else np.nan
    # direction-free orientation check: sign of value vs side (AUC-like, no threshold)
    sgn = u[np.isfinite(u.value)]
    sign_agree = ((sgn.value < 0) == (sgn.ez == "L")).mean() if len(sgn) else np.nan
    bal_sign = 0.5 * ((((sgn.value < 0) == (sgn.ez == "L"))[sgn.ez == "L"]).mean() +
                      (((sgn.value < 0) == (sgn.ez == "L"))[sgn.ez == "R"]).mean()) if nL and nR else np.nan
    return dict(statistic=label, n=len(u), nL=nL, nR=nR, threshold=round(float(thr), 3),
                agree=int(agree.sum()), balanced_acc=round(float(0.5 * (sensL + sensR)), 3),
                majority_baseline=round(max(nL, nR) / len(u), 3) if len(u) else np.nan,
                wilson_lo=round(wilson(int(agree.sum()), len(u))[0], 3), wilson_hi=round(wilson(int(agree.sum()), len(u))[1], 3),
                n_called=len(called), acc_among_called=round(float(acc_called), 3) if len(called) else np.nan,
                sign_agree_no_threshold=round(float(sign_agree), 3), balanced_acc_sign_only=round(float(bal_sign), 3)), t


def strata_table(t, gt, cfg):
    out = []
    for sid_mask_name, fn in [
        ("all", lambda g: True),
        ("MRI lesional", lambda g: (g.get("group") == "discrete") or ("MRI:" in str(g.get("evidence", "")) and "normal" not in str(g.get("evidence", "")).lower())),
        ("MRI-negative", lambda g: (g.get("group") == "negative") or ("normal" in str(g.get("evidence", "")).lower())),
        ("temporal", lambda g: (g.get("ez_lobe") == "temporal") or ("TLE" in str(g.get("thesis_dx", "")))),
        ("extratemporal/multilobar", lambda g: (g.get("ez_lobe") not in (None, "temporal")) or ("PLE" in str(g.get("thesis_dx", "")))),
        ("excl. post-surgical", lambda g: True),
    ]:
        ids = [i for i in t.id if i in gt and fn(gt[i])]
        if sid_mask_name == "excl. post-surgical":
            ids = [i for i in ids if i not in cfg.post_surgical]
        u = t[t.id.isin(ids) & t.ez.isin(["L", "R"])]
        if len(u) == 0:
            continue
        agree = (u.call == u.ez); nL = (u.ez == "L").sum(); nR = (u.ez == "R").sum()
        bal = 0.5 * (agree[u.ez == "L"].mean() + agree[u.ez == "R"].mean()) if nL and nR else np.nan
        out.append(dict(stratum=sid_mask_name, n=len(u), nL=int(nL), nR=int(nR), agree=int(agree.sum()),
                        balanced_acc=round(float(bal), 3), majority=round(max(nL, nR) / len(u), 3)))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); dataset_config.add_dataset_args(ap)
    ap.add_argument("--out", default=None); ap.add_argument("--t", nargs="*", type=float, default=[2.0, 2.5, 3.0])
    a = ap.parse_args()
    cfg = dataset_config.configure(a.dataset, perf_variant=getattr(a, "perf_variant", "tc"),
                                   band_policy=getattr(a, "band_policy", None))
    feat = BASE / "v2_features" / cfg.name
    out = Path(a.out) if a.out else BASE / "ez_analysis_mdt" / "v2" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    gt = cfg.ground_truth()
    patients = [p for p in cfg.patient_ids() if (feat / f"{p}.npz").exists()]
    bands = [g for g in cfg.available_groups() if len(cfg.control_ids(g)) >= 10]
    if cfg.is_inhouse:
        bands = ["FM_20_39", "FM_40_59", "FM_60_79"]
    else:
        bands = [g for g in bands if cfg._is_all_ages(g)] or bands[:1]
    ctrl_ids = {b: [c for c in cfg.control_ids(b) if (feat / f"{c}.npz").exists()] for b in bands}
    # patient -> band
    def band_of(pid):
        if not cfg.is_inhouse:
            return bands[0]
        # in-house: the band recorded by step 01 (results_zscore/patients/<ID>/<ID>_vs_<band>_summary.json)
        import glob, re
        for f in glob.glob(str(BASE / "results_zscore" / "patients" / pid / f"{pid}_vs_*_summary.json")):
            m = re.search(r"_vs_(FM_\d+_\d+)_summary", f)
            if m and m.group(1) in bands:
                return m.group(1)
        return None
    pband = {p: band_of(p) for p in patients}
    patients = [p for p in patients if pband[p]]
    print(f"dataset={cfg.name} bands={bands} controls={ {b: len(v) for b, v in ctrl_ids.items()} } patients={len(patients)}")
    summary_rows, strata_rows, per_patient = [], [], {}

    # ============================ R: regional ==================================
    ctrl_li = {b: pd.DataFrame([li_vector(load_roi(feat, c)) for c in ctrl_ids[b]], index=ctrl_ids[b]) for b in bands}
    ctrl_rel = {b: pd.DataFrame([rel_side_vector(load_roi(feat, c)) for c in ctrl_ids[b]], index=ctrl_ids[b]) for b in bands}
    sd_tab = pd.DataFrame({b: 100 * ctrl_li[b].std() for b in bands}); sd_tab.to_csv(out / "R_control_LI_sd_percent.csv")
    stats_R = {"R_mesial": MESIAL, "R_temporal": TEMPORAL, "R_all37": [n for _, _, n in PAIRS]}
    # LOO controls
    loo_rows = []
    for b in bands:
        for c in ctrl_ids[b]:
            z = zscore_frame(ctrl_li[b], ctrl_li[b].loc[c], loo_index=c)
            zr = zscore_frame(ctrl_rel[b], ctrl_rel[b].loc[c], loo_index=c)
            row = {"id": c, "band": b}
            for k, s in stats_R.items():
                row[k] = float(np.nanmean([z[n] for n in s]))
            row["R_mesial_dev"] = float(np.nanmean([zr[f"{n}_L"] for n in MESIAL]) - np.nanmean([zr[f"{n}_R"] for n in MESIAL]))
            loo_rows.append(row)
    loo_R = pd.DataFrame(loo_rows); loo_R.to_csv(out / "R_loo_controls.csv", index=False)
    pat_rows = []
    for p in patients:
        b = pband[p]; li = pd.Series(li_vector(load_roi(feat, p))); rel = pd.Series(rel_side_vector(load_roi(feat, p)))
        z = zscore_frame(ctrl_li[b], li); zr = zscore_frame(ctrl_rel[b], rel)
        row = {"id": p, "band": b, "ez": gt.get(p, {}).get("ez")}
        for k, s in stats_R.items():
            row[k] = float(np.nanmean([z[n] for n in s]))
        # offset-free direction: deviating side in the mesial set from GM-normalised CBF
        zl = float(np.nanmean([zr[f"{n}_L"] for n in MESIAL])); zrr = float(np.nanmean([zr[f"{n}_R"] for n in MESIAL]))
        row["R_mesial_relz_L"], row["R_mesial_relz_R"] = zl, zrr
        row["R_mesial_dev"] = zl - zrr
        dev_side = "L" if abs(zl) > abs(zrr) else "R"
        row["R_direction_side"] = dev_side; row["R_direction_class"] = ("hypo" if (zl if dev_side == "L" else zrr) < 0 else "hyper")
        row["R_n_roi_abs_z_gt2"] = int(np.nansum(np.abs(z.values) > 2))
        for n in ["hippocampus", "amygdala"]:
            row[f"z_{n}"] = float(z[n])
        pat_rows.append(row)
    pat_R = pd.DataFrame(pat_rows); pat_R.to_csv(out / "R_patients.csv", index=False)
    for k in list(stats_R) + ["R_mesial_dev"]:
        thr = float(np.nanpercentile(np.abs(loo_R[k]), 95))
        fp = float((np.abs(loo_R[k]) >= thr).mean())
        m, t = evaluate(pat_R, k, thr, gt, k); m["control_fp_call_rate"] = round(fp, 3); m["dataset"] = cfg.name
        summary_rows.append(m); st = strata_table(t, gt, cfg); st.insert(0, "statistic", k); strata_rows.append(st)
        per_patient[k] = t
        # uncalibrated sign rule as well
        m2, t2 = evaluate(pat_R, k, 0.0, gt, k + "_sign"); m2["control_fp_call_rate"] = 1.0; m2["dataset"] = cfg.name
        summary_rows.append(m2)

    # ============================ V: smoothed voxel-wise =======================
    def load_map(sid):
        z = np.load(feat / f"{sid}.npz"); return z["perf"].astype(np.float64), z["parc"]
    def ai_of(perf):
        L = perf; R = np.flip(L, 0); v = (L > 0) & (R > 0); ai = np.zeros_like(L); ai[v] = 100 * (L[v] - R[v]) / ((L[v] + R[v]) / 2); return ai, v
    # scope masks from the band's control parcellations (majority) -> symmetrised
    for b in bands:
        pass
    acc = {}
    for b in bands:
        S = SS = C = None; votes = None
        for c in ctrl_ids[b]:
            perf, parc = load_map(c); ai, v = ai_of(perf)
            if S is None:
                S = np.zeros_like(ai); SS = np.zeros_like(ai); C = np.zeros(ai.shape, np.int32); votes = np.zeros(ai.shape, np.int32); votes_t = np.zeros(ai.shape, np.int32)
            S[v] += ai[v]; SS[v] += ai[v] ** 2; C[v] += 1
            votes += np.isin(parc, SCOPE_LABELS); votes_t += np.isin(parc, TEMPORAL_CTX_LABELS + [17, 53, 18, 54])
        n = len(ctrl_ids[b]); scope = votes > n / 2; scope &= np.flip(scope, 0); tscope = votes_t > n / 2; tscope &= np.flip(tscope, 0)
        acc[b] = dict(S=S, SS=SS, C=C, scope=scope, tscope=tscope, n=n)
    def z_of(ai, v, b, loo=None):
        A = acc[b]; S, SS, C = A["S"].copy(), A["SS"].copy(), A["C"].copy()
        if loo is not None:
            S[v] -= ai[v]; SS[v] -= ai[v] ** 2; C[v] -= 1
        ok = C >= 10; m = np.where(ok, S / np.maximum(C, 1), 0); var = np.where(ok, SS / np.maximum(C, 1) - m ** 2, 0)
        var = var * C / np.maximum(C - 1, 1); sd = np.sqrt(np.maximum(var, 0)); sd = np.maximum(sd, np.maximum(sd, np.flip(sd, 0)))
        z = np.zeros_like(ai); good = ok & v & (sd > 1e-6); z[good] = (ai[good] - m[good]) / sd[good]; return np.clip(z, -20, 20), good
    left = np.zeros(acc[bands[0]]["S"].shape, bool); left[: left.shape[0] // 2] = True  # axis 0 index < centre = x < 0 = left (RAS)
    def masses(z, good, b, t, scope):
        m = good & scope
        hypoL = float(np.abs(z[m & left & (z <= -t)]).sum()); hypoR = float(np.abs(z[m & ~left & (z <= -t)]).sum())
        return hypoL, hypoR
    def vstats(z, good, b):
        row = {}
        for t in a.t:
            for nm, sc in (("all", acc[b]["scope"]), ("temporal", acc[b]["tscope"])):
                hl, hr = masses(z, good, b, t, sc)
                row[f"V_{nm}_t{t:g}"] = (hl - hr) / (hl + hr) if (hl + hr) > 0 else 0.0   # <0: left mass larger -> left hypo -> predict L
                row[f"V_{nm}_t{t:g}_massL"], row[f"V_{nm}_t{t:g}_massR"] = hl, hr
        gm = good & acc[b]["scope"]; row["V_frac_gm_abs_z_ge_1.96"] = float((np.abs(z[gm]) >= 1.96).mean()) if gm.any() else np.nan
        return row
    loo_rows = []
    for b in bands:
        for c in ctrl_ids[b]:
            perf, _ = load_map(c); ai, v = ai_of(perf); z, good = z_of(ai, v, b, loo=c)
            loo_rows.append({"id": c, "band": b, **vstats(z, good, b)})
    loo_V = pd.DataFrame(loo_rows); loo_V.to_csv(out / "V_loo_controls.csv", index=False)
    pat_rows = []
    for p in patients:
        b = pband[p]; perf, _ = load_map(p); ai, v = ai_of(perf); z, good = z_of(ai, v, b)
        pat_rows.append({"id": p, "band": b, "ez": gt.get(p, {}).get("ez"), **vstats(z, good, b)})
    pat_V = pd.DataFrame(pat_rows); pat_V.to_csv(out / "V_patients.csv", index=False)
    print(f"calibration: GM frac |z|>=1.96  controls(LOO) median {loo_V['V_frac_gm_abs_z_ge_1.96'].median():.3f}  patients median {pat_V['V_frac_gm_abs_z_ge_1.96'].median():.3f}")
    for k in [c for c in pat_V.columns if c.startswith("V_") and "mass" not in c and "frac" not in c]:
        thr = float(np.nanpercentile(np.abs(loo_V[k]), 95)); fp = float((np.abs(loo_V[k]) >= thr).mean())
        m, t = evaluate(pat_V, k, thr, gt, k); m["control_fp_call_rate"] = round(fp, 3); m["dataset"] = cfg.name
        summary_rows.append(m); st = strata_table(t, gt, cfg); st.insert(0, "statistic", k); strata_rows.append(st); per_patient[k] = t
        m2, _ = evaluate(pat_V, k, 0.0, gt, k + "_sign"); m2["control_fp_call_rate"] = 1.0; m2["dataset"] = cfg.name; summary_rows.append(m2)

    summ = pd.DataFrame(summary_rows); summ.to_csv(out / "summary.csv", index=False)
    pd.concat(strata_rows).to_csv(out / "strata.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n=== control LI SD (percent) per band, selected ROIs ===")
    print(sd_tab.loc[["hippocampus", "amygdala", "thalamus", "ctx15", "ctx30", "ctx28"]].round(1).to_string(), "\n median over 37:", sd_tab.median().round(1).to_dict())
    print("\n=== summary (calibrated calls: threshold = LOO-control 95th percentile of |stat|; *_sign = no threshold) ===")
    cols = ["statistic", "n", "nL", "nR", "threshold", "agree", "n_called", "acc_among_called", "balanced_acc", "majority_baseline", "wilson_lo", "wilson_hi", "control_fp_call_rate", "balanced_acc_sign_only"]
    print(summ[cols].to_string(index=False))
    print("\n=== strata (calibrated) ===")
    print(pd.concat(strata_rows).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
