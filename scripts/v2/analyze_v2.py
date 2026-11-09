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
     construction), lateralization index LI_M = (M_R - M_L)/(M_L + M_R) (negative = left
     carries more hypoperfused mass), predict the side with the larger mass; temporal variant.

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
LABELS = [l for l, _, _ in PAIRS] + [r for _, r, _ in PAIRS]


def load_roi(feat_dir, sid):
    with open(feat_dir / f"{sid}_roi.json") as f:
        return json.load(f)


# --- partial-volume correction (2026-11-04, improvement list) -----------------------------------
def pvc_map(feat_dir, sid, gm_min=0.3):
    """Gray-matter CBF on the subsampled grid: CBF_gm = (CBF - WM * CBF_wm) / GM where GM >= gm_min.
    CBF_wm is the subject's median CBF in nearly pure white matter (wm > 0.9). Needs the `gm`/`wm`
    probability maps written by prep_subject_features.py --tissue. Returns (map, parc)."""
    z = np.load(feat_dir / f"{sid}.npz")
    if "gm" not in z.files or "wm" not in z.files:
        raise RuntimeError(f"{sid}: no tissue maps in {feat_dir / (sid + '.npz')} (run prep_subject_features.py --tissue)")
    perf = z["perf"].astype(np.float64); parc = z["parc"]
    gm = z["gm"].astype(np.float64); wm = z["wm"].astype(np.float64)
    wm_pure = (wm > 0.9) & (perf > 0)
    cbf_wm = float(np.median(perf[wm_pure])) if wm_pure.sum() > 50 else 0.0
    out = np.zeros_like(perf); ok = (gm >= gm_min) & (perf > 0)
    out[ok] = (perf[ok] - wm[ok] * cbf_wm) / gm[ok]
    out = np.clip(out, 0, 300)   # negative after correction = invalid for the ratio -> 0 (excluded)
    return out, parc


def roi_from_map(perf, parc):
    """ROI means on the subsampled grid, same layout as load_roi (per-label mean/n, gm_mean)."""
    phys = (perf > 0) & (perf < 300); lab = np.where(phys, parc, 0).ravel().astype(np.int64)
    sums = np.bincount(lab, weights=perf.ravel(), minlength=2100); cnts = np.bincount(lab, minlength=2100)
    roi = {str(L): {"mean": float(sums[L] / cnts[L]) if cnts[L] > 0 else None, "n": int(cnts[L])} for L in LABELS}
    gm_lab = np.isin(parc, LABELS) & phys
    roi["gm_mean"] = float(perf[gm_lab].mean()) if gm_lab.any() else None
    roi["gm_n"] = int(gm_lab.sum())
    return roi


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
    ap.add_argument("--feat-dir", default=None, help="feature directory (default v2_features/<dataset>)")
    ap.add_argument("--pvc", action="store_true", help="use partial-volume-corrected gray-matter CBF (needs --tissue features)")
    ap.add_argument("--no-nested-cv", action="store_true", help="skip the nested leave-one-out statistic selection")
    a = ap.parse_args()
    cfg = dataset_config.configure(a.dataset, perf_variant=getattr(a, "perf_variant", "tc"),
                                   band_policy=getattr(a, "band_policy", None))
    feat = Path(a.feat_dir) if a.feat_dir else BASE / "v2_features" / cfg.name
    pvc = bool(a.pvc)
    if pvc:
        def load_roi(feat_dir, sid):  # noqa: F811 - ROI means of the PVC map on the subsampled grid
            perf, parc = pvc_map(feat_dir, sid); return roi_from_map(perf, parc)
    else:
        load_roi = globals()["load_roi"]
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
    # global offset (2026-11-04): subject GM-mean CBF z-scored against the band's controls, reported next to the direction class
    ctrl_gm = {b: pd.Series({c: load_roi(feat, c)["gm_mean"] for c in ctrl_ids[b]}, dtype=float) for b in bands}
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
            o = ctrl_gm[b].drop(index=c); row["global_offset_z"] = float((ctrl_gm[b][c] - o.mean()) / o.std(ddof=1))
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
        gmp = load_roi(feat, p)["gm_mean"]
        row["global_offset_z"] = float((gmp - ctrl_gm[b].mean()) / ctrl_gm[b].std(ddof=1)) if gmp else np.nan
        pat_rows.append(row)
    pat_R = pd.DataFrame(pat_rows); pat_R.to_csv(out / "R_patients.csv", index=False)
    offset_rho = np.nan
    try:
        from scipy import stats as _st
        okr = np.isfinite(pat_R.global_offset_z.values)
        if okr.sum() >= 5:
            offset_rho, offset_p = _st.spearmanr(pat_R.global_offset_z.values[okr], (pat_R.R_direction_class.values[okr] == "hyper").astype(float))
            print(f"global offset vs direction class (hyper = 1), GM-normalised second z: Spearman rho = {offset_rho:.2f}, p = {offset_p:.3g}; "
                  f"offset range {np.nanmin(pat_R.global_offset_z):+.2f} to {np.nanmax(pat_R.global_offset_z):+.2f}")
    except Exception as exc:  # noqa: BLE001
        print(f"offset check skipped: {exc!r}")
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
        if pvc:
            return pvc_map(feat, sid)
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
    # Hemisphere masks. The symmetric template affine has a NEGATIVE x step (index 0 = x = +90 mm = right),
    # verified against the parcellation labels (left cortex 1000-1035 lies at indices >= centre), so the
    # LEFT hemisphere is the upper half of axis 0 on the subsampled grid as well.
    nx0 = acc[bands[0]]["S"].shape[0]
    left = np.zeros(acc[bands[0]]["S"].shape, bool); left[nx0 // 2 + 1:] = True
    right = np.zeros_like(left); right[: nx0 // 2] = True
    def masses(z, good, b, t, scope):
        m = good & scope
        hypoL = float(np.abs(z[m & left & (z <= -t)]).sum()); hypoR = float(np.abs(z[m & right & (z <= -t)]).sum())
        return hypoL, hypoR
    def vstats(z, good, b):
        row = {}
        for t in a.t:
            for nm, sc in (("all", acc[b]["scope"]), ("temporal", acc[b]["tscope"])):
                hl, hr = masses(z, good, b, t, sc)
                # LI_M = (M_R - M_L)/(M_L + M_R): negative when the LEFT hemisphere carries the larger
                # hypoperfused z-mass (left lower than right) -> predict a left EZ, like the sign of LI.
                row[f"V_{nm}_t{t:g}"] = (hr - hl) / (hl + hr) if (hl + hr) > 0 else 0.0
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
        # LI_M is bounded at +-1: when the control 95th percentile sits at the boundary, the rule is
        # "call whenever all hypoperfused mass is one-sided" and is NOT a 5 % calibration -> flag it
        m["threshold_at_boundary"] = bool(thr >= 0.999)
        summary_rows.append(m); st = strata_table(t, gt, cfg); st.insert(0, "statistic", k); strata_rows.append(st); per_patient[k] = t
        m2, _ = evaluate(pat_V, k, 0.0, gt, k + "_sign"); m2["control_fp_call_rate"] = 1.0; m2["dataset"] = cfg.name; summary_rows.append(m2)

    # ===================== 2026-11-04 extensions (improvement list, deck slide "What could be improved") =====
    from scipy import stats as _st
    from scipy import ndimage as _nd
    ext = {"pvc": pvc, "feat_dir": str(feat)}
    zmaps = {}   # id -> (z, good, band); leave-one-out for controls
    for b in bands:
        for c in ctrl_ids[b]:
            perf, _ = load_map(c); ai, v = ai_of(perf); z, good = z_of(ai, v, b, loo=c); zmaps[c] = (z, good, b)
    for p in patients:
        b = pband[p]; perf, _ = load_map(p); ai, v = ai_of(perf); z, good = z_of(ai, v, b); zmaps[p] = (z, good, b)
    ctrl_all = [c for b in bands for c in ctrl_ids[b]]

    # (1) heavier-tailed / empirical null: t* such that the median LOO-control GM fraction |z| >= t* is 5 %
    def gm_frac(z, good, b, t):
        m = good & acc[b]["scope"]; return float((np.abs(z[m]) >= t).mean()) if m.any() else np.nan
    ts = np.arange(1.5, 4.001, 0.05)
    med = np.array([np.nanmedian([gm_frac(*zmaps[c], t) for c in ctrl_all]) for t in ts])
    t_star = float(ts[int(np.argmin(np.abs(med - 0.05)))])
    ext["t_star_5pct"] = round(t_star, 3)
    ext["ctrl_median_gm_frac_at_1.96"] = round(float(np.nanmedian([gm_frac(*zmaps[c], 1.96) for c in ctrl_all])), 4)
    ext["ctrl_median_gm_frac_at_t_star"] = round(float(med[int(np.argmin(np.abs(med - 0.05)))]), 4)
    n_ctrl = min(len(ctrl_ids[b]) for b in bands)
    ext["t_cut_student_97.5"] = round(float(_st.t.ppf(0.975, n_ctrl - 2)), 4)

    # (2) cluster-level inference: largest hypoperfused cluster per hemisphere, extent counted AT the threshold,
    #     26-connectivity, minimum 2 voxels (>= 1 native ASL voxel of 15.6 mm3 on the 2.4-mm grid), null from LOO controls
    struct = np.ones((3, 3, 3), bool); MIN_EXT = 2
    def max_cluster(z, good, t, scope):
        m = good & scope; res = {}
        for side, hm in (("L", left), ("R", right)):
            lab, n = _nd.label((z <= -t) & m & hm, structure=struct)
            if n == 0:
                res[side] = 0; continue
            sizes = np.bincount(lab.ravel())[1:]; mx = int(sizes.max()); res[side] = mx if mx >= MIN_EXT else 0
        return res
    def cluster_rows(sid, t):
        z, good, b = zmaps[sid]; row = {}
        for nm, sc in (("all", acc[b]["scope"]), ("temporal", acc[b]["tscope"])):
            mc = max_cluster(z, good, t, sc); l, r = mc["L"], mc["R"]
            row[f"C_{nm}_t{t:g}_maxL"], row[f"C_{nm}_t{t:g}_maxR"] = l, r
            row[f"C_{nm}_t{t:g}"] = (r - l) / (l + r) if (l + r) > 0 else 0.0     # negative = larger hypo cluster on the left -> predict L
            row[f"C_{nm}_t{t:g}_max"] = max(l, r)
        return row
    for t in sorted({2.0, round(t_star, 2)}):
        loo_C = pd.DataFrame([{"id": c, **cluster_rows(c, t)} for c in ctrl_all])
        pat_C = pd.DataFrame([{"id": p, "ez": gt.get(p, {}).get("ez"), **cluster_rows(p, t)} for p in patients])
        loo_C.to_csv(out / f"C_loo_controls_t{t:g}.csv", index=False); pat_C.to_csv(out / f"C_patients_t{t:g}.csv", index=False)
        for nm in ("all", "temporal"):
            k = f"C_{nm}_t{t:g}"
            null95 = float(np.nanpercentile(loo_C[f"{k}_max"], 95))
            def call_of(df, k=k, null95=null95):
                big = df[[f"{k}_maxL", f"{k}_maxR"]].max(axis=1).values > null95
                return np.where(big, np.where(df[k].values < 0, "L", "R"), "none")
            fp = float(np.mean(call_of(loo_C) != "none"))
            df = pat_C.copy(); df["value"] = df[k]
            m, tt = evaluate(df, "value", 1e-9, gt, f"{k}_sign"); m["control_fp_call_rate"] = 1.0; m["dataset"] = cfg.name
            m["note"] = "side of the larger hypoperfused cluster (extent at t, 26-conn, >= 1 native voxel)"; summary_rows.append(m)
            per_patient[f"{k}_sign"] = tt
            calls = call_of(pat_C); sel = pat_C.ez.isin(["L", "R"]).values; u = pat_C[sel]; cu = calls[sel]
            nL, nR = int((u.ez == "L").sum()), int((u.ez == "R").sum())
            agree = int((cu == u.ez.values).sum()); ncalled = int((cu != "none").sum())
            sensL = ((cu == "L") & (u.ez.values == "L")).sum() / nL if nL else np.nan
            sensR = ((cu == "R") & (u.ez.values == "R")).sum() / nR if nR else np.nan
            summary_rows.append(dict(statistic=f"{k}_clusternull", n=len(u), nL=nL, nR=nR, threshold=round(null95, 1), agree=agree,
                                     balanced_acc=round(float(0.5 * (sensL + sensR)), 3), majority_baseline=round(max(nL, nR) / len(u), 3) if len(u) else np.nan,
                                     wilson_lo=round(wilson(agree, len(u))[0], 3), wilson_hi=round(wilson(agree, len(u))[1], 3),
                                     n_called=ncalled, acc_among_called=round(agree / ncalled, 3) if ncalled else np.nan,
                                     control_fp_call_rate=round(fp, 3), dataset=cfg.name,
                                     note=f"calibrated: larger cluster must exceed the 95th pct of the LOO-control largest hypo cluster ({null95:.0f} voxels at 2.4 mm)"))
            ext[f"{k}_null95_voxels"] = round(null95, 1)

    # (3) threshold families at the smoothed scale, ranked by control false-positive coverage
    def fam_thresholds(z, good, b):
        m = good & acc[b]["scope"]; az = np.abs(z[m]); N = int(az.size)
        p = 2 * (1 - _st.norm.cdf(az)); ps = np.sort(p); kk = np.arange(1, N + 1)
        okf = ps <= 0.05 * kk / N; fdr_p = float(ps[okf].max()) if okf.any() else 0.0
        th = {"fixed_1.96": 1.96, "fixed_2.5": 2.5, "fixed_3": 3.0, "student_t": ext["t_cut_student_97.5"], "empirical_t*": t_star,
              "FDR_q0.05": float(_st.norm.ppf(1 - fdr_p / 2)) if fdr_p > 0 else np.inf,
              "Bonferroni_0.05": float(_st.norm.ppf(1 - 0.025 / max(N, 1)))}
        c1, c2 = float(np.percentile(az, 25)), float(np.percentile(az, 90))
        for _ in range(30):   # two-means split of |z| (data-driven, Otsu-like, as M9)
            lab = az > (c1 + c2) / 2
            c1n = float(az[~lab].mean()) if (~lab).any() else c1; c2n = float(az[lab].mean()) if lab.any() else c2
            if abs(c1n - c1) < 1e-4 and abs(c2n - c2) < 1e-4:
                break
            c1, c2 = c1n, c2n
        th["two_means_|z|"] = float((c1 + c2) / 2)
        return th
    FAMS = ["fixed_1.96", "fixed_2.5", "fixed_3", "student_t", "empirical_t*", "FDR_q0.05", "Bonferroni_0.05", "two_means_|z|"]
    def fam_stats(sid):
        z, good, b = zmaps[sid]; th = fam_thresholds(z, good, b); m = good & acc[b]["scope"]; row = {}
        for name, t in th.items():
            if np.isfinite(t):
                cov = float((np.abs(z[m]) >= t).mean()) if m.any() else 0.0
                hl = int(((z <= -t) & m & left).sum()); hr = int(((z <= -t) & m & right).sum())
            else:
                cov, hl, hr = 0.0, 0, 0
            row[f"{name}_cov"] = cov; row[f"{name}_side"] = (hr - hl) / (hl + hr) if (hl + hr) > 0 else 0.0; row[f"{name}_t"] = t
        return row
    fam_c = pd.DataFrame([{"id": c, **fam_stats(c)} for c in ctrl_all])
    fam_p = pd.DataFrame([{"id": p, "ez": gt.get(p, {}).get("ez"), **fam_stats(p)} for p in patients])
    fam_rows = []
    for name in FAMS:
        u = fam_p[fam_p.ez.isin(["L", "R"])]; v = u[f"{name}_side"].values; ez = u.ez.values
        pred = np.where(v < 0, "L", np.where(v > 0, "R", "none")); agree = int((pred == ez).sum()); called = int((pred != "none").sum())
        cov_c = fam_c[f"{name}_cov"]; cov_p = u[f"{name}_cov"]
        thr = float(np.nanpercentile(np.abs(fam_c[f"{name}_side"]), 95)); cal = np.abs(v) >= thr
        tvals = fam_p[f"{name}_t"].replace(np.inf, np.nan)
        fam_rows.append(dict(dataset=cfg.name, family=name, threshold_median=round(float(np.nanmedian(tvals)), 3) if tvals.notna().any() else np.nan,
                             control_cov_median=round(float(cov_c.median()), 4), control_cov_p95=round(float(cov_c.quantile(0.95)), 4),
                             patient_cov_median=round(float(cov_p.median()), 4),
                             cov_auc=round(float(np.mean(cov_p.values[:, None] > cov_c.values[None, :]) + 0.5 * np.mean(cov_p.values[:, None] == cov_c.values[None, :])), 3),
                             n=len(u), called=called, agree=agree, precision=round(agree / called, 3) if called else np.nan,
                             calibrated_threshold=round(thr, 3), calibrated_called=int(cal.sum()), calibrated_correct=int((cal & (pred == ez)).sum()),
                             control_fp=round(float((np.abs(fam_c[f"{name}_side"]) >= thr).mean()), 3)))
    fam_tab = pd.DataFrame(fam_rows); fam_tab.to_csv(out / "threshold_families.csv", index=False)

    # (4) nested leave-one-out over patients: choose the statistic on the other patients, score the held-out one
    if not a.no_nested_cv:
        cand = {"R_temporal": pat_R.set_index("id")["R_temporal"], "R_all37": pat_R.set_index("id")["R_all37"],
                "R_mesial": pat_R.set_index("id")["R_mesial"], "V_all_t2": pat_V.set_index("id")["V_all_t2"]}
        ids = [p for p in patients if gt.get(p, {}).get("ez") in ("L", "R") and all(np.isfinite(cand[k][p]) for k in cand)]
        ezs = {p: gt[p]["ez"] for p in ids}
        correct = {k: {p: bool((("L" if cand[k][p] < 0 else "R") == ezs[p])) for p in ids} for k in cand}
        hits = 0; picks = []
        for p in ids:
            others = [q for q in ids if q != p]
            best = max(cand, key=lambda k: np.mean([correct[k][q] for q in others])); picks.append(best); hits += int(correct[best][p])
        picked = {k: int(v) for k, v in pd.Series(picks).value_counts().items()}
        nL = sum(v == "L" for v in ezs.values()); nR = len(ids) - nL
        ext["nested_loo_sign"] = {"agree": hits, "n": len(ids), "picked": picked}
        summary_rows.append(dict(statistic="nested_LOO_best_statistic_sign", n=len(ids), nL=nL, nR=nR, threshold=np.nan, agree=hits,
                                 balanced_acc=np.nan, majority_baseline=round(max(nL, nR) / len(ids), 3) if ids else np.nan,
                                 wilson_lo=round(wilson(hits, len(ids))[0], 3) if ids else np.nan, wilson_hi=round(wilson(hits, len(ids))[1], 3) if ids else np.nan,
                                 n_called=len(ids), acc_among_called=round(hits / len(ids), 3) if ids else np.nan,
                                 control_fp_call_rate=1.0, dataset=cfg.name,
                                 note="statistic chosen on the other patients (sign rules); picked " + ", ".join(f"{k}:{v}" for k, v in picked.items())))
    ext["offset_vs_direction_spearman"] = None if not np.isfinite(offset_rho) else round(float(offset_rho), 3)
    with open(out / "extensions.json", "w") as fh:
        json.dump(ext, fh, indent=1, default=str)
    print(f"\n=== extensions: t* (median control GM fraction 5 %) = {t_star:.2f} (fraction at 1.96: {ext['ctrl_median_gm_frac_at_1.96']:.3f}) | "
          f"Student-t 97.5 % cut = {ext['t_cut_student_97.5']:.3f} | nested LOO: {ext.get('nested_loo_sign')}")
    print("\n=== threshold families (smoothed scale) ===")
    print(fam_tab.to_string(index=False))

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
