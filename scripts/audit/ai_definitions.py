#!/usr/bin/env python3
"""Does the form of the asymmetry index matter?  (audit 2026-11-04)

The pipeline uses AI = 200 (L - R)/(L + R) = 100 (L - R)/mean(L, R): the left-right difference as a percent
of the pair mean.  The constant is cosmetic: z = (AI - mu)/sigma is invariant to any constant factor, so
AI/200 = LI in [-1, 1] gives the identical z map, cluster set and rule output (checked numerically below).
What CAN change the result is the functional form, the normalisation and the scale.  Six definitions are
scored identically on both cohorts, at two scales, against the same-dataset controls with leave-one-out
(LOO) calibration, exactly like scripts/v2/analyze_v2.py:

  ratio      (L - R)/(L + R)                      current (x200 = AI)
  logratio   ln(L / R)                            symmetric in the log domain; ~2 x ratio for small asymmetry
  diff       L - R                                ml/100 g/min; scale depends on global CBF
  diff_gm    (L - R)/GM mean of the subject       global-offset-free difference
  zdiff_raw  z_L - z_R, z per side vs controls    analysis 3 (perfusion z) turned into an asymmetry
  zdiff_gm   same on CBF/GM-mean                  analysis 3 without the global offset

Scales: regional (ROI means, v2_features/<dataset>/<ID>_roi.json; temporal set of 10 pairs, mesial 2, all 37)
and voxel-wise (8-mm smoothed CBF on the mirror-symmetric 2.4-mm grid, v2_features/<dataset>/<ID>.npz;
hypoperfused z-mass LI at |z| >= 2 in the temporal and the 37-ROI scope).  For every statistic: sign agreement
with the reference side (negative = left lower = predict L), balanced accuracy, calibrated call at the LOO-control
95th percentile with its control false-positive rate, detection AUC (|stat| patients vs LOO controls) and the best
balanced operating point max_t min(sensitivity, specificity).

Output: ez_analysis_mdt/audit/ai_definitions/{summary.csv, per_subject_<dataset>.csv, identity_check.json}
Usage: python scripts/audit/ai_definitions.py [--dataset chalfont|inhouse|both]
"""
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE / "scripts" / "v2"))
import dataset_config  # noqa: E402
import analyze_v2 as A  # noqa: E402

OUT = BASE / "ez_analysis_mdt" / "audit" / "ai_definitions"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
REG_DEFS = ["ratio", "logratio", "diff", "diff_gm", "zdiff_raw", "zdiff_gm"]
VOX_DEFS = ["ratio", "logratio", "diff", "diff_gm", "zdiff_raw", "zdiff_gm"]
T = 2.0


# ----------------------------------------------------------------------------- scoring helpers
def auc_ci(pos, neg, n=1000):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan, np.nan
    def _a(p, q): return float(np.mean(p[:, None] > q[None, :]) + 0.5 * np.mean(p[:, None] == q[None, :]))
    a = _a(pos, neg)
    bs = [_a(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n)]
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def best_point(s_pat, ez, s_ctl):
    s_pat = np.asarray(s_pat, float); s_ctl = np.asarray(s_ctl, float); ez = np.asarray(ez)
    ok = np.isfinite(s_pat); s_pat, ez = s_pat[ok], ez[ok]; s_ctl = s_ctl[np.isfinite(s_ctl)]
    correct = np.where(s_pat < 0, "L", "R") == ez
    best = (-1.0, np.nan, np.nan, np.nan)
    for t in np.unique(np.concatenate([np.abs(s_pat), np.abs(s_ctl), [0.0]])):
        sens = float(np.mean((np.abs(s_pat) >= t) & correct)); spec = float(np.mean(np.abs(s_ctl) < t))
        if min(sens, spec) > best[0]:
            best = (min(sens, spec), t, sens, spec)
    return best


def score(name, dataset, scale, pat, ctl, gt):
    """pat: DataFrame id, value; ctl: DataFrame id, value (LOO controls)."""
    gtm = {p: gt.get(p, {}).get("ez") for p in pat.id}
    u = pat[[gtm[p] in ("L", "R") for p in pat.id]].copy(); u["ez"] = [gtm[p] for p in u.id]
    u = u[np.isfinite(u.value)]
    nL, nR = int((u.ez == "L").sum()), int((u.ez == "R").sum())
    pred = np.where(u.value < 0, "L", "R"); agree = pred == u.ez.values
    sensL = agree[u.ez.values == "L"].mean() if nL else np.nan; sensR = agree[u.ez.values == "R"].mean() if nR else np.nan
    thr = float(np.nanpercentile(np.abs(ctl.value), 95))
    called = np.abs(u.value.values) >= thr
    fp = float(np.mean(np.abs(ctl.value.values) >= thr))
    auc, lo, hi = auc_ci(np.abs(u.value.values), np.abs(ctl.value.values))
    bp = best_point(u.value.values, u.ez.values, ctl.value.values)
    lo_w, hi_w = A.wilson(int(agree.sum()), len(u))
    return dict(dataset=dataset, scale=scale, definition=name, n=len(u), nL=nL, nR=nR,
                sign_agree=int(agree.sum()), sign_acc=round(float(agree.mean()), 3),
                balanced=round(float(0.5 * (sensL + sensR)), 3), wilson=f"{lo_w:.2f}-{hi_w:.2f}",
                thr=round(thr, 4), n_called=int(called.sum()), called_correct=int((called & agree).sum()),
                control_fp=round(fp, 3), det_auc=round(auc, 3), det_auc_ci=f"{lo:.2f}-{hi:.2f}",
                best_min_sens_spec=round(bp[0], 3), best_t=round(float(bp[1]), 4) if np.isfinite(bp[1]) else np.nan,
                ctrl_sd_raw=round(float(np.nanstd(ctl.value.values)), 4))


# ----------------------------------------------------------------------------- regional
def pair_values(roi, kind):
    g = roi["gm_mean"]; out = {}
    for l, r, n in A.PAIRS:
        a, b = roi[str(l)]["mean"], roi[str(r)]["mean"]
        if not (a and b and a > 0 and b > 0 and g):
            out[n] = np.nan; continue
        if kind == "ratio": out[n] = (a - b) / (a + b)
        elif kind == "ratio200": out[n] = 200 * (a - b) / (a + b)
        elif kind == "logratio": out[n] = float(np.log(a / b))
        elif kind == "diff": out[n] = a - b
        elif kind == "diff_gm": out[n] = (a - b) / g
        elif kind == "L_raw": out[n] = a
        elif kind == "R_raw": out[n] = b
        elif kind == "L_gm": out[n] = a / g
        elif kind == "R_gm": out[n] = b / g
    return out


def regional(cfg, feat, gt, patients, pband, bands, ctrl_ids, rows, per_subject, identity):
    sets = {"temporal": A.TEMPORAL, "mesial": A.MESIAL, "all37": [n for _, _, n in A.PAIRS]}
    roi = {s: A.load_roi(feat, s) for b in bands for s in ctrl_ids[b]}
    roi.update({p: A.load_roi(feat, p) for p in patients})
    frames = {}
    for kind in ["ratio", "ratio200", "logratio", "diff", "diff_gm", "L_raw", "R_raw", "L_gm", "R_gm"]:
        frames[kind] = {b: pd.DataFrame([pair_values(roi[c], kind) for c in ctrl_ids[b]], index=ctrl_ids[b]) for b in bands}

    def first_level(kind, sid, b, loo):
        """asymmetry value vector (one per pair) for subject sid; controls LOO where a z is involved."""
        if kind in ("ratio", "ratio200", "logratio", "diff", "diff_gm"):
            return pd.Series(pair_values(roi[sid], kind))
        side = "raw" if kind == "zdiff_raw" else "gm"
        L = pd.Series(pair_values(roi[sid], f"L_{side}")); R = pd.Series(pair_values(roi[sid], f"R_{side}"))
        zL = A.zscore_frame(frames[f"L_{side}"][b], L, loo_index=loo); zR = A.zscore_frame(frames[f"R_{side}"][b], R, loo_index=loo)
        return zL - zR

    for kind in REG_DEFS + ["ratio200"]:
        ctl_first = {b: pd.DataFrame([first_level(kind, c, b, c) for c in ctrl_ids[b]], index=ctrl_ids[b]) for b in bands}
        stat_c, stat_p = {k: [] for k in sets}, {k: [] for k in sets}
        zc_store = {}
        for b in bands:
            for c in ctrl_ids[b]:
                z = A.zscore_frame(ctl_first[b], ctl_first[b].loc[c], loo_index=c); zc_store[c] = z
                for k, s in sets.items():
                    stat_c[k].append((c, float(np.nanmean([z[n] for n in s]))))
        for p in patients:
            b = pband[p]; z = A.zscore_frame(ctl_first[b], first_level(kind, p, b, None)); zc_store[p] = z
            for k, s in sets.items():
                stat_p[k].append((p, float(np.nanmean([z[n] for n in s]))))
        if kind == "ratio200":
            identity["regional_max_abs_dz_ratio_vs_ratio200"] = float(max(
                np.nanmax(np.abs(zc_store[s].values - identity["_reg_ratio_z"][s].values)) for s in zc_store))
            continue
        if kind == "ratio":
            identity["_reg_ratio_z"] = dict(zc_store)
        for k in sets:
            pat = pd.DataFrame(stat_p[k], columns=["id", "value"]); ctl = pd.DataFrame(stat_c[k], columns=["id", "value"])
            rows.append(score(f"{kind}", cfg.name, f"regional_{k}", pat, ctl, gt))
            for _, r in pat.iterrows(): per_subject.append(dict(dataset=cfg.name, id=r.id, role="patient", scale=f"regional_{k}", definition=kind, value=r.value))
            for _, r in ctl.iterrows(): per_subject.append(dict(dataset=cfg.name, id=r.id, role="loo_control", scale=f"regional_{k}", definition=kind, value=r.value))


# ----------------------------------------------------------------------------- voxel-wise (8-mm smoothed grid)
class Acc:
    def __init__(self, shape):
        self.S = np.zeros(shape); self.SS = np.zeros(shape); self.C = np.zeros(shape, np.int32)
    def add(self, x, v): self.S[v] += x[v]; self.SS[v] += x[v] ** 2; self.C[v] += 1
    def z(self, x, v, loo=False, floor=0.0):
        S, SS, C = self.S.copy(), self.SS.copy(), self.C.copy()
        if loo: S[v] -= x[v]; SS[v] -= x[v] ** 2; C[v] -= 1
        ok = C >= 10; m = np.where(ok, S / np.maximum(C, 1), 0); var = np.where(ok, SS / np.maximum(C, 1) - m ** 2, 0)
        var = var * C / np.maximum(C - 1, 1); sd = np.sqrt(np.maximum(var, 0)); sd = np.maximum(sd, np.flip(sd, 0)); sd = np.maximum(sd, floor)
        z = np.zeros_like(x); good = ok & v & (sd > 1e-9); z[good] = (x[good] - m[good]) / sd[good]
        return np.clip(z, -20, 20), good


def voxel(cfg, feat, gt, patients, pband, bands, ctrl_ids, rows, per_subject, identity):
    maps, parcs = {}, {}
    for b in bands:
        for c in ctrl_ids[b]:
            z = np.load(feat / f"{c}.npz"); maps[c] = z["perf"].astype(np.float32); parcs[c] = z["parc"]
    for p in patients:
        z = np.load(feat / f"{p}.npz"); maps[p] = z["perf"].astype(np.float32)
    gm = {s: A.load_roi(feat, s)["gm_mean"] for s in maps}
    shape = next(iter(maps.values())).shape
    nx0 = shape[0]; left = np.zeros(shape, bool); left[nx0 // 2 + 1:] = True; right = np.zeros(shape, bool); right[: nx0 // 2] = True
    scope, tscope = {}, {}
    for b in bands:
        votes = np.zeros(shape, np.int32); votes_t = np.zeros(shape, np.int32)
        for c in ctrl_ids[b]:
            votes += np.isin(parcs[c], A.SCOPE_LABELS); votes_t += np.isin(parcs[c], A.TEMPORAL_CTX_LABELS + [17, 53, 18, 54])
        n = len(ctrl_ids[b]); s = votes > n / 2; s &= np.flip(s, 0); t = votes_t > n / 2; t &= np.flip(t, 0)
        scope[b], tscope[b] = s, t
    valid = {s: (maps[s] > 0) & (np.flip(maps[s], 0) > 0) for s in maps}

    # first-level per-voxel perfusion z (analysis 3) for the zdiff definitions
    perf_acc = {b: Acc(shape) for b in bands}; perfgm_acc = {b: Acc(shape) for b in bands}
    for b in bands:
        for c in ctrl_ids[b]:
            v = maps[c] > 0; perf_acc[b].add(maps[c].astype(np.float64), v); perfgm_acc[b].add(maps[c].astype(np.float64) / gm[c], v)

    def asym(kind, s, b, loo):
        L = maps[s].astype(np.float64); R = np.flip(L, 0); v = valid[s]; out = np.zeros(shape)
        if kind in ("ratio", "ratio200"):
            f = 200.0 if kind == "ratio200" else 1.0; out[v] = f * (L[v] - R[v]) / (L[v] + R[v])
        elif kind == "logratio": out[v] = np.log(L[v] / R[v])
        elif kind == "diff": out[v] = L[v] - R[v]
        elif kind == "diff_gm": out[v] = (L[v] - R[v]) / gm[s]
        elif kind in ("zdiff_raw", "zdiff_gm"):
            acc = perf_acc[b] if kind == "zdiff_raw" else perfgm_acc[b]
            x = L if kind == "zdiff_raw" else L / gm[s]
            zp, g = acc.z(x, L > 0, loo=loo); zp[~g] = 0.0
            d = zp - np.flip(zp, 0); vv = v & g & np.flip(g, 0); out[vv] = d[vv]; return out, vv
        return out, v

    def stat_of(z, good, b):
        row = {}
        for nm, sc in (("temporal", tscope[b]), ("all", scope[b])):
            m = good & sc
            hl = float(np.abs(z[m & left & (z <= -T)]).sum()); hr = float(np.abs(z[m & right & (z <= -T)]).sum())
            row[nm] = (hr - hl) / (hl + hr) if (hl + hr) > 0 else 0.0
        return row

    for kind in VOX_DEFS + ["ratio200"]:
        acc = {b: Acc(shape) for b in bands}
        amaps = {}
        for b in bands:
            for c in ctrl_ids[b]:
                a, v = asym(kind, c, b, loo=True); amaps[c] = (a, v); acc[b].add(a, v)
        stat_c, stat_p = {"temporal": [], "all": []}, {"temporal": [], "all": []}
        zstore = {}
        for b in bands:
            for c in ctrl_ids[b]:
                a, v = amaps[c]; z, g = acc[b].z(a, v, loo=True); zstore[c] = z
                for k, val in stat_of(z, g, b).items(): stat_c[k].append((c, val))
        for p in patients:
            b = pband[p]; a, v = asym(kind, p, b, loo=False); z, g = acc[b].z(a, v); zstore[p] = z
            for k, val in stat_of(z, g, b).items(): stat_p[k].append((p, val))
        if kind == "ratio200":
            identity["voxel_max_abs_dz_ratio_vs_ratio200"] = float(max(np.max(np.abs(zstore[s] - identity["_vox_ratio_z"][s])) for s in zstore))
            continue
        if kind == "ratio":
            identity["_vox_ratio_z"] = zstore
        for k in ("temporal", "all"):
            pat = pd.DataFrame(stat_p[k], columns=["id", "value"]); ctl = pd.DataFrame(stat_c[k], columns=["id", "value"])
            rows.append(score(kind, cfg.name, f"voxel8mm_{k}_t{T:g}", pat, ctl, gt))
            for _, r in pat.iterrows(): per_subject.append(dict(dataset=cfg.name, id=r.id, role="patient", scale=f"voxel8mm_{k}", definition=kind, value=r.value))
            for _, r in ctl.iterrows(): per_subject.append(dict(dataset=cfg.name, id=r.id, role="loo_control", scale=f"voxel8mm_{k}", definition=kind, value=r.value))


# ----------------------------------------------------------------------------- driver
def run(dataset, rows, identity_all):
    cfg = dataset_config.configure(dataset, perf_variant="tc", band_policy=None)
    feat = BASE / "v2_features" / cfg.name
    gt = cfg.ground_truth()
    patients = [p for p in cfg.patient_ids() if (feat / f"{p}.npz").exists()]
    bands = [g for g in cfg.available_groups() if len(cfg.control_ids(g)) >= 10]
    if cfg.is_inhouse:
        bands = ["FM_20_39", "FM_40_59", "FM_60_79"]
    else:
        bands = [g for g in bands if cfg._is_all_ages(g)] or bands[:1]
    ctrl_ids = {b: [c for c in cfg.control_ids(b) if (feat / f"{c}.npz").exists()] for b in bands}

    def band_of(pid):
        if not cfg.is_inhouse:
            return bands[0]
        for f in glob.glob(str(BASE / "results_zscore" / "patients" / pid / f"{pid}_vs_*_summary.json")):
            m = re.search(r"_vs_(FM_\d+_\d+)_summary", f)
            if m and m.group(1) in bands:
                return m.group(1)
        return None
    pband = {p: band_of(p) for p in patients}; patients = [p for p in patients if pband[p]]
    print(f"dataset={cfg.name} bands={ {b: len(v) for b, v in ctrl_ids.items()} } patients={len(patients)} "
          f"unilateral={sum(gt.get(p, {}).get('ez') in ('L', 'R') for p in patients)}", flush=True)
    per_subject = []; identity = {}
    regional(cfg, feat, gt, patients, pband, bands, ctrl_ids, rows, per_subject, identity)
    print("  regional done", flush=True)
    voxel(cfg, feat, gt, patients, pband, bands, ctrl_ids, rows, per_subject, identity)
    print("  voxel done", flush=True)
    pd.DataFrame(per_subject).to_csv(OUT / f"per_subject_{cfg.name}.csv", index=False)
    identity_all[cfg.name] = {k: v for k, v in identity.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="both"); a = ap.parse_args()
    rows, identity = [], {}
    for ds in (["chalfont", "inhouse"] if a.dataset == "both" else [a.dataset]):
        run(ds, rows, identity)
    S = pd.DataFrame(rows); S.to_csv(OUT / "summary.csv", index=False)
    json.dump(identity, open(OUT / "identity_check.json", "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print("\nidentity check (z with x200 vs x1):", identity)
    cols = ["dataset", "scale", "definition", "n", "sign_agree", "balanced", "n_called", "called_correct", "control_fp", "det_auc", "best_min_sens_spec"]
    print(S[cols].to_string(index=False))


if __name__ == "__main__":
    main()
