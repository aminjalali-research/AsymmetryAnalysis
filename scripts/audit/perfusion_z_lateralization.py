#!/usr/bin/env python3
"""Method #3 of the study protocol (direct perfusion z, CBF vs controls) evaluated for
lateralization and lobe-level localization, side by side with method #4 (asymmetry z).

Protocol note (G. Winston): #1 region LI on FreeSurfer parcels; #2 voxel-wise AI in symmetric
space; #3 z(v) = (CBF(v) - mean_ctrl(v)) / sd_ctrl(v); #4 z of the voxel-wise AI. Pipeline B
computes #3 in 01_build_control_normative.py but only used it to classify the hypo/hyper
direction of #4 clusters. Here #3 is scored on its own.

Per subject (patients and leave-one-out controls), within the 37-ROI gray-matter scope:
  raw z and offset-corrected z' = z - median_GM(z)  (removes the whole-brain offset vs the band)
  * hypoperfused voxels (z <= -2) per hemisphere, per lobe; mean z per hemisphere / temporal lobe
  * rules: hypo-volume side (GM, temporal), lower-mean-z side (temporal; calibrated on LOO controls)
  * localization: lobe with most hypoperfused voxels on the predicted side and on the true side
  * same lobe statistic for #4 (|zAI| >= 1.96 voxels per lobe on a side)
Outputs: ez_analysis_mdt/perfz/<dataset>/{subjects.csv, summary.csv, localization.csv}
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from ez_ground_truth import EZ_GROUND_TRUTH, get_unilateral_patients  # noqa: E402

LOBE_CTX = {"temporal": [1, 6, 7, 9, 15, 16, 30, 33, 34], "frontal": [3, 12, 14, 17, 18, 19, 20, 24, 27, 28, 32],
            "parietal": [8, 22, 25, 29, 31], "occipital": [5, 11, 13, 21], "insula_cingulate": [2, 10, 23, 26, 35]}
def lobe_labels(side):
    base = 1000 if side == "L" else 2000
    d = {k: [base + i for i in v] for k, v in LOBE_CTX.items()}
    d["temporal"] += [17, 18] if side == "L" else [53, 54]
    return d
SCOPE = {10, 49, 17, 53, 18, 54} | set(range(1001, 1036)) | set(range(2001, 2036))
ZT = 2.0; AIT = 1.96

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

class Scope:
    def __init__(self, parc):
        self.parc = parc; self.gm = np.isin(parc, list(SCOPE))
        self.side = {s: np.isin(parc, [l for ls in lobe_labels(s).values() for l in ls] + ([10] if s == "L" else [49])) for s in "LR"}
        self.lobes = {s: {k: np.isin(parc, v) for k, v in lobe_labels(s).items()} for s in "LR"}

def features(z, valid, sc: Scope, prefix):
    out = {}
    g = sc.gm & valid
    for tag, zz in (("raw", z), ("oc", z - np.median(z[g]) if g.any() else z)):
        hypo = (zz <= -ZT) & g; hyper = (zz >= ZT) & g
        for s in "LR":
            m = sc.side[s] & g
            out[f"{prefix}{tag}_hypo_{s}"] = int((hypo & sc.side[s]).sum()); out[f"{prefix}{tag}_hyper_{s}"] = int((hyper & sc.side[s]).sum())
            out[f"{prefix}{tag}_meanz_{s}"] = float(zz[m].mean()) if m.any() else np.nan
            t = sc.lobes[s]["temporal"] & g
            out[f"{prefix}{tag}_hypo_temporal_{s}"] = int((hypo & t).sum()); out[f"{prefix}{tag}_meanz_temporal_{s}"] = float(zz[t].mean()) if t.any() else np.nan
            for lobe, lm in sc.lobes[s].items():
                out[f"{prefix}{tag}_hypo_{lobe}_{s}"] = int((hypo & lm & g).sum())
    return out

def ai_features(zai, sc: Scope):
    out = {}; g = sc.gm & (zai != 0); sig = (np.abs(zai) >= AIT) & g
    for s in "LR":
        for lobe, lm in sc.lobes[s].items(): out[f"ai_sig_{lobe}_{s}"] = int((sig & lm).sum())
        out[f"ai_sig_{s}"] = int((sig & sc.side[s]).sum())
    return out

def loo_z(x, S, SS, C):
    n = C - 1; valid = (n >= 2) & (x > 0)
    mean = np.zeros_like(x); m2 = np.zeros_like(x); sd = np.zeros_like(x)
    mean[valid] = (S[valid] - x[valid]) / n[valid]
    m2[valid] = SS[valid] - x[valid] ** 2 - n[valid] * mean[valid] ** 2
    sd[valid] = np.sqrt(np.maximum(m2[valid], 0) / (n[valid] - 1)); sd = np.maximum(sd, 0.01)
    z = np.zeros_like(x); z[valid] = (x[valid] - mean[valid]) / sd[valid]; return np.clip(z, -20, 20), valid

def build_inhouse_accumulators(band, ids, cache):
    if cache.exists():
        d = np.load(cache); return d["S"], d["SS"], d["C"]
    S = SS = C = None
    for i, cid in enumerate(ids):
        x = nib.load(str(ROOT / "symreg/sym_perf" / f"{cid}_perf_sym.nii.gz")).get_fdata(dtype=np.float32).astype(np.float64)
        if S is None: S = np.zeros_like(x); SS = np.zeros_like(x); C = np.zeros(x.shape, np.int32)
        v = x > 0; S[v] += x[v]; SS[v] += x[v] ** 2; C[v] += 1
        print(f"    acc {band} {i+1}/{len(ids)}", flush=True)
    np.savez(cache, S=S, SS=SS, C=C); return S, SS, C

def run(dataset, out):
    rows = []
    if dataset == "inhouse":
        bands = {"FM_20_39": ["P013", "P015", "P016", "P017", "P019", "P020", "P026", "P028", "P029", "P030"],
                 "FM_40_59": ["P018", "P021", "P022", "P023", "P024", "P027"], "FM_60_79": ["P025"]}
        uni = set(get_unilateral_patients())
        for band, pats in bands.items():
            parc = nib.load(str(ROOT / "results_zscore/groups" / band / "consensus_parcellation.nii.gz")).get_fdata().astype(int); sc = Scope(parc)
            for p in pats:
                z = nib.load(str(ROOT / "results_zscore/patients" / p / f"{p}_vs_{band}_zscore.nii.gz")).get_fdata(dtype=np.float32)
                zai = nib.load(str(ROOT / "results_zscore/asymmetry/patients" / p / f"{p}_asymmetry_zscore.nii.gz")).get_fdata(dtype=np.float32)
                gt = EZ_GROUND_TRUTH[p]; side = gt["ez"] if p in uni else None
                rows.append(dict(id=p, kind="patient", band=band, truth=side, lobe=("parietal" if p == "P025" else "temporal") if side else None,
                                 group="inhouse", **features(z, z != 0, sc, "p_"), **ai_features(zai, sc))); print("   ", p, flush=True)
            ids = sorted(d.name for d in (ROOT / "DatasetControls" / band).iterdir() if d.is_dir())
            S, SS, C = build_inhouse_accumulators(band, ids, ROOT / "results_zscore/groups" / band / "perf_loo_accumulators.npz")
            for i, cid in enumerate(ids):
                x = nib.load(str(ROOT / "symreg/sym_perf" / f"{cid}_perf_sym.nii.gz")).get_fdata(dtype=np.float32).astype(np.float64)
                z, valid = loo_z(x, S, SS, C); rows.append(dict(id=cid, kind="control", band=band, truth=None, lobe=None, group="control", **features(z.astype(np.float32), valid, sc, "p_")))
                if i % 10 == 0: print(f"    ctrl {band} {i+1}/{len(ids)}", flush=True)
    else:
        E = ROOT / "external_chalfont"; parc = nib.load(str(E / "atlas/consensus_parcellation_sym.nii.gz")).get_fdata().astype(int); sc = Scope(parc)
        ev = pd.read_csv(E / "results_zscore/lateralization/eval_per_patient.csv"); ev = ev[(ev.tier == "clinical") & (ev.scope == "37roi") & (ev.rule == "direction")].set_index("patient")
        pats = sorted(d.name for d in (E / "results_zscore/patients").iterdir() if d.is_dir())
        for p in pats:
            z = nib.load(str(E / "results_zscore/patients" / p / f"{p}_vs_CH_ALL_zscore.nii.gz")).get_fdata(dtype=np.float32)
            zai = nib.load(str(E / "results_zscore/asymmetry/patients" / p / f"{p}_asymmetry_zscore.nii.gz")).get_fdata(dtype=np.float32)
            truth = ev.truth.get(p); lobe = ev.ez_lobe.get(p); grp = ev.group.get(p, p[0])
            rows.append(dict(id=p, kind="patient", band="CH_ALL", truth=truth if isinstance(truth, str) else None, lobe=lobe if isinstance(lobe, str) else None, group=grp,
                             **features(z, z != 0, sc, "p_"), **ai_features(zai, sc))); print("   ", p, flush=True)
        acc = np.load(E / "results_zscore/asymmetry/groups/CH_ALL/loo_accumulators.npz"); S, SS, C = acc["Sp"], acc["SSp"], acc["Cp"]
        for i, cid in enumerate(acc["ids"]):
            x = nib.load(str(E / "sym_perf" / f"{cid}_perf_sym.nii.gz")).get_fdata(dtype=np.float32).astype(np.float64)
            z, valid = loo_z(x, S, SS, C); rows.append(dict(id=cid, kind="control", band="CH_ALL", truth=None, lobe=None, group="control", **features(z.astype(np.float32), valid, sc, "p_")))
            if i % 10 == 0: print(f"    ctrl {i+1}/{len(acc['ids'])}", flush=True)
    df = pd.DataFrame(rows); out.mkdir(parents=True, exist_ok=True); df.to_csv(out / "subjects.csv", index=False); return df

def evaluate(df, out, dataset):
    pat = df[(df.kind == "patient") & df.truth.isin(["L", "R"])].copy(); ctl = df[df.kind == "control"].copy()
    nL, nR = (pat.truth == "L").sum(), (pat.truth == "R").sum(); maj = max(nL, nR); maj_side = "L" if nL >= nR else "R"
    res = []
    def score(name, pred, stat_pat=None, stat_ctl=None, thr=None, ctl_called=None):
        ok = (pred == pat.truth); called = pred.isin(["L", "R"])
        sL = ok[pat.truth == "L"].mean() if nL else np.nan; sR = ok[pat.truth == "R"].mean() if nR else np.nan
        lo, hi = wilson(int(ok.sum()), len(pat))
        res.append(dict(dataset=dataset, rule=name, n=len(pat), nL=nL, nR=nR, agree=int(ok.sum()), balanced_acc=round(float(np.nanmean([sL, sR])), 3), sens_L=round(float(sL), 2), sens_R=round(float(sR), 2),
                        wilson_lo=round(lo, 3), wilson_hi=round(hi, 3), n_called=int(called.sum()), correct_called=int(ok[called].sum()),
                        majority_baseline=f"{maj}/{len(pat)} ({maj_side})", threshold=None if thr is None else round(float(thr), 3), control_call_rate=None if ctl_called is None else round(float(ctl_called), 3)))
    for tag in ("raw", "oc"):
        for scope_ in ("", "_temporal"):
            hL, hR = pat[f"p_{tag}_hypo{scope_}_L"], pat[f"p_{tag}_hypo{scope_}_R"]
            pred = np.where(hL > hR, "L", np.where(hR > hL, "R", "B")); score(f"#3 {tag}: more hypoperfused voxels (z<=-2){scope_ or ' GM'}", pd.Series(pred, index=pat.index))
            # lower mean z side, calibrated on controls
            d = pat[f"p_{tag}_meanz{scope_}_L"] - pat[f"p_{tag}_meanz{scope_}_R"]; dc = ctl[f"p_{tag}_meanz{scope_}_L"] - ctl[f"p_{tag}_meanz{scope_}_R"]
            score(f"#3 {tag}: lower mean z side (sign){scope_ or ' GM'}", pd.Series(np.where(d < 0, "L", "R"), index=pat.index))
            thr = np.nanpercentile(np.abs(dc), 95); pred = np.where(np.abs(d) < thr, "none", np.where(d < 0, "L", "R"))
            score(f"#3 {tag}: lower mean z side, calibrated 5% FP{scope_ or ' GM'}", pd.Series(pred, index=pat.index), thr=thr, ctl_called=(np.abs(dc) >= thr).mean())
        # hypo-volume LI calibrated
        hL, hR = pat[f"p_{tag}_hypo_L"], pat[f"p_{tag}_hypo_R"]; li = (hL - hR) / (hL + hR).replace(0, np.nan)
        cL, cR = ctl[f"p_{tag}_hypo_L"], ctl[f"p_{tag}_hypo_R"]; cli = ((cL - cR) / (cL + cR).replace(0, np.nan)).abs(); thr = np.nanpercentile(cli, 95)
        pred = np.where(li.abs() < thr, "none", np.where(li > 0, "L", "R")); score(f"#3 {tag}: hypo-volume LI calibrated 5% FP GM", pd.Series(pred, index=pat.index), thr=thr, ctl_called=(cli >= thr).mean())
    # #4 reference: more |zAI|>=1.96 voxels on a side is NOT meaningful (antisymmetric) -> report from existing evaluations instead
    summ = pd.DataFrame(res); summ.to_csv(out / "summary.csv", index=False)
    # localization
    loc = []
    lob = list(LOBE_CTX)
    for _, r in pat.iterrows():
        if r.lobe not in lob: continue
        for tag in ("raw", "oc"):
            for which, side in (("true side", r.truth), ("pred side", "L" if r[f"p_{tag}_hypo_L"] > r[f"p_{tag}_hypo_R"] else "R")):
                counts = {l: r[f"p_{tag}_hypo_{l}_{side}"] for l in lob}; best = max(counts, key=counts.get) if sum(counts.values()) > 0 else "none"
                loc.append(dict(id=r.id, method=f"#3 {tag}", basis=which, ez_lobe=r.lobe, pred_lobe=best, hit=int(best == r.lobe)))
        for which, side in (("true side", r.truth),):
            counts = {l: r[f"ai_sig_{l}_{side}"] for l in lob}; best = max(counts, key=counts.get) if sum(counts.values()) > 0 else "none"
            loc.append(dict(id=r.id, method="#4 |zAI|>=1.96", basis=which, ez_lobe=r.lobe, pred_lobe=best, hit=int(best == r.lobe)))
    loc = pd.DataFrame(loc); loc.to_csv(out / "localization_per_patient.csv", index=False)
    if len(loc):
        agg = loc.groupby(["method", "basis"]).agg(n=("hit", "size"), hits=("hit", "sum")).reset_index(); agg["rate"] = (agg.hits / agg.n).round(3)
        base = loc[loc.method == "#4 |zAI|>=1.96"].ez_lobe.value_counts(); agg["majority_lobe"] = f"{base.index[0]} {base.iloc[0]}/{base.sum()} ({base.iloc[0]/base.sum():.2f})"
        agg["pred_lobe_mix"] = agg.apply(lambda a: loc[(loc.method == a.method) & (loc.basis == a.basis)].pred_lobe.value_counts().to_dict(), axis=1)
        agg.to_csv(out / "localization.csv", index=False); print(agg.to_string())
    print(summ.to_string())
    # control FP of the hypo-volume rule at the asymmetric-count level: fraction of controls with |LI_hypo| >= patients' median
    ctl_desc = {f"{tag}_hypo_total_median": float((ctl[f"p_{tag}_hypo_L"] + ctl[f"p_{tag}_hypo_R"]).median()) for tag in ("raw", "oc")}
    ctl_desc |= {f"{tag}_hypo_total_median_patients": float((pat[f"p_{tag}_hypo_L"] + pat[f"p_{tag}_hypo_R"]).median()) for tag in ("raw", "oc")}
    json.dump(ctl_desc, open(out / "hypo_volume_descriptives.json", "w"), indent=1); print(ctl_desc)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["inhouse", "chalfont"], required=True); ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args(); out = ROOT / "ez_analysis_mdt/perfz" / a.dataset
    df = pd.read_csv(out / "subjects.csv") if a.reuse and (out / "subjects.csv").exists() else run(a.dataset, out)
    evaluate(df, out, a.dataset)
