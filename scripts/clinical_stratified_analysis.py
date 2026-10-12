#!/usr/bin/env python3
"""Clinically oriented secondary analyses of the Pipeline-B (voxel zAI) output.

Answers the questions a presurgical MDT would ask of the cluster reports in
``results_zscore/asymmetry/patients/<PID>/<PID>_asymmetry_cluster_report.csv``:

1. Decision rule: how does the direction-aware rule compare with (a) the
   classical hypoperfusion-only reading and (b) the legacy polarity-flip rule?
2. Spatial scope: does restricting the lateralization decision to the temporal
   lobe (temporal cortex + hippocampus + amygdala), i.e. a syndrome-informed
   prior, change concordance with the MDT side?
3. Threshold tier: research (|z|>=3, >=50 vox), clinical (|z|>=4, >=100 vox),
   surgical (|z|>=5, >=200 vox).
4. Stratification of the default rule by MRI status (lesional vs MRI-negative),
   MDT confidence, reference-standard tier, and with the post-resection patient
   (P024) removed.
5. Agreement between Pipeline A (temporal signed LI) and Pipeline B, and the
   accuracy of the side prediction when the two pipelines agree.
6. Per-patient direction-class profile (hypo / hyper / subtle / mixed).

Writes ``ez_analysis_mdt/clinical_stratified_summary.csv`` and
``ez_analysis_mdt/clinical_per_patient_predictions.csv`` and prints a report.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from ez_ground_truth import EZ_GROUND_TRUTH  # noqa: E402

_spec = importlib.util.spec_from_file_location("ci", BASE / "06_clinical_interpretation.py")
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)

CLUSTER_DIR = BASE / "results_zscore/asymmetry/patients"
OUT_DIR = BASE / "ez_analysis_mdt"

# ---------------------------------------------------------------------------
# Patient-level clinical metadata (Rai 2026 thesis Tables 11 and 14).
# ---------------------------------------------------------------------------
# MRI status for the epileptogenic zone: 'lesional' when the clinical or
# research MRI showed an abnormality concordant with the MDT side; otherwise
# 'MRI-negative' (including non-specific findings judged unrelated).
MRI_STATUS = {
    "P013": "lesional", "P015": "lesional", "P016": "MRI-negative",
    "P017": "lesional", "P018": "lesional", "P019": "lesional",
    "P020": "lesional", "P021": "MRI-negative", "P022": "MRI-negative",
    "P023": "lesional", "P024": "lesional", "P025": "lesional",
    "P026": "lesional", "P027": "MRI-negative", "P028": "MRI-negative",
    "P029": "lesional", "P030": "MRI-negative",
}
# Reference-standard tier (Scarabello et al. 2026 convention):
#   'defined'  = side supported by resective surgery with histopathology or
#                by stereo-EEG recording of seizures,
#   'probable' = MDT consensus without invasive confirmation,
#   'uncertain'= MDT declined to lateralize (B/U).
REFERENCE_TIER = {
    "P016": "defined",   # bilateral sEEG: right mesial temporal onset
    "P017": "defined",   # left temporal lobectomy 12/2025, histopathology
    "P023": "defined",   # right anterior temporal resection 08/2025
    "P025": "defined",   # lesion resection 03/2026, histopathology
    "P026": "defined",   # left temporal biopsy 11/2025 (MVNT), lesional
    "P018": "uncertain", "P019": "uncertain", "P020": "uncertain", "P021": "uncertain",
}
POST_SURGICAL = {"P024"}  # imaged after a prior right temporal resection

# Pipeline A (temporal signed LI, Youden threshold 0.040, empirically oriented)
# per-patient predictions on the unilateral cohort (ez_analysis_mdt output).
PIPELINE_A_PRED = {
    "P013": "L", "P015": "R", "P016": "R", "P017": "R", "P022": "L",
    "P023": "R", "P024": "R", "P025": "L", "P026": "L", "P027": "L",
    "P028": "L", "P029": "R", "P030": "R",
}

TIERS = {"research": (3.0, 50), "clinical": (4.0, 100), "surgical": (5.0, 200),
         # >= 20 native 2.5-mm ASL voxels (20 x 15.6 mm^3 = 312 mm^3 = 610 voxels at 0.8 mm)
         "native20": (4.0, 610)}
TEMPORAL_LOBES = {"temporal", "hippocampus", "amygdala"}


def load_clusters(pid: str) -> pd.DataFrame:
    df = pd.read_csv(CLUSTER_DIR / pid / f"{pid}_asymmetry_cluster_report.csv")
    normed = df["primary_region"].apply(ci._normalize_region)
    df["region_side"] = normed.apply(lambda t: t[0])
    df["region_canon"] = normed.apply(lambda t: t[1])
    df["region_lobe"] = df["region_canon"].apply(lambda n: ci._lobe_of(n) if n else "other")
    df["in_scope"] = df["region_canon"].apply(lambda n: bool(n) and n in ci.CLINICAL_NAMES)
    df["peak_abs"] = df["peak_z"].abs()
    df["flip_side"] = df["direction"].map({"left-dominant": "R", "right-dominant": "L"})
    df["dir_side"] = df["ez_side_pred"].astype(str).str.upper().str[:1]
    df.loc[~df["dir_side"].isin(["L", "R"]), "dir_side"] = df["flip_side"]
    return df


def predict(df: pd.DataFrame, peak: float, size: int, scope: str, rule: str) -> tuple[str, int, int]:
    """Return (predicted side, L voxels, R voxels)."""
    d = df[(df["peak_abs"] >= peak) & (df["size_voxels"] >= size) & df["in_scope"]].copy()
    if scope == "temporal":
        d = d[d["region_lobe"].isin(TEMPORAL_LOBES)]
    if rule == "direction":
        d["side"] = d["dir_side"]
    elif rule == "flip":
        # one member per mirrored pair: keep the left-hemisphere member; a
        # left-dominant asymmetry there implies right hypoperfusion (EZ right).
        d = d[d["cluster_hemi"] == "L"]
        d["side"] = d["flip_side"]
    elif rule == "hypo_only":
        # one member per mirrored pair; side = hemisphere of the hypoperfused member
        d = d[(d["direction_class"] == "hypo") & (d["cluster_hemi"] == d["ez_side_pred"])]
        d["side"] = d["cluster_hemi"]
    elif rule == "hyper_only":
        d = d[(d["direction_class"] == "hyper") & (d["cluster_hemi"] == d["ez_side_pred"])]
        d["side"] = d["cluster_hemi"]
    elif rule == "direction_no_subtle":
        d = d[d["direction_class"].isin(["hypo", "hyper"])]
        d["side"] = d["dir_side"]
    else:
        raise ValueError(rule)
    lv = int(d.loc[d["side"] == "L", "size_voxels"].sum())
    rv = int(d.loc[d["side"] == "R", "size_voxels"].sum())
    if lv == 0 and rv == 0:
        return "Unclear", lv, rv
    if lv >= 1.05 * rv:
        return "L", lv, rv
    if rv >= 1.05 * lv:
        return "R", lv, rv
    return "B", lv, rv


def predict_production(df: pd.DataFrame, peak: float, size: int) -> tuple[str, str, int, int]:
    """Reproduce the pre-specified surgeon-report rule of 06_clinical_interpretation.py
    (mesial-temporal priority, then whole-scope fallback). Returns
    (prediction, confidence, L voxels, R voxels)."""
    d = df[(df["peak_abs"] >= peak) & (df["size_voxels"] >= size) & df["in_scope"]].copy()
    d["ez_side"] = d["dir_side"]
    per = {"L": {}, "R": {}}
    for _, r in d.iterrows():
        e = per[r["ez_side"]].setdefault(r["region_lobe"], {"clusters": 0, "voxels": 0, "peak": 0.0})
        e["clusters"] += 1
        e["voxels"] += int(r["size_voxels"])
        e["peak"] = max(e["peak"], float(r["peak_abs"]))
    conf, pred, _ = ci._compute_confidence(per)
    lv = int(d.loc[d.ez_side == "L", "size_voxels"].sum())
    rv = int(d.loc[d.ez_side == "R", "size_voxels"].sum())
    return pred, conf, lv, rv


def auc_with_ci(scores: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 20260714):
    """AUC for 'higher score = Left' with a stratified bootstrap 95% CI."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    auc = roc_auc_score(y, scores)
    li, ri = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(li, len(li)), rng.choice(ri, len(ri))])
        boots.append(roc_auc_score(y[idx], scores[idx]))
    return auc, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    pids = sorted(p.name for p in CLUSTER_DIR.iterdir() if (p / f"{p.name}_asymmetry_cluster_report.csv").exists())
    pids = [p for p in pids if p in EZ_GROUND_TRUTH]
    data = {p: load_clusters(p) for p in pids}
    unilateral = [p for p in pids if EZ_GROUND_TRUTH[p]["ez"] in ("L", "R")]

    # ---- 1-3. decision rule x scope x tier ---------------------------------
    rows = []
    per_patient = []
    for tier, (peak, size) in TIERS.items():
        for scope in ("37roi", "temporal"):
            for rule in ("direction", "direction_no_subtle", "hypo_only", "hyper_only", "flip"):
                agree = agree_ex = n_ex = unclear = 0
                for p in unilateral:
                    pred, lv, rv = predict(data[p], peak, size, scope, rule)
                    truth = EZ_GROUND_TRUTH[p]["ez"]
                    ok = int(pred == truth)
                    agree += ok
                    unclear += int(pred == "Unclear")
                    if p not in POST_SURGICAL:
                        agree_ex += ok
                        n_ex += 1
                    per_patient.append(dict(tier=tier, scope=scope, rule=rule, patient=p,
                                            mdt=truth, pred=pred, L_vox=lv, R_vox=rv, correct=ok))
                sub = [r for r in per_patient if r["tier"] == tier and r["scope"] == scope and r["rule"] == rule]
                sens_l = np.mean([r["pred"] == "L" for r in sub if r["mdt"] == "L"])
                sens_r = np.mean([r["pred"] == "R" for r in sub if r["mdt"] == "R"])
                rows.append(dict(tier=tier, peak=peak, size=size, scope=scope, rule=rule,
                                 n=len(unilateral), agree=agree, unclear=unclear,
                                 left_correct=int(round(sens_l * 8)), right_correct=int(round(sens_r * 5)),
                                 balanced_acc=round((sens_l + sens_r) / 2, 3),
                                 agree_excl_P024=agree_ex, n_excl_P024=n_ex))
    summary = pd.DataFrame(rows)
    pp = pd.DataFrame(per_patient)
    summary.to_csv(OUT_DIR / "clinical_stratified_summary.csv", index=False)
    pp.to_csv(OUT_DIR / "clinical_per_patient_predictions.csv", index=False)

    print("\n=== Rule x scope x tier: unilateral concordance (n=13; n=12 without P024) ===")
    print(summary.to_string(index=False))

    # ---- production rule (pre-specified surgeon report) + continuous index --
    print("\n=== Pre-specified production rule (mesial-temporal priority) and volume-based lateralization index ===")
    prod_rows = []
    for tier, (peak, size) in TIERS.items():
        y, s = [], []
        n_ok = 0
        for p in unilateral:
            pred, conf, lv, rv = predict_production(data[p], peak, size)
            truth = EZ_GROUND_TRUTH[p]["ez"]
            li = (lv - rv) / (lv + rv) if (lv + rv) else 0.0
            prod_rows.append(dict(tier=tier, patient=p, mdt=truth, pred=pred, confidence=conf,
                                  L_vox=lv, R_vox=rv, vol_LI=round(li, 3), correct=int(pred == truth)))
            n_ok += int(pred == truth)
            y.append(1 if truth == "L" else 0)
            s.append(li)
        y, s = np.array(y), np.array(s)
        auc, lo, hi = auc_with_ci(s, y)
        keep = np.array([p not in POST_SURGICAL for p in unilateral])
        auc2, lo2, hi2 = auc_with_ci(s[keep], y[keep])
        print(f"  {tier:<9} production rule {n_ok}/13 concordant; volume LI AUC {auc:.3f} "
              f"[{lo:.2f}, {hi:.2f}]  (without P024: {auc2:.3f} [{lo2:.2f}, {hi2:.2f}])")
    prod = pd.DataFrame(prod_rows)
    prod.to_csv(OUT_DIR / "clinical_production_rule_by_tier.csv", index=False)
    print(prod[prod.tier == "clinical"].to_string(index=False))

    # ---- 4. stratification of the pre-specified rule (clinical tier) --------
    default = prod[prod.tier == "clinical"].set_index("patient")
    print("\n=== Pre-specified rule (clinical tier, 37-ROI scope, direction-aware): strata ===")

    def strat(name, members):
        m = [p for p in members if p in default.index]
        k = int(default.loc[m, "correct"].sum())
        print(f"  {name:<40} {k}/{len(m)}")
        return name, k, len(m)

    strata = []
    strata.append(strat("MRI lesional", [p for p in unilateral if MRI_STATUS[p] == "lesional"]))
    strata.append(strat("MRI-negative", [p for p in unilateral if MRI_STATUS[p] == "MRI-negative"]))
    strata.append(strat("MDT High confidence", [p for p in unilateral if EZ_GROUND_TRUTH[p]["confidence"] == "High"]))
    strata.append(strat("MDT Moderate confidence", [p for p in unilateral if EZ_GROUND_TRUTH[p]["confidence"] == "Moderate"]))
    strata.append(strat("Reference: defined (surgery/sEEG)", [p for p in unilateral if REFERENCE_TIER.get(p) == "defined"]))
    strata.append(strat("Reference: probable (MDT only)", [p for p in unilateral if REFERENCE_TIER.get(p) is None]))
    strata.append(strat("Temporal lobe epilepsy", [p for p in unilateral if p != "P025"]))
    strata.append(strat("Extratemporal (parietal)", ["P025"]))
    strata.append(strat("Left EZ", [p for p in unilateral if EZ_GROUND_TRUTH[p]["ez"] == "L"]))
    strata.append(strat("Right EZ", [p for p in unilateral if EZ_GROUND_TRUTH[p]["ez"] == "R"]))
    strata.append(strat("Excluding post-resection P024", [p for p in unilateral if p not in POST_SURGICAL]))
    pd.DataFrame(strata, columns=["stratum", "concordant", "n"]).to_csv(OUT_DIR / "clinical_strata_default_rule.csv", index=False)

    # ---- 5. Pipeline A vs B agreement ---------------------------------------
    print("\n=== Pipeline A (temporal signed LI) vs Pipeline B (default rule) ===")
    both_agree = both_correct = a_only = b_only = 0
    for p in unilateral:
        a = PIPELINE_A_PRED[p]
        b = default.loc[p, "pred"]
        t = EZ_GROUND_TRUTH[p]["ez"]
        if a == b:
            both_agree += 1
            both_correct += int(a == t)
        print(f"  {p}: MDT={t}  A={a}  B={b}  {'agree' if a == b else 'DISAGREE'}")
    print(f"  pipelines agree in {both_agree}/{len(unilateral)}; correct when they agree: {both_correct}/{both_agree}")

    # ---- 6. direction-class profile (all clusters in the research-grade CSV) -
    print("\n=== Direction-class profile per patient (all clusters, |z|>=1.96, >=50 vox) ===")
    # The asymmetry map is antisymmetric (AI(v) = -AI(mirror v)), so every
    # abnormality appears as a mirrored pair of clusters with the same
    # direction class and the same predicted side. Count one member per pair:
    # the cluster lying in the hemisphere the inference assigns the finding to.
    prof = []
    for p in pids:
        d = data[p]
        d = d[d["cluster_hemi"].astype(str) == d["ez_side_pred"].astype(str)]
        vc = d["direction_class"].value_counts()
        hypo, hyper = int(vc.get("hypo", 0)), int(vc.get("hyper", 0))
        hv = int(d.loc[d.direction_class == "hypo", "size_voxels"].sum())
        hyv = int(d.loc[d.direction_class == "hyper", "size_voxels"].sum())
        phen = "hypo" if hv > 2 * hyv else ("hyper" if hyv > 2 * hv else "mixed")
        prof.append(dict(patient=p, mdt=EZ_GROUND_TRUTH[p]["ez"], n_clusters=len(d), hypo=hypo, hyper=hyper,
                         subtle=int(vc.get("subtle", 0)), mixed=int(vc.get("mixed", 0)),
                         hypo_vox=hv, hyper_vox=hyv, phenotype=phen))
    prof = pd.DataFrame(prof)
    print(prof.to_string(index=False))
    prof.to_csv(OUT_DIR / "clinical_direction_profile.csv", index=False)

    # Lobe of the largest in-scope cluster at the clinical tier
    print("\n=== Largest in-scope cluster at the clinical tier (|z|>=4, >=100 vox) ===")
    for p in pids:
        d = data[p]
        d = d[(d.peak_abs >= 4) & (d.size_voxels >= 100) & d.in_scope]
        if d.empty:
            print(f"  {p}: none"); continue
        top = d.sort_values("size_voxels", ascending=False).iloc[0]
        print(f"  {p}: MDT={EZ_GROUND_TRUTH[p]['ez']}  {top.primary_region:<22} {int(top.size_voxels):>6} vox  "
              f"class={top.direction_class:<6} EZ-side={top.dir_side}")


if __name__ == "__main__":
    main()
