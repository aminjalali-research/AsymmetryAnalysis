#!/usr/bin/env python3
"""External-validation evaluation of the zAI lateralization rules (Chalfont).

Reads the per-patient cluster reports written by ``02_compute_zai.py`` and
applies exactly the in-house rules (pair-aware: one member per mirrored
cluster pair, ``cluster_hemi == ez_side_pred``) from
``scripts/clinical_stratified_analysis.py`` / ``06_clinical_interpretation.py``:

1. Unilateral patients (ez_side L or R), per rule x tier (x scope): agreement
   count, accuracy with Wilson 95% CI, balanced accuracy, majority-class
   baseline, and the same stratified by group (discrete vs negative) and by
   ez_lobe (temporal vs extratemporal).
2. Pre-specified production rule (06, mesial-temporal priority): call-strength
   strata (Strong / Moderate / Weak / Bilateral / Unclear) with agreement.
3. Direction profile: hypo- vs hyper-perfusion-dominant patients (pair-aware).
4. Control false positives: the same rules applied to every control scored as
   a pseudo-patient (``02_compute_zai.py --pseudo-controls``, leave-one-out by
   default) -> how often a healthy control receives a Strong/Moderate (or any)
   lateralizing call. Reported with Wilson CIs; the optimism of a full-stats
   (non-LOO) pass is flagged from the pseudo-control summary.
5. QC (``metadata/qc_exclusions.json``): technically failed subjects are out of
   every analysis (dataset_config); flagged-but-kept subjects stay in the
   primary cohort and the rule x tier table is repeated without them
   (strata ``cohort=without_*``); their calls are listed in ``eval_qc_flagged.csv``.

Outputs (``<results>/lateralization/``): ``eval_rule_tier.csv``,
``eval_rule_tier_strata.csv``, ``eval_call_strength.csv``,
``eval_direction_profile.csv``, ``eval_control_false_positives.csv``,
``eval_per_patient.csv``, ``eval_summary.txt``.

    python scripts/chalfont/evaluate_lateralization.py --dataset chalfont [--perf-variant flow]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import dataset_config  # noqa: E402

_spec = importlib.util.spec_from_file_location("csa", ROOT / "scripts" / "clinical_stratified_analysis.py")
csa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(csa)

RULES = ("direction", "direction_no_subtle", "hypo_only", "hyper_only", "flip")
SCOPES = ("37roi", "temporal")
LATERALIZING_CONF = ("Strong", "Moderate", "Weak")

# ---------------------------------------------------------------------------
# PRE-SPECIFIED ENDPOINTS. Fixed 2026-10-20 before any Chalfont lateralization
# output was read (audit 2026-10-19, finding 2; scripts/chalfont/CHALFONT_PLAN.md).
# The rule x tier x scope x variant x band-policy grid below is SENSITIVITY
# analysis; the single external-validation result is PRIMARY_ENDPOINT.
# ---------------------------------------------------------------------------
PRIMARY_ENDPOINT = {
    "rule": "production",      # 06 rule: mesial-temporal priority with call strength
    "tier": "clinical",        # |zAI| >= 4, >= 100 voxels (PASCOM Z4C analogue)
    "scope": "37roi",          # 3 subcortical + 34 Desikan-Killiany pairs
    "perf_variant": "tc",      # transit-time-corrected CBF
    "band_policy": "all",      # pooled CH_ALL controls
    "cohort": "primary",       # QC technical failures out; flagged subjects kept
    "patients": "unilateral",  # clinical lateralization L or R
    "metric": ("balanced accuracy (mean of L and R sensitivities) and raw agreement "
               "with Wilson 95% CI, against the majority-side baseline; B/Unclear count as wrong"),
}
SECONDARY_ENDPOINTS = (
    {"name": "direction_rule", "rule": "direction", "tier": "clinical", "scope": "37roi",
     "note": "in-house-selected direction-aware volume majority (confirmatory)"},
    {"name": "control_false_positives", "rule": "production", "tier": "clinical",
     "note": "Strong/Moderate lateralizing call in a healthy control, leave-one-out, Wilson 95% CI"},
)


def primary_endpoint_report(cfg, cs: pd.DataFrame, rule_df: pd.DataFrame, fp: pd.DataFrame,
                            truth: dict, unilateral: list) -> list[str]:
    """Lines for the pre-specified primary and secondary endpoints (printed first)."""
    pe = PRIMARY_ENDPOINT
    is_primary = (cfg.perf_variant == pe["perf_variant"] and cfg.band_policy == pe["band_policy"])
    status = ("PRIMARY ANALYSIS" if is_primary else
              f"SENSITIVITY ANALYSIS (perf_variant={cfg.perf_variant}, band_policy={cfg.band_policy}); "
              f"the primary endpoint is perf_variant={pe['perf_variant']}, band_policy={pe['band_policy']}")
    n_l = sum(truth[p] == "L" for p in unilateral)
    n_r = sum(truth[p] == "R" for p in unilateral)
    out = ["=== PRE-SPECIFIED PRIMARY ENDPOINT (fixed 2026-10-20 before results were read) ===",
           f"status: {status}",
           f"endpoint: {pe['rule']} rule, {pe['tier']} tier, {pe['scope']} scope, {pe['patients']} patients, "
           f"{pe['cohort']} cohort",
           f"metric: {pe['metric']}",
           f"n={len(unilateral)} (L={n_l}, R={n_r})"]
    row = cs[(cs.tier == pe["tier"]) & (cs.strength == "ALL")]
    if len(row):
        r = row.iloc[0]
        maj_acc = r.majority_agree / r.n if r.n else float("nan")
        out.append(f"PRIMARY: agree {int(r.agree)}/{int(r.n)} = {r.accuracy:.3f} "
                   f"[{r.ci_low:.3f}, {r.ci_high:.3f}]  balanced accuracy {r.balanced_acc:.3f}  "
                   f"majority baseline ({r.majority_side}) {int(r.majority_agree)}/{int(r.n)} = {maj_acc:.3f}")
        parts = []
        for st in ("Strong", "Moderate", "Weak", "Bilateral", "Unclear"):
            q = cs[(cs.tier == pe["tier"]) & (cs.strength == st)]
            if len(q):
                parts.append(f"{st} {int(q.iloc[0].agree)}/{int(q.iloc[0].n)}")
        out.append("  call strength (agree/n): " + ", ".join(parts))
    else:
        out.append("PRIMARY: not computable (no production-rule rows at the clinical tier)")
    sec = SECONDARY_ENDPOINTS[0]
    q = rule_df[(rule_df.rule == sec["rule"]) & (rule_df.tier == sec["tier"]) & (rule_df.scope == sec["scope"])]
    if len(q):
        r = q.iloc[0]
        out.append(f"SECONDARY 1 ({sec['note']}): agree {int(r.agree)}/{int(r.n)} = {r.accuracy:.3f} "
                   f"[{r.acc_ci_low:.3f}, {r.acc_ci_high:.3f}]  balanced accuracy {r.balanced_acc:.3f}  "
                   f"majority {int(r.majority_agree)}/{int(r.n)}")
    sec = SECONDARY_ENDPOINTS[1]
    q = fp[fp.tier == sec["tier"]] if len(fp) and "tier" in fp.columns else pd.DataFrame()
    if len(q):
        r = q.iloc[0]
        out.append(f"SECONDARY 2 ({sec['note']}): {int(r.strong_or_moderate)}/{int(r.n_controls)} = "
                   f"{r.fp_rate_strong_moderate:.3f} [{r.fp_ci_low:.3f}, {r.fp_ci_high:.3f}]  "
                   f"any lateralizing call {r.fp_rate_any:.3f}  normative statistics: {r.normative_stats}")
    else:
        out.append("SECONDARY 2: no pseudo-control results")
    out.append("everything below this block is sensitivity analysis")
    out.append("")
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Proportion with Wilson score 95% CI."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def balanced_accuracy(preds: dict, truth: dict) -> float:
    sens = []
    for side in ("L", "R"):
        ids = [p for p in preds if truth[p] == side]
        if ids:
            sens.append(np.mean([preds[p] == side for p in ids]))
    return float(np.mean(sens)) if sens else float("nan")


def majority_baseline(truth: dict, ids) -> tuple[str, int]:
    n_l = sum(truth[p] == "L" for p in ids)
    n_r = sum(truth[p] == "R" for p in ids)
    return ("L" if n_l >= n_r else "R"), max(n_l, n_r)


def lobe_class(ez_lobe: str) -> str:
    if not ez_lobe:
        return "unknown"
    return "temporal" if "temporal" in ez_lobe.lower() else "extratemporal"


def group_class(group: str) -> str:
    g = (group or "").lower()
    if g.startswith("discrete"):
        return "discrete"
    if g.startswith("negative"):
        return "negative"
    return g or "unknown"


def evaluate_rules(data: dict, truth: dict, meta: dict, strata_defs: dict):
    """Rule x tier x scope table with strata (strata_defs: name -> list of ids)."""
    rows, strata_rows, per_patient = [], [], []
    ids = list(data)
    for tier, (peak, size) in csa.TIERS.items():
        for scope in SCOPES:
            for rule in RULES:
                preds = {}
                for p in ids:
                    pred, lv, rv = csa.predict(data[p], peak, size, scope, rule)
                    preds[p] = pred
                    per_patient.append(dict(tier=tier, scope=scope, rule=rule, patient=p,
                                            truth=truth[p], pred=pred, L_vox=lv, R_vox=rv,
                                            correct=int(pred == truth[p]),
                                            group=meta[p]["group"], ez_lobe=meta[p]["ez_lobe"]))

                def summarise(name, members):
                    m = [p for p in members if p in preds]
                    n = len(m)
                    k = sum(preds[p] == truth[p] for p in m)
                    acc, lo, hi = wilson(k, n)
                    maj_side, maj_k = majority_baseline(truth, m)
                    maj_acc, mlo, mhi = wilson(maj_k, n)
                    return dict(tier=tier, peak=peak, size=size, scope=scope, rule=rule,
                                stratum=name, n=n, n_left=sum(truth[p] == "L" for p in m),
                                n_right=sum(truth[p] == "R" for p in m), agree=k,
                                unclear=sum(preds[p] == "Unclear" for p in m),
                                bilateral_call=sum(preds[p] == "B" for p in m),
                                accuracy=round(acc, 3), acc_ci_low=round(lo, 3), acc_ci_high=round(hi, 3),
                                balanced_acc=round(balanced_accuracy({p: preds[p] for p in m}, truth), 3),
                                majority_side=maj_side, majority_agree=maj_k,
                                majority_acc=round(maj_acc, 3),
                                majority_ci_low=round(mlo, 3), majority_ci_high=round(mhi, 3))

                rows.append(summarise("all", ids))
                for name, members in strata_defs.items():
                    strata_rows.append(summarise(name, members))
    return pd.DataFrame(rows), pd.DataFrame(strata_rows), pd.DataFrame(per_patient)


def production_calls(data: dict) -> pd.DataFrame:
    rows = []
    for tier, (peak, size) in csa.TIERS.items():
        for p, df in data.items():
            pred, conf, lv, rv = csa.predict_production(df, peak, size)
            rows.append(dict(tier=tier, id=p, pred=pred, confidence=conf,
                             strength=conf.split("-")[0], L_vox=lv, R_vox=rv))
    return pd.DataFrame(rows)


def direction_profile(data: dict, truth: dict, meta: dict) -> pd.DataFrame:
    rows = []
    for p, d in data.items():
        d = d[d["cluster_hemi"].astype(str) == d["ez_side_pred"].astype(str)]
        vc = d["direction_class"].value_counts()
        hv = int(d.loc[d.direction_class == "hypo", "size_voxels"].sum())
        hyv = int(d.loc[d.direction_class == "hyper", "size_voxels"].sum())
        phen = "hypo" if hv > 2 * hyv else ("hyper" if hyv > 2 * hv else "mixed")
        rows.append(dict(id=p, truth=truth.get(p), group=meta[p]["group"], ez_lobe=meta[p]["ez_lobe"],
                         n_pairs=len(d), hypo=int(vc.get("hypo", 0)), hyper=int(vc.get("hyper", 0)),
                         subtle=int(vc.get("subtle", 0)), mixed=int(vc.get("mixed", 0)),
                         hypo_vox=hv, hyper_vox=hyv, phenotype=phen))
    return pd.DataFrame(rows)


def control_false_positives(cfg, log, exclude=frozenset()) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    pdir = cfg.pseudo_control_dir
    if not pdir.exists():
        return pd.DataFrame(), pd.DataFrame(), "no pseudo-control cluster reports (run 02 --pseudo-controls)"
    cids = sorted(d.name for d in pdir.iterdir()
                  if d.is_dir() and (d / f"{d.name}_asymmetry_cluster_report.csv").exists()
                  and d.name not in exclude)
    loo_note = "unknown"
    summaries = [s for s in pdir.glob("*_pseudo_control_summary.csv") if s.stat().st_size > 0]
    n_listed = 0
    if summaries:
        sm = pd.concat([pd.read_csv(s) for s in summaries], ignore_index=True)
        n_listed = len(sm)
        if "leave_one_out" in sm.columns:
            loo_note = ("leave-one-out" if sm["leave_one_out"].all()
                        else "FULL statistics (optimistic: control included in its own reference)")
    # controls with no surviving cluster at all have no CSV: count them as Unclear
    no_csv = sorted(d.name for d in pdir.iterdir()
                    if d.is_dir() and d.name not in cids and d.name not in exclude)
    data = {c: csa.load_clusters(c, pdir) for c in cids}
    rows = []
    for tier, (peak, size) in csa.TIERS.items():
        strengths = {}
        for c in cids:
            _, conf, _, _ = csa.predict_production(data[c], peak, size)
            strengths[c] = conf.split("-")[0]
        for c in no_csv:
            strengths[c] = "Unclear"
        n = len(strengths)
        counts = {s: sum(v == s for v in strengths.values())
                  for s in ("Strong", "Moderate", "Weak", "Bilateral", "Unclear")}
        k_sm = counts["Strong"] + counts["Moderate"]
        k_any = k_sm + counts["Weak"]
        p1, l1, h1 = wilson(k_sm, n)
        p2, l2, h2 = wilson(k_any, n)
        # direction-rule volume majority (37-ROI): any side call
        k_dir = sum(csa.predict(data[c], peak, size, "37roi", "direction")[0] in ("L", "R") for c in cids)
        p3, l3, h3 = wilson(k_dir, n)
        rows.append(dict(tier=tier, peak=peak, size=size, n_controls=n, **counts,
                         strong_or_moderate=k_sm, fp_rate_strong_moderate=round(p1, 3),
                         fp_ci_low=round(l1, 3), fp_ci_high=round(h1, 3),
                         any_lateralizing=k_any, fp_rate_any=round(p2, 3),
                         fp_any_ci_low=round(l2, 3), fp_any_ci_high=round(h2, 3),
                         direction_rule_side_call=k_dir, direction_rule_call_rate=round(p3, 3),
                         direction_rule_ci_low=round(l3, 3), direction_rule_ci_high=round(h3, 3),
                         specificity_strong_moderate=round(1 - p1, 3),
                         normative_stats=loo_note))
        log(f"  {tier:<9} controls n={n}: Strong {counts['Strong']}, Moderate {counts['Moderate']}, "
            f"Weak {counts['Weak']}, Bilateral {counts['Bilateral']}, Unclear {counts['Unclear']}  "
            f"-> Strong/Moderate FP {p1:.2f} [{l1:.2f}, {h1:.2f}]  any {p2:.2f}")
    per_ctrl = production_calls(data)
    return pd.DataFrame(rows), per_ctrl, f"{loo_note}; {len(cids)} with clusters, {len(no_csv)} without ({n_listed} in summary)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dataset_config.add_dataset_args(ap)
    args = ap.parse_args(argv)
    cfg = dataset_config.configure(args.dataset, perf_variant=args.perf_variant, band_policy=args.band_policy)
    out = cfg.analysis_dir
    out.mkdir(parents=True, exist_ok=True)
    gt = cfg.ground_truth()
    cdir = cfg.cluster_report_dir
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    pids = sorted(d.name for d in cdir.iterdir()
                  if d.is_dir() and (d / f"{d.name}_asymmetry_cluster_report.csv").exists()) if cdir.exists() else []
    pids = [p for p in pids if p in gt]
    meta = {p: {"group": group_class(gt[p].get("group", "")), "ez_lobe": gt[p].get("ez_lobe", "")} for p in pids}
    truth = {p: gt[p]["ez"] for p in pids}
    qc_excl = cfg.qc_excluded()                      # technical failures: out of everything
    qc_sens = cfg.qc_sensitivity_sets()              # flagged, kept in the primary cohort
    qc_flags = dict(cfg.qc_flags())                  # registration_qc.csv (reported)
    for name, ids in qc_sens.items():
        for i in ids:
            qc_flags[i] = (qc_flags.get(i, "") + ";" + name.replace("without_", "")).strip(";")
    unilateral_all = [p for p in pids if truth[p] in ("L", "R")]
    unilateral = [p for p in unilateral_all if p not in qc_excl]
    log(f"dataset={cfg.name} perf_variant={cfg.perf_variant} band_policy={cfg.band_policy}")
    log(f"patients with cluster reports: {len(pids)}  unilateral (L/R): {len(unilateral_all)}  "
        f"L={sum(truth[p] == 'L' for p in unilateral)} R={sum(truth[p] == 'R' for p in unilateral)}  "
        f"B/U/NA excluded from accuracy: {len(pids) - len(unilateral_all)}")
    log(f"QC: technically excluded {sorted(qc_excl)}; primary cohort keeps flagged subjects; "
        f"sensitivity cohorts: { {k: sorted(v) for k, v in qc_sens.items()} }")
    if not unilateral:
        log("No unilateral patients: nothing to evaluate (check subjects.csv ez_side).")
        (out / "eval_summary.txt").write_text("\n".join(lines) + "\n")
        return 0

    data = {p: csa.load_clusters(p, cdir) for p in pids}
    uni_data = {p: data[p] for p in unilateral}
    strata_defs = {
        "group=discrete": [p for p in unilateral if meta[p]["group"] == "discrete"],
        "group=negative": [p for p in unilateral if meta[p]["group"] == "negative"],
        "lobe=temporal": [p for p in unilateral if lobe_class(meta[p]["ez_lobe"]) == "temporal"],
        "lobe=extratemporal": [p for p in unilateral if lobe_class(meta[p]["ez_lobe"]) == "extratemporal"],
        "lobe=unknown": [p for p in unilateral if lobe_class(meta[p]["ez_lobe"]) == "unknown"],
        "excluding_post_surgical": [p for p in unilateral if p not in cfg.post_surgical],
    }
    for name, drop in qc_sens.items():
        strata_defs[f"cohort={name}"] = [p for p in unilateral if p not in drop]
        strata_defs[f"cohort={name}&group=discrete"] = [p for p in unilateral if p not in drop and meta[p]["group"] == "discrete"]
        strata_defs[f"cohort={name}&group=negative"] = [p for p in unilateral if p not in drop and meta[p]["group"] == "negative"]
    rule_df, strata_df, pp = evaluate_rules(uni_data, truth, meta, strata_defs)
    rule_df.to_csv(out / "eval_rule_tier.csv", index=False)
    # QC-flagged subjects: report their calls (production rule, every tier) separately
    flagged = sorted(p for p in pids if p in qc_flags)
    if flagged:
        fl = production_calls({p: data[p] for p in flagged})
        fl["truth"] = fl["id"].map(truth)
        fl["qc_flag"] = fl["id"].map(qc_flags)
        fl["in_primary_cohort"] = ~fl["id"].isin(qc_excl)
        fl.to_csv(out / "eval_qc_flagged.csv", index=False)
        log("\n=== QC-flagged patients (kept in the primary cohort; production rule, clinical tier) ===")
        log(fl[fl.tier == "clinical"][["id", "qc_flag", "in_primary_cohort", "truth", "pred", "confidence"]]
            .to_string(index=False))
    strata_df.to_csv(out / "eval_rule_tier_strata.csv", index=False)
    pp.to_csv(out / "eval_per_patient.csv", index=False)

    log("\n=== Rule x tier (37-ROI scope), unilateral patients ===")
    cols = ["tier", "rule", "n", "agree", "unclear", "accuracy", "acc_ci_low", "acc_ci_high",
            "balanced_acc", "majority_side", "majority_agree", "majority_acc"]
    log(rule_df[rule_df.scope == "37roi"][cols].to_string(index=False))
    log("\n=== Rule x tier (temporal scope) ===")
    log(rule_df[rule_df.scope == "temporal"][cols].to_string(index=False))
    log("\n=== Direction rule, clinical tier, by stratum (incl. QC sensitivity cohorts) ===")
    sel = strata_df[(strata_df.rule == "direction") & (strata_df.tier == "clinical") & (strata_df.scope == "37roi")]
    log(sel[["stratum", "n", "n_left", "n_right", "agree", "accuracy", "acc_ci_low", "acc_ci_high",
             "balanced_acc", "majority_agree"]].to_string(index=False))

    # production rule + call strength
    prod = production_calls(uni_data)
    prod["truth"] = prod["id"].map(truth)
    prod["correct"] = (prod["pred"] == prod["truth"]).astype(int)
    prod["group"] = prod["id"].map(lambda p: meta[p]["group"])
    prod["lobe"] = prod["id"].map(lambda p: lobe_class(meta[p]["ez_lobe"]))
    cs_rows = []
    for tier, g in prod.groupby("tier", sort=False):
        for strength in ("Strong", "Moderate", "Weak", "Bilateral", "Unclear"):
            sub = g[g.strength == strength]
            k, n = int(sub.correct.sum()), int(len(sub))
            acc, lo, hi = wilson(k, n)
            cs_rows.append(dict(tier=tier, strength=strength, n=n, agree=k, accuracy=round(acc, 3),
                                ci_low=round(lo, 3), ci_high=round(hi, 3)))
        k, n = int(g.correct.sum()), int(len(g))
        acc, lo, hi = wilson(k, n)
        maj_side, maj_k = majority_baseline(truth, list(g.id))
        cs_rows.append(dict(tier=tier, strength="ALL", n=n, agree=k, accuracy=round(acc, 3),
                            ci_low=round(lo, 3), ci_high=round(hi, 3),
                            balanced_acc=round(balanced_accuracy(dict(zip(g.id, g.pred)), truth), 3),
                            majority_side=maj_side, majority_agree=maj_k))
    cs = pd.DataFrame(cs_rows)
    cs.to_csv(out / "eval_call_strength.csv", index=False)
    prod.to_csv(out / "eval_production_rule_per_patient.csv", index=False)
    log("\n=== Production rule (06, mesial-temporal priority): call strength x tier ===")
    log(cs.to_string(index=False))

    # direction profile
    prof = direction_profile(data, truth, meta)
    prof.to_csv(out / "eval_direction_profile.csv", index=False)
    log("\n=== Direction phenotype (pair-aware; hypo-/hyper-dominant = >2x voxel volume) ===")
    if len(prof):
        tab = prof.groupby(["group", "phenotype"]).size().unstack(fill_value=0)
        log(tab.to_string())
        log(f"  cohort: hypo-dominant {int((prof.phenotype == 'hypo').sum())}, "
            f"hyper-dominant {int((prof.phenotype == 'hyper').sum())}, mixed {int((prof.phenotype == 'mixed').sum())} "
            f"of {len(prof)}; unilateral: hyper-dominant "
            f"{int(((prof.phenotype == 'hyper') & prof.truth.isin(['L', 'R'])).sum())}/{len(unilateral)}")

    # controls
    log("\n=== Control false positives (controls scored as pseudo-patients, production rule) ===")
    fp, per_ctrl, note = control_false_positives(cfg, log, exclude=frozenset(qc_excl))
    log(f"  normative statistics for pseudo-controls: {note}")
    if qc_excl:
        log(f"  QC-excluded controls (never scored): {sorted(c for c in qc_excl if c.startswith(cfg.control_prefixes))}")
    fp.to_csv(out / "eval_control_false_positives.csv", index=False)
    per_ctrl.to_csv(out / "eval_control_calls_per_control.csv", index=False)

    # pre-specified endpoints: printed last (computed last) but written first
    primary = primary_endpoint_report(cfg, cs, rule_df, fp, truth, unilateral)
    print("\n" + "\n".join(primary))
    (out / "eval_primary_endpoint.txt").write_text("\n".join(primary) + "\n")
    lines[:0] = primary

    (out / "eval_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nOutputs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
