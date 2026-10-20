#!/usr/bin/env python3
"""Write external_chalfont/RESULTS_CHALFONT.md from the Chalfont pipeline outputs.

Reads results_zscore/ (primary TC-CBF, pooled CH_ALL), results_zscore_flow/
(plain CBF) and results_zscore_bands/ (age-band sensitivity) and tabulates:
control leave-one-out calibration (vs in-house 6.9%), patient calibration,
lateralization per rule x tier (pooled / lesional D / MRI-negative N) with
balanced accuracy, majority baseline and Wilson CIs, QC sensitivity cohorts,
call-strength strata, hypo/hyper-dominant counts, control false-positive rates,
the Flow-vs-TC comparison and the age-band sensitivity.

    python scripts/chalfont/write_results_summary.py [--out external_chalfont/RESULTS_CHALFONT.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import dataset_config  # noqa: E402

INHOUSE_CONTROL_LOO = 6.9     # symreg/PIPELINE_B_FACTS.md
INHOUSE_PATIENT_MEAN = 12.9
RULE_ORDER = ["direction", "direction_no_subtle", "hypo_only", "hyper_only", "flip"]
TIER_ORDER = ["research", "clinical", "surgical", "native20"]


def _read(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text()) if path.exists() else None
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else None


def fmt_ci(row, acc="accuracy", lo="acc_ci_low", hi="acc_ci_high"):
    return f"{row[acc]:.2f} [{row[lo]:.2f}, {row[hi]:.2f}]"


def section_calibration(cfgs: dict) -> list[str]:
    lines = ["## 1. Calibration (gray matter, %|zAI| >= 1.96)", "",
             "| Run | Band | Controls (LOO) mean | median | range | full-stats mean | Patients n | mean | median |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, cfg in cfgs.items():
        cal = _read(cfg.calibration_dir / "calibration_summary.json")
        if not cal:
            lines.append(f"| {name} | (not available) | | | | | | | |")
            continue
        for band, b in cal["bands"].items():
            pats = _read(cfg.calibration_dir / "patient_gm_calibration.csv")
            pp = pats[pats.band == band] if pats is not None else pd.DataFrame()
            pm = f"{pp.pct_abs_zai_ge_1_96.mean():.1f}%" if len(pp) else "-"
            pmed = f"{pp.pct_abs_zai_ge_1_96.median():.1f}%" if len(pp) else "-"
            lines.append(f"| {name} | {band} (n={b['n_controls']}) | {b['control_loo_mean_pct']:.1f}% | "
                         f"{b['control_loo_median_pct']:.1f}% | {b['control_loo_range_pct'][0]:.1f}-{b['control_loo_range_pct'][1]:.1f}% | "
                         f"{b['control_full_mean_pct']:.1f}% | {len(pp)} | {pm} | {pmed} |")
    lines += ["", f"In-house reference (HCP controls, symmetric space): control leave-one-out {INHOUSE_CONTROL_LOO}%, "
                  f"patient mean {INHOUSE_PATIENT_MEAN}% (n=17). Nominal expectation for a calibrated normative model: ~5%.", ""]
    return lines


def rule_table(cfg, scope="37roi", strata=("all", "group=discrete", "group=negative"), tiers=TIER_ORDER,
               rules=RULE_ORDER) -> list[str]:
    rt = _read(cfg.analysis_dir / "eval_rule_tier.csv")
    st = _read(cfg.analysis_dir / "eval_rule_tier_strata.csv")
    if rt is None:
        return ["(no evaluation outputs)"]
    rt = rt.assign(stratum="all")
    df = pd.concat([rt, st], ignore_index=True) if st is not None else rt
    df = df[df.scope == scope]
    lines = ["| Tier | Rule | Cohort | n (L/R) | agree | accuracy [95% CI] | bal. acc | majority (side) | unclear |",
             "|---|---|---|---|---|---|---|---|---|"]
    for tier in tiers:
        for rule in rules:
            for stratum in strata:
                r = df[(df.tier == tier) & (df.rule == rule) & (df.stratum == stratum)]
                if r.empty:
                    continue
                r = r.iloc[0]
                lines.append(f"| {tier} | {rule} | {stratum} | {int(r.n)} ({int(r.n_left)}/{int(r.n_right)}) | {int(r.agree)} | "
                             f"{fmt_ci(r)} | {r.balanced_acc:.2f} | {int(r.majority_agree)}/{int(r.n)} = {r.majority_acc:.2f} ({r.majority_side}) | {int(r.unclear)} |")
    return lines


def section_lateralization(cfg) -> list[str]:
    lines = ["## 2. Lateralization, primary cohort (TC-CBF, pooled CH_ALL controls)", "",
             "Unilateral patients (ez_side L or R); pair-aware rules (one member per mirrored cluster pair). "
             "Tiers: research |zAI|>=3 & >=50 vox; clinical >=4 & >=100; surgical >=5 & >=200; native20 >=4 & >=610.", "",
             "### 2a. 37-ROI scope, pooled / lesional (D) / MRI-negative (N)", ""]
    lines += rule_table(cfg, "37roi")
    lines += ["", "### 2b. Temporal scope (temporal cortex + hippocampus + amygdala), clinical tier", ""]
    lines += rule_table(cfg, "temporal", tiers=["clinical"])
    lines += ["", "### 2c. QC sensitivity cohorts (direction rule, clinical tier, 37-ROI scope)", ""]
    st = _read(cfg.analysis_dir / "eval_rule_tier_strata.csv")
    if st is not None:
        sel = st[(st.rule == "direction") & (st.tier == "clinical") & (st.scope == "37roi")]
        lines += ["| Stratum | n (L/R) | agree | accuracy [95% CI] | bal. acc | majority |", "|---|---|---|---|---|---|"]
        for _, r in sel.iterrows():
            lines.append(f"| {r.stratum} | {int(r.n)} ({int(r.n_left)}/{int(r.n_right)}) | {int(r.agree)} | {fmt_ci(r)} | "
                         f"{r.balanced_acc:.2f} | {int(r.majority_agree)}/{int(r.n)} |")
    fl = _read(cfg.analysis_dir / "eval_qc_flagged.csv")
    if fl is not None:
        lines += ["", "QC-flagged patients (kept in the primary cohort), production rule, clinical tier:", "",
                  "| id | flag | truth | pred | confidence |", "|---|---|---|---|---|"]
        for _, r in fl[fl.tier == "clinical"].iterrows():
            lines.append(f"| {r.id} | {r.qc_flag} | {r.truth} | {r.pred} | {r.confidence} |")
    return lines


def section_call_strength(cfg) -> list[str]:
    cs = _read(cfg.analysis_dir / "eval_call_strength.csv")
    lines = ["## 3. Pre-specified production rule (06_clinical_interpretation.py): call strength", ""]
    if cs is None:
        return lines + ["(not available)"]
    lines += ["| Tier | Strength | n | agree | accuracy [95% CI] |", "|---|---|---|---|---|"]
    for _, r in cs.iterrows():
        extra = f" bal. acc {r.balanced_acc:.2f}, majority {int(r.majority_agree)}/{int(r.n)}" if r.strength == "ALL" and pd.notna(r.get("balanced_acc")) else ""
        acc = f"{r.accuracy:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]" if pd.notna(r.accuracy) else "-"
        lines.append(f"| {r.tier} | {r.strength} | {int(r.n)} | {int(r.agree)} | {acc}{extra} |")
    return lines


def section_direction(cfg) -> list[str]:
    prof = _read(cfg.analysis_dir / "eval_direction_profile.csv")
    lines = ["## 4. Direction of abnormality (pair-aware; dominant = >2x voxel volume of the other class)", ""]
    if prof is None or prof.empty:
        return lines + ["(not available)"]
    prof["cohort"] = prof.truth.isin(["L", "R"]).map({True: "unilateral", False: "B/U/NA"})
    tab = prof.groupby(["group", "phenotype"]).size().unstack(fill_value=0)
    lines += ["| Group | hypo-dominant | hyper-dominant | mixed | n |", "|---|---|---|---|---|"]
    for g, r in tab.iterrows():
        lines.append(f"| {g} | {int(r.get('hypo', 0))} | {int(r.get('hyper', 0))} | {int(r.get('mixed', 0))} | {int(r.sum())} |")
    lines.append(f"| all | {int((prof.phenotype == 'hypo').sum())} | {int((prof.phenotype == 'hyper').sum())} | "
                 f"{int((prof.phenotype == 'mixed').sum())} | {len(prof)} |")
    lines += ["", f"Cluster pairs (research tier, all patients): hypo {int(prof.hypo.sum())}, hyper {int(prof.hyper.sum())}, "
                  f"subtle {int(prof.subtle.sum())}, mixed {int(prof.mixed.sum())}. In-house: 3/13 unilateral patients hyperperfusion-dominant."]
    return lines


def section_controls(cfg) -> list[str]:
    fp = _read(cfg.analysis_dir / "eval_control_false_positives.csv")
    lines = ["## 5. Control false positives / specificity (controls scored as pseudo-patients)", ""]
    if fp is None:
        return lines + ["(not available)"]
    lines += [f"Normative statistics: {fp.normative_stats.iloc[0]}.", "",
              "| Tier | n | Strong | Moderate | Weak | Bilateral | Unclear | FP Strong/Moderate [95% CI] | specificity | FP any lateralizing call | direction-rule side call |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in fp.iterrows():
        lines.append(f"| {r.tier} | {int(r.n_controls)} | {int(r.Strong)} | {int(r.Moderate)} | {int(r.Weak)} | {int(r.Bilateral)} | {int(r.Unclear)} | "
                     f"{r.fp_rate_strong_moderate:.2f} [{r.fp_ci_low:.2f}, {r.fp_ci_high:.2f}] | {r.specificity_strong_moderate:.2f} | "
                     f"{r.fp_rate_any:.2f} [{r.fp_any_ci_low:.2f}, {r.fp_any_ci_high:.2f}] | {r.direction_rule_call_rate:.2f} |")
    return lines


def key_metrics(cfg) -> dict:
    out = {}
    cal = _read(cfg.calibration_dir / "calibration_summary.json")
    if cal:
        out["control_loo"] = cal["control_loo_mean_pct"]
        out["patient_mean"] = cal.get("patient_mean_pct")
    rt = _read(cfg.analysis_dir / "eval_rule_tier.csv")
    if rt is not None:
        r = rt[(rt.tier == "clinical") & (rt.scope == "37roi") & (rt.rule == "direction")].iloc[0]
        out.update(n=int(r.n), agree=int(r.agree), acc=r.accuracy, lo=r.acc_ci_low, hi=r.acc_ci_high,
                   bal=r.balanced_acc, maj=r.majority_acc)
    cs = _read(cfg.analysis_dir / "eval_call_strength.csv")
    if cs is not None:
        r = cs[(cs.tier == "clinical") & (cs.strength == "ALL")].iloc[0]
        out.update(prod_agree=int(r.agree), prod_n=int(r.n), prod_acc=r.accuracy, prod_bal=r.balanced_acc)
    fp = _read(cfg.analysis_dir / "eval_control_false_positives.csv")
    if fp is not None:
        r = fp[fp.tier == "clinical"].iloc[0]
        out.update(fp_sm=r.fp_rate_strong_moderate, fp_any=r.fp_rate_any, n_ctrl=int(r.n_controls))
    return out


def section_comparison(cfgs: dict) -> list[str]:
    lines = ["## 6. Plain-Flow vs TC-Flow and age-band sensitivity (clinical tier, 37-ROI scope)", "",
             "| Run | controls LOO | patients mean | direction rule agree/n | accuracy [CI] | bal. acc | majority | production rule agree/n (bal. acc) | control FP Strong/Mod | FP any |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, cfg in cfgs.items():
        m = key_metrics(cfg)
        if "n" not in m:
            lines.append(f"| {name} | (not available) | | | | | | | | |")
            continue
        lines.append(f"| {name} | {m.get('control_loo', float('nan')):.1f}% | {m.get('patient_mean') or float('nan'):.1f}% | "
                     f"{m['agree']}/{m['n']} | {m['acc']:.2f} [{m['lo']:.2f}, {m['hi']:.2f}] | {m['bal']:.2f} | {m['maj']:.2f} | "
                     f"{m.get('prod_agree', '-')}/{m.get('prod_n', '-')} ({m.get('prod_bal', float('nan')):.2f}) | "
                     f"{m.get('fp_sm', float('nan')):.2f} | {m.get('fp_any', float('nan')):.2f} |")
    # per-patient agreement TC vs Flow
    tc, fl = cfgs.get("TC-CBF (primary)"), cfgs.get("plain Flow")
    if tc and fl:
        a = _read(tc.analysis_dir / "eval_production_rule_per_patient.csv")
        b = _read(fl.analysis_dir / "eval_production_rule_per_patient.csv")
        if a is not None and b is not None:
            a = a[a.tier == "clinical"].set_index("id"); b = b[b.tier == "clinical"].set_index("id")
            common = a.index.intersection(b.index)
            same = int((a.loc[common, "pred"] == b.loc[common, "pred"]).sum())
            both_ok = int(((a.loc[common, "pred"] == a.loc[common, "truth"]) & (b.loc[common, "pred"] == b.loc[common, "truth"])).sum())
            lines += ["", f"TC vs Flow per-patient agreement of the production-rule call (clinical tier): {same}/{len(common)} identical calls; "
                          f"both correct in {both_ok}/{len(common)}."]
    bands = cfgs.get("TC-CBF, age bands (sensitivity)")
    if bands:
        st = _read(bands.analysis_dir / "eval_rule_tier.csv")
        if st is not None:
            lines += ["", "Age-band sensitivity: patients aged 20-59 compared with their own 20-year band "
                          "(CH_20_39 n=29, CH_40_59 n=27; CH_60_79 n=5 skipped). Same table as 2a:", ""]
            lines += rule_table(bands, "37roi", tiers=["clinical"])
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    tc = dataset_config.configure("chalfont", "tc", band_policy="all")
    cfgs = {"TC-CBF (primary)": tc,
            "plain Flow": dataset_config.configure("chalfont", "flow", band_policy="all"),
            "TC-CBF, age bands (sensitivity)": dataset_config.configure("chalfont", "tc", band_policy="specific")}
    out = args.out or (tc.output_dir.parent / "RESULTS_CHALFONT.md")
    gt = tc.ground_truth()
    pats = tc.patient_ids()
    from collections import Counter
    ez = Counter(gt[p]["ez"] for p in pats)
    grp = Counter(gt[p]["group"] for p in pats)
    ex = tc.qc_exclusions()
    lines = ["# Chalfont external validation — results", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/chalfont/write_results_summary.py` from "
             f"`{tc.output_dir.name}/`, `{cfgs['plain Flow'].output_dir.name}/`, `{cfgs['TC-CBF, age bands (sensitivity)'].output_dir.name}/`.", "",
             "## 0. Cohort and method", "",
             f"- Controls: CH_ALL n={len(tc.control_ids('CH_ALL'))} (pooled, sex-pooled, all ages) = primary normative group; "
             f"QC-excluded controls {ex.get('controls_excluded', [])}; duplicates {ex.get('duplicate_person_excluded', [])}.",
             f"- Patients with registered perfusion: {len(pats)} ({dict(grp)}); ez_side {dict(ez)}; "
             f"technically excluded {ex.get('patients_excluded_technical', [])}; flagged but kept "
             f"{ex.get('patients_flagged_lr_asymmetric', [])} (L/R asymmetric), {ex.get('registration_qc_flagged', [])} (registration QC).",
             "- GE 3T eASL CBF (ml/100g/min), transit-time-corrected (primary) and plain Flow; FSL flirt+fnirt to the in-house "
             "symmetric template (0.8 mm); ONE fixed Desikan-Killiany atlas (mode of 180 HCP aparc_sym) for GM masking, "
             "37-ROI scope and region names; pipeline logic identical to the in-house run "
             "(01 -> 02 -> 03 -> 06, |zAI| clipped at 20, SD floor 5%, mirror-pooled SD, pair-aware voting).",
             "- Reference standard: subjects.csv ez_side (L/R scored; B/U/NA kept in voxel-wise outputs, excluded from accuracy).", ""]
    lines += section_calibration(cfgs)
    lines += section_lateralization(tc) + [""]
    lines += section_call_strength(tc) + [""]
    lines += section_direction(tc) + [""]
    lines += section_controls(tc) + [""]
    lines += section_comparison(cfgs) + [""]
    lines += ["## 7. Commands", "", "```bash",
              "python scripts/chalfont/build_template_atlas.py",
              "bash scripts/chalfont/run_chalfont_pipeline.sh tc all      # primary (results_zscore/)",
              "bash scripts/chalfont/run_chalfont_pipeline.sh flow all    # plain Flow (results_zscore_flow/)",
              'CHALFONT_BANDS="CH_20_39 CH_40_59" bash scripts/chalfont/run_chalfont_pipeline.sh tc specific   # age bands (results_zscore_bands/)',
              "python scripts/chalfont/write_results_summary.py", "```", ""]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
