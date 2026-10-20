"""tests/test_chalfont_synthetic_pipeline.py — end-to-end run of the Chalfont
driver (scripts/chalfont/run_chalfont_pipeline.sh, tc pass) on a small
synthetic root written by scripts/chalfont/make_synthetic_chalfont.py.

Checks that every stage runs in --dataset chalfont mode with the fixed template
atlas and ID-list discovery, that the planted unilateral abnormalities are
lateralized correctly (D001 L, D002 R, N003 R), that NA/B patients are kept in
the voxel-wise outputs but excluded from accuracy, and that the leave-one-out
control calibration and pseudo-control false-positive outputs are written.
Takes ~30 s; set ASYM_SKIP_SLOW=1 to skip.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(os.environ.get("ASYM_SKIP_SLOW") == "1", reason="ASYM_SKIP_SLOW=1")


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("chalfont") / "external_chalfont"
    subprocess.run([sys.executable, str(ROOT / "scripts/chalfont/make_synthetic_chalfont.py"),
                    str(root), "--n-controls", "8", "--single-band"], check=True, cwd=ROOT)
    env = dict(os.environ, CHALFONT_ROOT=str(root), CHALFONT_MIN_CONTROLS_PER_BAND="5")
    res = subprocess.run(["bash", str(ROOT / "scripts/chalfont/run_chalfont_pipeline.sh"), "tc"],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    return root, root / "results_zscore"


def test_stages_wrote_outputs(synthetic_run):
    root, out = synthetic_run
    assert (out / "groups/CH_ALL/consensus_parcellation.nii.gz").exists()
    meta = json.loads((out / "groups/CH_ALL/metadata.json").read_text())
    assert meta["n_subjects"] == 8 and meta["sym"] is True
    for pid in ("D001", "D002", "N001", "N002", "N003"):
        assert (out / f"patients/{pid}/{pid}_vs_CH_ALL_zscore.nii.gz").exists()
        assert (out / f"asymmetry/patients/{pid}/{pid}_asymmetry_zscore.nii.gz").exists()
        assert (out / f"clinical/{pid}/{pid}_clinical_zai_gm.nii.gz").exists()
    assert (out / "calibration/control_loo_calibration.csv").exists()
    assert (out / "calibration/patient_gm_calibration.csv").exists()
    assert (out / "asymmetry/pseudo_controls/CH_ALL_pseudo_control_summary.csv").exists()
    for f in ("eval_rule_tier.csv", "eval_rule_tier_strata.csv", "eval_call_strength.csv",
              "eval_direction_profile.csv", "eval_control_false_positives.csv", "eval_summary.txt",
              "eval_primary_endpoint.txt",
              "clinical_stratified_summary.csv"):
        assert (out / "lateralization" / f).exists(), f


def test_planted_lateralization_recovered(synthetic_run):
    root, out = synthetic_run
    cc = pd.read_csv(out / "clinical/cohort_concordance.csv").set_index("patient_id")
    assert cc.loc["D001", "algo_prediction"] == "L" and cc.loc["D001", "concordance"] == "agree"
    assert cc.loc["D002", "algo_prediction"] == "R" and cc.loc["D002", "concordance"] == "agree"
    assert cc.loc["N003", "algo_prediction"] == "R"
    prof = pd.read_csv(out / "lateralization/eval_direction_profile.csv").set_index("id")
    assert prof.loc["D001", "phenotype"] == "hypo" and prof.loc["D002", "phenotype"] == "hyper"
    rt = pd.read_csv(out / "lateralization/eval_rule_tier.csv")
    row = rt[(rt.tier == "clinical") & (rt.scope == "37roi") & (rt.rule == "direction")].iloc[0]
    assert row.n == 3 and row.agree == 3 and row.majority_agree == 2   # NA + B excluded; L=1 R=2
    assert 0 <= row.acc_ci_low <= row.accuracy <= row.acc_ci_high <= 1


def test_control_calibration_and_false_positives(synthetic_run):
    root, out = synthetic_run
    cal = json.loads((out / "calibration/calibration_summary.json").read_text())
    assert cal["bands"]["CH_ALL"]["n_controls"] == 8
    assert 0 < cal["control_loo_mean_pct"] < 15
    assert cal["control_loo_mean_pct"] > cal["control_full_mean_pct"]   # LOO is the honest number
    fp = pd.read_csv(out / "lateralization/eval_control_false_positives.csv").set_index("tier")
    assert int(fp.loc["clinical", "n_controls"]) == 8
    assert fp.loc["clinical", "normative_stats"] == "leave-one-out"
    assert "band_policy=all" in (out / "lateralization/eval_summary.txt").read_text()
    assert 0 <= fp.loc["clinical", "fp_rate_strong_moderate"] <= 0.5


def test_primary_endpoint_written_first(synthetic_run):
    """Audit 2026-10-19 finding 2: one pre-specified result, printed before the sensitivity grid."""
    root, out = synthetic_run
    pe = (out / "lateralization/eval_primary_endpoint.txt").read_text()
    summary = (out / "lateralization/eval_summary.txt").read_text()
    assert pe.startswith("=== PRE-SPECIFIED PRIMARY ENDPOINT")
    assert summary.startswith(pe.splitlines()[0])
    assert "status: PRIMARY ANALYSIS" in pe            # tc / all is the primary configuration
    assert "PRIMARY: agree " in pe and "balanced accuracy" in pe
    assert "SECONDARY 1" in pe and "SECONDARY 2" in pe
    assert "n=3 (L=1, R=2)" in pe                      # NA + B excluded
