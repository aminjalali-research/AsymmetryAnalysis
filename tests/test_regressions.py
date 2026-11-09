"""
tests/test_regressions.py — regression tests for previously fixed bugs and
core invariants of the Pipeline A/B code.

Covers:
  a. ez_ground_truth label tally (L=8, R=5, B=2, U=2; P014 excluded).
  b. 02_compute_zai._classify_direction direction-of-abnormality rules.
  c. Antisymmetry of the voxelwise AI under left-right mirroring.
  d. 06_clinical_interpretation.ClinicalInterpreter._mdt_concordance.
  e. src/thresholding M3/M4/M7 do not collapse to an empty mask when the
     zAI map contains a large block of zero (no-data) voxels.
  f. interactive_viewer all-layer view (options 17-18, --subject/--all):
     dataset from the ID, layer files for both layouts, peak voxel with zAI
     fallback, FSLeyes command (centred, only template + clinical on), PNG.
  g. interactive_viewer menu launcher: commands built through _launch in
     record mode (template underlay, stated threshold = display threshold,
     missing layers skipped, centred on the peak) and the reference line.
  h. interactive_viewer registration views (options 19-25): per-dataset files
     of the chain, missing inputs reported, consensus outlines.
"""

import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _load_script(name: str, filename: str):
    """Import a top-level script whose filename is not a valid module name."""
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# a. Ground-truth labels
# ---------------------------------------------------------------------------

def test_ground_truth_label_tally():
    from ez_ground_truth import EZ_GROUND_TRUTH, DISCARDED

    assert len(EZ_GROUND_TRUTH) == 17
    counts = Counter(v["ez"] for v in EZ_GROUND_TRUTH.values())
    assert counts == {"L": 8, "R": 5, "B": 2, "U": 2}
    assert counts["L"] + counts["R"] == 13  # unilateral discrimination cohort


def test_p014_excluded():
    from ez_ground_truth import EZ_GROUND_TRUTH, DISCARDED

    assert "P014" in DISCARDED
    assert "P014" not in EZ_GROUND_TRUTH


# ---------------------------------------------------------------------------
# b. Direction-of-abnormality classification
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def classify_direction():
    return _load_script("compute_zai_02", "02_compute_zai.py")._classify_direction


def test_direction_hypo_on_cluster_side(classify_direction):
    cls, side, _ = classify_direction(-3.0, 0.5, "L", "R")
    assert (cls, side) == ("hypo", "L")


def test_direction_hyper_on_mirror_side(classify_direction):
    cls, side, _ = classify_direction(0.3, 3.5, "L", "R")
    assert (cls, side) == ("hyper", "R")


def test_direction_both_deviating_is_mixed(classify_direction):
    cls, _, _ = classify_direction(-3.0, 2.5, "R", "L")
    assert cls == "mixed"


def test_direction_neither_deviating_is_subtle(classify_direction):
    cls, _, _ = classify_direction(0.8, -1.2, "R", "L")
    assert cls == "subtle"


def test_direction_exact_tie_gives_no_side(classify_direction):
    _, side, _ = classify_direction(0.0, 0.0, "L", "R")
    assert side == ""


# ---------------------------------------------------------------------------
# c. AI antisymmetry under left-right mirroring
# ---------------------------------------------------------------------------

def test_ai_antisymmetry():
    rng = np.random.default_rng(0)
    L = rng.uniform(10.0, 80.0, size=(20, 24, 18))
    R = np.flip(L, axis=0)
    ai = 100.0 * (L - R) / ((L + R) / 2.0)
    np.testing.assert_allclose(ai, -np.flip(ai, axis=0), atol=1e-10)


# ---------------------------------------------------------------------------
# d. MDT concordance rule
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mdt_concordance():
    mod = _load_script("clinical_interp_06", "06_clinical_interpretation.py")
    return mod.ClinicalInterpreter._mdt_concordance


def test_mdt_concordance_agree(mdt_concordance):
    assert mdt_concordance("L", "L") == "agree"


def test_mdt_concordance_disagree(mdt_concordance):
    assert mdt_concordance("R", "L") == "disagree"


def test_mdt_concordance_bilateral(mdt_concordance):
    assert mdt_concordance("B", "B") == "agree"


def test_mdt_concordance_unclear_label(mdt_concordance):
    # signature: (prediction, mdt_ez)
    assert mdt_concordance("L", "U") == "mdt-unclear"


# ---------------------------------------------------------------------------
# e. M3/M4/M7 zero-voxel sigma regression
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def zero_heavy_data():
    """zAI volume whose brain mask is >50% zero (no-data) voxels plus a blob.

    Before the fix, the half-normal noise sigma was estimated over all
    quality-mask voxels; with a zero majority the median |zAI| was 0, so
    sigma_noise collapsed to 0 and M3/M4/M7 returned an empty mask.
    """
    from thresholding import AsymData

    shape = (32, 32, 32)
    rng = np.random.RandomState(7)
    brain = np.zeros(shape, bool)
    brain[2:-2, 2:-2, 2:-2] = True

    zai = np.zeros(shape, np.float32)
    data_region = brain.copy()
    data_region[:18] = False  # x < 18 inside brain stays exactly zero (~57%)
    zai[data_region] = rng.randn(int(data_region.sum())).astype(np.float32)

    blob = np.zeros(shape, bool)
    blob[20:28, 10:18, 10:18] = True
    zai[blob] = 8.0 + 0.5 * rng.randn(int(blob.sum())).astype(np.float32)

    frac_zero = np.mean(zai[brain] == 0)
    assert frac_zero > 0.5, frac_zero  # precondition for the regression

    perf = np.ones(shape, np.float32)  # uniform perfusion -> quality mask = brain
    return AsymData(zai, brain, perfusion=perf), blob


@pytest.mark.parametrize("method_name", ["m3_fdr", "m4_bonferroni", "m7_permutation"])
def test_mc_methods_survive_zero_voxels(zero_heavy_data, method_name):
    import thresholding

    d, blob = zero_heavy_data
    mask = getattr(thresholding, method_name)(d)
    assert mask.dtype == bool and mask.shape == blob.shape
    assert mask.any(), f"{method_name} returned an empty mask"
    assert mask[blob].mean() > 0.5, f"{method_name} missed the planted blob"


# ---------------------------------------------------------------------------
# e. Mirror-AI validity: one-sided non-positive CBF must not be accepted
#    (audit 2026-10-19; previously mean_lr > 0.001 admitted L=-30/R=+40 -> AI=-1400%)
# ---------------------------------------------------------------------------

def test_ai_valid_requires_both_sides_positive():
    mod = _load_script("compute_zai_02", "02_compute_zai.py")
    perf = np.full((6, 4, 4), 40.0)
    perf[0, 0, 0] = -30.0          # negative on one side, mirror (index 5) is +40
    perf[1, 0, 0] = 0.0            # zero on one side
    ai, valid = mod.compute_asymmetry_map(perf)
    assert not valid[0, 0, 0] and not valid[5, 0, 0]
    assert not valid[1, 0, 0] and not valid[4, 0, 0]
    assert valid[2, 0, 0] and valid[3, 0, 0]
    assert np.all(np.abs(ai) <= 200.0 + 1e-9)   # |AI| is bounded when both sides > 0
    assert ai[0, 0, 0] == 0.0


# ---------------------------------------------------------------------------
# f. interactive_viewer all-layer view
# ---------------------------------------------------------------------------
_VIEW_SHAPE = (12, 14, 10)


def _view_save(path, data):
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.diag([-0.8, 0.8, 0.8, 1])), str(path))


@pytest.fixture(autouse=True)
def _private_view_cache(tmp_path, monkeypatch):
    """Masked copies written by the viewer go to the test's own folder, never the shared cache."""
    import interactive_viewer as iv
    monkeypatch.setattr(iv, "VIEW_CACHE", tmp_path / "asym_view")
    monkeypatch.setattr(iv, "_MASKS", {})


def _view_tree(tmp, sid="D002", peak=(3, 5, 6), clinical=True):
    """Minimal Chalfont layout: template, CBF, #3, #4, clinical map, atlas, concordance row."""
    import interactive_viewer as iv
    t1 = np.zeros(_VIEW_SHAPE); t1[2:10, 2:12, 2:8] = 100
    _view_save(tmp / iv.SYM_T1, t1)
    res = tmp / "external_chalfont/results_zscore"
    _view_save(tmp / f"external_chalfont/sym_perf/{sid}_perf_sym.nii.gz", t1 * 0.5)
    _view_save(res / f"patients/{sid}/{sid}_vs_CH_ALL_zscore.nii.gz", np.full(_VIEW_SHAPE, -2.5))
    zai = np.zeros(_VIEW_SHAPE); zai[8, 8, 4] = -6.0
    _view_save(res / f"asymmetry/patients/{sid}/{sid}_asymmetry_zscore.nii.gz", zai)
    if clinical:
        clin = np.zeros(_VIEW_SHAPE); clin[peak] = 4.3; clin[1, 1, 1] = -2.0
        _view_save(res / f"clinical/{sid}/{sid}_clinical_zai_lateralized_significant.nii.gz", clin)
        both = clin.copy(); both[_VIEW_SHAPE[0] - 1 - peak[0], peak[1], peak[2]] = -4.3   # mirror member
        _view_save(res / f"clinical/{sid}/{sid}_clinical_zai_significant_clusters.nii.gz", both)
    _view_save(tmp / "external_chalfont/atlas/consensus_parcellation_sym.nii.gz", np.ones(_VIEW_SHAPE))
    (res / "clinical").mkdir(parents=True, exist_ok=True)
    (res / "clinical/cohort_concordance.csv").write_text(
        "patient_id,mdt_ez,mdt_confidence,algo_prediction,algo_confidence,concordance,top_lobe,top_voxels,peak_zai\n"
        f"{sid},L,,B,Bilateral,disagree,L-parietal,582,4.33\n")
    return tmp


def test_all_layers_dataset_and_files(tmp_path):
    import interactive_viewer as iv
    assert iv.dataset_of("P016") == "inhouse" and iv.dataset_of("D002") == "chalfont"
    v = iv.AllLayersView("D002", base=_view_tree(tmp_path))
    assert v.band == "CH_ALL" and v.missing() == []
    assert "reference L" in v.summary() and "B (Bilateral)" in v.summary()


def test_all_layers_peak_voxel_and_fallback(tmp_path):
    import interactive_viewer as iv
    assert iv.AllLayersView("D002", base=_view_tree(tmp_path)).peak_voxel() == (3, 5, 6)
    v = iv.AllLayersView("D002", base=_view_tree(tmp_path / "noclin", clinical=False))
    assert v.peak_voxel() == (8, 8, 4)   # no clinical map: largest |zAI|


def test_all_layers_fsleyes_command(tmp_path):
    import interactive_viewer as iv
    cmd = iv.AllLayersView("D002", base=_view_tree(tmp_path)).fsleyes_command((3, 5, 6))
    assert cmd[:6] == ["fsleyes", "-vl", "3", "5", "6", iv.SYM_T1]
    clin = next(i for i, c in enumerate(cmd) if c.endswith("significant_clusters.nii.gz"))
    assert "-d" not in cmd[clin:]          # clinical clusters (both pair members) visible
    assert cmd.count("-d") == 4            # atlas, CBF, #3, #4 start hidden


def test_all_layers_png(tmp_path):
    import interactive_viewer as iv
    v = iv.AllLayersView("D002", base=_view_tree(tmp_path, clinical=False))
    out = v.render_png(v.peak_voxel(), tmp_path / "out" / "D002.png")
    assert out.exists() and out.stat().st_size > 1000
    assert v.missing() == ["clinical"]


# ---------------------------------------------------------------------------
# g. interactive_viewer menu launcher (options 3-16 share _launch)
# ---------------------------------------------------------------------------
def _menu_viewer(tmp_path, monkeypatch):
    import interactive_viewer as iv
    _view_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(iv, "RESULTS_ROOT", "external_chalfont/results_zscore")
    monkeypatch.setattr(iv, "FIXED_ATLAS", "external_chalfont/atlas/consensus_parcellation_sym.nii.gz")
    monkeypatch.setattr(iv, "DATASET", "chalfont")
    monkeypatch.setattr(iv, "rule_focus", lambda sid, ds: (None, None))   # no 06_ report in the fixture
    v = iv.InteractiveOverlayViewer("D002")
    v._collect = []
    return iv, v


def test_menu_clinical_view_command(tmp_path, monkeypatch):
    iv, v = _menu_viewer(tmp_path, monkeypatch)
    v.view_zai_clinical()
    (title, argv), = v._collect
    assert argv[:5] == ["fsleyes", "-vl", "3", "5", "6"]            # opened on the clinical peak
    assert argv[5] == iv.SYM_T1                                     # template underlay
    both = argv.index(next(a for a in argv if a.endswith("significant_clusters.nii.gz")))
    assert argv[both + 1:both + 8] == ["-cm", "red-yellow", "-nc", "blue-lightblue", "-un", "-dr", "3"]
    nxt = argv.index(next(a for a in argv if a.endswith("lateralized_significant.nii.gz")))
    assert "-d" not in argv[both:nxt] and argv[nxt - 1] != "-d"       # both members visible ...
    assert "-d" in argv[nxt:nxt + 13]                                # ... higher-member map hidden
    assert not any(a.endswith("left_dominant.nii.gz") for a in argv)  # missing masks skipped, not passed


def test_menu_opens_on_rule_focus(tmp_path, monkeypatch):
    iv, v = _menu_viewer(tmp_path, monkeypatch)
    monkeypatch.setattr(iv, "rule_focus", lambda sid, ds: ((7, 8, 4), "rule's top cluster"))
    v.view_zai_clinical()
    (_, argv), = v._collect
    assert argv[1:5] == ["-vl", "7", "8", "4"]                      # the rule's cluster wins over the map peak


def test_menu_perfusion_z_threshold(tmp_path, monkeypatch):
    iv, v = _menu_viewer(tmp_path, monkeypatch)
    v.view_zscore_control_map()
    (_, argv), = v._collect
    z = argv.index(next(a for a in argv if a.endswith("_vs_CH_ALL_zscore.nii.gz")))
    assert argv[z + 6:z + 9] == ["-dr", "1.96", "6"]                # |z| < 1.96 transparent
    assert "-vl" not in argv                                        # whole-brain map: not centred


def test_reference_line(monkeypatch):
    import interactive_viewer as iv
    monkeypatch.setattr(iv, "EZ_GROUND_TRUTH", {
        "P016": {"ez": "R", "confidence": "Moderate"},
        "D002": {"ez": "L", "confidence": None, "group": "discrete", "ez_lobe": "frontal", "pathology": "FCD"}})
    v = iv.InteractiveOverlayViewer.__new__(iv.InteractiveOverlayViewer)
    v.patient_id = "P016"
    assert v._reference_line() == "MDT EZ: R (Moderate confidence)"
    v.patient_id = "D002"
    assert v._reference_line() == "Reference side: L (discrete; frontal; FCD)"


# ---------------------------------------------------------------------------
# h. interactive_viewer registration views (options 19-25)
# ---------------------------------------------------------------------------
def test_registration_view_files(tmp_path):
    import interactive_viewer as iv
    base = _view_tree(tmp_path)
    rv = iv.RegistrationView("D002", base=base)
    assert rv.t1 == base / "external_chalfont/staged/D002/T1_brain.nii.gz"
    assert rv.asl2t1 == base / "external_chalfont/sym_perf/D002_asl2t1.mat"
    assert rv.labels_sym == base / "external_chalfont/atlas/consensus_parcellation_sym.nii.gz"
    miss = rv.missing()                      # fixture has CBF on T_sym but no staged inputs / warp
    assert any(m.endswith("T1_brain.nii.gz") for m in miss) and any(m.endswith("_warp.nii.gz") for m in miss)
    assert not any(m.endswith("D002_perf_sym.nii.gz") for m in miss)
    ih = iv.RegistrationView("P016", base=base)
    assert ih.asl_ref is None and ih.asl2t1 is None                 # ASL -> T1 done by HCP-ASL
    assert ih.t1 == base / "symreg/patient_brains/P016_brain.nii.gz"


def test_menu_outlines_use_consensus(tmp_path, monkeypatch):
    iv, v = _menu_viewer(tmp_path, monkeypatch)
    assert v.overlay_parc.endswith("consensus_parcellation_sym.nii.gz")   # not the patient's own warped aparc


def test_brain_masked_copies(tmp_path):
    import interactive_viewer as iv
    mask = np.zeros(_VIEW_SHAPE); mask[2:10, 2:12, 2:8] = 7          # labels inside the brain
    _view_save(tmp_path / "a/atlas.nii.gz", mask)
    z = np.full(_VIEW_SHAPE, 3.0)
    _view_save(tmp_path / "a/D002_vs_CH_ALL_zscore.nii.gz", z)
    _view_save(tmp_path / "b/D002_vs_CH_ALL_zscore.nii.gz", -z)      # same file name, other dataset
    ma = iv.brain_masked(tmp_path / "a/D002_vs_CH_ALL_zscore.nii.gz", tmp_path / "a/atlas.nii.gz")
    mb = iv.brain_masked(tmp_path / "b/D002_vs_CH_ALL_zscore.nii.gz", tmp_path / "a/atlas.nii.gz")
    assert ma != mb                                                   # no cache collision
    da = np.asanyarray(nib.load(str(ma)).dataobj)
    assert da[0, 0, 0] == 0 and da[5, 5, 5] == 3.0                    # outside zeroed, inside kept
    assert np.asanyarray(nib.load(str(mb)).dataobj)[5, 5, 5] == -3.0
