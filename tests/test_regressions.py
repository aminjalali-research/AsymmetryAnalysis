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
  f. interactive_viewer all-layer view (menu 'All layers', --subject/--all):
     dataset from the ID, layer files for both layouts, peak voxel with zAI
     fallback, FSLeyes command (centred, only template + clinical on), PNG.
  g. interactive_viewer menu launcher: commands built through _launch in
     record mode (template underlay, stated threshold = display threshold,
     missing layers skipped, centred on the peak) and the reference line.
  h. interactive_viewer registration views (menu 'registration'): per-dataset
     files of the chain, missing inputs reported, consensus outlines.
  i. interactive_viewer menu in the order of the Methods (option numbers from
     MENU, none hard-coded), the analysis #2 voxel AI (= 02_'s index,
     antisymmetric), regional painting sign, direction-class maps rebuilt from
     the 02_ masks, and the regional normative z equal to the v2 table.
  j. interactive_viewer misdetected cases: the side-from-text rule, the label-file
     parser, and on the real outputs every method's count equal to the published
     one, the control-range split and the reasons for the in-house misses.
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
# g. interactive_viewer menu launcher (every FSLeyes view goes through _launch)
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
# h. interactive_viewer registration views
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


# ---------------------------------------------------------------------------
# i. interactive_viewer menu in Methods order, analysis #1-#2 and direction views
# ---------------------------------------------------------------------------
def test_menu_follows_methods_order():
    import inspect
    import re
    import interactive_viewer as iv
    titles = [t for t, _ in iv.MENU]
    sections = ["Inputs", "Analysis #1", "Left-right symmetric registration", "Analyses #2 and #4", "Thresholding",
                "Direction", "Lateralization decision", "Post hoc", "All layers", "Misdetected cases", "Display aid"]
    assert [next(i for i, t in enumerate(titles) if t.startswith(sec)) for sec in sections] == list(range(len(sections)))
    names = [n for _, items in iv.MENU for n, _ in items]
    assert len(names) == len(set(names)) == len(iv.OPTION) == 30
    assert all(callable(getattr(iv.InteractiveOverlayViewer, n)) for n in names) and iv.PRINT_ONLY <= set(names)
    # numbers quoted by the deck and docs/pipeline/08-interactive-viewer.md
    quoted = {"view_regional_li": "3", "view_reg_template": "4", "render_reg_png": "10", "view_voxel_ai": "11",
              "view_control_normative": "12", "view_zai_asymmetry_map": "13", "view_zai_significant": "14",
              "view_zai_clinical": "15", "view_direction_classes": "17", "view_clinical_report": "18",
              "view_combined_presurgical": "21", "view_regional_normative": "22", "view_zscore_clusters": "23",
              "view_all_layers": "24", "render_all_layers_png": "25", "view_miss_overview": "26",
              "view_miss_case": "27", "show_quick_commands": "30"}
    assert {n: iv.OPTION[n] for n in quoted} == quoted
    body = inspect.getsource(iv).replace(iv.__doc__, "")
    assert not re.search(r"[Oo]ptions? \d", body)       # every "option N" in a note comes from OPTION


def test_voxel_ai_is_02s_index_and_antisymmetric(tmp_path):
    import interactive_viewer as iv
    cbf = np.random.default_rng(0).uniform(10, 80, _VIEW_SHAPE)
    cbf[4, 6, 5] = 0                                                  # no AI where one side is not positive
    _view_save(tmp_path / "cbf.nii.gz", cbf)
    brain = np.zeros(_VIEW_SHAPE); brain[2:10, 2:12, 2:8] = 1; brain[2, 2, 2] = 0   # not mirror-symmetric
    _view_save(tmp_path / "brain.nii.gz", brain)
    ai = np.asanyarray(nib.load(str(iv.voxel_ai(tmp_path / "cbf.nii.gz", tmp_path / "brain.nii.gz", "T"))).dataobj)
    ref, _ = iv.load_script("02_compute_zai.py").compute_asymmetry_map(
        np.asanyarray(nib.load(str(tmp_path / "cbf.nii.gz")).dataobj))
    assert ai.shape == _VIEW_SHAPE
    inside = ai != 0
    assert inside.sum() > 0 and np.allclose(ai[inside], ref[inside], atol=1e-3)
    assert np.allclose(ai, -np.flip(ai, axis=0), atol=1e-3)           # AI(mirror v) = -AI(v) everywhere
    assert ai[9, 2, 2] == 0 and ai[4, 6, 5] == 0 and ai[7, 6, 5] == 0


def test_paint_regions_sign(tmp_path):
    import interactive_viewer as iv
    parc = np.zeros(_VIEW_SHAPE); parc[1:3] = 17; parc[9:11] = 53; parc[5, 5, 5] = 1001; parc[6, 5, 5] = 2001
    _view_save(tmp_path / "parc.nii.gz", parc)
    d = np.asanyarray(nib.load(str(iv.paint_regions({(17, 53): -0.2, (1001, 2001): float("nan")},
                                                    tmp_path / "parc.nii.gz", "x.nii.gz"))).dataobj)
    assert np.allclose(d[parc == 17], -0.2) and np.allclose(d[parc == 53], 0.2)   # left lower: left blue, right red
    assert d[5, 5, 5] == 0 and d[0, 0, 0] == 0


def test_direction_class_maps_match_report(tmp_path, monkeypatch):
    iv, v = _menu_viewer(tmp_path, monkeypatch)
    pos = np.zeros(_VIEW_SHAPE); pos[1:3, 1:3, 1:3] = 1; pos[1:3, 8:11, 5:7] = 1    # 02_ clusters 1 (8 vox), 2 (12)
    neg = np.flip(pos, axis=0)                                                       # their mirrors, same ids
    _view_save(tmp_path / v.zai_files["left_dominant"], pos)
    _view_save(tmp_path / v.zai_files["right_dominant"], neg)
    (tmp_path / v.zai_dir / "D002_asymmetry_cluster_report.csv").write_text(
        "direction,cluster_id,size_voxels,cluster_hemi,ez_side_pred,direction_class\n"
        "left-dominant,1,8,L,L,hypo\nleft-dominant,2,12,L,R,hyper\n"
        "right-dominant,1,8,R,L,hypo\nright-dominant,2,12,R,R,hyper\n"
        "left-dominant,3,99,L,R,subtle\n")                                         # no such cluster
    masks, other, table, mismatch = v.direction_class_maps()
    hypo = np.asanyarray(nib.load(str(masks["hypo"])).dataobj) > 0
    hyper = np.asanyarray(nib.load(str(masks["hyper"])).dataobj) > 0
    rest = np.asanyarray(nib.load(str(other)).dataobj) > 0
    a = np.zeros(_VIEW_SHAPE, bool); a[1:3, 1:3, 1:3] = True
    b = np.zeros(_VIEW_SHAPE, bool); b[1:3, 8:11, 5:7] = True
    assert (hypo == a).all() and (hyper == np.flip(b, axis=0)).all()      # one member per pair, on the predicted side
    assert (rest == (b | np.flip(a, axis=0))).all() and set(masks) == {"hypo", "hyper"}
    assert table.loc["hypo", "L"] == 1 and table.loc["hyper", "R"] == 1 and mismatch == 1


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "v2_features/inhouse/P016_roi.json").exists(),
                    reason="needs the in-house v2 features")
def test_regional_normative_equals_v2_table(monkeypatch):
    import interactive_viewer as iv
    import pandas as pd
    base = Path(iv.BASE_DIR)
    monkeypatch.chdir(base)
    iv.configure_dataset("inhouse")
    v = iv.InteractiveOverlayViewer("P016")
    _, zbar, ctrl = v.regional_normative(base / "v2_features/inhouse")
    row = pd.read_csv(base / "ez_analysis_mdt/v2/inhouse/R_patients.csv").set_index("id").loc["P016"]
    assert len(ctrl) == 60
    assert np.isclose(zbar["mesial"], row["R_mesial"]) and np.isclose(zbar["temporal"], row["R_temporal"])
    assert np.isclose(zbar["all 37"], row["R_all37"])


# ---------------------------------------------------------------------------
# j. interactive_viewer misdetected cases
# ---------------------------------------------------------------------------
def test_side_from_text():
    import interactive_viewer as iv
    assert iv._side_from_text("Left temporal spikes") == "L"
    assert iv._side_from_text("R TLE") == "R"
    assert iv._side_from_text("right > left temporal slowing") == "B"       # both sides named
    assert iv._side_from_text("bitemporal onset") == "B"
    assert iv._side_from_text("Normal") == "neg"
    assert iv._side_from_text("generalised discharges") == "NL"
    assert iv._side_from_text("N/A") == "" and iv._side_from_text(None) == ""
    assert iv._side_from_text("left a message with family") == "NL"        # "left" here is not a side


def test_chalfont_label_files_parser(tmp_path):
    import interactive_viewer as iv
    script = tmp_path / "build_subjects.py"
    script.write_text(
        "k2 = src('seeg_sheet2', 'baris/x.ods [Sheet2] (N001-N059: Lateralisation)')\n"
        "    put(i, 'lateralisation', r[ci['Lateralisation']].strip(), k2)\n"
        "k = src('qc', 'gavin/qc.csv (ID, Laterality 1=right/0=left)')\n"
        "    if r[3] in ('D', 'N'): put(i, 'lat_code', {'1': 'R', '0': 'L'}.get(r[4].strip()), k)\n"
        "k = src('anon', 'oscar/anon.csv (MRI Report, EEG)')\n"
        "    put(i, 'mri_report', r['MRI Report'].strip(), k); put(i, 'eeg', r['EEG Findings'].strip(), k)\n")
    fields, files = iv.chalfont_label_files(script)
    assert fields == {"lateralisation": ["seeg_sheet2"], "lat_code": ["qc"], "mri_report": ["anon"], "eeg": ["anon"]}
    assert files["seeg_sheet2"] == "baris/x.ods [Sheet2]" and files["qc"] == "gavin/qc.csv"


_HAVE_MISS_INPUTS = all((Path(__file__).resolve().parents[1] / f).exists() for f in (
    "ez_analysis_mdt/v2/inhouse/R_patients.csv", "results_zscore/clinical/cohort_concordance.csv",
    "ez_analysis_mdt/v2/chalfont/R_patients.csv", "external_chalfont/metadata/subjects.csv"))


@pytest.mark.skipif(not _HAVE_MISS_INPUTS, reason="needs the pipeline outputs of both cohorts")
def test_miss_analysis_reproduces_published_counts(monkeypatch):
    import interactive_viewer as iv
    monkeypatch.chdir(iv.BASE_DIR)
    # Chalfont counts = the outputs on disk with the BBR ASL -> T1 matrices (2026-11-18; the polish was rejected) (manuscript and deck slides 35-62 still
    # quote the old-matrix values: rule 58, regional_z 61, regional_z_flow 65, voxel_mass 59, perfz 59, thesis 26,
    # sequential 68; see CLAUDE.md, Chalfont step 1).
    published = {"inhouse": dict(rule=7, regional_li=9, regional_z=8, voxel_mass=8, perfz=6),
                 "chalfont": dict(rule=49, regional_z=60, regional_z_flow=64, voxel_mass=57, perfz=55, thesis=29,
                                  hipp_volume=62, sequential=67)}
    for ds, counts in published.items():
        ma = iv.MissAnalysis(ds)
        u = ma.unilateral
        assert len(u) == (13 if ds == "inhouse" else 93)
        for m, k in counts.items():
            assert int((u[f"{m}_ok"] == 1).sum()) == k, (ds, m)
    # Chalfont: nearly every wrong-side call of the rule lies inside the leave-one-out controls' range (30/30 with the
    # old matrices, 37/39 now); regional z beyond the range: 12 right, 2 wrong (1 before)
    ms = ma.method_summary().set_index("method")
    assert ms.loc["rule", "wrong"] == 39 and ms.loc["rule", "within_wrong"] == 37
    assert ms.loc["regional_z", "beyond_right"] == 12 and ms.loc["regional_z", "beyond_wrong"] == 2


@pytest.mark.skipif(not _HAVE_MISS_INPUTS, reason="needs the in-house pipeline outputs")
def test_inhouse_rule_miss_reasons(monkeypatch):
    import interactive_viewer as iv
    monkeypatch.chdir(iv.BASE_DIR)
    c = iv.MissAnalysis("inhouse").c
    assert sorted(c.index[c["rule_miss"]]) == ["P015", "P022", "P025", "P027", "P028", "P030"]
    assert c.loc["P025", "f_extratemporal"] and c.loc["P025", "f_others_right"] and c.loc["P025", "f_top_on_ref"]
    assert c.loc["P027", "f_none_right"] and not c.loc["P027", "f_offset"]      # left-lower even without the offset
    assert c.loc["P015", "global_offset_z"] > 1.5 and c.loc["P015", "vote_subtle"] > 0.8
    # only clusters with an extreme deviation (|z| >= 2) voting: P030 turns to the reference side, P025 to no call
    assert c.loc["P030", "f_subtle"] and c.loc["P030", "rule_extreme_only"] == "R"
    assert c.loc["P025", "rule_extreme_only"] == "-" and not c.loc["P015", "f_subtle"]
    # removing the global offset (as single-subject ASL normalizes CBF) reverses P015 and P025 to the reference side
    assert c.loc["P015", "f_offset"] and c.loc["P015", "rule_offset_removed"] == "L"
    assert c.loc["P025", "rule_offset_removed"] == "L" and c.loc["P023", "rule_offset_removed"] == "L"
    assert c.loc["P028", "rule"] == "-" and c.loc["P028", "f_weak"]
    assert c.loc["P023", "regional_li"] == "R" and c.loc["P023", "regional_li_on_cut"]   # score equals the fitted cut
    assert not c.loc[["P013", "P016", "P017", "P023", "P024", "P026", "P029"], "rule_miss"].any()


@pytest.mark.skipif(not _HAVE_MISS_INPUTS, reason="needs the in-house pipeline outputs")
def test_bilateral_unclear_kept_apart(monkeypatch):
    """The patients the MDT gave no side get their own table: never scored, every method's call, agreement, control range."""
    import interactive_viewer as iv
    monkeypatch.chdir(iv.BASE_DIR)
    ma = iv.MissAnalysis("inhouse")
    b = ma.bilateral_unclear()
    assert sorted(b.index) == ["P018", "P019", "P020", "P021"] and b["ref"].isin(["B", "U"]).all()
    assert set(b.index).isdisjoint(ma.unilateral.index) and set(b.index).isdisjoint(ma.method_summary()["missed"].str.split().sum())
    assert b.loc["P019", "blind_agree"].startswith("all R")              # every blind method on the stereo-EEG majority side
    assert b.loc["P018", "rule"] == "L" and b.loc["P018", "perfz"] == "R"  # split: rule with the MRI / HD-EEG, not the stereo-EEG
    assert b.loc["P018", "sources_R"] == "interictal EEG ictal EEG invasive" and "MRI" in b.loc["P018", "sources_L"]
    # the asymmetry is no larger than healthy controls produce, as a cancelling bilateral abnormality predicts
    assert b.loc["P020", "inside_range"] == "2/3" and b.loc[["P018", "P019", "P021"], "inside_range"].eq("3/3").all()
    assert (b["regional_z_cal"] == "-").sum() == 3 and b.loc["P020", "regional_z_cal"] == "R"


def test_chalfont_step1_is_brain_to_brain():
    """2026-11-18: bet on the PD reference kept the whole head, so ASL -> T1 was head-to-brain normmi (D042 8.5 deg
    off). The script now refines the normmi initialisation with BBR on the GIF white matter and guards divergence."""
    s = (ROOT / "scripts/chalfont/reg_chalfont_subject.sh").read_text()
    cmds = [ln for ln in s.splitlines() if not ln.lstrip().startswith("#")]
    assert not any(ln.lstrip().startswith("bet ") for ln in cmds)
    assert any("-cost bbr" in ln and "-wmseg" in ln for ln in cmds)
    joined = " ".join(ln.strip() for ln in cmds)
    assert 'chalfont_io.py" step1pick' in joined and '"$ST" 15 8' in joined   # guard: 15 deg / 8 mm + fit not worse
    assert "_asl2t1_init.mat" in s and 'MODE="${2:-full}"' in s          # provenance + step1-only re-run
    assert "step1polish" not in joined   # the polish was rejected on independent checks (2026-11-19); BBR is final
    import importlib.util
    spec = importlib.util.spec_from_file_location("cio", ROOT / "scripts/chalfont/chalfont_io.py")
    cio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cio)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        a = np.eye(4)
        th = np.radians(10.0)
        b = np.array([[1, 0, 0, 0], [0, np.cos(th), -np.sin(th), 0], [0, np.sin(th), np.cos(th), 0], [0, 0, 0, 1.0]])
        np.savetxt(f"{td}/a.mat", a)
        np.savetxt(f"{td}/b.mat", b)
        img = nib.Nifti1Image(np.zeros((128, 128, 40), np.float32), np.diag([1.875, 1.875, 4.0, 1.0]))
        nib.save(img, f"{td}/mov.nii.gz")
        rot, mm = cio.rigid_gap(f"{td}/a.mat", f"{td}/b.mat", f"{td}/mov.nii.gz")
        assert abs(rot - 10.0) < 1e-6
        centre = np.array([120.0, 120.0, 80.0, 1.0])                      # dims x pixdims / 2
        assert abs(mm - np.linalg.norm(((b - a) @ centre)[:3])) < 1e-6
    # the polish perturbs about the ASL centre: a pure rotation leaves the centre where it is, a translation moves it
    c = np.array([120.0, 120.0, 80.0])
    R = cio._rigid_about(c, rx=3.0)
    assert np.allclose((R @ np.r_[c, 1.0])[:3], c) and abs(np.degrees(np.arccos((np.trace(R[:3, :3]) - 1) / 2)) - 3.0) < 1e-9
    T = cio._rigid_about(c, tz=1.0)
    assert np.allclose((T @ np.r_[c, 1.0])[:3], c + [0, 0, 1.0])


def test_per_index_table_ranked_by_the_weaker_side():
    """Table S2: ranked by min(sensitivity, specificity), so an index that works for one side only cannot lead."""
    import pandas as pd
    df = pd.read_csv(ROOT / "ez_analysis_mdt/per_index_performance_exclude.csv")
    assert (df["min_sens_spec"] - df[["sensitivity", "specificity"]].min(axis=1)).abs().max() < 1e-9
    assert (df["min_sens_spec"].diff().dropna() <= 1e-9).all()
    top = df.iloc[0]
    assert top["index"] == "glass_delta_asymmetry" and (top["left_correct"], top["right_correct"]) == (6, 3)
    cvr = df.set_index("index").loc["cv_ratio"]
    assert cvr["auc"] == df["auc"].max() and cvr["min_sens_spec"] == 0.4      # highest AUC, one-sided operating point


@pytest.mark.skipif(not all((Path(__file__).resolve().parents[1] / f).exists() for f in (
    "Dataset/reference_sources.csv", "Dataset/reference_sources_quotes.csv", "manuscript/Thesis.pdf",
    "manuscript/Manuscript_Complete.tex")) or __import__("shutil").which("pdftotext") is None,
                    reason="needs the private source coding, the thesis PDF and pdftotext")
def test_inhouse_sources_verified_against_thesis():
    import interactive_viewer as iv
    rows = iv.thesis_rows()
    assert all(sorted(rows[t]) == list(range(1, 18)) for t in (11, 12, 13, 14))   # one row per thesis patient
    v = iv.verify_inhouse_sources()
    bad = v[~v["ok"]]
    assert bad.empty, bad[["kind", "id", "column", "code", "quote"]].to_string()
    assert (v["kind"] == "tier").sum() == 17 and (v["kind"] == "quote").sum() >= 140
