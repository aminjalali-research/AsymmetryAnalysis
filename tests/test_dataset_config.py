"""tests/test_dataset_config.py — the --dataset {inhouse,chalfont} switch.

a. configure('inhouse') reproduces the hard-coded defaults of 01/02/03/06 and
   applying it leaves their module globals unchanged.
b. configure('chalfont') with a temporary root: groups from groups.json,
   ID-list subject discovery, age-band matching, ground truth from
   subjects.csv (NA -> None), fixed-atlas parcellation, perf variant.
c. Applying the chalfont config to the scripts' globals (SYM forced, paths).
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dataset_config  # noqa: E402


@pytest.fixture(autouse=True)
def _small_bands(monkeypatch):
    """The temp fixture has 5 controls; allow bands of >= 3."""
    monkeypatch.setattr(dataset_config, "MIN_CONTROLS_PER_BAND", 3)


def _load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# a. in-house defaults
# ---------------------------------------------------------------------------

def test_inhouse_config_matches_script_defaults():
    cfg = dataset_config.configure("inhouse")
    m01 = _load_script("cfg_01", "01_build_control_normative.py")
    m02 = _load_script("cfg_02", "02_compute_zai.py")
    m03 = _load_script("cfg_03", "03_clinical_maps.py")
    m06 = _load_script("cfg_06", "06_clinical_interpretation.py")
    assert cfg.is_inhouse and cfg.sym_mode is None and cfg.group_definitions is None
    assert cfg.perf_suffix == "_perf_sym"
    assert cfg.controls_base_dir == m01.CONTROLS_BASE_DIR.resolve()
    assert cfg.patient_dataset_dir == m01.PATIENT_DATASET_DIR.resolve()
    assert cfg.output_dir == m01.OUTPUT_DIR.resolve() == m03.RESULTS_DIR.resolve()
    assert cfg.output_dir / "asymmetry" == m02.OUTPUT_DIR.resolve()
    assert cfg.sym_dir == m01.SYM_DIR.resolve() == m02.SYM_DIR.resolve() == m03.SYM_DIR.resolve()
    assert cfg.demographics_path == m01.CLINICAL_SPREADSHEET.resolve()
    assert cfg.cluster_report_dir == m06.CLUSTER_REPORT_DIR.resolve()
    assert cfg.clinical_dir == m06.DEFAULT_OUTPUT_ROOT.resolve()
    assert cfg.analysis_dir == ROOT / "ez_analysis_mdt"
    assert cfg.ground_truth() is m06.EZ_GROUND_TRUTH
    assert set(dataset_config.INHOUSE_GROUP_DEFINITIONS) == set(m01.GROUP_DEFINITIONS)
    for k, g in m01.GROUP_DEFINITIONS.items():
        assert dataset_config.INHOUSE_GROUP_DEFINITIONS[k]["subdirs"] == g["subdirs"]
        assert dataset_config.INHOUSE_GROUP_DEFINITIONS[k]["age_range"] == g["age_range"]


def test_apply_inhouse_leaves_globals_unchanged():
    m01 = _load_script("cfg_01b", "01_build_control_normative.py")
    before = {k: m01.__dict__[k] for k in ("CONTROLS_BASE_DIR", "PATIENT_DATASET_DIR", "OUTPUT_DIR",
                                           "SYM_DIR", "SYM_MODE", "PERF_SUFFIX", "GROUP_DEFINITIONS")}
    m01._apply_dataset_config(dataset_config.configure("inhouse"))
    for k, v in before.items():
        assert m01.__dict__[k] == v or Path(m01.__dict__[k]) == Path(v).resolve()
    assert m01.GROUP_DEFINITIONS is before["GROUP_DEFINITIONS"]
    assert m01.SYM_MODE is False


def test_invalid_arguments():
    with pytest.raises(ValueError):
        dataset_config.configure("nonsense")
    with pytest.raises(ValueError):
        dataset_config.configure("inhouse", perf_variant="flow")
    with pytest.raises(ValueError):
        dataset_config.configure("chalfont", perf_variant="bogus")


# ---------------------------------------------------------------------------
# b. chalfont from a temporary root
# ---------------------------------------------------------------------------

@pytest.fixture
def chalfont_root(tmp_path):
    root = tmp_path / "external_chalfont"
    (root / "metadata").mkdir(parents=True)
    (root / "sym_perf").mkdir()
    (root / "atlas").mkdir()
    groups = {"CH_ALL": ["C001", "C002", "C003", "C004", "C005", "C006"],
              "CH_20_39": ["C001", "C002"], "CH_40_59": ["C003"],
              "CH_AGE_UNKNOWN": ["C004"], "_notes": ["free text"],
              "age_summary": {"mean": 40.0}}
    (root / "metadata" / "groups.json").write_text(json.dumps(groups))
    rows = ["id,group,sex,age_at_scan,ez_side,ez_lobe,pathology",
            "C001,control,F,25,NA,,",
            "C002,control,M,30,NA,,",
            "C003,control,F,45,NA,,",
            "C004,control,M,70,NA,,",          # no perfusion file -> excluded
            "C005,control,F,66,NA,,",
            "C006,control,M,38,NA,,",
            "D001,discrete,F,33.7,L,temporal,HS",
            "D002,discrete,M,50,R,frontal,FCD",
            "N001,negative,F,28,NA,,",
            "N002,negative,M,41,B,temporal,",
            "N003,negative,F,,U,,",            # missing age -> CH_ALL only
            "N004,negative,M,60,L,temporal,"]  # no perfusion file -> excluded
    (root / "metadata" / "subjects.csv").write_text("\n".join(rows) + "\n")
    for sid in ("C001", "C002", "C003", "C005", "C006", "D001", "D002", "N001", "N002", "N003"):
        (root / "sym_perf" / f"{sid}_perf_sym.nii.gz").write_bytes(b"")
    for sid in ("C001", "D001"):
        (root / "sym_perf" / f"{sid}_perfflow_sym.nii.gz").write_bytes(b"")
    return root


def test_chalfont_groups_and_subjects(chalfont_root):
    cfg = dataset_config.configure("chalfont", chalfont_root=chalfont_root)
    assert not cfg.is_inhouse and cfg.sym_mode is True
    assert cfg.band_policy == "all"                      # primary: pooled CH_ALL
    assert cfg.output_dir == chalfont_root / "results_zscore"
    assert cfg.qc_csv == chalfont_root / "qc" / "registration_qc.csv"
    assert cfg.qc_json == chalfont_root / "metadata" / "qc_exclusions.json"
    assert cfg.qc_flags() == {} and cfg.qc_excluded() == set() and cfg.qc_sensitivity_sets() == {}
    assert cfg.parcellation == "fixed_atlas"
    assert cfg.atlas_path == chalfont_root / "atlas" / "consensus_parcellation_sym.nii.gz"
    assert cfg.subject_parc_path("D001") == cfg.atlas_path
    assert cfg.perf_path("D001") == chalfont_root / "sym_perf" / "D001_perf_sym.nii.gz"
    assert set(cfg.group_definitions) == {"CH_ALL", "CH_20_39", "CH_40_59"}
    assert cfg.group_definitions["CH_ALL"]["age_range"] == (0, 200)
    assert cfg.group_definitions["CH_20_39"]["age_range"] == (20, 39)
    assert cfg.group_definitions["CH_ALL"]["sex"] == "FM"
    assert cfg.available_groups() == ["CH_ALL"]          # small bands (< MIN_CONTROLS_PER_BAND) unused
    assert cfg.control_ids("CH_ALL") == ["C001", "C002", "C003", "C005", "C006"]  # C004: no perfusion
    assert cfg.control_ids("CH_ALL", require_perf=False) == ["C001", "C002", "C003", "C004", "C005", "C006"]
    assert cfg.patient_ids() == ["D001", "D002", "N001", "N002", "N003"]  # N004 has no perfusion
    assert cfg.demographics()["D001"] == {"age": 33, "sex": "F"}   # 33.7 -> completed years
    assert cfg.demographics()["N003"] == {"age": None, "sex": "F"}


def test_chalfont_band_matching(chalfont_root):
    cfg = dataset_config.configure("chalfont", chalfont_root=chalfont_root)
    assert cfg.patients_for_group("CH_ALL") == ["D001", "D002", "N001", "N002", "N003"]
    assert cfg.patients_for_group("CH_20_39") == []           # band unused (< MIN_CONTROLS_PER_BAND)
    assert cfg.groups_for_patient("D001") == ["CH_ALL"]       # CH_20_39 too small to be used
    assert cfg.groups_for_patient("N003") == ["CH_ALL"]       # unknown age
    dataset_config.MIN_CONTROLS_PER_BAND, saved = 1, dataset_config.MIN_CONTROLS_PER_BAND
    try:
        assert cfg.available_groups() == ["CH_ALL", "CH_20_39", "CH_40_59"]
        assert cfg.groups_for_patient("D001") == ["CH_ALL"]  # default policy 'all'
        cfg = dataset_config.configure("chalfont", chalfont_root=chalfont_root, band_policy="specific")
        assert cfg.output_dir == chalfont_root / "results_zscore_bands"
        # age bands take priority; CH_ALL is the fallback (unknown age / uncovered age)
        assert cfg.groups_for_patient("D001") == ["CH_20_39"]
        assert cfg.groups_for_patient("N003") == ["CH_ALL"]
        assert cfg.patients_for_group("CH_20_39") == ["D001", "N001"]
        assert cfg.patients_for_group("CH_40_59") == ["D002", "N002"]
        assert cfg.patients_for_group("CH_ALL") == ["N003"]
        assert cfg.pseudo_control_ids("CH_20_39") == ["C001", "C002"]
        assert cfg.pseudo_control_ids("CH_ALL") == ["C005"]   # age 66: no age band -> fallback
        cfg_all = dataset_config.configure("chalfont", chalfont_root=chalfont_root, band_policy="all")
        assert cfg_all.groups_for_patient("D001") == ["CH_ALL"]
        assert cfg_all.patients_for_group("CH_20_39") == []
        assert cfg_all.pseudo_control_ids("CH_ALL") == ["C001", "C002", "C003", "C005", "C006"]
        with pytest.raises(ValueError):
            dataset_config.configure("chalfont", chalfont_root=chalfont_root, band_policy="bogus")
    finally:
        dataset_config.MIN_CONTROLS_PER_BAND = saved
    # only CH_ALL present -> every patient maps to it
    (chalfont_root / "metadata" / "groups.json").write_text(
        json.dumps({"CH_ALL": ["C001", "C002", "C003", "C005", "C006"]}))
    cfg2 = dataset_config.configure("chalfont", chalfont_root=chalfont_root)
    assert cfg2.available_groups() == ["CH_ALL"]
    assert cfg2.patients_for_group("CH_ALL") == cfg2.patient_ids()


def test_chalfont_ground_truth(chalfont_root):
    cfg = dataset_config.configure("chalfont", chalfont_root=chalfont_root)
    gt = cfg.ground_truth()
    assert set(gt) == {"D001", "D002", "N001", "N002", "N003", "N004"}
    assert gt["D001"]["ez"] == "L" and gt["D002"]["ez"] == "R"
    assert gt["N001"]["ez"] is None          # NA: kept, but not scoreable
    assert gt["N002"]["ez"] == "B" and gt["N003"]["ez"] == "U"
    assert gt["D001"]["group"] == "discrete" and gt["D001"]["ez_lobe"] == "temporal"
    assert cfg.discarded == {} and cfg.post_surgical == set()
    uni = [p for p in cfg.patient_ids() if gt[p]["ez"] in ("L", "R")]
    assert uni == ["D001", "D002"]


def test_chalfont_perf_variant(chalfont_root):
    cfg = dataset_config.configure("chalfont", perf_variant="flow", chalfont_root=chalfont_root)
    assert cfg.perf_suffix == "_perfflow_sym"
    assert cfg.output_dir == chalfont_root / "results_zscore_flow"
    assert cfg.analysis_dir == chalfont_root / "results_zscore_flow" / "lateralization"
    assert cfg.perf_path("D001").name == "D001_perfflow_sym.nii.gz"
    assert cfg.control_ids("CH_ALL") == ["C001"]
    assert cfg.patient_ids() == ["D001"]


def test_chalfont_qc_exclusions(chalfont_root):
    (chalfont_root / "qc").mkdir()
    (chalfont_root / "qc" / "registration_qc.csv").write_text(
        "id,group,ncc_t1_template,flag\nC001,controls,0.82,\nC002,controls,0.85,lr_asymmetric;low_cbf\n"
        "D001,discrete,0.67,low_ncc\nN001,negative,0.80,low_cbf\n")
    (chalfont_root / "metadata" / "qc_exclusions.json").write_text(json.dumps({
        "rule": "text", "controls_excluded": ["C002"], "patients_excluded_technical": ["N002"],
        "patients_flagged_lr_asymmetric": ["D002"], "registration_qc_flagged": ["C002", "D001", "N001"],
        "duplicate_person_excluded": ["C099"]}))
    cfg = dataset_config.configure("chalfont", chalfont_root=chalfont_root)
    assert cfg.qc_flags() == {"C002": "lr_asymmetric;low_cbf", "D001": "low_ncc", "N001": "low_cbf"}
    assert cfg.qc_excluded() == {"C002", "N002", "C099"}
    assert cfg.control_ids("CH_ALL") == ["C001", "C003", "C005", "C006"]      # C002 out of the normative group
    assert cfg.patient_ids() == ["D001", "D002", "N001", "N003"]              # N002 out of all patient analyses
    assert "N002" not in cfg.patients_for_group("CH_ALL")
    sens = cfg.qc_sensitivity_sets()
    assert sens["without_lr_asymmetric"] == {"D002"}
    assert sens["without_registration_qc_flagged"] == {"D001", "N001"}          # C002 already excluded
    assert sens["without_any_qc_flag"] == {"D001", "D002", "N001"}
    # Audit finding 5 (2026-10-19): the exclusion must reach the group table that
    # 01/02 discover_subjects read, not only control_ids().
    assert "C002" not in cfg.group_definitions["CH_ALL"]["subject_ids"]
    assert "C002" not in cfg.group_definitions["CH_20_39"]["subject_ids"]
    assert "(n=5)" in cfg.group_definitions["CH_ALL"]["description"]
    m01 = _load_script("cfg_01q", "01_build_control_normative.py")
    m01._apply_dataset_config(cfg)
    assert [s.name for s in m01.ControlGroupBuilder("CH_ALL").discover_subjects()] == ["C001", "C003", "C005", "C006"]
    m02 = _load_script("cfg_02q", "02_compute_zai.py")
    m02._apply_dataset_config(cfg)
    assert [s.name for s in m02.discover_subjects("CH_ALL")] == ["C001", "C003", "C005", "C006"]


def test_chalfont_without_metadata(tmp_path):
    """Configuration must resolve before the metadata files exist."""
    cfg = dataset_config.configure("chalfont", chalfont_root=tmp_path / "nope")
    assert cfg.group_definitions is None
    with pytest.raises(FileNotFoundError):
        cfg.patient_ids()


# ---------------------------------------------------------------------------
# c. applying the chalfont config to the scripts
# ---------------------------------------------------------------------------

def test_apply_chalfont_to_scripts(chalfont_root):
    cfg = dataset_config.configure("chalfont", chalfont_root=chalfont_root)
    m01 = _load_script("cfg_01c", "01_build_control_normative.py")
    m01._apply_dataset_config(cfg)
    assert m01.SYM_MODE is True
    assert m01.OUTPUT_DIR == chalfont_root / "results_zscore"
    assert m01.SYM_DIR == chalfont_root / "sym_perf"
    assert set(m01.GROUP_DEFINITIONS) == {"CH_ALL", "CH_20_39", "CH_40_59"}
    assert m01._sym_perf_path("C001").exists() and not m01._sym_perf_path("C004").exists()
    assert m01._patient_available("D001") and not m01._patient_available("N004")
    builder = m01.ControlGroupBuilder("CH_20_39")
    assert [s.name for s in builder.discover_subjects()] == ["C001", "C002"]

    m02 = _load_script("cfg_02c", "02_compute_zai.py")
    m02._apply_dataset_config(cfg)
    assert m02.RESULTS_DIR == chalfont_root / "results_zscore"
    assert m02.OUTPUT_DIR == chalfont_root / "results_zscore" / "asymmetry"
    assert [s.name for s in m02.discover_subjects("CH_ALL")] == ["C001", "C002", "C003", "C005", "C006"]

    m03 = _load_script("cfg_03c", "03_clinical_maps.py")
    m03._apply_dataset_config(cfg)
    assert m03.ATLAS_PATH == cfg.atlas_path
    assert m03.ZAI_PATIENTS_DIR == chalfont_root / "results_zscore" / "asymmetry" / "patients"
    assert m03.GROUP_KEY == "CH_ALL"

    m06 = _load_script("cfg_06c", "06_clinical_interpretation.py")
    m06._apply_dataset_config(cfg)
    assert m06.CLUSTER_REPORT_DIR == chalfont_root / "results_zscore" / "asymmetry" / "patients"
    assert m06.EZ_GROUND_TRUTH["D002"]["ez"] == "R"
    assert m06.ClinicalInterpreter._mdt_concordance("L", None) == "no-mdt-label"
