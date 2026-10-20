#!/usr/bin/env python3
"""Dataset switch for the Pipeline-B scripts: in-house cohort vs external validation.

``configure(dataset)`` returns a :class:`DatasetConfig` that the pipeline scripts
(01, 02, 03, 06, calibration, stratified analysis) apply to their module globals
in ``main()``. The ``inhouse`` configuration reproduces the hard-coded defaults
of those scripts exactly (tests/test_dataset_config.py asserts this), so the
default behaviour and outputs are unchanged.

``chalfont`` (external validation, GE 3T eASL CBF):
  * all subjects live flat in ``external_chalfont/sym_perf/<ID>_perf_sym.nii.gz``
    (symmetric-template space, same grid as ``symreg/sym_perf``); subjects are
    discovered from ID lists, never from directories;
  * control bands come from ``external_chalfont/metadata/groups.json``
    (``{"CH_ALL": [ids], "CH_20_39": [ids], ..., "age_summary": {...}}``);
  * demographics and the EZ reference standard come from
    ``external_chalfont/metadata/subjects.csv`` (columns ``id, group, sex,
    age_at_scan, ez_side, ez_lobe, pathology, ...``);
  * there is no per-subject FreeSurfer parcellation: ONE fixed template-space
    atlas (``external_chalfont/atlas/consensus_parcellation_sym.nii.gz``, built by
    ``scripts/chalfont/build_template_atlas.py``) is used as the consensus
    parcellation of every band;
  * ``--perf-variant flow`` switches to the plain (non transit-corrected) CBF
    maps ``<ID>_perfflow_sym.nii.gz`` with outputs in the sibling directory
    ``external_chalfont/results_zscore_flow``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent

DATASETS = ("inhouse", "chalfont")
PERF_VARIANTS = {"tc": "_perf_sym", "flow": "_perfflow_sym"}  # suffix of <ID><suffix>.nii.gz
CHALFONT_ROOT = BASE_DIR / "external_chalfont"
CHALFONT_PATIENT_GROUPS = ("discrete", "negative")
CHALFONT_CONTROL_GROUPS = ("control", "controls")
VALID_EZ = ("L", "R", "B", "U")
# External bands with fewer registered controls are not used (CH_60_79, n=5, is skipped).
# CHALFONT_MIN_CONTROLS_PER_BAND overrides it (tests / small fixtures).
MIN_CONTROLS_PER_BAND = int(os.environ.get("CHALFONT_MIN_CONTROLS_PER_BAND", "10"))
# Band policy when groups.json holds BOTH an all-ages band (CH_ALL) and age bands:
#   'all' (default, PRIMARY analysis): every subject is compared with the pooled
#                                      all-ages band (CH_ALL, n=66 after QC).
#   'specific' (sensitivity analysis): a subject is assigned to the age band
#                                      covering its age; the all-ages band only
#                                      takes subjects no usable age band covers.
#                                      Outputs go to results_zscore[_flow]_bands/.
# Override with --band-policy or the CHALFONT_BAND_POLICY environment variable.
BAND_POLICIES = ("all", "specific")
# QC exclusions (external_chalfont/metadata/qc_exclusions.json, pre-specified rule in
# scripts/chalfont/asl_quality.py):
#   controls_excluded / duplicate_person_excluded -> never in a normative group,
#   patients_excluded_technical                   -> dropped from every patient analysis,
#   patients_flagged_lr_asymmetric, registration_qc_flagged
#                                                 -> kept in the PRIMARY cohort; the
#                                                    evaluation repeats the accuracy
#                                                    without them (sensitivity cohorts).
# qc/registration_qc.csv flags are reported only.
QC_SENSITIVITY_KEYS = ("patients_flagged_lr_asymmetric", "registration_qc_flagged")


def _age_range_from_key(key: str) -> tuple[int, int]:
    """``CH_20_39`` -> (20, 39); ``CH_ALL`` (or any non-numeric key) -> (0, 200)."""
    parts = key.split("_")
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return int(parts[-2]), int(parts[-1])
    return 0, 200


def _norm_sex(s) -> str:
    s = "" if s is None else str(s).strip().upper()
    if s.startswith("F"):
        return "F"
    if s.startswith("M"):
        return "M"
    return ""


def _norm_ez(s) -> Optional[str]:
    s = "" if s is None else str(s).strip().upper()
    if s in VALID_EZ:
        return s
    if s.startswith("LEFT"):
        return "L"
    if s.startswith("RIGHT"):
        return "R"
    if s.startswith("BI"):
        return "B"
    if s.startswith("UN"):
        return "U"
    return None  # NA / missing: no reference standard


def _to_int(v) -> Optional[int]:
    """Age in completed years (floor of a decimal age, e.g. 39.6 -> 39)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return int(math.floor(f))


@dataclass
class DatasetConfig:
    name: str
    base_dir: Path
    controls_base_dir: Path
    patient_dataset_dir: Path
    output_dir: Path                 # results root (results_zscore for in-house)
    sym_dir: Path                    # <ID><perf_suffix>.nii.gz, <ID>_aparc_sym.nii.gz
    sym_mode: Optional[bool]         # None = decided by --sym (in-house); True = forced
    perf_variant: str = "tc"
    perf_suffix: str = "_perf_sym"
    parcellation: str = "per_subject"   # 'per_subject' | 'fixed_atlas'
    atlas_path: Optional[Path] = None
    demographics_path: Optional[Path] = None   # clinical_spreadsheet.xlsx | subjects.csv
    demographics_kind: str = "clinical_spreadsheet"  # | 'subjects_csv'
    groups_json: Optional[Path] = None
    group_definitions: Optional[dict] = None    # None = keep the script's own table
    patient_prefixes: tuple = ("P",)
    control_prefixes: tuple = ("HCA", "HCD")
    analysis_dir: Path = field(default_factory=lambda: BASE_DIR / "ez_analysis_mdt")
    band_policy: str = "all"
    qc_csv: Optional[Path] = None
    qc_json: Optional[Path] = None
    _subjects: Optional[list] = field(default=None, repr=False)

    # ------------------------------------------------------------------ paths
    @property
    def is_inhouse(self) -> bool:
        return self.name == "inhouse"

    @property
    def cluster_report_dir(self) -> Path:
        return self.output_dir / "asymmetry" / "patients"

    @property
    def pseudo_control_dir(self) -> Path:
        """02_compute_zai.py --pseudo-controls output (controls run as patients)."""
        return self.output_dir / "asymmetry" / "pseudo_controls"

    @property
    def groups_dir(self) -> Path:
        return self.output_dir / "groups"

    @property
    def clinical_dir(self) -> Path:
        return self.output_dir / "clinical"

    @property
    def calibration_dir(self) -> Path:
        return self.output_dir / "calibration"

    def perf_path(self, subject_id: str) -> Path:
        return self.sym_dir / f"{subject_id}{self.perf_suffix}.nii.gz"

    def subject_parc_path(self, subject_id: str) -> Path:
        if self.parcellation == "fixed_atlas":
            return self.atlas_path
        return self.sym_dir / f"{subject_id}_aparc_sym.nii.gz"

    # --------------------------------------------------------------- metadata
    def load_subjects(self) -> list[dict]:
        """Rows of subjects.csv (chalfont) with normalised fields; cached."""
        if self._subjects is not None:
            return self._subjects
        if self.demographics_kind != "subjects_csv":
            raise RuntimeError(f"{self.name}: no subjects.csv; use the clinical spreadsheet")
        if self.demographics_path is None or not self.demographics_path.exists():
            raise FileNotFoundError(f"subjects.csv not found: {self.demographics_path}")
        import csv
        rows = []
        with open(self.demographics_path, newline="") as f:
            reader = csv.DictReader(f)
            cols = {c.lower().strip(): c for c in reader.fieldnames or []}
            id_col = cols.get("id") or cols.get("subject_id") or cols.get("subject")
            if id_col is None:
                raise ValueError(f"subjects.csv has no 'id' column: {reader.fieldnames}")
            for raw in reader:
                g = lambda k: raw.get(cols[k]) if k in cols else None  # noqa: E731
                sid = str(raw[id_col]).strip()
                if not sid:
                    continue
                group = (g("group") or "").strip().lower()
                rows.append({
                    "id": sid,
                    "group": group,
                    "is_control": group.startswith(CHALFONT_CONTROL_GROUPS)
                                  or (not group and sid.startswith(self.control_prefixes)),
                    "sex": _norm_sex(g("sex")),
                    "age": _to_int(g("age_at_scan") if "age_at_scan" in cols else g("age")),
                    "ez": _norm_ez(g("ez_side")),
                    "ez_raw": (g("ez_side") or "").strip(),
                    "ez_lobe": (g("ez_lobe") or "").strip().lower(),
                    "pathology": (g("pathology") or "").strip(),
                    "post_surgical": str(g("post_surgical") or "").strip().lower()
                                     in ("1", "true", "yes", "y"),
                    "raw": raw,
                })
        self._subjects = rows
        return rows

    def demographics(self) -> dict:
        """``{id: {"age": int|None, "sex": "F"|"M"|""}}`` for every subject (chalfont)."""
        return {r["id"]: {"age": r["age"], "sex": r["sex"]} for r in self.load_subjects()}

    def patient_ids(self, require_perf: bool = True) -> list[str]:
        """Discrete + negative patients (chalfont) that have a registered perfusion map."""
        if self.is_inhouse:
            from ez_ground_truth import EZ_GROUND_TRUTH
            ids = sorted(EZ_GROUND_TRUTH)
        else:
            excluded = self.qc_excluded()
            ids = sorted(r["id"] for r in self.load_subjects()
                         if not r["is_control"] and r["id"] not in excluded)
        if require_perf:
            ids = [i for i in ids if self.perf_path(i).exists()]
        return ids

    def control_ids(self, group_key: str, require_perf: bool = True) -> list[str]:
        gdef = self._gdef(group_key)
        if gdef.get("subject_ids") is not None:
            excluded = self.qc_excluded()
            ids = [i for i in gdef["subject_ids"] if i not in excluded]
        else:  # in-house: one directory per control under the band subdir(s)
            ids = []
            for sd in gdef["subdirs"]:
                d = self.controls_base_dir / sd
                if d.exists():
                    ids.extend(sorted(e.name for e in d.iterdir() if e.is_dir()))
        if require_perf and (self.sym_mode or not self.is_inhouse):
            ids = [i for i in ids if self.perf_path(i).exists()]
        return ids

    def _gdef(self, group_key: str) -> dict:
        gdefs = self.group_definitions
        if gdefs is None:  # in-house scripts own their table; mirror 01's for lookups
            gdefs = INHOUSE_GROUP_DEFINITIONS
        if group_key not in gdefs:
            raise KeyError(f"Unknown group '{group_key}'. Available: {list(gdefs)}")
        return gdefs[group_key]

    def available_groups(self) -> list[str]:
        gdefs = self.group_definitions if self.group_definitions is not None else INHOUSE_GROUP_DEFINITIONS
        out = []
        for key in gdefs:
            try:
                ids = self.control_ids(key, require_perf=not self.is_inhouse)
            except FileNotFoundError:
                continue
            if ids and (self.is_inhouse or len(ids) >= MIN_CONTROLS_PER_BAND):
                out.append(key)
        return out

    def _is_all_ages(self, group_key: str) -> bool:
        return tuple(self._gdef(group_key)["age_range"]) == (0, 200)

    def bands_for_age(self, age: Optional[int]) -> list[str]:
        """Available bands a subject of this age is assigned to (sex-pooled).

        Age bands take priority over an all-ages band (band_policy 'specific');
        the all-ages band is the fallback for ages no band covers and for
        unknown ages. With band_policy 'all' only all-ages bands are used.
        """
        avail = self.available_groups()
        all_ages = [g for g in avail if self._is_all_ages(g)]
        if self.band_policy == "all":
            return all_ages if all_ages else avail
        if age is not None:
            specific = [g for g in avail if not self._is_all_ages(g)
                        and self._gdef(g)["age_range"][0] <= age <= self._gdef(g)["age_range"][1]]
            if specific:
                return specific
        return all_ages if all_ages else avail

    def patients_for_group(self, group_key: str) -> list[str]:
        """Patients (with perfusion) assigned to this band (see bands_for_age)."""
        demos = self.demographics()
        return sorted(p for p in self.patient_ids()
                      if group_key in self.bands_for_age(demos.get(p, {}).get("age")))

    def groups_for_patient(self, patient_id: str) -> list[str]:
        return self.bands_for_age(self.demographics().get(patient_id, {}).get("age"))

    def pseudo_control_ids(self, group_key: str) -> list[str]:
        """Controls of this band to score as pseudo-patients: those the band
        assignment rule would send to this band (so each control is scored once)."""
        if self.is_inhouse:
            return self.control_ids(group_key)
        demos = self.demographics()
        return [c for c in self.control_ids(group_key)
                if group_key in self.bands_for_age(demos.get(c, {}).get("age"))]

    def ground_truth(self) -> dict:
        """``{id: {"ez": L|R|B|U|None, "confidence", "evidence", ...}}``.

        In-house: ``ez_ground_truth.EZ_GROUND_TRUTH``. Chalfont: built from
        subjects.csv ``ez_side`` (NA -> ``ez=None``: kept in the voxel-wise
        steps, excluded from lateralization accuracy).
        """
        if self.is_inhouse:
            from ez_ground_truth import EZ_GROUND_TRUTH
            return EZ_GROUND_TRUTH
        gt = {}
        for r in self.load_subjects():
            if r["is_control"]:
                continue
            gt[r["id"]] = {
                "ez": r["ez"],
                "confidence": None,
                "evidence": "; ".join(x for x in (
                    f"group={r['group']}" if r["group"] else "",
                    f"ez_lobe={r['ez_lobe']}" if r["ez_lobe"] else "",
                    f"pathology={r['pathology']}" if r["pathology"] else "",
                    f"ez_side={r['ez_raw']}" if r["ez"] is None and r["ez_raw"] else "") if x),
                "group": r["group"],
                "ez_lobe": r["ez_lobe"],
                "pathology": r["pathology"],
            }
        return gt

    @property
    def discarded(self) -> dict:
        if self.is_inhouse:
            from ez_ground_truth import DISCARDED
            return DISCARDED
        return {}

    @property
    def post_surgical(self) -> set:
        if self.is_inhouse:
            return {"P024"}
        try:
            return {r["id"] for r in self.load_subjects() if r["post_surgical"]}
        except FileNotFoundError:
            return set()

    def qc_flags(self) -> dict:
        """``{id: flag_string}`` for subjects with a non-empty registration-QC flag."""
        if self.qc_csv is None or not self.qc_csv.exists():
            return {}
        import csv
        out = {}
        with open(self.qc_csv, newline="") as f:
            for row in csv.DictReader(f):
                flag = (row.get("flag") or "").strip()
                if flag:
                    out[str(row.get("id", "")).strip()] = flag
        return out

    def qc_exclusions(self) -> dict:
        """Contents of qc_exclusions.json (lists of IDs per key); {} when absent."""
        if self.qc_json is None or not self.qc_json.exists():
            return {}
        with open(self.qc_json) as f:
            raw = json.load(f)
        return {k: [str(x) for x in v] for k, v in raw.items() if isinstance(v, list)}

    def qc_excluded(self) -> set:
        """Subjects removed from EVERY analysis: failed controls, duplicate persons,
        technically failed patients."""
        return _qc_excluded_ids(self.qc_json)

    def qc_sensitivity_sets(self) -> dict:
        """``{cohort_name: ids_to_drop}`` for the sensitivity analyses (flagged but
        kept in the primary cohort)."""
        ex = self.qc_exclusions()
        sets = {}
        for key in QC_SENSITIVITY_KEYS:
            ids = set(ex.get(key, [])) - self.qc_excluded()
            if ids:
                sets[f"without_{key.replace('patients_flagged_', '')}"] = ids
        if len(sets) > 1:
            sets["without_any_qc_flag"] = set().union(*sets.values())
        return sets

    def summary(self) -> str:
        lines = [f"dataset={self.name} perf_variant={self.perf_variant} suffix={self.perf_suffix} "
                 f"band_policy={self.band_policy}",
                 f"  sym_dir={self.sym_dir}", f"  output_dir={self.output_dir}",
                 f"  parcellation={self.parcellation} atlas={self.atlas_path}",
                 f"  demographics={self.demographics_path}"]
        if self.group_definitions is not None:
            for k, g in self.group_definitions.items():
                n = len(g.get("subject_ids") or [])
                n_perf = len(self.control_ids(k))
                used = "used" if k in self.available_groups() else f"UNUSED (<{MIN_CONTROLS_PER_BAND})"
                lines.append(f"  group {k}: ages {g['age_range']}  listed {n}  with perfusion {n_perf}  {used}")
            try:
                pats = self.patient_ids()
                gt = self.ground_truth()
                from collections import Counter
                c = Counter(gt[p]["ez"] for p in pats if p in gt)
                lines.append(f"  patients with perfusion: {len(pats)}  ez: {dict(c)}")
                lines.append(f"  QC excluded everywhere: {sorted(self.qc_excluded())}")
                for k, v in self.qc_sensitivity_sets().items():
                    lines.append(f"  sensitivity cohort {k}: drop {sorted(v)}")
                lines.append(f"  registration-QC flags (reported): {sorted(self.qc_flags())}")
                for k in self.available_groups():
                    lines.append(f"  band {k}: {len(self.patients_for_group(k))} patients, "
                                 f"{len(self.pseudo_control_ids(k))} pseudo-controls")
            except FileNotFoundError as e:
                lines.append(f"  patients: (metadata missing: {e})")
        return "\n".join(lines)


# In-house control bands (mirrors 01_build_control_normative.py GROUP_DEFINITIONS;
# the scripts keep their own copy so the default behaviour is untouched).
INHOUSE_GROUP_DEFINITIONS = {
    "F_20_39": {"subdirs": ["20_29F", "30_39F"], "sex": "F", "age_range": (20, 39)},
    "F_40_59": {"subdirs": ["40_49F", "50_59F"], "sex": "F", "age_range": (40, 59)},
    "F_60_79": {"subdirs": ["60_69F", "70_79F"], "sex": "F", "age_range": (60, 79)},
    "M_20_39": {"subdirs": ["20_29M", "30_39M"], "sex": "M", "age_range": (20, 39)},
    "M_40_59": {"subdirs": ["40_49M", "50_59M"], "sex": "M", "age_range": (40, 59)},
    "M_60_79": {"subdirs": ["60_69M", "70_79M"], "sex": "M", "age_range": (60, 79)},
    "FM_20_39": {"subdirs": ["FM_20_39"], "sex": "FM", "age_range": (20, 39)},
    "FM_40_59": {"subdirs": ["FM_40_59"], "sex": "FM", "age_range": (40, 59)},
    "FM_60_79": {"subdirs": ["FM_60_79"], "sex": "FM", "age_range": (60, 79)},
}


BAND_KEY_RE = re.compile(r"^CH_(ALL|\d+_\d+)$")


QC_EXCLUDED_KEYS = ("controls_excluded", "duplicate_person_excluded", "patients_excluded_technical")


def _qc_excluded_ids(qc_json: Optional[Path]) -> set:
    """IDs removed from every analysis according to qc_exclusions.json ({} when absent)."""
    if qc_json is None or not Path(qc_json).exists():
        return set()
    with open(qc_json) as f:
        raw = json.load(f)
    return {str(x) for k in QC_EXCLUDED_KEYS for x in raw.get(k, []) if isinstance(raw.get(k), list)}


def load_chalfont_groups(groups_json: Path, excluded=()) -> dict:
    """Build GROUP_DEFINITIONS from groups.json.

    Bands are the keys ``CH_ALL`` and ``CH_<lo>_<hi>`` whose value is an ID
    list; everything else (``age_summary``, ``_notes``, ``CH_AGE_UNKNOWN`` ...)
    is ignored. Controls of unknown age are expected in ``CH_ALL``. IDs in
    ``excluded`` (QC failures, duplicate persons) are dropped from every band
    here, so that 01/02 ``discover_subjects`` and ``control_ids()`` agree even
    when groups.json still lists them (audit finding 5, 2026-10-19).
    """
    if not groups_json.exists():
        raise FileNotFoundError(f"groups.json not found: {groups_json}")
    with open(groups_json) as f:
        raw = json.load(f)
    excluded = set(excluded)
    gdefs = {}
    for key, val in raw.items():
        if not isinstance(val, list) or not BAND_KEY_RE.match(key):
            continue  # e.g. "age_summary", "_notes", "CH_AGE_UNKNOWN"
        lo, hi = _age_range_from_key(key)
        ids = [str(v) for v in val if str(v) not in excluded]
        gdefs[key] = {
            "subdirs": [],
            "sex": "FM",
            "age_range": (lo, hi),
            "description": (f"Chalfont controls, all ages (n={len(ids)})" if (lo, hi) == (0, 200)
                            else f"Chalfont controls aged {lo}-{hi} (n={len(ids)})"),
            "subject_ids": ids,
        }
    if not gdefs:
        raise ValueError(f"{groups_json}: no control ID lists found")
    return gdefs


def configure(dataset: str = "inhouse", perf_variant: str = "tc",
              chalfont_root: Optional[Path] = None, base_dir: Optional[Path] = None,
              band_policy: Optional[str] = None) -> DatasetConfig:
    """Return the configuration for ``dataset`` ('inhouse' | 'chalfont')."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'; expected one of {DATASETS}")
    if perf_variant not in PERF_VARIANTS:
        raise ValueError(f"Unknown perf variant '{perf_variant}'; expected one of {list(PERF_VARIANTS)}")
    band_policy = band_policy or os.environ.get("CHALFONT_BAND_POLICY") or "all"
    if band_policy not in BAND_POLICIES:
        raise ValueError(f"Unknown band policy '{band_policy}'; expected one of {BAND_POLICIES}")
    base = Path(base_dir) if base_dir else BASE_DIR

    if dataset == "inhouse":
        if perf_variant != "tc":
            raise ValueError("--perf-variant applies to the chalfont dataset only")
        return DatasetConfig(
            name="inhouse", base_dir=base,
            controls_base_dir=base / "DatasetControls",
            patient_dataset_dir=base / "Dataset",
            output_dir=base / "results_zscore",
            sym_dir=base / "symreg" / "sym_perf",
            sym_mode=None,
            demographics_path=base / "clinical_spreadsheet.xlsx",
            demographics_kind="clinical_spreadsheet",
            group_definitions=None,
            analysis_dir=base / "ez_analysis_mdt",
        )

    # CHALFONT_ROOT env var lets the driver script / tests target another root
    # (e.g. a synthetic fixture) without touching the real external_chalfont/.
    root = Path(chalfont_root or os.environ.get("CHALFONT_ROOT") or CHALFONT_ROOT)
    out_name = "results_zscore" + ("" if perf_variant == "tc" else f"_{perf_variant}") \
        + ("" if band_policy == "all" else "_bands")
    out = root / out_name
    groups_json = root / "metadata" / "groups.json"
    qc_json = root / "metadata" / "qc_exclusions.json"
    gdefs = load_chalfont_groups(groups_json, excluded=_qc_excluded_ids(qc_json)) \
        if groups_json.exists() else None
    return DatasetConfig(
        name="chalfont", base_dir=base,
        controls_base_dir=root / "sym_perf",
        patient_dataset_dir=root / "sym_perf",
        output_dir=out,
        sym_dir=root / "sym_perf",
        sym_mode=True,
        perf_variant=perf_variant,
        perf_suffix=PERF_VARIANTS[perf_variant],
        parcellation="fixed_atlas",
        atlas_path=root / "atlas" / "consensus_parcellation_sym.nii.gz",
        demographics_path=root / "metadata" / "subjects.csv",
        demographics_kind="subjects_csv",
        groups_json=groups_json,
        group_definitions=gdefs,
        patient_prefixes=("D", "N"),
        control_prefixes=("C",),
        analysis_dir=out / "lateralization",
        band_policy=band_policy,
        qc_csv=root / "qc" / "registration_qc.csv",
        qc_json=qc_json,
    )


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--dataset`` / ``--perf-variant`` to a script's argparse parser."""
    parser.add_argument("--dataset", choices=DATASETS, default="inhouse",
                        help="Which cohort to run on (default: inhouse = unchanged behaviour). "
                             "'chalfont' = external validation under external_chalfont/.")
    parser.add_argument("--perf-variant", choices=list(PERF_VARIANTS), default="tc",
                        help="chalfont only: 'tc' = transit-time-corrected CBF (<ID>_perf_sym, "
                             "primary), 'flow' = plain CBF (<ID>_perfflow_sym; outputs in "
                             "results_zscore_flow).")
    parser.add_argument("--band-policy", choices=list(BAND_POLICIES), default=None,
                        help="chalfont only: 'all' = pooled CH_ALL controls for every patient "
                             "(primary, default); 'specific' = age-band sensitivity analysis "
                             "(outputs in results_zscore[_flow]_bands).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect the dataset configuration.")
    ap.add_argument("dataset", nargs="?", choices=DATASETS, default="inhouse")
    ap.add_argument("--perf-variant", choices=list(PERF_VARIANTS), default="tc")
    ap.add_argument("--band-policy", choices=list(BAND_POLICIES), default=None)
    ap.add_argument("--list-groups", action="store_true", help="print available group keys, one per line")
    ap.add_argument("--list-run-bands", action="store_true",
                    help="print the bands that receive patients or pseudo-controls under the band policy")
    ap.add_argument("--list-patients", action="store_true", help="print patient IDs with perfusion")
    ap.add_argument("--patients-for-group", metavar="BAND", help="print the patients assigned to BAND")
    ap.add_argument("--pseudo-controls-for-group", metavar="BAND", help="print the controls scored in BAND")
    ap.add_argument("--output-dir", action="store_true", help="print the results root")
    args = ap.parse_args(argv)
    cfg = configure(args.dataset, perf_variant=args.perf_variant, band_policy=args.band_policy)
    if args.list_groups:
        print("\n".join(cfg.available_groups()))
    elif args.list_run_bands:
        print("\n".join(b for b in cfg.available_groups()
                        if cfg.patients_for_group(b) or cfg.pseudo_control_ids(b)))
    elif args.list_patients:
        print("\n".join(cfg.patient_ids()))
    elif args.patients_for_group:
        print("\n".join(cfg.patients_for_group(args.patients_for_group)))
    elif args.pseudo_controls_for_group:
        print("\n".join(cfg.pseudo_control_ids(args.pseudo_controls_for_group)))
    elif args.output_dir:
        print(cfg.output_dir)
    else:
        print(cfg.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
