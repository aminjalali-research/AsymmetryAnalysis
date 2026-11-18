#!/usr/bin/env python3
"""
Interactive Overlay Viewer — zAI canonical (post-2026-05-04 pivot)

Menu-driven launcher for FSLeyes that visualizes the canonical
Pipeline B outputs:

  - Raw anatomy in ORIGINAL space (Dataset/<PID>/; optional legacy
    "Dataset MNI/<PID>/" perfusion if present)
  - Pipeline B Step 1-2: raw voxel z-score maps (results_zscore/patients/)
  - Pipeline B Step 3: voxel zAI maps (results_zscore/asymmetry/patients/)
  - Pipeline B Step 3 (clinical): gray-matter zAI maps (results_zscore/clinical/)

Space: the canonical results_zscore/ maps live in the left-right SYMMETRIC
template space (symreg/). Whenever zAI / clinical maps are displayed, the
underlay T1 is symreg/templates/T1_sym.nii.gz and the parcellation is
symreg/sym_perf/<PID>_aparc_sym.nii.gz (when both exist). Pass --orig to
force the original-space Dataset/<PID>/ T1 + aparc+aseg instead (NOT
voxel-aligned with the symmetric-space zAI maps; for debugging only).
The input views (1-2) and the in-house regional view (3) use original-space anatomy.

Reading zAI: the zAI map is antisymmetric, zAI(v) = -zAI(mirror v).
Red (positive) = this voxel is MORE perfused than its mirror voxel; blue
(negative) = LESS perfused than its mirror. Positive zAI means L>R only for
voxels in the left hemisphere. The predicted EZ side comes from
direction-of-abnormality inference (cluster report columns direction_class /
ez_side_pred: the hemisphere that deviates from controls), not from the sign
of zAI alone.

The viewer does not read results/, results_voxel/ or other legacy result
directories.

Menu: 30 numbered options plus S (switch patient) and 0 (exit), in the order of
the manuscript Methods (and of the deck's procedure slides), so the views can be
shown in the order the methods are presented. The numbers come from MENU below:
  1-2   inputs: T1 + FreeSurfer parcellation, CBF (original space; in-house only)
  3     analysis #1, regional laterality index of the 37 pairs (Pipeline A)
  4-10  left-right symmetric registration: 4 template, 5 inputs, 6 ASL -> T1,
        7 T1 -> T_sym affine, 8 fnirt + mirror check + QC, 9 CBF and labels on
        T_sym, 10 the chain as one PNG
  11-13 analyses #2 and #4: 11 voxel AI of the patient alone, 12 control
        normative (mean and SD of AI), 13 zAI map + calibration
  14-15 thresholding: 14 clusters at |zAI| >= 1.96 (02_), 15 clinical maps (03_)
  16-17 direction of the abnormality: 16 #3 perfusion z (the rule's input),
        17 clusters coloured by direction class
  18-21 lateralization decision: 18 patient report, 19 top-3 clinical-tier
        clusters, 20 37-region view, 21 combined presurgical view
  22-23 post hoc: 22 regional normative z_LI (enhanced estimator i),
        23 #3 hypo/hyper clusters on its own
  24-25 all layers of one subject (CBF, #3, #4, clinical clusters, atlas)
        centred on the strongest clinical cluster: 24 FSLeyes, 25 PNG
  26-27 misdetected cases (Results): 26 every method's misses in the cohort
        (tables, figures, report), 27 this patient's label file, sources, method
        calls and why the primary rule differs (also --misses)
  28-30 display aid and utilities: 28 island-cleaned overlay, 29 file list,
        30 FSLeyes commands

Every FSLeyes view goes through InteractiveOverlayViewer._launch: the
symmetric template is the underlay (control mean CBF as a hidden layer),
signed maps are shown from the stated threshold (|value| below it is
transparent), focal views open on their peak voxel (11-13, 15, 17, 21, 24 and 25
on the strongest clinical-tier cluster of the 06_ report, on its predicted side), missing layers are
reported, the command is printed for copy/paste and FSLeyes start-up errors
are shown instead of discarded. Paths are relative to the project root; the
script changes to it, so it can be started from any directory.

The same all-layer view runs without the menu (dataset taken from the ID):
  python interactive_viewer.py --subject D002               # FSLeyes
  python interactive_viewer.py --subject D002 --png --open  # PNG, opened in VS Code
  python interactive_viewer.py --subject D002 --print       # FSLeyes command only
  python interactive_viewer.py --subject P016 --at 60 150 90
  python interactive_viewer.py --dataset chalfont --all     # PNG for every patient
PNGs: visual_analysis/subject_views/<dataset>/<ID>.png.
"""

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path
import time
import tempfile

try:
    from ez_ground_truth import EZ_GROUND_TRUTH
except Exception:
    EZ_GROUND_TRUTH = {}

# Dataset switch (2026-11-04): `--dataset chalfont` points every results_zscore/
# path at external_chalfont/results_zscore/, takes the reference sides from
# external_chalfont/metadata/subjects.csv (via dataset_config) and uses the
# fixed template atlas as the parcellation underlay (Chalfont has no per-subject
# aparc+aseg). The in-house layout is the default and is unchanged.
RESULTS_ROOT = "results_zscore"
FIXED_ATLAS = None
DATASET = "inhouse"


def configure_dataset(name):
    """Point the viewer at the in-house or the external (Chalfont) outputs."""
    global RESULTS_ROOT, FIXED_ATLAS, EZ_GROUND_TRUTH, DATASET
    DATASET = name
    if name == "chalfont":
        import dataset_config
        cfg = dataset_config.configure("chalfont")
        RESULTS_ROOT = os.path.relpath(cfg.output_dir, Path(__file__).resolve().parent)  # external_chalfont/results_zscore
        FIXED_ATLAS = "external_chalfont/atlas/consensus_parcellation_sym.nii.gz"
        EZ_GROUND_TRUTH = cfg.ground_truth()
    else:
        RESULTS_ROOT = "results_zscore"
        FIXED_ATLAS = None
        from ez_ground_truth import EZ_GROUND_TRUTH as inhouse   # restore after a Chalfont run in the same process
        EZ_GROUND_TRUTH = inhouse

# numpy / nibabel / pandas: needed by every view that reads or writes a map
# (all but the plain FSLeyes launches); the menu itself runs without them.
try:
    import numpy as _np
    import nibabel as _nib
    import pandas as _pd
    _HAVE_SCI = True
except Exception:
    _HAVE_SCI = False


# ---------------------------------------------------------------------------
# Clinical ROI labels (Desikan-Killiany; 34 cortical pairs + 3 subcortical
# pairs = 37 paired regions, mirrored L/R = 76 FreeSurfer label IDs).
#
# Source: CLAUDE.md "34 Cortical + 6 Subcortical Paired Regions" (the 6
# subcortical labels are 3 pairs: thalamus, hippocampus, amygdala). The
# remaining 3 pairs in the CLAUDE.md doc (caudate / putamen / pallidum) are
# basal ganglia and not part of the clinically relevant EZ region set, so they
# are intentionally NOT included here.
# ---------------------------------------------------------------------------
SUBCORTICAL_LABELS = [10, 49, 17, 53, 18, 54]                 # thalamus, hippocampus, amygdala (L+R)
CORTICAL_LABELS = list(range(1001, 1036)) + list(range(2001, 2036))  # 34 cortical L+R
CLINICAL_ROI_LABELS = SUBCORTICAL_LABELS + CORTICAL_LABELS    # 76 labels = 37 ROI pairs
# Desikan-Killiany cortical names by label - 1001 (index 3, label 1004 corpuscallosum, is not a region pair)
DK_CORTICAL_NAMES = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "corpuscallosum", "cuneus", "entorhinal",
    "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate", "lateraloccipital",
    "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine", "postcentral", "posteriorcingulate",
    "precentral", "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole", "temporalpole", "transversetemporal",
    "insula",
]
SUBCORTICAL_NAMES = {10: "thalamus", 17: "hippocampus", 18: "amygdala"}

# Symmetric-template anatomy (Pipeline B canonical space; see symreg/).
SYM_T1 = "symreg/templates/T1_sym.nii.gz"
SYM_APARC_FMT = "symreg/sym_perf/{pid}_aparc_sym.nii.gz"


def _cluster_physical_side(row, mean_col):
    """Physical hemisphere with HIGHER perfusion for a zAI cluster row.

    zAI is antisymmetric, so a positive cluster in the left hemisphere and a
    negative cluster in the right hemisphere both mean L > R. Uses the CSV's
    dominant_side column when present; otherwise derives it from the cluster
    hemisphere (cluster_hemi column, else the "L-"/"R-" primary_region
    prefix, else centroid_x sign in RAS mm).
    """
    ds = row.get("dominant_side") if hasattr(row, "get") else None
    if isinstance(ds, str) and ds.strip().upper() in ("L", "R"):
        return ds.strip().upper()
    hemi = row.get("cluster_hemi") if hasattr(row, "get") else None
    if not (isinstance(hemi, str) and hemi.strip().upper() in ("L", "R")):
        hemi = None
        region = str(row.get("primary_region", ""))
        if region[:2] in ("L-", "R-"):
            hemi = region[0]
        else:
            try:
                hemi = "L" if float(row.get("centroid_x")) < 0 else "R"
            except (TypeError, ValueError):
                return "?"
    hemi = hemi.strip().upper()
    return "L" if (float(row[mean_col]) > 0) == (hemi == "L") else "R"


# ---------------------------------------------------------------------------
# All layers of one subject (menu section 'All layers' and the --subject / --all
# command line). Everything lives on the 0.8-mm symmetric template grid
# (227 x 272 x 227), so FSLeyes and the PNG show the same voxel:
#   CBF            symreg/sym_perf/<ID>_perf_sym | external_chalfont/sym_perf/<ID>_perf_sym (TC-Flow)
#   #3 perfusion z <results>/patients/<ID>/<ID>_vs_<band>_zscore
#   #4 zAI         <results>/asymmetry/patients/<ID>/<ID>_asymmetry_zscore
#   clinical       <results>/clinical/<ID>/<ID>_clinical_zai_lateralized_significant
#   atlas          band consensus parcellation (in-house) | fixed template atlas (Chalfont)
# The dataset is taken from the ID (P0xx in-house, C/D/N Chalfont) unless given.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ALL_LAYER_DATASETS = {
    "inhouse": dict(results="results_zscore", perf="symreg/sym_perf", atlas=None),
    "chalfont": dict(results="external_chalfont/results_zscore", perf="external_chalfont/sym_perf",
                     atlas="external_chalfont/atlas/consensus_parcellation_sym.nii.gz"),
}
# key in AllLayersView.files, PNG row title, display range (|z| below the lower bound is transparent)
ALL_LAYERS = [
    ("cbf", "CBF (ml/100 g/min)", (0, 80)),
    ("perfz", "#3 perfusion z vs controls", (1.96, 6)),
    ("zai", "#4 asymmetry z (zAI)", (1.96, 6)),
    ("clinical", "clinical clusters (|zAI| >= 3)", (3, 8)),
]
ALL_LAYER_PNG_DIR = BASE_DIR / "visual_analysis" / "subject_views"


VIEW_CACHE = Path(tempfile.gettempdir()) / "asym_view"
_MASKS = {}


def brain_masked(path, mask_path):
    """Copy of a map with everything outside the brain (mask_path > 0) set to 0, cached in VIEW_CACHE.

    The #3 perfusion z map, its hypo/hyper cluster maps and the CBF are computed over the scanner field of
    view, which includes the scalp: P016's #3 map has 4.3 M non-zero voxels for a 3.5 M-voxel template brain.
    Outside the brain these values are noise and showed as salt-and-pepper islands around the skull. The
    viewer shows them inside the consensus parcellation only (the brain the analysis used); the files on
    disk are unchanged. zAI and the clinical maps are already inside the brain (0.0 % outside).
    """
    if not _HAVE_SCI or not path or not mask_path or not Path(path).exists() or not Path(mask_path).exists():
        return path
    import hashlib
    src, msk = Path(path).resolve(), Path(mask_path).resolve()
    tag = hashlib.md5(f"{src}|{msk}".encode()).hexdigest()[:10]       # unique per source + mask
    out = VIEW_CACHE / f"{tag}_{src.name}"
    img = _nib.load(str(src))
    if out.exists() and out.stat().st_mtime >= max(src.stat().st_mtime, msk.stat().st_mtime) \
            and _nib.load(str(out)).shape == img.shape:
        return out
    key = str(msk)
    if key not in _MASKS:
        _MASKS[key] = _np.asanyarray(_nib.load(str(msk)).dataobj) > 0
    d = _np.asanyarray(img.dataobj)
    if d.shape != _MASKS[key].shape:
        return path
    out.parent.mkdir(parents=True, exist_ok=True)
    _nib.save(_nib.Nifti1Image(_np.where(_MASKS[key], d, 0).astype(d.dtype), img.affine, img.header), str(out))
    return out


def dataset_of(subject_id):
    return "inhouse" if subject_id.upper().startswith("P") else "chalfont"


_SCRIPTS = {}


def load_script(rel_path):
    """Import a pipeline script by path (02_compute_zai.py etc. cannot be imported by name), once."""
    if rel_path not in _SCRIPTS:
        import importlib.util
        import sys
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        spec = importlib.util.spec_from_file_location("_iv_" + Path(rel_path).stem, str(BASE_DIR / rel_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SCRIPTS[rel_path] = mod
    return _SCRIPTS[rel_path]


def save_view_map(name, data, ref):
    """Write a map computed for display to VIEW_CACHE/<name> (float32, grid of ref) and return its path."""
    out = VIEW_CACHE / name
    out.parent.mkdir(parents=True, exist_ok=True)
    _nib.save(_nib.Nifti1Image(data.astype(_np.float32), ref.affine, ref.header), str(out))
    return out


def voxel_ai(cbf_path, brain_path, tag):
    """Analysis #2: AI(v) = 200 (CBF(v) - CBF(mirror v)) / (CBF(v) + CBF(mirror v)) inside the brain.

    The AI comes from 02_compute_zai.compute_asymmetry_map (mirror = flip of axis 0, zero unless both voxels
    are positive), the function 02_ applies before the control z; 02_ does not save it. Cached per tag.
    """
    out = VIEW_CACHE / f"{tag}_voxel_ai.nii.gz"
    if out.exists() and all(out.stat().st_mtime >= Path(f).stat().st_mtime for f in (cbf_path, brain_path)):
        return out
    img = _nib.load(str(cbf_path))
    ai, _valid = load_script("02_compute_zai.py").compute_asymmetry_map(_np.asanyarray(img.dataobj, dtype=_np.float32))
    brain = _np.asanyarray(_nib.load(str(brain_path)).dataobj) > 0
    brain &= _np.flip(brain, axis=0)            # a mirror-symmetric mask keeps AI(mirror v) = -AI(v)
    return save_view_map(out.name, _np.where(brain, ai, 0), img)


# direction class -> (FSLeyes colour map, matplotlib colour); the order gives the label codes 1-4
DIRECTION_CLASSES = {"hypo": ("blue", "#2b6cff"), "hyper": ("red", "#e8432e"), "mixed": ("green", "#2ca02c"),
                     "subtle": ("yellow", "#e3c000")}


def direction_classes(zai_dir, pid):
    """The 02_ clusters of one subject labelled by direction class: (class volume, other members, image, report, n_mismatch).

    02_ labels the positive and the negative zAI voxels (|zAI| >= 1.96) separately with the 18-connected structure
    of src.thresholding and numbers the clusters of >= 50 voxels 1, 2, ... in label order. Labelling the saved masks
    (<ID>_asymmetry_left/right_dominant) again gives the same numbers; each is checked against the report's
    size_voxels. The class volume holds code 1-4 (DIRECTION_CLASSES order) on the member of each pair that lies in
    the predicted hemisphere (cluster_hemi == ez_side_pred); 'other' marks the remaining members.
    """
    from scipy import ndimage
    load_script("02_compute_zai.py")                             # puts the project root on sys.path for src/
    from src.thresholding import get_structure_3d
    zai_dir = Path(zai_dir)
    df = _pd.read_csv(zai_dir / f"{pid}_asymmetry_cluster_report.csv")
    code = {c: i for i, c in enumerate(DIRECTION_CLASSES, 1)}
    cls_vol, other, ref, mismatch = None, None, None, 0
    for direction, key in (("left-dominant", "left_dominant"), ("right-dominant", "right_dominant")):
        ref = _nib.load(str(zai_dir / f"{pid}_asymmetry_{key}.nii.gz"))
        lab, n = ndimage.label(_np.asanyarray(ref.dataobj) > 0, structure=get_structure_3d())
        sizes = _np.bincount(lab.ravel(), minlength=n + 1)
        lut_cls, lut_other = _np.zeros(n + 1, _np.uint8), _np.zeros(n + 1, bool)
        for r in df[df["direction"] == direction].itertuples():
            cid = int(r.cluster_id)
            if cid > n or sizes[cid] != int(r.size_voxels):
                mismatch += 1
            elif r.cluster_hemi == r.ez_side_pred:
                lut_cls[cid] = code.get(r.direction_class, 0)
            else:
                lut_other[cid] = True
        cls_vol = lut_cls[lab] if cls_vol is None else _np.maximum(cls_vol, lut_cls[lab])
        other = lut_other[lab] if other is None else (other | lut_other[lab])
    return cls_vol, other, ref, df, mismatch


def paint_regions(values, parc_path, name):
    """Map of region-pair values on a parcellation: the left label gets v, the right label -v.

    values: {(left label, right label): v} with v a left-minus-right quantity, so a region is blue where it is
    lower than its homologue and red where higher, as in the zAI maps (voxel minus mirror).
    """
    img = _nib.load(str(parc_path))
    parc = _np.asanyarray(img.dataobj).astype(_np.int32)
    lut = _np.zeros(max(int(parc.max()), 2100) + 1, _np.float32)
    for (left, right), v in values.items():
        if v is not None and _np.isfinite(v):
            lut[left], lut[right] = v, -v
    return save_view_map(name, lut[_np.clip(parc, 0, None)], img)


class AllLayersView:
    """Resolve one subject's layer files and show them in FSLeyes or as a PNG."""

    def __init__(self, subject_id, dataset=None, base=BASE_DIR):
        self.id = subject_id
        self.dataset = dataset or dataset_of(subject_id)
        self.base = Path(base)
        cfg = ALL_LAYER_DATASETS[self.dataset]
        res = self.base / cfg["results"]
        perfz = sorted((res / "patients" / subject_id).glob(f"{subject_id}_vs_*_zscore.nii.gz"))
        self.band = perfz[0].name[len(subject_id) + 4:-len("_zscore.nii.gz")] if perfz else None
        # consensus parcellation on the template (the analysis' GM / 37 pairs); individual warped aparcs
        # spill outside the average template brain
        atlas = cfg["atlas"] or (f"{cfg['results']}/groups/{self.band}/consensus_parcellation.nii.gz"
                                 if self.band else None)
        files = {
            "t1": self.base / SYM_T1,
            "cbf": self.base / cfg["perf"] / f"{subject_id}_perf_sym.nii.gz",
            "perfz": perfz[0] if perfz else None,
            "zai": res / "asymmetry" / "patients" / subject_id / f"{subject_id}_asymmetry_zscore.nii.gz",
            "clinical": next((p for p in (res / "clinical" / subject_id / f"{subject_id}_clinical_zai_{s}.nii.gz"
                                          for s in ("significant_clusters", "lateralized_significant"))
                              if p.exists()), None),     # both pair members; older runs: higher member only
            "atlas": self.base / atlas if atlas else None,
        }
        self.files = {k: (v if v is not None and Path(v).exists() else None) for k, v in files.items()}
        if self.files["t1"] is None:
            raise FileNotFoundError(f"template missing: {self.base / SYM_T1}")
        self.info = self._concordance_row(res / "clinical" / "cohort_concordance.csv")

    def _concordance_row(self, path):
        if not path.exists():
            return {}
        import csv
        with open(path) as fh:
            return next((r for r in csv.DictReader(fh) if r["patient_id"] == self.id), {})

    def summary(self):
        i = self.info
        ref = i.get("mdt_ez") or "?"
        call = f"{i.get('algo_prediction', '?')} ({i.get('algo_confidence', '?')})" if i else "no rule output"
        return (f"{self.id} [{self.dataset}{', ' + self.band if self.band else ''}]  reference {ref}  "
                f"rule {call}  top {i.get('top_lobe', '-')}  peak |zAI| {i.get('peak_zai', '-')}")

    def missing(self):
        return [k for k, f in self.files.items() if f is None]

    def peak_voxel(self):
        """Voxel of the largest |value| in the clinical map, else in zAI, else the template centre."""
        for key in ("clinical", "zai"):
            if self.files[key] is not None:
                d = _np.abs(_np.asanyarray(_nib.load(str(self.files[key])).dataobj, dtype=_np.float32))
                if d.max() > 0:
                    return tuple(int(v) for v in _np.unravel_index(_np.argmax(d), d.shape))
        return tuple(int(s) // 2 for s in _nib.load(str(self.files["t1"])).shape)

    def focus(self):
        """(voxel, description): the strongest clinical-tier cluster when 06_ has a report, else the map peak."""
        voxel, label = rule_focus(self.id, self.dataset)
        if voxel is not None:
            return voxel, label
        source = next((k for k in ("clinical", "zai") if self.files[k] is not None), None)
        return self.peak_voxel(), {"clinical": "peak |zAI| of the clinical map", "zai": "peak |zAI| of the zAI map",
                                   None: "template centre (no zAI maps)"}[source]

    # ---- FSLeyes ---------------------------------------------------------------------------------
    def fsleyes_command(self, voxel):
        shown = dict(self.files)
        for k in ("cbf", "perfz"):            # scalp / background values hidden (see brain_masked)
            shown[k] = brain_masked(shown[k], self.files["atlas"])
        f = {k: (str(Path(v).relative_to(self.base)) if Path(v).is_relative_to(self.base) else str(v)) if v else None
             for k, v in shown.items()}
        signed = ["-cm", "red-yellow", "-nc", "blue-lightblue", "-un"]
        cmd = ["fsleyes", "-vl", *map(str, voxel), f["t1"], "-dr", "0", "0.95", "-n", "T1 (symmetric template)"]
        if f["atlas"]:
            cmd += [f["atlas"], "-ot", "label", "-l", "freesurfercolorlut", "-o", "-n", "regions (outline)", "-d"]
        if f["cbf"]:
            cmd += [f["cbf"], *CBF_ARGS, "-a", "60", "-n", "CBF (ml/100 g/min)", "-d"]
        if f["perfz"]:
            cmd += [f["perfz"], *signed, "-dr", "1.96", "6", "-n", f"#3 perfusion z vs {self.band}", "-d"]
        if f["zai"]:
            cmd += [f["zai"], *signed, "-dr", "1.96", "6", "-n", "#4 zAI (|zAI| >= 1.96)", "-d"]
        if f["clinical"]:
            cmd += [f["clinical"], *signed, "-dr", "3", "8", "-n", "clinical zAI clusters, both members (|zAI| >= 3)"]
        return cmd

    def open_fsleyes(self, voxel):
        import shutil
        cmd = self.fsleyes_command(voxel)
        if shutil.which("fsleyes") is None:
            print("fsleyes not on PATH; run this where FSL is installed:\n  " + shlex.join(cmd))
            return
        subprocess.Popen(cmd, cwd=self.base, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("FSLeyes started (template + clinical clusters on; tick the eye icons for CBF, #3, #4, atlas).")

    # ---- PNG -------------------------------------------------------------------------------------
    def render_png(self, voxel, out):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        np = _np
        t1 = np.asanyarray(_nib.load(str(self.files["t1"])).dataobj, dtype=np.float32)
        t1 /= np.percentile(t1[t1 > 0], 99)
        blues = LinearSegmentedColormap.from_list("blue-lightblue", ["#1f4fff", "#9fd8ff"])
        rows = [(k, title, rng) for k, title, rng in ALL_LAYERS if self.files[k] is not None]
        fig, axs = plt.subplots(len(rows), 3, figsize=(10, 3.0 * len(rows) + 0.5), dpi=150, squeeze=False,
                                gridspec_kw=dict(width_ratios=[272, 227, 227]))
        fig.patch.set_facecolor("black")
        fig.subplots_adjust(left=0.05, right=1, bottom=0, top=1 - 0.5 / (3.0 * len(rows) + 0.5),
                            wspace=0.02, hspace=0.04)
        brain = (np.asanyarray(_nib.load(str(self.files["atlas"])).dataobj) > 0) if self.files["atlas"] else t1 > 0.2
        for r, (key, title, (lo, hi)) in enumerate(rows):
            vol = np.asanyarray(_nib.load(str(self.files[key])).dataobj, dtype=np.float32)
            if vol.shape == brain.shape:
                vol = np.where(brain, vol, 0)     # scalp / background values are noise (see brain_masked)
            if key == "cbf":  # per-subject range: HCP-ASL and GE eASL CBF differ in scale
                lo, hi = (float(x) for x in np.percentile(vol[(vol > 0) & (t1 > 0.05)], [2, 98]))
                title = f"CBF {lo:.0f}–{hi:.0f} ml/100 g/min"
            for c, plane in enumerate(("sag", "cor", "ax")):
                ax = axs[r, c]
                under = _view_slice(t1, voxel, plane)
                ax.imshow(under, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
                s = _view_slice(vol, voxel, plane)
                if key == "cbf":  # restricted to the template brain
                    ax.imshow(np.ma.masked_where((s <= 0) | (under < 0.05), s), cmap="hot", vmin=lo, vmax=hi,
                              alpha=0.8)
                else:  # signed z: red-yellow positive, blue-lightblue negative, |z| < lo transparent
                    ax.imshow(np.ma.masked_where(s < lo, s), cmap="autumn", vmin=lo, vmax=hi)
                    ax.imshow(np.ma.masked_where(s > -lo, -s), cmap=blues, vmin=lo, vmax=hi)
                col, row = _view_marker(voxel, vol.shape, plane)
                ax.axvline(col, color="#39ff14", lw=0.4, alpha=0.6)
                ax.axhline(row, color="#39ff14", lw=0.4, alpha=0.6)
                ax.axis("off")
                if plane != "sag":
                    ax.text(0.02, 0.03, "L", color="w", fontsize=8, weight="bold", transform=ax.transAxes)
                    ax.text(0.94, 0.03, "R", color="w", fontsize=8, weight="bold", transform=ax.transAxes)
            axs[r, 0].text(-0.04, 0.5, title, color="w", fontsize=9, rotation=90, va="center", ha="right",
                           transform=axs[r, 0].transAxes)
        where = getattr(self, "where", "")
        fig.suptitle(self.summary() + f"\ncrosshair = voxel {voxel}" + (f" ({where})" if where else "")
                     + "\nred-yellow: above controls (#3) or above the mirror voxel (zAI); blue: below  ·  L on the left",
                     color="w", fontsize=9)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        return out


def rule_focus(subject_id, dataset):
    """(voxel, description) of the strongest clinical-tier cluster (first row of the 06_ report, by |peak|),
    on that cluster's predicted side; (None, None) without a report. No files are written. The patient-level
    call (the patient report) weighs all clusters and can differ from this cluster's side.

    The clinical maps draw each finding on its higher-perfused member, so their peak can be the mirror of
    the cluster the rule assigns to the EZ side (e.g. a left hypoperfused cluster); this is the rule's side.
    The voxel is the cluster centroid on the template grid.
    """
    if not _HAVE_SCI:
        return None, None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_clin_interp_focus", str(BASE_DIR / "06_clinical_interpretation.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if dataset != "inhouse":
            import dataset_config
            mod._apply_dataset_config(dataset_config.configure(dataset))
        ci = mod.ClinicalInterpreter(subject_id)
        report = ci.analyze() if ci.has_data() else None
    except Exception:
        return None, None
    top = (report or {}).get("top_clusters") or []
    if not top:
        return None, None
    c = top[0]
    inv = _np.linalg.inv(_nib.load(str(BASE_DIR / SYM_T1)).affine)
    voxel = tuple(int(round(v)) for v in (inv @ _np.array([*c["centroid"], 1.0]))[:3])
    return voxel, (f"strongest clinical-tier cluster {c['cluster_id']}: {c['primary_region']}, "
                   f"{c['direction_class']}, peak zAI {c['peak_z']:+.2f}, this cluster's side {c['ez_side']} (centroid)")


def _view_slice(vol, voxel, plane):
    """2-D slice for display: index 0 of x is the right hemisphere, so x is reversed (L on the left)."""
    i, j, k = voxel
    if plane == "sag":
        s = vol[i, :, :]
    elif plane == "cor":
        s = vol[::-1, j, :]
    else:
        s = vol[::-1, :, k]
    return s.T[::-1]


def _view_marker(voxel, shape, plane):
    """(column, row) of the voxel in the slice returned by _view_slice."""
    i, j, k = voxel
    nx, ny, nz = shape
    if plane == "sag":
        return j, nz - 1 - k
    if plane == "cor":
        return nx - 1 - i, nz - 1 - k
    return nx - 1 - i, ny - 1 - j


def all_layer_patient_ids(dataset, base=BASE_DIR):
    clin = Path(base) / ALL_LAYER_DATASETS[dataset]["results"] / "clinical"
    return sorted(p.name for p in clin.iterdir() if p.is_dir())


def open_file(path):
    """Open a file in VS Code when `code` is on PATH, else with the desktop default."""
    import shutil
    for opener in ("code", "xdg-open"):
        if shutil.which(opener):
            subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return


def run_all_layers(subject_id, dataset=None, png=False, open_png=False, print_only=False, at=None,
                   out_dir=ALL_LAYER_PNG_DIR, quiet=False):
    """One subject: FSLeyes (default), PNG (png=True) or the FSLeyes command (print_only=True)."""
    v = AllLayersView(subject_id, dataset)
    voxel, where = (tuple(at), "voxel given with --at") if at else v.focus()
    v.where = where
    if not quiet:
        print(v.summary())
        print(f"opened at voxel {voxel}: {where}")
        print("missing layers: " + (", ".join(v.missing()) or "none"))
    if print_only:
        print(shlex.join(v.fsleyes_command(voxel)))
    elif png:
        out = v.render_png(voxel, Path(out_dir) / v.dataset / f"{subject_id}.png")
        print(out)
        if open_png:
            open_file(out)
        return out
    else:
        v.open_fsleyes(voxel)
    return None


# ---------------------------------------------------------------------------
# Registration chain of one subject (menu section 'registration', --registration).
# In-house (symreg/reg_subject.sh): T1 brain on the MNI grid -> flirt 12 dof -> fnirt -> applywarp; the ASL
# -> T1 -> MNI step was done earlier by HCP-ASL. Chalfont (scripts/chalfont/reg_chalfont_subject.sh): PD/M0
# reference -> T1 (flirt 6 dof, normmi) -> T1 -> T_sym (flirt 6 then 12 dof, fnirt) -> one applywarp of the
# CBF with the composed transform. The registration scripts deleted the affine matrix and the warped T1;
# they are recomputed here with the same FSL commands and cached in REG_CACHE/<ID>/.
# ---------------------------------------------------------------------------
REG_CACHE = Path(tempfile.gettempdir()) / "asym_registration"
# CBF in FSLeyes: hot 10-80 ml/100 g/min, values below 10 hidden (noise outside the brain is not painted)
CBF_ARGS = ["-cm", "hot", "-dr", "10", "80", "-cr", "10", "100000"]
SYM_T1_AVG = "symreg/templates/T1_avg.nii.gz"


class RegistrationView:
    """Files, recomputed intermediates and QC of one subject's registration to the symmetric template."""

    def __init__(self, subject_id, dataset=None, base=BASE_DIR):
        self.id = subject_id
        self.dataset = dataset or dataset_of(subject_id)
        self.base = b = Path(base)
        self.sym = b / SYM_T1
        self.cache = REG_CACHE / subject_id
        if self.dataset == "chalfont":
            st, sp = b / "external_chalfont/staged" / subject_id, b / "external_chalfont/sym_perf"
            self.t1, self.asl_ref = st / "T1_brain.nii.gz", st / "asl_ref.nii.gz"
            self.cbf_native, self.labels_native = st / "tcflow.nii.gz", st / "gif_parc.nii.gz"
            self.asl2t1 = sp / f"{subject_id}_asl2t1.mat"
            self.asl2t1_init = sp / f"{subject_id}_asl2t1_init.mat"     # the normmi initialisation (since 2026-11-18)
            self.t1_mask = st / "brain_mask.nii.gz"
            self.warp, self.cbf_sym = sp / f"{subject_id}_warp.nii.gz", sp / f"{subject_id}_perf_sym.nii.gz"
            self.script = "scripts/chalfont/reg_chalfont_subject.sh"
            self.affine_desc = "flirt 6 dof (full search), then 12 dof -init -nosearch"
        else:
            sp = b / "symreg/sym_perf"
            self.t1 = b / "symreg/patient_brains" / f"{subject_id}_brain.nii.gz"
            self.asl_ref = self.asl2t1 = None          # ASL -> T1 -> MNI was done by HCP-ASL
            self.cbf_native = b / "Dataset" / subject_id / f"{subject_id}_perfusion_calib_resampled_to_T1w.nii.gz"
            self.labels_native = b / "Dataset" / subject_id / "aparc+aseg.nii.gz"
            self.warp, self.cbf_sym = sp / f"{subject_id}_warp.nii.gz", sp / f"{subject_id}_perf_sym.nii.gz"
            self.script = "symreg/reg_subject.sh"
            self.affine_desc = "flirt 12 dof -nosearch (the T1 is already on the MNI grid)"
        self.labels_sym = AllLayersView(subject_id, self.dataset, b).files["atlas"]

    def missing(self):
        need = [self.sym, self.t1, self.cbf_native, self.warp, self.cbf_sym] + \
               ([self.asl_ref, self.asl2t1] if self.dataset == "chalfont" else [])
        return [str(p) for p in need if p is None or not Path(p).exists()]

    # ---- FSL steps (cached) --------------------------------------------------------------------
    def _fsl(self, *cmd):
        if shutil.which(cmd[0]) is None:
            raise RuntimeError(f"{cmd[0]} (FSL) is not on PATH; source $FSLDIR/etc/fslconf/fsl.sh")
        subprocess.run([str(c) for c in cmd], check=True, capture_output=True,
                       env=dict(os.environ, FSLOUTPUTTYPE="NIFTI_GZ"))

    def _cached(self, name, make):
        out = self.cache / name
        if not out.exists():
            self.cache.mkdir(parents=True, exist_ok=True)
            make(out)
        return out

    def step1_method(self):
        """How the saved asl2t1.mat was estimated (Chalfont): 'bbr' = whole-head normmi initialisation refined by
        boundary-based registration on the GIF white matter (since 2026-11-18); 'normmi' = no GIF parcellation or
        the refinement was rejected; 'normmi_wholehead' = a matrix from before 2026-11-18 (bet on the PD kept the
        whole head, so the fit was head-to-brain)."""
        if self.dataset != "chalfont":
            return None
        if not self.asl2t1_init.exists():
            return "normmi_wholehead"
        return "normmi" if _np.allclose(_np.loadtxt(self.asl2t1_init), _np.loadtxt(self.asl2t1)) else "bbr"

    STEP1_TEXT = {
        "bbr": "flirt 6 dof normmi (whole head, +-30 deg search) as initialisation, refined by boundary-based "
               "registration on the GIF white matter (flirt -cost bbr, global_abs)",
        "normmi": "flirt 6 dof normmi (whole head, +-30 deg search); the BBR refinement was not applied (no GIF "
                  "parcellation, or rejected as a divergence)",
        "normmi_wholehead": "matrix from before 2026-11-18: whole-head normmi only (bet on the PD reference kept "
                            "the whole head); re-estimate with scripts/chalfont/run_registration.sh --redo-step1",
    }

    def asl_in_t1(self, image="asl_ref"):
        """Step 1 output in the T1, with the saved rigid matrix: 'asl_ref' (trilinear), 'cbf' as the pipeline resamples
        it (spline) inside the T1 brain + 2 mm, or 'cbf_native' (nearest neighbour: the 1.9 x 1.9 x 4 mm voxels,
        unmasked). Cached per matrix (file name carries the matrix's modification time)."""
        src = self.asl_ref if image == "asl_ref" else self.cbf_native
        interp = {"asl_ref": "trilinear", "cbf": "spline", "cbf_native": "nearestneighbour"}[image]
        tag = int(Path(self.asl2t1).stat().st_mtime)

        def make(o):
            self._fsl("flirt", "-in", src, "-ref", self.t1, "-applyxfm", "-init", self.asl2t1, "-interp", interp,
                      "-out", o)
            if image == "cbf":
                from scipy import ndimage
                img = _nib.load(str(o))
                d = _np.asanyarray(img.dataobj, dtype=_np.float32)
                m = ndimage.binary_dilation(_np.asanyarray(_nib.load(str(self.t1_mask)).dataobj) > 0, iterations=2)
                _nib.save(_nib.Nifti1Image(d * m, img.affine), str(o))
        return self._cached(f"{image}_in_t1_{tag}.nii.gz", make)

    def affine_matrix(self):
        def make(o):
            print(f"    computing the T1 -> T_sym affine ({self.affine_desc}); ~30-45 s, cached in {self.cache}")
            if self.dataset == "chalfont":
                init = self.cache / "init6.mat"
                self._fsl("flirt", "-in", self.t1, "-ref", self.sym, "-dof", "6", "-omat", init)
                self._fsl("flirt", "-in", self.t1, "-ref", self.sym, "-dof", "12", "-init", init, "-nosearch", "-omat", o)
            else:
                self._fsl("flirt", "-in", self.t1, "-ref", self.sym, "-dof", "12", "-nosearch", "-omat", o)
        return self._cached("t1_to_sym_affine.mat", make)

    def t1_affine(self):
        """Step 2 output: the T1 after the affine registration, on the template grid."""
        mat = self.affine_matrix()
        return self._cached("t1_affine.nii.gz", lambda o: self._fsl(
            "flirt", "-in", self.t1, "-ref", self.sym, "-applyxfm", "-init", mat, "-out", o))

    def t1_warped(self):
        """Step 3 output: the T1 after fnirt (affine + non-linear warp), on the template grid."""
        return self._cached("t1_warped.nii.gz", lambda o: self._fsl(
            "applywarp", "-i", self.t1, "-r", self.sym, "-w", self.warp, "-o", o))

    def t1_warped_mirror(self):
        """The warped T1 flipped left-right on the template grid (mirror check: should match the original)."""
        def make(o):
            img = _nib.load(str(self.t1_warped()))
            d = _np.asanyarray(img.dataobj, dtype=_np.float32)
            _nib.save(_nib.Nifti1Image(_np.flip(d, 0), img.affine, img.header), str(o))
        return self._cached("t1_warped_mirrored.nii.gz", make)

    @staticmethod
    def template_outline():
        """Label image of the template (1 = brain, 2 = white matter) drawn as outlines over a registered T1."""
        out = REG_CACHE / "template_outline.nii.gz"
        if not out.exists():
            REG_CACHE.mkdir(parents=True, exist_ok=True)
            img = _nib.load(str(BASE_DIR / SYM_T1))
            t = _np.asanyarray(img.dataobj, dtype=_np.float32)
            p98 = _np.percentile(t[t > 0], 98)
            lab = (t > 0.2 * p98).astype(_np.uint8) + (t > 0.8 * p98).astype(_np.uint8)
            _nib.save(_nib.Nifti1Image(lab, img.affine), str(out))
        return out

    @staticmethod
    def template_asymmetry():
        """|T_avg - flip(T_avg)| / mean (%) inside the brain: what averaging with the mirror removed."""
        out = REG_CACHE / "template_asymmetry_removed.nii.gz"
        if not out.exists():
            REG_CACHE.mkdir(parents=True, exist_ok=True)
            img = _nib.load(str(BASE_DIR / SYM_T1_AVG))
            a = _np.asanyarray(img.dataobj, dtype=_np.float32)
            a = a / _np.percentile(a[a > 0], 98)
            f = _np.flip(a, 0)
            m = (a + f) / 2
            ok = m > 0.15
            r = _np.zeros_like(a)
            r[ok] = 100 * _np.abs(a[ok] - f[ok]) / m[ok]
            _nib.save(_nib.Nifti1Image(r, img.affine), str(out))
        return out

    # ---- QC ------------------------------------------------------------------------------------
    def qc(self):
        """NCC of the warped T1 with the template and mirror Dice of its tissue mask (0.8-mm grid);
        Chalfont also reports its registration_qc.csv row (2-mm grid, the pre-specified QC)."""
        t = _np.asanyarray(_nib.load(str(self.sym)).dataobj, dtype=_np.float32)
        w = _np.asanyarray(_nib.load(str(self.t1_warped())).dataobj, dtype=_np.float32)
        brain = t > 0.2 * _np.percentile(t[t > 0], 98)
        x, y = t[brain], w[brain]
        ncc = float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std() + 1e-9))
        tis = w > 0.3 * _np.percentile(w[w > 0], 98)
        dice = float(2 * (tis & _np.flip(tis, 0)).sum() / max(1, 2 * tis.sum()))
        out = {"NCC with T_sym": round(ncc, 3), "mirror Dice of the T1 tissue mask": round(dice, 3)}
        csv_path = self.base / "external_chalfont/qc/registration_qc.csv"
        if self.dataset == "chalfont" and csv_path.exists():
            import csv
            with open(csv_path) as fh:
                row = next((r for r in csv.DictReader(fh) if r["id"] == self.id), None)
            if row:
                out.update({"QC csv (2 mm): NCC": row["ncc_t1_template"], "mirror Dice": row["mirror_dice_t1"],
                            "L/R coverage": row["coverage_lr_ratio"]})
                if row.get("asl_t1_gm_corr"):
                    out.update({"ASL->T1 CBF-cortex corr": row["asl_t1_gm_corr"],
                                "ventricle/cortex CBF": row["asl_t1_vent_ratio"]})
                out["flag"] = row.get("flag") or "none"
        return out

    # ---- one figure of the whole chain ---------------------------------------------------------
    def render_png(self, out):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def canon(path):
            img = _nib.as_closest_canonical(_nib.load(str(path)))
            return _np.asanyarray(img.dataobj, dtype=_np.float32), img.affine

        def norm(v):
            return v / max(float(_np.percentile(v[v > 0], 99)), 1e-6) if (v > 0).any() else v

        def vox(aff, mm):
            return [int(round(c)) for c in (_np.linalg.inv(aff) @ _np.array([*mm, 1.0]))[:3]]

        def sl(v, at, plane):        # RAS canonical: x left->right, superior / anterior up
            return v[:, at[1], :].T[::-1] if plane == "cor" else v[:, :, at[2]].T[::-1]

        t1n, a_n = canon(self.t1)
        centre = [int(c) for c in _np.round(_np.argwhere(t1n > 0).mean(0))]
        tmpl, a_t = canon(self.sym)
        at_t = vox(a_t, (0, -20, 8))
        outline, _ = canon(self.template_outline())
        cols = [("Input T1\n(own grid)", t1n, None, centre, False)]
        if self.dataset == "chalfont":
            how = {"bbr": "normmi init + BBR on GIF white matter", "normmi": "flirt 6 dof, normmi",
                   "normmi_wholehead": "whole-head normmi, pre-2026-11-18"}[self.step1_method()]
            cols.append((f"1 · CBF -> T1, rigid\n({how};\nspline, inside the T1 brain + 2 mm)", t1n,
                         canon(self.asl_in_t1("cbf"))[0], centre, False))
        else:
            cols.append(("1 · CBF on the input T1\n(ASL -> T1 -> MNI by HCP-ASL)", t1n, canon(self.cbf_native)[0],
                         centre, False))
        cols += [("2 · T1 -> T_sym, affine\n(" + ("flirt 6 + 12 dof" if self.dataset == "chalfont"
                                                else "flirt 12 dof") + "; template outline)",
                  canon(self.t1_affine())[0], None, at_t, True),
                 ("3 · T1 -> T_sym, non-linear\n(fnirt; template outline)", canon(self.t1_warped())[0], None, at_t, True),
                 # the CBF on T_sym covers the scalp (spline-resampled slab): shown inside the template brain, as the
                 # viewer's step-4 option and every analysis step do
                 ("4 · CBF on T_sym\n(applywarp, one resampling;\ninside the template brain)", tmpl,
                  canon(self.cbf_sym)[0] * (outline > 0), at_t, False)]
        fig, axs = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 6.6), dpi=150, squeeze=False)
        fig.patch.set_facecolor("black")
        for c, (title, under, over, at, draw_outline) in enumerate(cols):
            u = norm(under)
            for r, plane in enumerate(("cor", "ax")):
                ax = axs[r, c]
                ax.imshow(sl(u, at, plane), cmap="gray", vmin=0, vmax=1.05, interpolation="bilinear")
                if over is not None:
                    o = sl(norm(over), at, plane)
                    ax.imshow(_np.ma.masked_where(o <= 0.05, o), cmap="hot", vmin=0, vmax=1, alpha=0.6)
                if draw_outline:
                    ax.contour(sl(outline, at, plane), levels=[0.5, 1.5], colors=["#ff9f1c", "#2ec4b6"], linewidths=0.5)
                ax.axis("off")
                ax.text(0.02, 0.03, "L", color="w", fontsize=7, weight="bold", transform=ax.transAxes)
                ax.text(0.92, 0.03, "R", color="w", fontsize=7, weight="bold", transform=ax.transAxes)
            axs[0, c].set_title(title, color="w", fontsize=8)
        q = self.qc()
        fig.suptitle(f"{self.id} [{self.dataset}] registration chain ({self.script})   ·   "
                     + "   ".join(f"{k}: {v}" for k, v in q.items())
                     + "\norange = template brain outline, teal = template white matter; coronal y = -20, "
                       "axial z = +8 on the template; L on the left", color="w", fontsize=8)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        return out


def _peak_scale(path):
    """98th percentile of the non-zero voxels: display maximum for a T1 of unknown intensity scale."""
    d = _np.asanyarray(_nib.load(str(path)).dataobj, dtype=_np.float32)
    return round(float(_np.percentile(d[d > 0], 98)), 3) if (d > 0).any() else 1


def run_registration_png(subject_id, dataset=None, out_dir=ALL_LAYER_PNG_DIR, open_png=False):
    """PNG of one subject's registration chain (last registration option, --subject ID --registration)."""
    rv = RegistrationView(subject_id, dataset)
    if rv.missing():
        print("[!] cannot draw the registration chain; missing: " + ", ".join(rv.missing()))
        return None
    out = rv.render_png(Path(out_dir) / rv.dataset / f"{subject_id}_registration.png")
    print(out)
    if open_png:
        open_file(out)
    return out


# ---------------------------------------------------------------------------
# Misdetected cases (menu section 'Misdetected cases', --misses). For every patient of a cohort: the reference side,
# the file it came from and the side each clinical source pointed to; the call of every method; for each wrong call
# whether the method's statistic lies inside the range its leave-one-out controls produce (no signal: the sign is
# noise) or beyond it (a confident call on the other side); and for the primary rule the clusters that carried its
# vote, by direction class. Tables go to ez_analysis_mdt/misdetected/<dataset>/, figures and a report to
# visual_analysis/misdetected/<dataset>/. Both folders, and the hand-coded in-house sources
# (Dataset/reference_sources.csv), are gitignored: they hold patient-level clinical detail.
# ---------------------------------------------------------------------------
MISS_TABLES = BASE_DIR / "ez_analysis_mdt" / "misdetected"
MISS_FIGURES = BASE_DIR / "visual_analysis" / "misdetected"
INHOUSE_SOURCES = BASE_DIR / "Dataset" / "reference_sources.csv"
MTL_LOBES = ("temporal", "hippocampus", "amygdala")          # the lobes the primary rule votes on (06_)
# (column, label, short label); methods without data in a cohort are left out
MISS_METHODS = [
    ("rule", "#4 primary rule: zAI clusters, clinical tier, temporal priority", "#4 rule"),
    ("regional_li", "#1 regional LI, temporal signed (Pipeline A, fitted cut)", "#1 LI"),
    ("regional_z", "regional normative z, temporal mean (sign)", "reg. z"),
    ("regional_z_cal", "regional normative z, calibrated at 5 % FP", "reg. z cal."),
    ("regional_z_flow", "regional normative z on plain Flow (sign)", "reg. z Flow"),
    ("voxel_mass", "smoothed voxel hypoperfused mass (sign)", "vox. mass"),
    ("perfz", "#3 perfusion z: side with more hypoperfused gray matter", "#3 z"),
    ("thesis", "thesis rule: more temporal regions at |AI| >= 10 %", "thesis"),
    ("hipp_volume", "hippocampal volume AI, FSL FIRST (sign)", "hipp. vol."),
    ("hipp_volume_cal", "hippocampal volume AI, calibrated at 5 % FP", "hipp. vol. cal."),
    ("sequential", "volume call where calibrated, else Flow regional z", "vol.->Flow"),
    # the thesis's own readings of the same in-house scans (Rai 2026, Tables 12-13), made beside the presumed EZ
    ("thesis_visual", "thesis visual reading (Table 12; presumed EZ known)", "thesis visual"),
    ("thesis_roi", "thesis ROI reading (Table 12; presumed EZ known)", "thesis ROI"),
    ("thesis_voxel", "thesis voxel-wise reading (Table 13; presumed EZ known)", "thesis voxel"),
]
DERIVED_METHODS = {"regional_z_cal", "hipp_volume_cal", "sequential"}   # thresholded or combined versions
UNBLINDED_METHODS = {"thesis_visual", "thesis_roi", "thesis_voxel"}   # readings that knew the presumed EZ
STRENGTH = {"Bilateral": 0, "Unclear": 0, "Weak": 1, "Moderate": 2, "Strong": 3}   # 06_ call grades
SOURCE_SIDE = {"L": "L", "L>R": "L", "L?": "L", "R": "R", "R>L": "R", "R?": "R"}   # codes that point to one side
INHOUSE_SOURCE_COLS = [("mri", "MRI"), ("interictal_eeg", "interictal EEG"), ("ictal_eeg", "ictal EEG"),
                       ("meg_hd_eeg", "MEG / HD-EEG"), ("neuropsych", "neuropsych."), ("invasive", "invasive")]
CHALFONT_SOURCE_COLS = [("mri_report", "MRI report"), ("eeg", "EEG"), ("pet", "PET"), ("spect", "SPECT"),
                        ("ic_telemetry", "IC telemetry")]
SIDE_COLOURS = {"L": "#3b7dd8", "R": "#e07b39", "L?": "#a9c9f0", "R?": "#f3c4a2", "B": "#8e6bbf", "neg": "#d9d9d9",
                "NL": "#d9d9d9", "": "#ffffff"}                       # '?' = possible (one event, or on review only)


def _side_colour(code):
    return SIDE_COLOURS.get(code, SIDE_COLOURS.get(SOURCE_SIDE.get(code, code), "#d9d9d9"))


def _side_from_text(t):
    """L / R / B from clinical free text, 'neg' for a normal study, 'NL' for text without a side, '' for no text.

    The side test is that of external_chalfont/metadata/build_subjects.py::side_from_text, which derived the labels.
    """
    import re
    t = (t or "").strip().lower()
    if not t or t in ("n/a", "na", "nd", "-"):
        return ""
    t = re.sub(r"(messages? left|left (a )?message|left on answerphone|left voicemail|lives in|nothing left)", " ", t)
    if re.search(r"\bbilat|bitemporal|both hemisph|bilateral", t):
        return "B"
    left = bool(re.search(r"\b(left|l|lhs|lh)\b|^l[ -]|\bl\s*(hs|tle|ftle|mfg|ifg|sfg)", t))
    right = bool(re.search(r"\b(right|r|rhs|rh|righr)\b|^r[ -]|\br\s*(hs|tle|ftle|mfg|ifg|sfg)", t))
    if left or right:
        return "B" if left and right else ("L" if left else "R")
    return "neg" if re.search(r"normal|negative|unremarkable|no (focal |significant |definite )?abnormal", t) else "NL"


INHOUSE_QUOTES = BASE_DIR / "Dataset" / "reference_sources_quotes.csv"   # one thesis quote per coded side (private)
THESIS_PDF = BASE_DIR / "manuscript" / "Thesis.pdf"
MANUSCRIPT_TEX = BASE_DIR / "manuscript" / "Manuscript_Complete.tex"
QUOTED_COLUMNS = [k for k, _ in INHOUSE_SOURCE_COLS] + ["thesis_visual", "thesis_roi", "thesis_voxel", "thesis_asl_useful"]
# where each thesis table starts and ends, and how its rows begin (pdftotext -raw, whitespace collapsed)
THESIS_TABLES = {
    11: ("Table 11. Patient Demographics and Clinical Details", "Patient recruited prior to project",
         r"(?<![\d/.])(\d{1,2})[ab]? \d{2}/[MF]\b"),
    12: ("Table 12. Qualitative and Quantitative ROI Analyses", "Note: Only ROIs",
         r"(?<![\d/.-])(\d{1,2}) (?=LTLE|Left \?posterio|RTLE|TLE[;,]|BTLE|LPLE)"),
    13: ("Table 13. Voxel-Wise Cluster Based Analysis", "Abbreviations",
         r"(?<![\d/.-])(\d{1,2}) (?=LTLE|Left \?posterio|RTLE|TLE[;,]|BTLE|LPLE)"),
    14: ("Table 14. Clinical Utility of ASL", "Abbreviations", r"(?<![\d/.-])(\d{1,2}) (?=(?:Yes|No) (?:Yes|No)\.)"),
}


def thesis_rows(pdf=THESIS_PDF):
    """{table: {thesis patient number: row text}} of thesis Tables 11-14, or {} without pdftotext / the PDF."""
    import re
    import subprocess
    if not Path(pdf).exists() or shutil.which("pdftotext") is None:
        return {}
    text = re.sub(r"\s+", " ", subprocess.run(["pdftotext", "-raw", str(pdf), "-"], capture_output=True,
                                              text=True).stdout)
    out = {}
    for table, (start, end, row_re) in THESIS_TABLES.items():
        i = text.rfind(start)                         # the table itself, not its line in the list of tables
        seg = text[i:text.find(end, i + len(start))]
        starts = [(int(m.group(1)), m.start()) for m in re.finditer(row_re, seg)]
        out[table] = {n: seg[a:(starts[k + 1][1] if k + 1 < len(starts) else len(seg))]
                      for k, (n, a) in enumerate(starts)}
    return out


def verify_inhouse_sources(sources=INHOUSE_SOURCES, quotes=INHOUSE_QUOTES, pdf=THESIS_PDF, tex=MANUSCRIPT_TEX):
    """Check the hand-coded in-house sources against the thesis and the manuscript.

    Every code in Dataset/reference_sources.csv must have a quote in Dataset/reference_sources_quotes.csv with the
    same code, and the quote must appear word for word (whitespace collapsed) in that patient's row of the cited
    thesis table. Every reference tier must match the 'Reference standard' column of manuscript Table 1. Returns
    one row per check (kind, id, column, code, table, quote, ok).
    """
    import re
    src = _pd.read_csv(sources, dtype=str).fillna("").set_index("id")
    q = _pd.read_csv(quotes, dtype=str).fillna("")
    rows, table_rows = [], thesis_rows(pdf)
    norm = lambda t: re.sub(r"\s+", " ", t).strip()   # noqa: E731
    for r in q.itertuples():
        coded = src.loc[r.id, r.column] if r.id in src.index else None
        block = table_rows.get(int(r.thesis_table), {}).get(int(src.loc[r.id, "thesis_no"]) if r.id in src.index else -1)
        rows.append(dict(kind="quote", id=r.id, column=r.column, code=r.code, table=f"thesis Table {r.thesis_table}",
                         quote=r.quote, ok=bool(coded == r.code and block is not None and norm(r.quote) in block)))
    quoted = set(zip(q["id"], q["column"]))
    for pid, s in src.iterrows():                       # every coded side has its quote
        for col in QUOTED_COLUMNS:
            if s.get(col, "") and (pid, col) not in quoted:
                rows.append(dict(kind="missing quote", id=pid, column=col, code=s[col], table="", quote="", ok=False))
    if Path(tex).exists():                              # reference tier against manuscript Table 1
        body = Path(tex).read_text()
        body = body[body.find("\\label{tab:patients}"):]
        for pid, s in src.iterrows():
            m = re.search(rf"^{pid} & .*$", body, flags=re.M)
            cell = m.group(0).split(" & ")[5].strip() if m else ""
            rows.append(dict(kind="tier", id=pid, column="tier", code=s["tier"], table="manuscript Table 1",
                             quote=cell, ok=bool(cell) and cell[0] == s["tier"]))
    return _pd.DataFrame(rows)


def chalfont_label_files(path=None):
    """({field: [source key, ...]}, {source key: file}) parsed from external_chalfont/metadata/build_subjects.py.

    build_subjects.py registers every sheet with src(key, description) and stores each field with
    put(i, field, value, key-variable); walking its lines in order resolves which sheets can supply each field.
    """
    import re
    path = Path(path or BASE_DIR / "external_chalfont/metadata/build_subjects.py")
    if not path.exists():
        return {}, {}
    var, files, field_src = {}, {}, {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(k\w*)\s*=\s*src\('(\w+)',\s*([\"'])(.*?)\3\)", line)
        if m:
            var[m.group(1)] = m.group(2)
            files[m.group(2)] = m.group(4).split(" (")[0]
        for field, v in re.findall(r"put\(i, '(\w+)', .*?, (k\w*)\)", line):
            if v in var and var[v] not in field_src.setdefault(field, []):
                field_src[field].append(var[v])
    return field_src, files


def chalfont_source_sides():
    """{id: {source column: side code}} from the anonymised MRI-negative-study sheet (oscar Patient_Details_Anon.csv).

    Seizure semiology is left out: motor signs are contralateral to the focus and a text rule would invert them.
    """
    import csv
    try:
        sheet = load_script("scripts/chalfont/chalfont_io.py").SOURCE_ROOT.parent / \
            "oscar/MRnegStudy/Patient_Details_Anon.csv"
    except Exception:                                       # noqa: BLE001 - the sources are optional context
        return {}
    if not sheet.exists():
        return {}
    cols = {"mri_report": "MRI Report", "eeg": "EEG Findings", "pet": "PET findings", "spect": "SPECT findings",
            "ic_telemetry": "IC telemetry"}
    with open(sheet, newline="") as fh:
        return {r["Subject"].strip(): {k: _side_from_text(r.get(c, "")) for k, c in cols.items()}
                for r in csv.DictReader(fh)}


def _fresh_06(dataset):
    """06_clinical_interpretation loaded anew (its dataset switch changes module globals) and pointed at dataset."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_06_{dataset}", str(BASE_DIR / "06_clinical_interpretation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if dataset != "inhouse":
        import dataset_config
        mod._apply_dataset_config(dataset_config.configure(dataset))
    return mod


def _loo_values(folder, col, kind="R"):
    """Leave-one-out control values of one enhanced-estimator statistic (<folder>/<kind>_loo_controls.csv)."""
    f = Path(folder) / f"{kind}_loo_controls.csv"
    return _pd.read_csv(f)[col].to_numpy(dtype=float) if f.exists() else _np.array([])


COHORT_NAME = {"inhouse": "In-house (HCP-ASL, n = 17)", "chalfont": "External (Chalfont, GE eASL)"}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("default")              # 05_ imports seaborn, which restyles matplotlib globally
    return plt


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    _pyplot().close(fig)
    return out


def _textwrap(text, width):
    import textwrap
    return textwrap.wrap(str(text), width) or [""]


def _wilson(k, n, z=1.96):
    """Wilson 95 % interval of k / n."""
    if n == 0:
        return _np.nan, _np.nan
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * _np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def _call(stat):
    """'L' for a negative statistic, 'R' for a positive one, '-' for zero or missing (no call)."""
    if stat is None or not _np.isfinite(stat) or stat == 0:
        return "-"
    return "L" if stat < 0 else "R"


class MissAnalysis:
    """Every method's call against the reference for one cohort, and why each miss happened."""

    def __init__(self, dataset):
        if DATASET != dataset:
            configure_dataset(dataset)
        self.dataset = dataset
        self.tables, self.figures = MISS_TABLES / dataset, MISS_FIGURES / dataset
        self.controls, self.cut95, self.reports = {}, {}, {}
        self.c = self._reference()
        self._rule()
        self._other_methods()
        self._control_ranges()
        self._outcomes()
        self._flags()

    # ---- the reference side, its file and the clinical sources ---------------------------------------------
    def _reference(self):
        if self.dataset == "inhouse":
            from ez_ground_truth import EZ_GROUND_TRUTH as gt
            src = (_pd.read_csv(INHOUSE_SOURCES, dtype=str).fillna("").set_index("id")
                   if INHOUSE_SOURCES.exists() else _pd.DataFrame())
            ages = {}
            try:
                x = _pd.read_excel(BASE_DIR / "clinical_spreadsheet.xlsx")
                ages = dict(zip(x["ID"].astype(str).str.replace("sub-", "", regex=False), x["AGE (SCAN)"]))
            except Exception:                               # noqa: BLE001 - age is context only
                pass
            rows = []
            for pid, g in sorted(gt.items()):
                s = src.loc[pid].to_dict() if pid in src.index else {}
                mri = s.get("mri", "")
                rows.append(dict(
                    id=pid, ref=g["ez"], ref_lobe="parietal" if "PARIETAL" in g.get("note", "") else "temporal",
                    certainty=g.get("confidence", ""), tier=s.get("tier", ""), age=ages.get(pid, _np.nan),
                    group="MRI-negative" if mri in ("neg", "L?", "R?") else "lesional", diagnosis=g.get("thesis_dx", ""),
                    label_file="ez_ground_truth.py <- thesis Table 11 (Rai 2026), 'Clinical Diagnosis' (MDT consensus)",
                    label_method="transcribed by hand", label_note=g.get("note", ""),
                    evidence=g.get("evidence", ""), invasive_detail=s.get("invasive_detail", ""),
                    source_note=s.get("source", ""), thesis_no=s.get("thesis_no", ""),
                    thesis_asl_useful=s.get("thesis_asl_useful", ""),
                    **{f"{m}_code": s.get(m, "") for m in UNBLINDED_METHODS},
                    **{k: s.get(k, "") for k, _ in INHOUSE_SOURCE_COLS}))
            return _pd.DataFrame(rows).set_index("id")
        subj = _pd.read_csv(BASE_DIR / "external_chalfont/metadata/subjects.csv").set_index("id")
        ids = sorted(_pd.read_csv(BASE_DIR / f"ez_analysis_mdt/v2/{self.dataset}/R_patients.csv")["id"])
        field_src, files = chalfont_label_files()
        method_field = {"lateralisation_column": "lateralisation", "laterality_code": "lat_code",
                        "side_column": "side_col"}
        sides = chalfont_source_sides()
        rows = []
        for pid in ids:
            s = subj.loc[pid]
            method = str(s.get("ez_side_method", ""))
            field = method_field.get(method, method.split(":", 1)[1] if method.startswith("text_regex:") else "")
            have = set(str(s.get("source", "")).split(";"))
            cand = [k for k in field_src.get(field, []) if k in have]
            path = str(s.get("pathology", "")) if _pd.notna(s.get("pathology")) else ""
            lobe = str(s.get("ez_lobe")) if _pd.notna(s.get("ez_lobe")) else "unknown"
            rows.append(dict(
                id=pid, ref=s["ez_side"], ref_lobe=lobe, certainty="", tier=str(s.get("reference_tier", "")),
                age=s.get("age_at_scan", _np.nan),
                group=("MRI-negative" if s["group"] == "negative" else ("HS" if path.startswith("HS") else "other lesion")),
                diagnosis=path,   # build_subjects.py keeps the first non-empty value, in the order it reads the sheets
                label_file=(("first non-empty of: " if len(cand) > 1 else "") + "; ".join(files.get(k, k) for k in cand)
                            or "metadata/subjects.csv"),
                label_method=method, label_note=str(s.get("notes", "")) if _pd.notna(s.get("notes")) else "",
                evidence="", invasive_detail="", source_note="",
                **{k: sides.get(pid, {}).get(k, "") for k, _ in CHALFONT_SOURCE_COLS}))
        return _pd.DataFrame(rows).set_index("id")

    # ---- the primary rule and the clusters that carried its vote -------------------------------------------
    def _rule_one(self, ci):
        rep, filt = ci.analyze(), ci._filter(ci._load())
        mtl = filt[filt["region_lobe"].isin(MTL_LOBES)]
        row = {f"mtl_{s}_{c}": int(mtl[(mtl.ez_side == s) & (mtl.direction_class == c)].size_voxels.sum())
               for s in "LR" for c in DIRECTION_CLASSES}
        L, R = (sum(row[f"mtl_{s}_{c}"] for c in DIRECTION_CLASSES) for s in "LR")
        call = rep["algo_prediction"]
        row.update(rule="-" if call in ("B", "Unclear") else call, rule_strength=rep["algo_confidence"],
                   rule_rationale=rep["rationale"], top_lobe=rep["top_lobe"] or "", mtl_L=L, mtl_R=R,
                   rule_stat=float(_np.log2((R + 1) / (L + 1))))
        side = call if call in ("L", "R") else ("L" if L >= R else "R")
        deciding = mtl[mtl.ez_side == side].sort_values("size_voxels", ascending=False)
        if len(deciding):
            d = deciding.iloc[0]
            row.update(decide_region=d["primary_region"], decide_class=d["direction_class"],
                       decide_size=int(d["size_voxels"]), decide_peak=float(d["peak_z"]),
                       decide_mm=(float(d["centroid_x"]), float(d["centroid_y"]), float(d["centroid_z_coord"])))
        return row, rep

    def _rule_offset_removed(self, mod, pid, offset):
        """The primary rule with the patient's global offset removed (the field's remedy for a global CBF difference).

        Single-subject ASL comparisons divide CBF by the subject's gray-matter mean before testing against controls
        (Boscolo Galazzo et al. 2015, PLoS One, doi 10.1371/journal.pone.0123975). In z units the analogue is the
        correction of analysis #3: subtract the patient's median gray-matter z. Here it is subtracted from both
        per-side cluster z of 02_, the class and side are recomputed with 02_._classify_direction and the vote with
        06_. Global normalization can itself bias when the global level differs (Borghammer et al. 2008,
        NeuroImage, doi 10.1016/j.neuroimage.2007.12.057), so this is a sensitivity check, not a correction.
        """
        classify = load_script("02_compute_zai.py")._classify_direction
        ci = mod.ClinicalInterpreter(pid)
        df = ci._load().copy()
        other = {"L": "R", "R": "L"}
        new = [classify(zc - offset, zm - offset, h, other[h]) if h in other and _np.isfinite(zc) and _np.isfinite(zm)
               else (cl, ez, "") for zc, zm, h, cl, ez in zip(df["z_perf_cluster_side"], df["z_perf_mirror_side"],
                                                            df["cluster_hemi"], df["direction_class"], df["ez_side_pred"])]
        df["direction_class"], df["ez_side_pred"] = [n[0] for n in new], [n[1] for n in new]
        ci._load = lambda: df
        rep = ci.analyze()
        return ("-" if rep["algo_prediction"] in ("B", "Unclear") else rep["algo_prediction"]), rep["algo_confidence"]

    def _rule_extreme_only(self, mod, pid):
        """The primary rule counting only clusters with an extreme deviation (|z| >= 2 against the controls on at least
        one side, i.e. the subtle class dropped), as normative-modelling studies count only extreme deviations
        (Rutherford et al. 2022, eLife, doi 10.7554/eLife.72904: Z > 2 or Z < -2)."""
        ci = mod.ClinicalInterpreter(pid)
        df = ci._load()
        df = df[df["direction_class"] != "subtle"].copy()
        tier = (df["peak_z"].abs() >= ci.peak_threshold) & (df["size_voxels"] >= ci.size_threshold)
        if not tier.any():                       # 06_ expects at least one clinical-tier cluster
            return "-", "no extreme cluster at the clinical tier"
        ci._load = lambda: df
        rep = ci.analyze()
        return ("-" if rep["algo_prediction"] in ("B", "Unclear") else rep["algo_prediction"]), rep["algo_confidence"]

    def _rule(self):
        mod = _fresh_06(self.dataset)
        P = _pd.read_csv(BASE_DIR / f"ez_analysis_mdt/perfz/{self.dataset}/subjects.csv").set_index("id")
        # analysis #3 subtracted the median gray-matter z: raw minus offset-corrected hemisphere mean recovers it
        median_z = (P["p_raw_meanz_L"] - P["p_oc_meanz_L"]).astype(float)
        rows = {}
        for pid in self.c.index:
            ci = mod.ClinicalInterpreter(pid)
            if ci.has_data():
                rows[pid], self.reports[pid] = self._rule_one(ci)
                rows[pid]["rule_extreme_only"], rows[pid]["rule_extreme_only_strength"] = \
                    self._rule_extreme_only(mod, pid)
                m = float(median_z.get(pid, _np.nan))
                rows[pid]["offset_median_z"] = m
                if _np.isfinite(m):
                    rows[pid]["rule_offset_removed"], rows[pid]["rule_offset_removed_strength"] = \
                        self._rule_offset_removed(mod, pid, m)
        self.c = self.c.join(_pd.DataFrame.from_dict(rows, orient="index"))
        pseudo = BASE_DIR / RESULTS_ROOT / "asymmetry" / "pseudo_controls"
        if pseudo.exists():                          # leave-one-out controls through the same rule
            stats, grades = [], []
            for d in sorted(pseudo.iterdir()):
                ci = mod.ClinicalInterpreter(d.name)
                ci.cluster_csv = d / f"{d.name}_asymmetry_cluster_report.csv"
                if ci.has_data():
                    r = self._rule_one(ci)[0]
                    stats.append(r["rule_stat"])
                    grades.append(STRENGTH.get(r["rule_strength"].split("-")[0], 0))
            self.controls["rule"] = _np.array(stats)
            grades = _np.array(grades)
            # share of controls whose call is at least as strong as a patient's (Weak / Moderate / Strong)
            self.control_rate = {g: float((grades >= g).mean()) for g in (1, 2, 3)} if grades.size else {}
            self.c["rule_control_rate"] = [self.control_rate.get(STRENGTH.get(str(x).split("-")[0], 0), _np.nan)
                                           for x in self.c["rule_strength"]]

    # ---- every other method, from its stored per-patient output --------------------------------------------
    def _other_methods(self):
        c, ds = self.c, self.dataset
        v2 = BASE_DIR / "ez_analysis_mdt" / "v2" / ds
        R = _pd.read_csv(v2 / "R_patients.csv").set_index("id")
        V = _pd.read_csv(v2 / "V_patients.csv").set_index("id")
        thr = float(_pd.read_csv(v2 / "summary.csv").set_index("statistic").loc["R_temporal", "threshold"])
        self.cut_regional_z = thr
        c["regional_z_stat"] = R["R_temporal"].reindex(c.index)
        c["regional_z"] = c["regional_z_stat"].map(_call)
        c["regional_z_cal"] = [_call(z) if _np.isfinite(z) and abs(z) >= thr else "-" for z in c["regional_z_stat"]]
        c["voxel_mass_stat"] = V["V_all_t2"].reindex(c.index)
        c["voxel_mass"] = c["voxel_mass_stat"].map(_call)
        c["global_offset_z"] = R["global_offset_z"].reindex(c.index)
        P = _pd.read_csv(BASE_DIR / f"ez_analysis_mdt/perfz/{ds}/subjects.csv").set_index("id")
        hypo_li = (P["p_raw_hypo_R"] - P["p_raw_hypo_L"]) / (P["p_raw_hypo_R"] + P["p_raw_hypo_L"]).replace(0, _np.nan)
        c["perfz_stat"] = hypo_li.reindex(c.index)
        c["perfz"] = c["perfz_stat"].map(_call)
        T = _pd.read_csv(BASE_DIR / f"ez_analysis_mdt/audit/thesis_roi_rule/{ds}.csv").set_index("id")
        c["thesis_stat"] = T["roi_count_LI"].reindex(c.index)
        c["thesis"] = c["thesis_stat"].map(_call)
        self.controls.update(regional_z=_loo_values(v2, "R_temporal"), voxel_mass=_loo_values(v2, "V_all_t2", "V"),
                             perfz=hypo_li[P["kind"] == "control"].to_numpy(dtype=float))
        # the thesis count index saturates at +-1, so its control range is degenerate: report how often it fires
        tc = T[T["kind"] == "control"]
        self.thesis_control_flag = float(((tc["n_hypo_L_temporal"] + tc["n_hypo_R_temporal"]) > 0).mean()) if len(tc) else _np.nan
        self.thesis_control_rate = float((tc["roi_count_LI"].fillna(0).abs() > 0).mean()) if len(tc) else _np.nan
        if ds == "inhouse":
            mdt = load_script("05_roi_discrimination.py").EZDiscriminatorMDT()
            perf = _pd.read_csv(BASE_DIR / "ez_analysis_mdt/method_performance_exclude.csv").set_index("method")
            cut = float(perf.loc["temporal_signed_li", "threshold"])   # 05_: flipped, call L when -score > cut
            self.cut_regional_li = cut
            score = {}
            for pid in c.index:
                f = BASE_DIR / f"results_analysis/{pid}/{pid}_comprehensive_asymmetry_analysis.csv"
                score[pid] = mdt.temporal_signed_li(_pd.read_csv(f)) if f.exists() else _np.nan
            c["regional_li_stat"] = _pd.Series(score)
            # the fitted cut is an observed score (P023's), so compare with a tolerance and flag the tie
            c["regional_li"] = ["L" if -s > cut + 1e-9 else "R" if _np.isfinite(s) else "-" for s in c["regional_li_stat"]]
            c["regional_li_on_cut"] = [bool(_np.isfinite(s) and abs(-s - cut) < 1e-6) for s in c["regional_li_stat"]]
            for m in sorted(UNBLINDED_METHODS):          # a reading with a side is a call; B / none / blank are not
                if f"{m}_code" in c and c[f"{m}_code"].astype(bool).any():
                    c[m] = [SOURCE_SIDE.get(str(v), "-") for v in c[f"{m}_code"]]
        else:
            F = _pd.read_csv(BASE_DIR / "ez_analysis_mdt/audit/flow_variant/R_patients.csv").set_index("id")
            c["regional_z_flow_stat"] = F["R_temporal"].reindex(c.index)
            c["regional_z_flow"] = c["regional_z_flow_stat"].map(_call)
            H = _pd.read_csv(BASE_DIR / "ez_analysis_mdt/audit/first_chalfont/volume_ai.csv").set_index("id")
            c["hipp_volume_stat"] = H["hipp_ai"].reindex(c.index)
            c["hipp_volume"] = c["hipp_volume_stat"].map(_call)
            c["hipp_volume_cal"] = [_call(a) if _np.isfinite(a) and abs(a) >= 0.087 else "-" for a in c["hipp_volume_stat"]]
            c["sequential"] = [v if v != "-" else f for v, f in zip(c["hipp_volume_cal"], c["regional_z_flow"])]
            self.controls.update(regional_z_flow=_loo_values(BASE_DIR / "ez_analysis_mdt/audit/flow_variant", "R_temporal"),
                                 hipp_volume=H.loc[H["group"] == "control", "hipp_ai"].to_numpy(dtype=float))
            qc = BASE_DIR / RESULTS_ROOT / "lateralization" / "eval_qc_flagged.csv"
            flagged = set(_pd.read_csv(qc).query("tier == 'clinical'")["id"]) if qc.exists() else set()
            c["qc_flag"] = [pid in flagged for pid in c.index]

    def _control_ranges(self):
        for m, v in self.controls.items():
            v = _np.abs(_np.asarray(v, dtype=float))
            v = v[_np.isfinite(v)]
            if v.size:
                self.cut95[m] = float(_np.percentile(v, 95))

    @property
    def methods(self):
        """(column, label, short) of the methods with data in this cohort, in MISS_METHODS order."""
        return [m for m in MISS_METHODS if m[0] in self.c.columns]

    @property
    def core(self):
        """Columns of the independent blind methods (no thresholded or combined versions, no thesis readings)."""
        return [m for m, _, _ in self.methods if m not in DERIVED_METHODS | UNBLINDED_METHODS]

    # ---- right / wrong / no call, and where each statistic lies against the controls ------------------------
    def _outcomes(self):
        c = self.c
        uni = c["ref"].isin(["L", "R"])
        for m, _, _ in self.methods:
            ok = _np.where(c[m].isin(["L", "R"]), (c[m] == c["ref"]).astype(float), _np.nan)
            c[f"{m}_ok"] = _np.where(uni, ok, _np.nan)
            if m in self.cut95 and f"{m}_stat" in c:
                c[f"{m}_beyond"] = c[f"{m}_stat"].abs() >= self.cut95[m]
        c["n_called"] = sum(c[f"{m}_ok"].notna().astype(int) for m in self.core)
        c["n_right"] = sum((c[f"{m}_ok"] == 1).astype(int) for m in self.core)

    # ---- why the primary rule missed ----------------------------------------------------------------------
    FLAGS = [  # (column, text) in report order
        ("f_weak", "weak or bilateral call (temporal volume ratio below 2)"),
        ("f_subtle", "the call changes when only clusters with an extreme deviation (|z| >= 2) vote"),
        ("f_hyper", "the called side's extreme clusters are mainly hyperperfused (the hyperperfused side is called)"),
        ("f_hypo", "the called side's extreme clusters are mainly hypoperfused (hypoperfusion opposite the reference)"),
        ("f_offset", "the call changes when the patient's global offset is removed (intensity normalization)"),
        ("f_within", "rule statistic inside the controls' 95 % range"),
        ("f_extratemporal", "EZ outside the temporal lobe (the rule gives temporal clusters priority)"),
        ("f_top_on_ref", "the largest abnormal volume lies on the reference side"),
        ("f_others_right", "most other methods call the reference side"),
        ("f_none_right", "no method calls the reference side"),
        ("f_label_weak", "label rests on clinical hypothesis or MDT consensus without invasive support"),
        ("f_source_agrees", "a clinical source points to the side the rule called"),
        ("f_age_gap", "age 22-35: no control of that age in the band (in-house)"),
        ("f_thesis_reader", "the thesis's reading (presumed EZ known) pointed to the reference side"),
        ("f_qc", "registration QC flag"),
    ]

    def _flags(self):
        c = self.c
        miss = c["ref"].isin(["L", "R"]) & (c["rule_ok"] != 1)
        c["rule_miss"] = miss
        src_cols = [k for k, _ in (INHOUSE_SOURCE_COLS if self.dataset == "inhouse" else CHALFONT_SOURCE_COLS)]
        for pid in c.index:
            r = c.loc[pid]
            side = r["rule"] if r["rule"] in ("L", "R") else None
            vols = {cl: (r[f"mtl_{side}_{cl}"] if side else r[f"mtl_L_{cl}"] + r[f"mtl_R_{cl}"]) for cl in DIRECTION_CLASSES}
            tot = sum(vols.values())
            share = {cl: (v / tot if tot else 0.0) for cl, v in vols.items()}
            others = [m for m in self.core if m != "rule" and _np.isfinite(r[f"{m}_ok"])]
            f = dict(
                f_weak=str(r["rule_strength"]).startswith(("Weak", "Bilateral")),
                f_subtle=r.get("rule_extreme_only", r["rule"]) != r["rule"],
                f_hyper=vols["hyper"] > vols["hypo"], f_hypo=vols["hypo"] > vols["hyper"],
                f_offset=r.get("rule_offset_removed", r["rule"]) != r["rule"],
                f_within=bool("rule" in self.cut95 and abs(r["rule_stat"]) < self.cut95["rule"]),
                f_extratemporal=r["ref_lobe"] in ("frontal", "parietal", "occipital"),
                f_top_on_ref=str(r["top_lobe"]).startswith(f"{r['ref']}-"),
                f_others_right=bool(others) and sum(r[f"{m}_ok"] == 1 for m in others) > len(others) / 2,
                f_none_right=r["n_right"] == 0,
                f_label_weak=(r["tier"] in ("P", "clinical_hypothesis")) or str(r["label_method"]).startswith(("text_regex", "none")),
                f_source_agrees=bool(side) and any(SOURCE_SIDE.get(str(r[k])) == side for k in src_cols),
                f_age_gap=self.dataset == "inhouse" and _np.isfinite(r["age"]) and 22 <= r["age"] <= 35,
                f_qc=bool(r.get("qc_flag", False)),
                f_thesis_reader=any(r.get(m) == r["ref"] for m in UNBLINDED_METHODS))
            for k, v in f.items():
                c.loc[pid, k] = bool(v) and bool(miss[pid])
            for cl, v in share.items():
                c.loc[pid, f"vote_{cl}"] = round(v, 3)
            c.loc[pid, "why"] = self.sentence(pid, share) if miss[pid] else ""

    def sentence(self, pid, share):
        """One line: what the rule saw and the main reasons it disagreed with the reference."""
        r = self.c.loc[pid]
        call = r["rule"] if r["rule"] in ("L", "R") else "no side"
        cls = ", ".join(f"{100 * v:.0f} % {k}" for k, v in sorted(share.items(), key=lambda kv: -kv[1]) if v >= 0.05)
        parts = [f"called {call} ({r['rule_strength']}) on temporal volume L {int(r['mtl_L'])} vs R {int(r['mtl_R'])} voxels"
                 + (f" ({cls})" if cls else "")]
        if _np.isfinite(r["global_offset_z"]):
            parts.append(f"global CBF {r['global_offset_z']:+.1f} SD vs controls")
        if r.get("rule_offset_removed") not in (None, r["rule"]) and isinstance(r.get("rule_offset_removed"), str):
            alt = r["rule_offset_removed"] if r["rule_offset_removed"] != "-" else "no side"
            parts.append(f"with the global offset removed it calls {alt}")
        if r["top_lobe"]:
            parts.append(f"largest volume {r['top_lobe']}")
        others = [m for m in self.core if m != "rule" and _np.isfinite(r[f"{m}_ok"])]
        if others:
            parts.append(f"{sum(r[f'{m}_ok'] == 1 for m in others)} of {len(others)} other methods call {r['ref']}")
        return "; ".join(parts)

    # ---- tables -------------------------------------------------------------------------------------------
    @property
    def unilateral(self):
        return self.c[self.c["ref"].isin(["L", "R"])]

    def method_summary(self):
        """Per method: right / wrong / no call among unilateral patients, balanced accuracy, the missed patients and,
        where controls exist, the calls split by whether the statistic lies beyond the controls' 95 % range."""
        u, rows = self.unilateral, []
        for m, label, short in self.methods:
            ok = u[f"{m}_ok"]
            right = {s: int((ok[u["ref"] == s] == 1).sum()) for s in "LR"}
            n = {s: int((u["ref"] == s).sum()) for s in "LR"}
            row = dict(method=m, label=label, n=len(u), right=int((ok == 1).sum()), wrong=int((ok == 0).sum()),
                       no_call=int(ok.isna().sum()), right_L=f"{right['L']}/{n['L']}", right_R=f"{right['R']}/{n['R']}",
                       balanced_accuracy=round((right["L"] / max(n["L"], 1) + right["R"] / max(n["R"], 1)) / 2, 3),
                       missed=" ".join(u.index[ok != 1]))
            if f"{m}_beyond" in u:
                b = u[f"{m}_beyond"].astype(bool)
                row.update(control_cut=round(self.cut95[m], 3), beyond_right=int(((ok == 1) & b).sum()),
                           beyond_wrong=int(((ok == 0) & b).sum()), within_right=int(((ok == 1) & ~b).sum()),
                           within_wrong=int(((ok == 0) & ~b).sum()))
            rows.append(row)
        return _pd.DataFrame(rows)

    def reason_counts(self):
        """How many primary-rule misses carry each flag."""
        miss = self.unilateral[self.unilateral["rule_miss"]]
        return _pd.DataFrame([dict(flag=k, reason=t, n=int(miss[k].sum()), of=len(miss), ids=" ".join(miss.index[miss[k]]))
                              for k, t in self.FLAGS])

    STRATA = {"inhouse": [("tier", "reference standard"), ("certainty", "MDT certainty"), ("group", "MRI"),
                          ("ref_lobe", "EZ lobe")],
              "chalfont": [("tier", "reference standard"), ("label_method", "how the side was read"),
                           ("group", "lesion"), ("ref_lobe", "EZ lobe")]}

    def strata(self):
        """Right calls by stratum of the label and the patient, for every independent method, with Wilson 95 % CIs."""
        u, rows = self.unilateral.copy(), []
        u["label_method"] = u["label_method"].astype(str).str.split(":").str[0]
        for var, title in self.STRATA[self.dataset]:
            for level, g in u.groupby(var):
                for m in self.core + [m for m in ("sequential",) if m in u]:
                    k = int((g[f"{m}_ok"] == 1).sum())
                    lo, hi = _wilson(k, len(g))
                    rows.append(dict(variable=title, level=level, method=m, n=len(g), right=k,
                                     rate=round(k / len(g), 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
        return _pd.DataFrame(rows)

    def agreement(self):
        """Share of unilateral patients on whom two methods call the same side, when both call."""
        u, core = self.unilateral, self.core
        a = _pd.DataFrame(index=core, columns=core, dtype=float)
        for m1 in core:
            for m2 in core:
                both = u[m1].isin(["L", "R"]) & u[m2].isin(["L", "R"])
                a.loc[m1, m2] = float((u.loc[both, m1] == u.loc[both, m2]).mean()) if both.any() else _np.nan
        return a

    # ---- the patients without a reference side: a separate look, never scored -------------------------------
    @property
    def nonunilateral(self):
        return self.c[~self.c["ref"].isin(["L", "R"])]

    @property
    def ranged(self):
        """Independent methods whose statistic has a leave-one-out control range."""
        return [m for m in self.core if m in self.cut95 and f"{m}_stat" in self.c]

    def bilateral_unclear(self):
        """One row per bilateral / unclear patient: each clinical source, every method's call (the rule with its grade and
        its two variants), how many blind methods call each side, how many statistics lie inside the control range, which
        sources point each way, and the thesis's readings. There is no side to be right about, so nothing is scored."""
        b = self.nonunilateral
        src = INHOUSE_SOURCE_COLS if self.dataset == "inhouse" else CHALFONT_SOURCE_COLS
        rows = []
        for pid, r in b.iterrows():
            calls = [r[m] for m in self.core]
            n = {s: calls.count(s) for s in ("L", "R")}
            if n["L"] and n["R"]:
                agree = f"split {n['L']} L : {n['R']} R"
            elif n["L"] or n["R"]:
                agree = f"all {'L' if n['L'] else 'R'} ({n['L'] or n['R']} of {len(calls)})"
            else:
                agree = "no call"
            with_range = [m for m in self.ranged if _np.isfinite(r[f"{m}_stat"])]
            inside = [m for m in with_range if not r[f"{m}_beyond"]]
            point = {s: [t for k, t in src if SOURCE_SIDE.get(str(r[k])) == s] for s in ("L", "R")}
            rows.append(dict(
                id=pid, ref=r["ref"], tier=r["tier"], diagnosis=r["diagnosis"], **{k: r[k] for k, _ in src},
                invasive_detail=r.get("invasive_detail", ""), **{m: r[m] for m, _, _ in self.methods},
                rule_strength=r["rule_strength"], rule_extreme_only=r.get("rule_extreme_only", "-"),
                rule_offset_removed=r.get("rule_offset_removed", "-"),
                blind_L=n["L"], blind_R=n["R"], blind_none=len(calls) - n["L"] - n["R"], blind_agree=agree,
                inside_range=f"{len(inside)}/{len(with_range)}",
                sources_L=" ".join(point["L"]), sources_R=" ".join(point["R"]),
                mtl_L=int(r["mtl_L"]), mtl_R=int(r["mtl_R"]), subtle_share=round(float(r["vote_subtle"]), 2),
                **{f"{m}_code": r[f"{m}_code"] for m in sorted(UNBLINDED_METHODS) if f"{m}_code" in r}))
        return _pd.DataFrame(rows).set_index("id") if rows else _pd.DataFrame()

    def write_tables(self):
        self.tables.mkdir(parents=True, exist_ok=True)
        keep = [c for c in self.c.columns if c not in ("decide_mm",)]
        self.c[keep].to_csv(self.tables / "cases.csv")
        self.method_summary().to_csv(self.tables / "method_summary.csv", index=False)
        self.reason_counts().to_csv(self.tables / "rule_miss_reasons.csv", index=False)
        self.strata().to_csv(self.tables / "strata.csv", index=False)
        self.agreement().round(3).to_csv(self.tables / "method_agreement.csv")
        bu = self.bilateral_unclear()
        if len(bu):
            bu.to_csv(self.tables / "bilateral_unclear.csv")
        cols = ["ref", "ref_lobe", "certainty", "tier", "group", "diagnosis", "label_file", "label_method", "label_note"]
        cols += [k for k, _ in (INHOUSE_SOURCE_COLS if self.dataset == "inhouse" else CHALFONT_SOURCE_COLS)]
        self.c[cols + (["invasive_detail", "evidence"] if self.dataset == "inhouse" else [])].to_csv(
            self.tables / "label_sources.csv")
        if self.verification is not None:
            self.verification.to_csv(self.tables / "source_verification.csv", index=False)
        return self.tables

    @property
    def verification(self):
        """In-house: every hand-coded source checked against a verbatim thesis quote (verify_inhouse_sources)."""
        if self.dataset != "inhouse" or not (INHOUSE_SOURCES.exists() and INHOUSE_QUOTES.exists()):
            return None
        if not hasattr(self, "_verification"):
            self._verification = verify_inhouse_sources()
        return self._verification

    # ---- figures ------------------------------------------------------------------------------------------
    OUTCOME_COLOURS = {1.0: "#2ca25f", 0.0: "#de2d26"}           # right, wrong; no call grey
    TIER_SHORT = {"I": "invasive", "P": "MDT", "U": "uncertain", "resected": "resected", "lesion_mdt": "lesion+MDT",
                  "clinical_hypothesis": "hypothesis", "lateralisation_column": "lat. column",
                  "laterality_code": "lat. code", "side_column": "side column", "text_regex": "text rule"}
    # in-house 'MRI-negative' = negative or non-specific, including findings seen on review only (Table 1)
    GROUP_SHORT = {"MRI-negative": "MRI-neg", "lesional": "lesional", "HS": "HS", "other lesion": "lesion"}

    def _order(self):
        """Unilateral patients, fewest methods right first, then bilateral / unclear."""
        u = self.unilateral.sort_values(["n_right", "ref"]).index.tolist()
        return u + [p for p in self.c.index if p not in u]

    def fig_matrix(self, out):
        plt = _pyplot()
        from matplotlib.patches import Rectangle
        c, order, meths = self.c, self._order(), self.methods
        info = [("ref", "reference"), ("tier", "standard"), ("group", "MRI")]
        ncol, nrow = len(info) + len(meths), len(order)
        fig, ax = plt.subplots(figsize=(1.3 + 0.55 * ncol, 0.17 * nrow + 1.9), dpi=200)
        for i, pid in enumerate(order):
            r = c.loc[pid]
            uni = r["ref"] in ("L", "R")
            for j, (k, _) in enumerate(info):
                txt = {"tier": self.TIER_SHORT, "group": self.GROUP_SHORT}.get(k, {}).get(str(r[k]), str(r[k]))
                ax.add_patch(Rectangle((j, i), 1, 1, fc=SIDE_COLOURS.get(txt, "#f2f2f2") if k == "ref" else "#f2f2f2",
                                       ec="white", lw=0.6))
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=4.6)
            for j, (m, _, _) in enumerate(meths, start=len(info)):
                call, ok = str(r[m]), r[f"{m}_ok"]
                fc = (self.OUTCOME_COLOURS.get(ok, "#d0d0d0") if uni else SIDE_COLOURS.get(call, "#d0d0d0"))
                if call in ("n/a", "nan", ""):
                    fc = "#ffffff"
                ax.add_patch(Rectangle((j, i), 1, 1, fc=fc, ec="white", lw=0.6))
                ax.text(j + 0.5, i + 0.5, "" if call == "nan" else call, ha="center", va="center", fontsize=4.8,
                        color="white" if fc in ("#2ca25f", "#de2d26") else "black")
        n_uni = len(self.unilateral)
        ax.axhline(n_uni, color="black", lw=0.8)
        ax.axvline(len(info), color="black", lw=0.8)
        unbl = [j for j, (m, _, _) in enumerate(meths, start=len(info)) if m in UNBLINDED_METHODS]
        if unbl:
            ax.axvline(min(unbl), color="black", lw=1.4)
            ax.text(min(unbl) + len(unbl) / 2, nrow + 0.4, "thesis readings: presumed EZ known", ha="center",
                    va="top", fontsize=5.2, style="italic")
        ax.set_xlim(0, ncol)
        ax.set_ylim(nrow, 0)
        ax.set_yticks([i + 0.5 for i in range(nrow)])
        ax.set_yticklabels([f"{p}{' *' if c.loc[p, 'rule_miss'] else ''}" for p in order], fontsize=4.8)
        ax.set_xticks([j + 0.5 for j in range(ncol)])
        ax.set_xticklabels([t for _, t in info] + [s for _, _, s in meths], fontsize=5.5, rotation=45, ha="left")
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{COHORT_NAME[self.dataset]}: every method's call. Green = reference side, red = other side, "
                     "grey = no call;\nrows sorted by the number of methods right (hardest first); * = missed by the "
                     "primary rule; below the line: bilateral / unclear (call coloured by side)", fontsize=6.5, pad=34)
        return _save(fig, out)

    def fig_control_range(self, out):
        plt = _pyplot()
        meths = [m for m in self.methods if m[0] in self.cut95]
        u = self.unilateral
        fig, axs = plt.subplots(len(meths), 1, figsize=(7.2, 1.0 * len(meths) + 0.9), dpi=200, squeeze=False)
        rng = _np.random.default_rng(0)
        for ax, (m, label, short) in zip(axs[:, 0], meths):
            cut = self.cut95[m]
            o = (u[f"{m}_stat"] * _np.where(u["ref"] == "R", 1, -1) / cut).astype(float)   # +: toward the reference
            ok = u[f"{m}_ok"]
            ax.axvspan(-1, 1, color="#ececec", zorder=0)
            ax.axvline(0, color="#888888", lw=0.6)
            x = o.clip(-3.4, 3.4)
            y = rng.uniform(-0.35, 0.35, len(o))
            ax.scatter(x, y, s=10 if len(o) < 40 else 6, c=[self.OUTCOME_COLOURS.get(v, "#9e9e9e") for v in ok],
                       edgecolors="none", zorder=3)
            wrong = ok == 0
            for pid in u.index[wrong & ((o < -1) | (len(u) < 40))]:
                ax.annotate(pid, (x[pid], y[list(u.index).index(pid)]), fontsize=4.5, xytext=(2, 3),
                            textcoords="offset points")
            b = u[f"{m}_beyond"].astype(bool)
            ax.set_title(f"{label}: wrong {int(wrong.sum())} = {int((wrong & ~b).sum())} inside the control range "
                         f"+ {int((wrong & b).sum())} beyond;  right {int((ok == 1).sum())} = "
                         f"{int(((ok == 1) & ~b).sum())} inside + {int(((ok == 1) & b).sum())} beyond",
                         fontsize=5.8, loc="left")
            ax.set_xlim(-3.6, 3.6)
            ax.set_ylim(-0.6, 0.6)
            ax.set_yticks([])
            ax.tick_params(labelsize=5.5)
        axs[-1, 0].set_xlabel("statistic / 95th percentile of the leave-one-out controls, signed so that + points to the "
                              "reference side\n(grey band = what healthy controls produce; |x| > 3.4 drawn at the edge)",
                              fontsize=6)
        fig.suptitle(f"{COHORT_NAME[self.dataset]}: are the misses signal or control noise?", fontsize=8)
        fig.tight_layout()
        return _save(fig, out)

    def fig_rule_vote(self, out):
        plt = _pyplot()
        c = self.c
        ids = [p for p in self._order() if c.loc[p, "rule_miss"] or c.loc[p, "ref"] not in ("L", "R")
               or self.dataset == "inhouse"]
        fig, ax = plt.subplots(figsize=(7.2, 0.2 * len(ids) + 1.4), dpi=200)
        for i, pid in enumerate(ids):
            r = c.loc[pid]
            for side, sign in (("L", -1), ("R", 1)):
                x0 = 0.0
                for cl, (_, col) in DIRECTION_CLASSES.items():
                    v = float(r[f"mtl_{side}_{cl}"])
                    ax.barh(i, sign * v, left=sign * x0, color=col, height=0.7, edgecolor="none")
                    x0 += v
            ax.text(1.02, i, f"ref {r['ref']} | call {r['rule']} ({r['rule_strength']})", transform=ax.get_yaxis_transform(),
                    fontsize=5, va="center", color="#de2d26" if r["rule_miss"] else "black")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(ids, fontsize=5.5)
        ax.invert_yaxis()
        tot = c.loc[ids, ["mtl_L", "mtl_R"]].to_numpy(dtype=float)
        lim = max(1.0, float(_np.nanpercentile(tot, 95)) * 1.15)
        for i, pid in enumerate(ids):                     # bars beyond the axis: value written at the edge
            for side, sign in (("L", -1), ("R", 1)):
                if c.loc[pid, f"mtl_{side}"] > lim:
                    ax.text(sign * lim * 0.98, i, f"{int(c.loc[pid, f'mtl_{side}']):,} >", fontsize=5,
                            ha="right" if sign > 0 else "left", va="center", color="white", weight="bold")
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        ax.set_xlabel("in-scope temporal (temporal, hippocampus, amygdala) cluster volume assigned to each side, voxels  "
                      "(left of 0: L, right: R)", fontsize=6)
        ax.tick_params(labelsize=5.5)
        handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col in DIRECTION_CLASSES.values()]
        ax.legend(handles, list(DIRECTION_CLASSES), fontsize=5.5, loc="lower right", frameon=False)
        ax.set_title(f"{COHORT_NAME[self.dataset]}: what the primary rule voted on (clinical-tier clusters, one member "
                     "per pair,\nby direction class); 2x the other side = Moderate, 1.05-2x = Weak, below = Bilateral; red "
                     "= missed", fontsize=6.5)
        fig.tight_layout()
        return _save(fig, out)

    def fig_offset(self, out):
        plt = _pyplot()
        prof = (BASE_DIR / "ez_analysis_mdt/clinical_direction_profile.csv" if self.dataset == "inhouse" else
                BASE_DIR / RESULTS_ROOT / "lateralization/eval_direction_profile.csv")
        if not prof.exists():
            return None
        d = _pd.read_csv(prof)
        d = d.rename(columns={"patient": "id"}).set_index("id")
        c = self.c.join(d[["hypo", "hyper"]], how="left")
        c["hyper_share"] = c["hyper"] / (c["hyper"] + c["hypo"]).replace(0, _np.nan)
        from scipy.stats import spearmanr
        ok = c[["global_offset_z", "hyper_share"]].dropna()
        rho = spearmanr(ok["global_offset_z"], ok["hyper_share"])[0] if len(ok) > 3 else _np.nan
        fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=200)
        for pid, r in c.iterrows():
            uni = r["ref"] in ("L", "R")
            col = self.OUTCOME_COLOURS.get(r["rule_ok"], "#9e9e9e") if uni else "#8e6bbf"
            ax.scatter(r["global_offset_z"], r["hyper_share"], s=22 if self.dataset == "inhouse" else 12,
                       c=col if uni else "none", edgecolors=col, lw=0.8)
            if self.dataset == "inhouse" or (r["rule_miss"] and abs(r["global_offset_z"]) >= 1):
                ax.annotate(pid, (r["global_offset_z"], r["hyper_share"]), fontsize=5, xytext=(3, 2),
                            textcoords="offset points")
        ax.axhline(0.5, color="#999999", lw=0.5, ls=":")
        ax.axvline(0, color="#999999", lw=0.5, ls=":")
        ax.set_xlabel("global CBF offset: patient's gray-matter CBF vs the controls (z)", fontsize=7)
        ax.set_ylabel("hyper / (hyper + hypo) cluster pairs", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{COHORT_NAME[self.dataset]}: the direction class follows the scanner offset (Spearman rho "
                     f"{rho:.2f})\ngreen = rule right, red = wrong, grey = no call, purple = bilateral / unclear",
                     fontsize=7)
        fig.tight_layout()
        return _save(fig, out)

    def _side_grid(self, ax, ids, cols, frame_wrong=True):
        """Coloured side codes: patients (rows) x columns [(key, title, kind)], kind 'src' / 'ref' / 'm' (method).
        With frame_wrong, a method's call on the other side of a unilateral reference gets a red frame."""
        from matplotlib.patches import Rectangle
        c = self.c
        for i, pid in enumerate(ids):
            r = c.loc[pid]
            for j, (k, _, kind) in enumerate(cols):
                v = str(r[k]) if _pd.notna(r[k]) else ""
                ec, lw = "white", 0.6
                if frame_wrong and kind == "m" and r["ref"] in ("L", "R") and v in ("L", "R") and v != r["ref"]:
                    ec, lw = "#de2d26", 1.6
                ax.add_patch(Rectangle((j, i), 1, 1, fc=_side_colour(v), ec=ec, lw=lw))
                ax.text(j + 0.5, i + 0.5, v, ha="center", va="center", fontsize=4.8,
                        weight="bold" if kind == "ref" else "normal")
        n_src = sum(kind == "src" for _, _, kind in cols)
        ax.axvline(n_src, color="black", lw=0.8)
        ax.axvline(n_src + 1, color="black", lw=0.8)
        ax.set_xlim(0, len(cols))
        ax.set_ylim(len(ids), 0)
        ax.set_yticks([i + 0.5 for i in range(len(ids))])
        ax.set_yticklabels(ids, fontsize=5.2)
        ax.set_xticks([j + 0.5 for j in range(len(cols))])
        ax.set_xticklabels([t for _, t, _ in cols], fontsize=5.5, rotation=45, ha="left")
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def fig_sources(self, out):
        plt = _pyplot()
        c = self.c
        src = INHOUSE_SOURCE_COLS if self.dataset == "inhouse" else CHALFONT_SOURCE_COLS
        ids = self._order() if self.dataset == "inhouse" else [p for p in self._order() if c.loc[p, "rule_miss"]]
        meths = [(m, s) for m, _, s in self.methods if m in self.core]
        cols = [(k, t, "src") for k, t in src] + [("ref", "reference", "ref")] + [(m, s, "m") for m, s in meths]
        cols += [(f"{m}_code", s, "m") for m, _, s in self.methods if m in UNBLINDED_METHODS]
        fig, ax = plt.subplots(figsize=(1.3 + 0.55 * len(cols), 0.2 * len(ids) + 1.8), dpi=200)
        self._side_grid(ax, ids, cols)
        what = ("thesis Table 11 + manuscript Table 1, coded by hand (Dataset/reference_sources.csv)"
                if self.dataset == "inhouse" else "the anonymised MRI-negative-study sheet, read by the label's text rule; "
                "blank = patient not in that sheet")
        ax.set_title(f"{COHORT_NAME[self.dataset]}: what each clinical source said, the reference, and each method "
                     f"(blue L, orange R, purple B, grey none / non-lateralizing;\nred frame = method on the other side). "
                     f"Sources: {what}" + ("" if self.dataset == "inhouse" else "; rows: primary-rule misses"),
                     fontsize=6, pad=30)
        return _save(fig, out)

    def fig_strata(self, out):
        plt = _pyplot()
        st = self.strata()
        show = ["rule", "sequential" if self.dataset == "chalfont" else "regional_z"]
        names = {m: s for m, _, s in self.methods}
        variables = list(dict.fromkeys(st["variable"]))
        fig, axs = plt.subplots(1, len(variables), figsize=(2.3 * len(variables), 2.8), dpi=200, squeeze=False)
        for ax, var in zip(axs[0], variables):
            g = st[(st["variable"] == var) & st["method"].isin(show)]
            levels = list(dict.fromkeys(g["level"]))
            for k, m in enumerate(show):
                h = g[g["method"] == m].set_index("level").reindex(levels)
                xs = _np.arange(len(levels)) + (k - 0.5) * 0.36
                ax.bar(xs, h["rate"], width=0.34, color=["#4c72b0", "#dd8452"][k], label=names.get(m, m))
                ax.errorbar(xs, h["rate"], yerr=[h["rate"] - h["ci_lo"], h["ci_hi"] - h["rate"]], fmt="none",
                            ecolor="black", lw=0.6, capsize=1.5)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels([f"{self.TIER_SHORT.get(str(l), str(l))}\n(n={int(g[g.level == l]['n'].iloc[0])})"
                                for l in levels], fontsize=5.2)
            ax.axhline(0.5, color="#999999", lw=0.5, ls=":")
            ax.set_ylim(0, 1.05)
            ax.set_title(var, fontsize=6.5)
            ax.tick_params(labelsize=5.5)
        axs[0, 0].set_ylabel("share called on the reference side", fontsize=6)
        h, lab = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, fontsize=6, frameon=False, loc="upper right", ncol=2)
        fig.suptitle(f"{COHORT_NAME[self.dataset]}: do misses follow weak labels? (95 % Wilson intervals)", fontsize=7.5)
        fig.tight_layout()
        return _save(fig, out)

    def fig_agreement(self, out):
        plt = _pyplot()
        a = self.agreement()
        names = {m: s for m, _, s in self.methods}
        u = self.unilateral
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=200, gridspec_kw=dict(width_ratios=[1.25, 1]))
        im = ax1.imshow(a.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1)
        ax1.set_xticks(range(len(a)))
        ax1.set_xticklabels([names[m] for m in a.columns], rotation=45, ha="right", fontsize=5.5)
        ax1.set_yticks(range(len(a)))
        ax1.set_yticklabels([names[m] for m in a.index], fontsize=5.5)
        for i in range(len(a)):
            for j in range(len(a)):
                v = a.iloc[i, j]
                if _np.isfinite(v):
                    ax1.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=4.8,
                             color="white" if v < 0.6 else "black")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)
        ax1.set_title("same side called by two methods (% of patients both call)", fontsize=6.5)
        k = len(self.core)
        counts = u["n_right"].value_counts().reindex(range(k + 1), fill_value=0)
        ax2.bar(counts.index, counts.values, color="#4c72b0")
        for x, v in counts.items():
            if v:
                ax2.text(x, v, str(int(v)), ha="center", va="bottom", fontsize=5.5)
        hard = list(u.index[u["n_right"] <= 1])
        ax2.set_xlabel(f"independent methods (of {k}) that call the reference side", fontsize=6)
        ax2.set_ylabel("unilateral patients", fontsize=6)
        ax2.tick_params(labelsize=5.5)
        ax2.set_title("how many methods get each patient right\n0-1: " + (", ".join(hard) if len(hard) <= 12 else
                      f"{len(hard)} patients"), fontsize=6.2)
        fig.suptitle(f"{COHORT_NAME[self.dataset]}: do the methods agree with each other more than with the labels?",
                     fontsize=7.5)
        fig.tight_layout()
        return _save(fig, out)

    def fig_bilateral_unclear(self, out):
        """The patients without a reference side: what each source said and each method called (top; no cell is wrong,
        there is no side to be right about), and how large each statistic is against the controls beside the unilateral
        patients (bottom): a bilateral abnormality cancels in a left-minus-right statistic, so it should sit in the band."""
        plt = _pyplot()
        b, u = self.nonunilateral, self.unilateral
        ids = list(b.index)
        src = INHOUSE_SOURCE_COLS if self.dataset == "inhouse" else CHALFONT_SOURCE_COLS
        names = {m: s for m, _, s in self.methods}
        cols = [(k, t, "src") for k, t in src] + [("ref", "reference", "ref")] + [(m, names[m], "m") for m in self.core]
        cols += [(k, t, "m") for k, t in (("rule_extreme_only", "#4 rule, |z| >= 2 only"),
                                          ("rule_offset_removed", "#4 rule, offset removed"),
                                          ("regional_z_cal", "reg. z calibrated"),
                                          ("hipp_volume_cal", "hipp. vol. calibrated")) if k in b]
        cols += [(f"{m}_code", names[m], "m") for m, _, _ in self.methods if m in UNBLINDED_METHODS]
        ranged = self.ranged
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(1.3 + 0.55 * len(cols), 0.3 * len(ids) + 4.4), dpi=200,
                                       gridspec_kw=dict(height_ratios=[0.3 * len(ids) + 1.3, 2.6]))
        self._side_grid(ax1, ids, cols, frame_wrong=False)
        ax1.set_title(f"{COHORT_NAME[self.dataset]}: the patients the MDT gave no side. What each source said, the "
                      "reference, and what each method called\n(blue L, orange R, purple B, grey none; no cell is wrong: "
                      "there is no side to be right about)", fontsize=6, pad=30)
        rng = _np.random.default_rng(0)
        step = 0.16
        for j, m in enumerate(ranged):
            cut = self.cut95[m]
            ux = (u[f"{m}_stat"].abs() / cut).clip(upper=3.4)
            ax2.scatter(j + rng.uniform(-0.3, 0.3, len(ux)), ux, s=7, c="#b8b8b8", edgecolors="none", zorder=2)
            for k, pid in enumerate(ids):
                v = abs(b.loc[pid, f"{m}_stat"]) / cut
                if _np.isfinite(v):
                    x, y = j + (k - (len(ids) - 1) / 2) * step, min(v, 3.4)
                    ax2.scatter(x, y, s=24, c=_side_colour(str(b.loc[pid, m])), edgecolors="black", lw=0.5, zorder=3)
                    ax2.annotate(pid, (x, y), fontsize=4.2, xytext=(0, 4), textcoords="offset points", ha="center")
        ax2.axhspan(0, 1, color="#ececec", zorder=0)
        ax2.set_xticks(range(len(ranged)))
        ax2.set_xticklabels([names[m] for m in ranged], fontsize=6)
        ax2.set_xlim(-0.6, len(ranged) - 0.4)
        ax2.set_ylim(0, 3.6)
        ax2.tick_params(labelsize=5.5)
        ax2.set_ylabel("|statistic| / control 95th percentile", fontsize=6)
        ax2.set_title("size of the asymmetry against the controls: grey band = what healthy controls produce, grey dots = "
                      "the unilateral patients,\ncoloured = these patients (colour = side called; |x| > 3.4 drawn at the "
                      "top)", fontsize=6, loc="left")
        fig.tight_layout()
        return _save(fig, out)

    def case_panel(self, pid, out):
        """One patient: the clusters the rule voted on (by class) through the deciding cluster, its vote, every method's
        call, the clinical sources and the reasons."""
        plt = _pyplot()
        from matplotlib.patches import Rectangle
        r = self.c.loc[pid]
        zai_dir = BASE_DIR / RESULTS_ROOT / "asymmetry" / "patients" / pid
        cls_vol, other, ref, _, _ = direction_classes(zai_dir, pid)
        t1 = _np.asanyarray(_nib.load(str(BASE_DIR / SYM_T1)).dataobj, dtype=_np.float32)
        t1 /= _np.percentile(t1[t1 > 0], 99)
        mm = r.get("decide_mm")
        if isinstance(mm, tuple):
            voxel = tuple(int(round(v)) for v in _np.linalg.inv(ref.affine) @ _np.array([*mm, 1.0]))[:3]
        else:
            voxel = tuple(s // 2 for s in cls_vol.shape)
        fig = plt.figure(figsize=(13, 7.2), dpi=150)
        gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], width_ratios=[272, 227, 227, 380], hspace=0.25, wspace=0.06)
        colours = [col for _, col in DIRECTION_CLASSES.values()]
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colours)
        for k, plane in enumerate(("sag", "cor", "ax")):
            ax = fig.add_subplot(gs[0, k])
            ax.imshow(_view_slice(t1, voxel, plane), cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            o = _view_slice(other.astype(_np.float32), voxel, plane)
            ax.imshow(_np.ma.masked_where(o == 0, o), cmap=ListedColormap(["#ffffff"]), alpha=0.45)
            s = _view_slice(cls_vol.astype(_np.float32), voxel, plane)
            ax.imshow(_np.ma.masked_where(s == 0, s), cmap=cmap, vmin=0.5, vmax=4.5, interpolation="nearest")
            col, row = _view_marker(voxel, cls_vol.shape, plane)
            ax.axvline(col, color="#39ff14", lw=0.4, alpha=0.7)
            ax.axhline(row, color="#39ff14", lw=0.4, alpha=0.7)
            ax.axis("off")
            if plane != "sag":
                ax.text(0.02, 0.03, "L", color="w", fontsize=8, weight="bold", transform=ax.transAxes)
                ax.text(0.92, 0.03, "R", color="w", fontsize=8, weight="bold", transform=ax.transAxes)
        ax = fig.add_subplot(gs[0, 3])
        ax.axis("off")
        lines = [f"{pid}  ({COHORT_NAME[self.dataset]})",
                 f"Reference: {r['ref']}  ·  {r['diagnosis']}",
                 f"  {self.TIER_SHORT.get(str(r['tier']), r['tier'])}" + (f", {r['certainty']} certainty" if r['certainty'] else "")
                 + f"  ·  {self.GROUP_SHORT.get(r['group'], r['group'])}"]
        lines += [f"  {'from: ' if k == 0 else '      '}{t}" for k, t in enumerate(_textwrap(r["label_file"], 62))]
        lines += [f"  read as: {r['label_method']}", "", f"Primary rule: {r['rule']}  ({r['rule_strength']})"]
        lines += [f"  {t}" for t in _textwrap(r["rule_rationale"], 66)]
        lines += [f"  deciding cluster: {r.get('decide_region', '')} ({r.get('decide_class', '')}, "
                 f"{int(r['decide_size']) if _pd.notna(r.get('decide_size')) else 0} vox)", ""]
        reasons = [t for k, t in self.FLAGS if bool(r.get(k, False))]
        lines += ["Why the rule differs:" if r["rule_miss"] else "Rule agrees with the reference." if r["ref"] in ("L", "R")
                  else "Reference bilateral / unclear (not scored)."]
        for t in reasons:
            lines += [f"  {'- ' if k == 0 else '  '}{w}" for k, w in enumerate(_textwrap(t, 64))]
        ax.text(0, 1, "\n".join(lines), va="top", fontsize=6.3, family="monospace", transform=ax.transAxes)
        # vote by class
        ax = fig.add_subplot(gs[1, 0])
        for k, side in enumerate("LR"):
            x0 = 0.0
            for cl, (_, colr) in DIRECTION_CLASSES.items():
                v = float(r[f"mtl_{side}_{cl}"])
                ax.bar(k, v, bottom=x0, color=colr, width=0.6)
                x0 += v
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"L{'  (ref)' if r['ref'] == 'L' else ''}", f"R{'  (ref)' if r['ref'] == 'R' else ''}"],
                           fontsize=7)
        ax.set_ylabel("temporal cluster volume (voxels)", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_title("what the rule voted on, by class", fontsize=7)
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=colr) for colr in colours], list(DIRECTION_CLASSES), fontsize=5.5,
                  frameon=False)
        # every method and every source
        ax = fig.add_subplot(gs[1, 1:])
        src = INHOUSE_SOURCE_COLS if self.dataset == "inhouse" else CHALFONT_SOURCE_COLS
        items = [(t, str(r[k])) for k, t in src] + [("reference", str(r["ref"]))] + \
                [(s, str(r[m])) for m, _, s in self.methods]
        for j, (t, v) in enumerate(items):
            wrong = j > len(src) and r["ref"] in ("L", "R") and v in ("L", "R") and v != r["ref"]
            ax.add_patch(Rectangle((j, 0), 1, 1, fc=_side_colour(v),
                                   ec="#de2d26" if wrong else "white", lw=2 if wrong else 0.8))
            ax.text(j + 0.5, 0.5, "" if v == "nan" else v, ha="center", va="center", fontsize=7,
                    weight="bold" if t == "reference" else "normal")
            ax.text(j + 0.5, 1.08, t, ha="left", va="bottom", rotation=50, fontsize=5.8)
        for x in (len(src), len(src) + 1):
            ax.plot([x, x], [-0.05, 1.05], color="black", lw=0.8)
        ax.set_xlim(0, len(items))
        ax.set_ylim(-1.6, 3.2)
        ax.axis("off")
        why = str(r.get("why", "")) or ""
        ax.text(0, -0.25, "\n".join(_textwrap(why, 125)), va="top", fontsize=6.4, transform=ax.transData)
        fig.suptitle(f"{pid}: clusters at |zAI| >= 1.96 coloured by direction class on the member that votes "
                     f"(white = mirror member); crosshair = largest temporal cluster on the called side; L on the left",
                     fontsize=8)
        return _save(fig, out)

    def case_text(self, pid):
        """Terminal summary of one patient: reference and its file, each source, each method, why the rule differs."""
        r = self.c.loc[pid]
        src = INHOUSE_SOURCE_COLS if self.dataset == "inhouse" else CHALFONT_SOURCE_COLS
        mark = lambda ok: {1.0: "right", 0.0: "WRONG"}.get(ok, "no call")   # noqa: E731
        lines = ["=" * 70, f"  {pid} · {COHORT_NAME[self.dataset]}",
                 f"  Reference: {r['ref']} · {r['diagnosis']} · {self.TIER_SHORT.get(str(r['tier']), r['tier'])}"
                 + (f", {r['certainty']} certainty" if r["certainty"] else "") + f" · {r['group']}",
                 f"  Label file: {r['label_file']}", f"  Read as: {r['label_method']}"
                 + (f" ({r['label_note']})" if r["label_note"] else ""),
                 "  Sources: " + ", ".join(f"{t} {r[k] or '-'}" for k, t in src)]
        if r.get("invasive_detail"):
            lines.append(f"  Follow-up (invasive / surgical): {r['invasive_detail']}")
        if r.get("thesis_asl_useful"):
            lines.append(f"  Thesis (patient #{r['thesis_no']}): ASL judged useful: {r['thesis_asl_useful']} "
                         "(Table 14; readings made beside the presumed EZ)")
        lines.append("  Methods:")
        for m, label, _ in self.methods:
            extra = ""
            if f"{m}_beyond" in r and _np.isfinite(r.get(f"{m}_stat", _np.nan)):
                extra = f"   ({'beyond' if r[f'{m}_beyond'] else 'inside'} the control range)"
            lines.append(f"    {label:58s} {r[m]:>2s}  {mark(r[f'{m}_ok']) if r['ref'] in ('L', 'R') else ''}{extra}")
        lines += [f"  Primary rule: {r['rule_rationale']}",
                  "  Vote by class: " + ", ".join(f"{s} {cl} {int(r[f'mtl_{s}_{cl}'])}" for s in "LR"
                                                  for cl in DIRECTION_CLASSES if r[f"mtl_{s}_{cl}"])]
        if r["rule_miss"]:
            lines += ["  Why the rule differs:"] + [f"    - {t}" for k, t in self.FLAGS if bool(r.get(k, False))]
            lines.append(f"  In one line: {r['why']}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def panel_ids(self):
        """Patients with a case panel: every primary-rule miss, every bilateral / unclear case and the hard cases."""
        c = self.c
        keep = c["rule_miss"] | ~c["ref"].isin(["L", "R"]) | (c["ref"].isin(["L", "R"]) & (c["n_right"] <= 1))
        return [p for p in self._order() if keep[p]]

    def write_figures(self, panels=True):
        self.figures.mkdir(parents=True, exist_ok=True)
        made = {}
        names = ["matrix", "control_range", "rule_vote", "offset", "sources", "strata", "agreement"]
        if len(self.nonunilateral):
            names.append("bilateral_unclear")
        for name in names:
            made[name] = getattr(self, f"fig_{name}")(self.figures / f"{name}.png")
        if panels:
            for pid in self.panel_ids():
                made[f"case_{pid}"] = self.case_panel(pid, self.figures / "cases" / f"{pid}.png")
        return made

    # ---- report -------------------------------------------------------------------------------------------
    def write_report(self):
        """REPORT.md beside the figures: definitions, label sources, every method's misses, and why."""
        import datetime
        u, ms, rc = self.unilateral, self.method_summary(), self.reason_counts()
        miss = u[u["rule_miss"]]
        wrong = int(ms["wrong"].sum())
        inside = int(ms.get("within_wrong", _pd.Series(dtype=float)).fillna(0).sum())
        with_range = int(ms.loc[ms["control_cut"].notna(), "wrong"].sum()) if "control_cut" in ms else 0
        L = [f"# Misdetected cases: {COHORT_NAME[self.dataset]}", "",
             f"Generated {datetime.date.today()} by `python interactive_viewer.py --misses --dataset {self.dataset}` "
             f"(menu section 'Misdetected cases'). Tables: `{self.tables.relative_to(BASE_DIR)}/`. Patient-level and "
             "private: this folder and the tables are gitignored.", "",
             "## 1. What counts as a miss", "",
             f"{len(u)} patients have a unilateral reference side"
             + (f"; {len(self.c) - len(u)} bilateral or unclear patients are not scored and appear below the line in "
                "the figures." if len(self.c) > len(u) else "."),
             "A method *misses* a patient when it calls the other side (wrong) or no side (no call). For every method "
             "with leave-one-out controls, the *control range* is the 95th percentile of the controls' |statistic| (the "
             "calibrated 5 % false-positive cut): a call inside it is no stronger than what healthy people produce.", "",
             _md_table(_pd.DataFrame([(s, lab) for _, lab, s in self.methods], columns=["method", "what it is"])), "",
             "## 2. Where the reference side comes from", ""]
        if self.dataset == "inhouse":
            L += ["Every side was transcribed by hand from Table 11 of the thesis (Rai 2026, 'Clinical Diagnosis': the "
                  "MDT consensus) into `ez_ground_truth.py`, with the evidence (MRI, EMU interictal and ictal, MEG / "
                  "HD-EEG, neuropsychology) and the certainty grade. The reference-standard tier (I invasively supported, "
                  "P MDT-probable, U uncertain) and the stereo-EEG and surgical findings are from the manuscript's "
                  "Table 1 (treating team's records). The side each source pointed to is coded by hand in "
                  "`Dataset/reference_sources.csv` (private) from thesis Tables 11 and 14 (clinical sources and "
                  "follow-up), together with the thesis's own readings of the scans (Tables 12-14), each code with a "
                  "verbatim quote in `Dataset/reference_sources_quotes.csv`.", ""]
            v = self.verification
            if v is not None and len(v):
                qv, tv = v[v["kind"] != "tier"], v[v["kind"] == "tier"]
                L += [f"Verification: {int(qv['ok'].sum())} of {len(qv)} coded sources are backed by a quote found word "
                      f"for word in that patient's row of thesis Tables 11-14, and {int(tv['ok'].sum())} of {len(tv)} "
                      "reference tiers match manuscript Table 1 (`source_verification.csv`, which lists every quote).",
                      ""]
            L += [
                  _md_table(u.groupby(["tier", "certainty"]).size().rename("patients").reset_index()), ""]
        else:
            t = u.copy()
            t["how"] = t["label_method"].astype(str)
            L += ["Every side was derived by `external_chalfont/metadata/build_subjects.py` from the clinical sheets in a "
                  "fixed order: an explicit lateralisation column, a numeric laterality code, a side column, then a text "
                  "rule over the recruitment log, lesion text, MRI report, diagnosis and EEG. The method and the "
                  "sheet(s) are stored per patient (`label_sources.csv`).", "",
                  _md_table(t.groupby(["how", "label_file"]).size().rename("patients").reset_index()), "",
                  _md_table(t.groupby("tier").size().rename("patients").reset_index()), ""]
        L += ["## 3. Every method's misses", "",
              _md_table(ms[[c for c in ("label", "right", "wrong", "no_call", "right_L", "right_R", "balanced_accuracy",
                                        "within_wrong", "beyond_wrong") if c in ms]]), "",
              "![every method's call](matrix.png)", "",
              "## 4. Signal or control noise?", "",
              f"Of the {with_range} wrong calls made by methods that have controls, {inside} lie inside the control "
              "range: in those patients the method had no signal beyond what healthy controls produce, and its side was "
              "decided by noise. Calls beyond the range are the informative ones: their accuracy is in the "
              "'beyond' columns of `method_summary.csv`.", "",
              "![misses against the control range](control_range.png)", ""]
        if getattr(self, "control_rate", None):
            L += [f"The primary rule run on the leave-one-out controls calls a side in "
                  f"{100 * self.control_rate[1]:.0f} % of them (Moderate or Strong in {100 * self.control_rate[2]:.0f} %, "
                  f"Strong in {100 * self.control_rate[3]:.0f} %).", ""]
        if _np.isfinite(getattr(self, "thesis_control_rate", _np.nan)):
            L += [f"The thesis rule flags a temporal region in {100 * self.thesis_control_flag:.0f} % of the controls and "
                  f"calls a side in {100 * self.thesis_control_rate:.0f} %; its count index saturates at +-1, so it has "
                  "no control range.", ""]
        sig = ms[ms["control_cut"].notna()] if "control_cut" in ms else ms.iloc[0:0]
        if len(sig):
            acc = lambda k, n: f"{k}/{n} ({100 * k / n:.0f} %)" if n else "-"   # noqa: E731
            L += ["Accuracy of the sign where the statistic is beyond the control range (signal) and inside it (noise):",
                  "", _md_table(_pd.DataFrame([dict(method=r.label,
                      beyond=acc(int(r.beyond_right), int(r.beyond_right + r.beyond_wrong)),
                      inside=acc(int(r.within_right), int(r.within_right + r.within_wrong))) for r in sig.itertuples()])),
                  ""]
        L += ["## 5. Why the primary rule missed", "",
              f"The rule sums the in-scope temporal, hippocampal and amygdalar cluster volume assigned to each side (one "
              f"member per mirrored pair) and calls the larger side when it is at least 1.05x the other. It missed "
              f"{len(miss)} of {len(u)}. Flags per miss (a miss can carry several):", "",
              _md_table(rc[["reason", "n", "ids"]]), "",
              "![what the rule voted on](rule_vote.png)", "", "![direction class and global offset](offset.png)", ""]
        if "rule_offset_removed" in u:
            ch = u["rule_offset_removed"] != u["rule"]
            fixed = int((ch & u["rule_miss"] & (u["rule_offset_removed"] == u["ref"])).sum())
            broken = int((ch & ~u["rule_miss"]).sum())
            L += ["**The global offset, handled as the field does.** Single-subject ASL studies divide CBF by the "
                  "subject's gray-matter mean before comparing with controls (Boscolo Galazzo et al. 2015, PLoS One, "
                  "[doi](https://doi.org/10.1371/journal.pone.0123975)); PET and SPECT normalize counts the same way, or "
                  "use measures a global shift cannot move: an asymmetry index thresholded within the subject (Gajdos "
                  "et al. 2021, [doi](https://doi.org/10.1038/s41598-021-89774-4); Soma et al. 2012, "
                  "[doi](https://doi.org/10.1007/s12149-012-0573-8)) or the lowest 0.5 % of the patient's own z (Han et "
                  "al. 2026, [doi](https://doi.org/10.1002/cns.70876)). Global normalization itself biases when the "
                  "global level differs (Borghammer et al. 2008, [doi](https://doi.org/10.1016/j.neuroimage.2007.12.057))."
                  " Here the patient's median gray-matter perfusion z (the offset analysis #3 removes) was subtracted from "
                  "both sides of every cluster and the same rule re-run: right calls "
                  f"{int((u['rule_offset_removed'] == u['ref']).sum())} of {len(u)} (as run "
                  f"{int((u['rule'] == u['ref']).sum())}); {int((ch & u['rule_miss']).sum())} of {len(miss)} misses "
                  f"change ({fixed} to the reference side) and {broken} correct calls change. A miss is marked "
                  "offset-dependent when its call changes; the offset explains single misses but removing it is not a "
                  "fix.", ""]
        if "rule_extreme_only" in u:
            ch = u["rule_extreme_only"] != u["rule"]
            L += ["**Sub-threshold clusters, counted as normative studies count deviations.** Normative models count a "
                  "deviation only when it is extreme, Z > 2 or Z < -2 (Rutherford et al. 2022, eLife, "
                  "[doi](https://doi.org/10.7554/eLife.72904)); the subtle class has no such deviation on either side. "
                  "Re-running the rule with only clusters that deviate by |z| >= 2 on at least one side gives "
                  f"{int((u['rule_extreme_only'] == u['ref']).sum())} of {len(u)} right (as run "
                  f"{int((u['rule'] == u['ref']).sum())}; {int((u['rule_extreme_only'] == '-').sum())} no call): "
                  f"{int((ch & u['rule_miss']).sum())} of {len(miss)} misses change "
                  f"({int((ch & u['rule_miss'] & (u['rule_extreme_only'] == u['ref'])).sum())} to the reference side) "
                  f"and {int((ch & ~u['rule_miss']).sum())} correct calls change. The rule's hits rest on "
                  "sub-threshold clusters as much as its misses.", ""]
        L += [
              "One line per miss:", ""]
        L += [f"- **{pid}** (ref {r['ref']}): {r['why']}" for pid, r in miss.iterrows()]
        if self.dataset == "inhouse" and "thesis_visual" in self.c:
            t = self.c[["ref", "thesis_visual_code", "thesis_roi_code", "thesis_voxel_code", "thesis_asl_useful", "rule",
                        "regional_z", "n_right"]].reset_index()
            t.columns = ["id", "reference", "thesis visual", "thesis ROI", "thesis voxel", "thesis: ASL useful",
                         "#4 rule (blind)", "reg. z (blind)", "blind methods right"]
            ok = {m: int((u[f"{m}_ok"] == 1).sum()) for m in UNBLINDED_METHODS}
            L += ["", "### How the thesis read the same scans", "",
                  "The thesis (Rai 2026) read every scan visually, by regional asymmetry and voxel-wise (Tables 12-13), "
                  "and graded each finding against the presumed EZ listed in the same table (Table 14, 'ASL utility'). "
                  "It does not state that the readings were blinded: the question was whether ASL showed an abnormality "
                  "consistent with the known EZ. Our methods answer a different question: they call a side for every "
                  "patient without the label, and are scored afterwards. On the 13 unilateral patients the thesis "
                  f"readings point to the reference side in {ok['thesis_visual']} (visual), {ok['thesis_roi']} (ROI) and "
                  f"{ok['thesis_voxel']} (voxel-wise) cases; the blind primary rule in "
                  f"{int((u['rule_ok'] == 1).sum())}. The readings are shown for comparison and are not counted among "
                  "the methods.", "", _md_table(t), ""]
        L += ["", "## 6. What each clinical source said", "", "![sources, reference and methods](sources.png)", "",
              "## 7. Do misses follow weak labels?", "", "![right calls by stratum](strata.png)", "",
              _md_table(self.strata().query("method in ['rule', 'sequential', 'regional_z']")), "",
              "## 8. Do the methods agree with each other?", "", "![agreement](agreement.png)", "",
              f"Patients no more than one independent method gets right: {', '.join(u.index[u['n_right'] <= 1]) or 'none'}.",
              ""]
        bu = self.bilateral_unclear()
        if len(bu):
            L += self._bilateral_section(bu)
        L += ["## 10. Case panels", ""]
        L += [f"- [{pid}](cases/{pid}.png): ref {self.c.loc[pid, 'ref']}, rule {self.c.loc[pid, 'rule']} "
              f"({self.c.loc[pid, 'rule_strength']})" for pid in self.panel_ids()]
        out = self.figures / "REPORT.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(L) + "\n")
        return out

    def _bilateral_section(self, bu):
        """Report lines for the patients without a reference side: what each method calls, whether the methods agree
        with each other and with the work-up, and how large the asymmetry is against the controls."""
        u, c = self.unilateral, self.c
        names = {m: s for m, _, s in self.methods}
        ranged = self.ranged

        def inside(d):
            n = sum(int(d[f"{m}_stat"].notna().sum()) for m in ranged)
            k = sum(int((~d[f"{m}_beyond"].astype(bool) & d[f"{m}_stat"].notna()).sum()) for m in ranged)
            return k, n
        k_b, n_b = inside(c.loc[bu.index])
        k_u, n_u = inside(u)
        by = {s: [p for p in bu.index if bu.loc[p, "ref"] == s] for s in ("B", "U")}
        cal = [(m, lab) for m, lab, _ in self.methods if m in DERIVED_METHODS and m.endswith("_cal") and m in bu]
        src = INHOUSE_SOURCE_COLS if self.dataset == "inhouse" else CHALFONT_SOURCE_COLS
        cols = ["ref", "tier"] + [k for k, _ in src] + self.core + ["rule_strength", "rule_extreme_only",
                                                                     "rule_offset_removed"]
        cols += [m for m, _ in cal] + ["blind_agree", "inside_range"]
        L = ["## 9. Bilateral and unclear patients: a separate look", "",
             f"The MDT gave no side to {len(bu)} patients: bilateral {', '.join(by['B']) or 'none'}; unclear "
             f"{', '.join(by['U']) or 'none'}. They are left out of every score above, because there is no side to be "
             "right about, and are kept in their own table here (`bilateral_unclear.csv`). Three questions: what does "
             "each method call, do the methods say the same thing, and how does that sit with the work-up?", "",
             "Every sign-based method has to call a side, so for these patients the call is not the finding; the size "
             "of the asymmetry is. If both hemispheres are abnormal, the left-minus-right difference partly cancels, so "
             "the statistics should sit inside the control range and a calibrated method should make no call. "
             f"{k_b} of {n_b} statistics with controls lie inside the range (unilateral patients: {k_u} of {n_u})"
             + "".join(f"; the {lab} calls a side in {int(bu[m].isin(['L', 'R']).sum())} of {len(bu)}" for m, lab in cal)
             + ". The supplement's magnitude analysis tests the same prediction with the regional |LI| over all "
             "patients.", "",
             _md_table(bu[[k for k in cols if k in bu]].reset_index()), "",
             "![bilateral and unclear patients](bilateral_unclear.png)", "", "One paragraph per patient:", ""]
        for pid, r in bu.iterrows():
            c0 = c.loc[pid]
            calls = ", ".join(f"{names[m]} {c0[m]}" + (f" ({c0['rule_strength']})" if m == "rule" else "")
                              for m in self.core)
            read = ", ".join(f"{names[m]} {c0[f'{m}_code'] or 'none'}" for m in sorted(UNBLINDED_METHODS)
                             if f"{m}_code" in c0)
            L.append(f"- **{pid}** (reference {r['ref']}, {self.TIER_SHORT.get(str(r['tier']), r['tier'])}; "
                     f"{r['diagnosis']}). Work-up: {c0['evidence'] or 'see label_sources.csv'}."
                     + (f" Follow-up: {c0['invasive_detail']}." if c0.get("invasive_detail") else "")
                     + f" Blind methods: {calls}: {r['blind_agree']}. Rule with only |z| >= 2 clusters voting: "
                     f"{r['rule_extreme_only']}; with the global offset removed: {r['rule_offset_removed']}"
                     + "".join(f"; {lab}: {r[m]}" for m, lab in cal)
                     + f". Statistics inside the control range: {r['inside_range']}. Sources pointing left: "
                     f"{r['sources_L'] or 'none'}; right: {r['sources_R'] or 'none'}."
                     + (f" Thesis readings (presumed EZ known): {read}." if read else ""))
        return L + [""]


def _md_table(df):
    """A DataFrame as a Markdown table (no extra dependency)."""
    df = df.copy()
    cols = list(df.columns)
    fmt = lambda v: ("-" if not _np.isfinite(v) else f"{v:.3g}") if isinstance(v, float) else str(v)   # noqa: E731
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(fmt(v).replace("|", "\\|") for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def run_misses(dataset, panels=True):
    """Tables, figures, case panels and the report for one cohort; returns the MissAnalysis."""
    ma = MissAnalysis(dataset)
    ma.write_tables()
    ma.write_figures(panels=panels)
    report = ma.write_report()
    ms = ma.method_summary()
    print(f"\n[*] MISDETECTED CASES — {COHORT_NAME[dataset]}")
    for r in ms.itertuples():
        extra = (f"; wrong inside the control range {int(r.within_wrong)}/{int(r.wrong)}"
                 if "within_wrong" in ms and _np.isfinite(r.within_wrong) else "")
        print(f"    {r.label:58s} right {r.right:3d} wrong {r.wrong:3d} no call {r.no_call:3d}{extra}")
    bu = ma.bilateral_unclear()
    if len(bu):
        print("    Bilateral / unclear (not scored; own table bilateral_unclear.csv and report section 9):")
        for pid, r in bu.iterrows():
            print(f"      {pid} ref {r['ref']}: rule {r['rule']} ({r['rule_strength']}); blind methods {r['blind_agree']}; "
                  f"statistics inside the control range {r['inside_range']}")
    print(f"    Tables : {ma.tables.relative_to(BASE_DIR)}/")
    print(f"    Figures: {ma.figures.relative_to(BASE_DIR)}/ (case panels in cases/)")
    print(f"    Report : {report.relative_to(BASE_DIR)}")
    return ma

# regional analysis #1, symmetric registration, AI and zAI (#2, #4), thresholding, direction (#3 as the
# rule's input), lateralization decision, the post hoc estimators, then summary views and utilities.
# Each section title names the Methods subsection it shows. Option numbers are positions in this list, so
# reordering it renumbers the menu, the dispatch table and every "option N" printed in a note (OPTION).
# Labels are format strings: {zai} {clin} {rawz} {anat} flag missing data, {step1} {affine} {band} vary by
# dataset; a trailing * means the option prints a table in the terminal first.
# ---------------------------------------------------------------------------
MENU = [
    ("Inputs · MRI acquisition and processing (original space, in-house only; not aligned with the template maps)", [
        ("view_basic_anatomy", "T1 + FreeSurfer parcellation{anat}"),
        ("view_perfusion_comparison", "CBF on the T1{anat}"),
    ]),
    ("Analysis #1 · regional asymmetry (Pipeline A)", [
        ("view_regional_li", "LI of the 37 region pairs on the parcellation + aggregation scores *"),
    ]),
    ("Left-right symmetric registration, ASL -> T1 -> T_sym (current patient)", [
        ("view_reg_template", "Template: mean of 30 HCP T1s vs T_sym (the asymmetry removed)"),
        ("view_reg_inputs", "Step 0 · inputs before registration (T1, ASL reference / CBF, labels)"),
        ("view_reg_asl_to_t1", "Step 1 · ASL -> T1, rigid ({step1})"),
        ("view_reg_affine", "Step 2 · T1 -> T_sym, affine ({affine}; first run ~30-45 s, cached)"),
        ("view_reg_nonlinear", "Step 3 · T1 -> T_sym, non-linear (fnirt) + mirror check + QC"),
        ("view_reg_output", "Step 4 · CBF and labels on T_sym (applywarp, one resampling)"),
        ("render_reg_png", "Registration chain as a PNG (steps 0-4 in one figure, for the IDE)"),
    ]),
    ("Analyses #2 and #4 · asymmetry index and normative z-score", [
        ("view_voxel_ai", "#2 voxel AI = 200 (v - mirror) / (v + mirror), patient alone (|AI| >= 10 %) *"),
        ("view_control_normative", "Control normative: per-voxel mean and SD of AI ({band}) *"),
        ("view_zai_asymmetry_map", "#4 zAI map (|zAI| >= 1.96; both members of each pair) + calibration *{zai}"),
    ]),
    ("Thresholding", [
        ("view_zai_significant", "Clusters at |zAI| >= 1.96, >= 50 voxels (02_; sign masks hidden){zai}"),
        ("view_zai_clinical", "Clinical maps: gray matter, 37 pairs, |zAI| >= 3, >= 50 voxels (03_){clin}"),
    ]),
    ("Direction-of-abnormality inference (uses #3, CBF z vs controls)", [
        ("view_zscore_control_map", "#3 perfusion z vs controls, the rule's input (|z| >= 1.96){rawz}"),
        ("view_direction_classes", "Clusters by direction class: hypo / hyper / mixed / subtle *{zai}"),
    ]),
    ("Lateralization decision and clinical scope (clinical tier: peak |zAI| >= 4, >= 100 voxels)", [
        ("view_clinical_report", "Patient report * (rule call, clinical-tier clusters, predicted side){zai}"),
        ("view_zai_dominant_clusters", "Top-3 clinical-tier clusters * (table + overlay){zai}"),
        ("view_zai_clinical_roi_mask", "37-region view, |zAI| >= 4 * (top regions + overlay){zai}"),
        ("view_combined_presurgical", "Combined presurgical view (RECOMMENDED){zai}"),
    ]),
    ("Post hoc · enhanced estimator (i) and analysis #3 on its own", [
        ("view_regional_normative", "Regional normative z_LI vs controls + calibrated call *"),
        ("view_zscore_clusters", "#3 hypo / hyper clusters (z beyond 1.96, >= 10 voxels) + region outlines{rawz}"),
    ]),
    ("All layers, one subject (CBF + #3 + #4 + clinical clusters, at the strongest clinical cluster)", [
        ("view_all_layers", "All layers in FSLeyes (template + clinical on; tick the others)"),
        ("render_all_layers_png", "All layers as a PNG (for the IDE; opens in VS Code when `code` is on PATH)"),
    ]),
    ("Misdetected cases: every method against the reference, and why (Results)", [
        ("view_miss_overview", "Every method's misses in this cohort: tables, figures, report *"),
        ("view_miss_case", "This patient: label and its file, each source, each method, why the rule differs *"),
    ]),
    ("Display aid and utilities (not analysis steps)", [
        ("view_zai_cleaned_overlay", "Island-cleaned clinical zAI (|zAI| >= 3, >= 50 voxels) + montage{zai}"),
        ("show_file_list", "File list"),
        ("show_quick_commands", "FSLeyes commands for this patient (copy / paste)"),
    ]),
]
OPTION = {name: str(i) for i, name in enumerate((n for _, items in MENU for n, _ in items), 1)}
# Options that only print (no FSLeyes window to wait for before the menu returns)
PRINT_ONLY = {"show_file_list", "show_quick_commands", "view_clinical_report", "render_all_layers_png",
              "render_reg_png", "view_miss_overview", "view_miss_case"}


class InteractiveOverlayViewer:
    """Interactive viewer for canonical pipeline outputs."""

    def __init__(self, patient_id=None, use_sym=True):
        self.base_dir = Path(".")
        self.patient_id = patient_id
        self.use_sym = use_sym
        self._missing = []      # files a layer builder could not find (reported by _launch)
        self._collect = None    # list -> _launch records commands instead of running them (FSLeyes commands)
        self._focus_label = None  # what the view was centred on (set by _focus, printed by _launch)

        if patient_id:
            self._set_patient_files(patient_id)
        else:
            # Default to P013 if no patient specified
            self._set_patient_files("P013")

    def _set_patient_files(self, patient_id):
        """Set file paths for a specific patient (canonical paths only)."""
        self.patient_id = patient_id

        # ORIGINAL-space raw anatomy & inputs (Dataset/). Used ONLY by the
        # raw-anatomy menu items (1-2); NOT voxel-aligned with the
        # symmetric-space zAI / clinical maps.
        self.freesurfer_files = {
            "T1w": f"Dataset/{patient_id}/T1w_acpc_dc_restore.nii.gz",
            "Parcellation": f"Dataset/{patient_id}/aparc+aseg.nii.gz",
            "Perfusion_Original": f"Dataset/{patient_id}/perfusion_calib.nii.gz",
            "Perfusion_Resampled": f"Dataset/{patient_id}/{patient_id}_perfusion_calib_resampled_to_T1w.nii.gz",
        }

        # Optional legacy original-space folder ("Dataset MNI/<PID>/"); only
        # some patients have it. Every use is guarded by an existence check.
        self.mni_t1w = f"Dataset MNI/{patient_id}/T1w_restore.nii.gz"
        self.mni_perfusion = f"Dataset MNI/{patient_id}/perfusion_calib.nii.gz"
        self.mni_parcellation = f"Dataset MNI/{patient_id}/aparc+aseg.nii.gz"

        # Underlay anatomy for zAI / clinical maps. The canonical
        # results_zscore/ maps are in the left-right SYMMETRIC template space,
        # so default to the symmetric T1 template + the patient's aparc warped
        # into it. Fall back to original space only with --orig or if the
        # symmetric files are missing.
        sym_t1 = SYM_T1
        sym_parc = SYM_APARC_FMT.format(pid=patient_id)
        if not Path(sym_parc).exists() and FIXED_ATLAS:
            sym_parc = FIXED_ATLAS        # external cohort: one template atlas for everyone
        if self.use_sym and Path(sym_t1).exists() and Path(sym_parc).exists():
            self.space = "symmetric"
            self.overlay_t1 = sym_t1
            self.overlay_parc = sym_parc
        else:
            self.space = "original"
            self.overlay_t1 = self.freesurfer_files["T1w"]
            self.overlay_parc = self.freesurfer_files["Parcellation"]

        # Pipeline B — raw voxel z-score maps (results_zscore/patients/<PID>/)
        # File pattern: <PID>_vs_<group>_zscore.nii.gz, _hyper_*, _hypo_*
        self.zscore_control_files = {}
        zscore_patient_dir = Path(f"{RESULTS_ROOT}/patients/{patient_id}")
        if zscore_patient_dir.exists():
            for f in sorted(zscore_patient_dir.glob(f"{patient_id}_vs_*_zscore.nii.gz")):
                fname = f.name
                group_key = fname.replace(f"{patient_id}_vs_", "").replace("_zscore.nii.gz", "")
                prefix = f"{patient_id}_vs_{group_key}"
                self.zscore_control_files[group_key] = {
                    "zscore": str(f),
                    "hyper": str(zscore_patient_dir / f"{prefix}_hyper_perfusion.nii.gz"),
                    "hypo": str(zscore_patient_dir / f"{prefix}_hypo_perfusion.nii.gz"),
                    "hyper_clusters": str(zscore_patient_dir / f"{prefix}_hyper_clusters.nii.gz"),
                    "hypo_clusters": str(zscore_patient_dir / f"{prefix}_hypo_clusters.nii.gz"),
                }
        self.zscore_group_dir = f"{RESULTS_ROOT}/groups"
        # The patient's control band (one per patient: FM_* in-house, CH_ALL for Chalfont)
        self.band = next(iter(self.zscore_control_files), None)
        self.group_dir = f"{self.zscore_group_dir}/{self.band}" if self.band else None

        # Region outlines and 37-pair masks in symmetric space come from the CONSENSUS parcellation (majority
        # vote of the controls on the template: the GM and scope the analysis used, and it follows the
        # template's cortex). The patient's own warped aparc (symreg/sym_perf/<ID>_aparc_sym) is individual
        # anatomy that a 10-mm warp cannot fit to an average template: up to 8 % of its cortex labels fall
        # outside the template brain, which drew outlines outside the brain (fixed 2026-11-09).
        self.own_parc = sym_parc
        if self.space == "symmetric":
            consensus = FIXED_ATLAS or (f"{self.group_dir}/consensus_parcellation.nii.gz" if self.group_dir else None)
            if consensus and Path(consensus).exists():
                self.overlay_parc = consensus

        # Pipeline B — zAI voxel maps (results_zscore/asymmetry/patients/<PID>/)
        # Canonical filenames: <PID>_asymmetry_zscore.nii.gz,
        #                      <PID>_asymmetry_significant.nii.gz,
        #                      <PID>_asymmetry_left_dominant.nii.gz,
        #                      <PID>_asymmetry_right_dominant.nii.gz
        self.zai_dir = Path(f"{RESULTS_ROOT}/asymmetry/patients/{patient_id}")
        self.zai_files = {
            "asymmetry_zscore": self.zai_dir / f"{patient_id}_asymmetry_zscore.nii.gz",
            "significant": self.zai_dir / f"{patient_id}_asymmetry_significant.nii.gz",
            "left_dominant": self.zai_dir / f"{patient_id}_asymmetry_left_dominant.nii.gz",
            "right_dominant": self.zai_dir / f"{patient_id}_asymmetry_right_dominant.nii.gz",
        }

        # Pipeline B — clinical (gray-matter) maps (results_zscore/clinical/<PID>/)
        self.clinical_dir = Path(f"{RESULTS_ROOT}/clinical/{patient_id}")

    # ------------------------------------------------------------------
    # AVAILABILITY DETECTION
    # ------------------------------------------------------------------

    def _detect_available_data(self):
        """Return dict of which canonical data is available for this patient."""
        pid = self.patient_id
        zai_path = self.zai_dir / f"{pid}_asymmetry_zscore.nii.gz"
        clin_zai = self.clinical_dir / f"{pid}_clinical_zai_lateralized_significant.nii.gz"
        return {
            "anatomy": Path(self.freesurfer_files["T1w"]).exists(),
            "sym_anatomy": self.space == "symmetric",
            "raw_zscore": bool(self.zscore_control_files),
            "zai": zai_path.exists(),
            "clinical": clin_zai.exists() or self.clinical_dir.exists(),
        }

    def _availability_str(self):
        """Compact one-line availability summary for the menu header."""
        avail = self._detect_available_data()
        labels = {
            "anatomy": "orig anatomy",
            "sym_anatomy": "sym-space underlay",
            "raw_zscore": "raw z-score",
            "zai": "zAI",
            "clinical": "clinical",
        }
        parts = []
        for k, lbl in labels.items():
            mark = "[+]" if avail[k] else "[-]"
            parts.append(f"{mark} {lbl}")
        return "  ".join(parts)

    # ------------------------------------------------------------------
    # PATIENT SELECTION
    # ------------------------------------------------------------------

    def get_available_patients(self):
        """Scan canonical zAI directory for patients with data."""
        zai_root = Path(f"{RESULTS_ROOT}/asymmetry/patients")
        anat_root = Path("Dataset")
        patients = set()

        if zai_root.exists():
            for d in sorted(zai_root.iterdir()):
                # in-house P0NN; external D (lesional), N (MRI-negative), C (control pseudo-patients)
                if d.is_dir() and d.name[:1] in "PDNC":
                    if (d / f"{d.name}_asymmetry_zscore.nii.gz").exists():
                        patients.add(d.name)

        # Also include patients with anatomy-only available
        if anat_root.exists():
            for d in sorted(anat_root.iterdir()):
                if d.is_dir() and d.name.startswith("P"):
                    if (d / "T1w_acpc_dc_restore.nii.gz").exists():
                        patients.add(d.name)

        return sorted(patients)

    def select_patient(self):
        """Interactive patient selection."""
        patients = self.get_available_patients()

        if not patients:
            print("[!] No patients found in Dataset/ or results_zscore/asymmetry/patients/.")
            return None

        print("\n" + "=" * 70)
        print("AVAILABLE PATIENTS")
        print("=" * 70)

        zai_root = Path(f"{RESULTS_ROOT}/asymmetry/patients")
        for i, patient in enumerate(patients, 1):
            tags = []
            if (zai_root / patient / f"{patient}_asymmetry_zscore.nii.gz").exists():
                tags.append("zAI")
            if Path(f"Dataset/{patient}/T1w_acpc_dc_restore.nii.gz").exists():
                tags.append("anat")
            ez = EZ_GROUND_TRUTH.get(patient, {})
            ez_tag = f"EZ={ez.get('ez', '?')}" if ez else "EZ=?"
            tag_str = f"[{', '.join(tags) or 'none'}]  {ez_tag}"
            print(f"   {i:2d}. {patient}   {tag_str}")

        print("\n   0. Exit")
        print("=" * 70)

        while True:
            choice = input(
                f"\nSelect patient (1-{len(patients)} or ID, e.g. {patients[0]}) or 0 to exit: "
            ).strip()

            if choice == "0":
                return None
            if choice.upper() in patients:
                choice = str(patients.index(choice.upper()) + 1)

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(patients):
                    selected = patients[idx]
                    print(f"\n[+] Selected patient: {selected}")
                    self._set_patient_files(selected)
                    return selected
                else:
                    print(f"[!] Invalid choice. Please enter 1-{len(patients)} or 0")
            except ValueError:
                print("[!] Invalid input. Please enter a number or a patient ID")

    # ------------------------------------------------------------------
    # FILE-LIST UTILITIES
    # ------------------------------------------------------------------

    def check_files(self):
        """Check which canonical files exist for the current patient."""
        pid = self.patient_id
        all_ok = True

        print("[*] Checking canonical file availability:")

        print(f"\n  zAI underlay ({self.space} space):")
        for name, path in [("T1", self.overlay_t1), ("Parcellation", self.overlay_parc)]:
            mark = "[+]" if Path(path).exists() else "[-]"
            print(f"     {mark} {name}: {path}")
        if self.space == "original":
            print("     [!] zAI maps are in SYMMETRIC space; the original-space underlay "
                  "is not voxel-aligned with them.")

        cbf = AllLayersView(pid, DATASET).files["cbf"] if DATASET in ALL_LAYER_DATASETS else None
        print(f"\n  CBF registered to the template (options {OPTION['view_reg_output']}, {OPTION['view_all_layers']}-{OPTION['render_all_layers_png']}):")
        print(f"     [{'+' if cbf else '-'}] {Path(cbf).relative_to(BASE_DIR) if cbf else 'not found'}")

        if DATASET != "inhouse" and not Path(self.freesurfer_files["T1w"]).exists():
            print("\n  Raw anatomy, ORIGINAL space: in-house only (no Dataset/<ID>/ for this cohort)")
        else:
            print(f"\n  Raw anatomy, ORIGINAL space (Dataset/; menu items {OPTION['view_basic_anatomy']}-{OPTION['view_regional_li']} only):")
            for name, path in self.freesurfer_files.items():
                if Path(path).exists():
                    size_mb = Path(path).stat().st_size / (1024**2)
                    print(f"     [+] {name}: {path} ({size_mb:.1f} MB)")
                else:
                    print(f"     [-] {name}: {path} - NOT FOUND")

        if Path(self.mni_t1w).parent.exists():
            print("\n  Legacy original-space folder (Dataset MNI/):")
            for name, path in [
                ("T1w", self.mni_t1w),
                ("Perfusion", self.mni_perfusion),
                ("Parcellation", self.mni_parcellation),
            ]:
                if Path(path).exists():
                    size_mb = Path(path).stat().st_size / (1024**2)
                    print(f"     [+] {name}: {path} ({size_mb:.1f} MB)")
                else:
                    print(f"     [-] {name}: {path} - NOT FOUND")

        ds_flag = " --dataset chalfont" if DATASET == "chalfont" else ""
        print(f"\n  #3 perfusion z vs controls ({RESULTS_ROOT}/patients/):")
        if self.zscore_control_files:
            for group, files in self.zscore_control_files.items():
                z_path = files["zscore"]
                if Path(z_path).exists():
                    size_mb = Path(z_path).stat().st_size / (1024**2)
                    print(f"     [+] vs {group}: {z_path} ({size_mb:.1f} MB)")
        else:
            print(f"     [-] No perfusion z maps for {pid}")
            print(f"         Run: python 01_build_control_normative.py --sym{ds_flag}")
            all_ok = False

        print(f"\n  #4 zAI maps ({RESULTS_ROOT}/asymmetry/patients/):")
        zai_found = 0
        for name, path in self.zai_files.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024**2)
                print(f"     [+] {name}: {path} ({size_mb:.1f} MB)")
                zai_found += 1
            else:
                print(f"     [-] {name}: {path} - NOT FOUND")
        if zai_found == 0:
            print(f"         Run: python 02_compute_zai.py --sym{ds_flag}")
            all_ok = False

        print(f"\n  Clinical gray-matter zAI ({RESULTS_ROOT}/clinical/):")
        if self.clinical_dir.exists():
            clin_files = sorted(self.clinical_dir.glob(f"{pid}_clinical*.nii.gz"))
            for f in clin_files:
                size_mb = f.stat().st_size / (1024**2)
                print(f"     [+] {f.name} ({size_mb:.1f} MB)")
            if not clin_files:
                print(f"     [-] No clinical NIfTI files in {self.clinical_dir}")
        else:
            print(f"     [-] {self.clinical_dir} not found")
            print(f"         Run: python 03_clinical_maps.py --sym{ds_flag}")

        return all_ok

    # ------------------------------------------------------------------
    # LAYER BUILDERS + ONE LAUNCHER (every FSLeyes option goes through _launch)
    # ------------------------------------------------------------------
    # A builder returns the FSLeyes arguments for one overlay, or [] when the
    # file is missing (recorded in self._missing and reported by _launch).
    # Signed maps use red-yellow (+) / blue-lightblue (-) with the lower
    # display bound as the threshold: |value| below it is transparent, so the
    # threshold printed in the menu is the threshold on screen.

    def _layer(self, path, args, name, hidden=False):
        if not path or not Path(path).exists():
            if path:
                self._missing.append(str(path))
            return []
        return [str(path), *args, "-n", name] + (["-d"] if hidden else [])

    def _zlayer(self, path, lo, hi, name, hidden=False, alpha=None):
        args = ["-cm", "red-yellow", "-nc", "blue-lightblue", "-un", "-dr", f"{lo:g}", f"{hi:g}"]
        if alpha is not None:
            args += ["-a", str(alpha)]
        return self._layer(path, args, name, hidden)

    def _mask_layer(self, path, colour, name, hidden=False, alpha=85):
        """Binary or label map shown in one colour (any value >= 1)."""
        return self._layer(path, ["-cm", colour, "-dr", "0.5", "1", "-a", str(alpha)], name, hidden)

    def _atlas_layer(self, path=None, hidden=True):
        """Parcellation as thin FreeSurfer-coloured outlines."""
        return self._layer(path or self.overlay_parc, ["-ot", "label", "-l", "freesurfercolorlut", "-o"],
                           "regions (outline)" if path else "regions (outline, consensus atlas)", hidden)

    def _background(self):
        """T1 underlay (template, or original T1 with --orig) + the band's control mean CBF, hidden."""
        bg = [str(self.overlay_t1), "-n", f"T1 ({self.space} space)"]
        if self.space == "symmetric":
            bg[1:1] = ["-dr", "0", "0.95"]          # template intensities 0-1.05; brighter cortical ribbon
        if self.group_dir:
            bg += self._layer(self._in_brain(f"{self.group_dir}/mean_perfusion.nii.gz"), [*CBF_ARGS, "-a", "60"],
                              f"control mean CBF ({self.band})", hidden=True)
        return bg

    def _in_brain(self, path):
        """Brain-masked copy for symmetric-space maps that extend over the scalp (#3, its clusters, CBF)."""
        return brain_masked(path, self.overlay_parc) if self.space == "symmetric" else path

    def _peak(self, path, within=None):
        """Voxel (i, j, k) of the largest |value| (optionally inside a mask), or None."""
        if not _HAVE_SCI or not path or not Path(path).exists():
            return None
        d = _np.abs(_np.asanyarray(_nib.load(str(path)).dataobj, dtype=_np.float32))
        if within is not None:
            d = _np.where(within, d, 0)
        if not d.max() > 0:
            return None
        return tuple(int(v) for v in _np.unravel_index(_np.argmax(d), d.shape))

    def _focus(self, fallback_map):
        """The strongest clinical-tier cluster (06_ report), else the peak of fallback_map."""
        voxel, self._focus_label = rule_focus(self.patient_id, DATASET)
        if voxel is None:
            self._focus_label = "peak of the clinical map (no rule report)"
            return self._peak(fallback_map)
        return voxel

    def _launch(self, title, layers, notes=(), centre=None, background=None):
        """Build the FSLeyes command, report skipped layers, launch it and surface start-up errors."""
        argv = ["fsleyes"]
        if centre is not None and (background is not None or self.space == "symmetric"):
            argv += ["-vl", *map(str, centre)]
        argv += (background if background is not None else self._background())
        argv += [a for layer in layers for a in layer]
        missing, self._missing = list(dict.fromkeys(self._missing)), []
        if self._collect is not None:          # FSLeyes-commands option: record, do not run
            self._collect.append((title, argv))
            return argv
        print(f"\n[*] {title}")
        for n in notes:
            print(f"    {n}")
        for m in missing:
            print(f"    [!] missing, layer skipped: {m}")
        if centre is not None:
            print(f"    Opened at voxel {centre}: {getattr(self, '_focus_label', None) or 'peak of the main layer'}.")
        self._focus_label = None
        print("    Hidden layers: tick the eye icon in the Overlay list (bottom left) to show them.")
        print(f"\n  Command (run from the project root):\n  {shlex.join(argv)}\n")
        if shutil.which("fsleyes") is None:
            print("  [!] fsleyes is not on PATH; run the command above where FSL is installed.")
            return argv
        log = Path(tempfile.gettempdir()) / f"fsleyes_{self.patient_id}.log"
        with open(log, "w") as fh:
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=fh, start_new_session=True)
        time.sleep(1.5)
        if proc.poll() not in (None, 0):
            print(f"  [!] FSLeyes exited with code {proc.returncode}; last lines of {log}:")
            for line in log.read_text().splitlines()[-6:]:
                print(f"      {line}")
        else:
            print("  [+] FSLeyes launched.")
        return argv

    def _has_native_anatomy(self):
        if Path(self.freesurfer_files["T1w"]).exists():
            return True
        print(f"[!] No original-space anatomy for {self.patient_id} (Dataset/{self.patient_id}/ is in-house only).")
        print(f"    Chalfont: option {OPTION['view_all_layers']} shows the template, the patient's CBF and every map; "
              f"{OPTION['show_file_list']} lists the files.")
        return False

    def _reference_line(self):
        """Reference side for the header and the printed tables (MDT for in-house, record for Chalfont)."""
        ez = EZ_GROUND_TRUTH.get(self.patient_id, {})
        if not ez:
            return "Reference side: (not in the reference table)"
        if ez.get("confidence"):
            return f"MDT EZ: {ez.get('ez', '?')} ({ez['confidence']} confidence)"
        detail = "; ".join(str(ez[k]) for k in ("group", "ez_lobe", "pathology") if ez.get(k))
        return f"Reference side: {ez.get('ez') or 'none'}" + (f" ({detail})" if detail else "")

    def _pick_zscore_group(self):
        """Pick a control group if multiple are available; return key or None."""
        groups = list(self.zscore_control_files.keys())
        if not groups:
            print("[!] No perfusion z-score (#3) maps found for this patient.")
            print(f"    Run: python 01_build_control_normative.py --sym"
                  f"{' --dataset chalfont' if DATASET == 'chalfont' else ''}")
            return None
        if len(groups) == 1:
            return groups[0]
        print("\n  Available control groups:")
        for i, g in enumerate(groups, 1):
            print(f"    {i}. {g}")
        choice = input(f"  Select group [1-{len(groups)}, default=1]: ").strip()
        idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(groups) else 0
        return groups[idx]

    # ------------------------------------------------------------------
    # INPUTS (original space): T1 + parcellation, CBF
    # ------------------------------------------------------------------

    def view_basic_anatomy(self):
        """T1 + FreeSurfer parcellation in ORIGINAL space (in-house only; not aligned with zAI)."""
        if not self._has_native_anatomy():
            return
        f = self.freesurfer_files
        self._launch(f"RAW ANATOMY, original space — {self.patient_id}",
                     [self._atlas_layer(f["Parcellation"], hidden=False)],
                     notes=["Original space: NOT voxel-aligned with the symmetric-space maps (registration section onward)."],
                     background=[f["T1w"], "-n", "T1 (original space)"])

    def view_perfusion_comparison(self):
        """The patient's CBF on the T1, ORIGINAL space (in-house only; not aligned with zAI)."""
        if not self._has_native_anatomy():
            return
        f = self.freesurfer_files
        self._launch(f"RAW PERFUSION, original space — {self.patient_id}",
                     [self._layer(f["Perfusion_Resampled"], [*CBF_ARGS, "-a", "60"], "CBF (ml/100 g/min)"),
                      self._layer(self.mni_perfusion, ["-cm", "cool", "-dr", "0", "80", "-a", "40"],
                                  "CBF, legacy Dataset MNI copy", hidden=True)],
                     notes=["Original space: NOT voxel-aligned with the symmetric-space maps.",
                            f"Option {OPTION['view_reg_output']} shows the same CBF after registration to the symmetric template."],
                     background=[f["T1w"], "-n", "T1 (original space)"])

    # ------------------------------------------------------------------
    # ANALYSIS #1: regional laterality index (Pipeline A); post hoc regional normative z (enhanced estimator i)
    # ------------------------------------------------------------------

    @staticmethod
    def _pair_name(left_label):
        return SUBCORTICAL_NAMES.get(left_label) or DK_CORTICAL_NAMES[left_label - 1001]

    def _regional_table(self):
        """LI = (L - R) / (L + R) of the 37 region pairs: (table, source, parcellation, background) or None.

        In-house: Pipeline A's table (results_analysis/<ID>/<ID>_comprehensive_asymmetry_analysis.csv: the
        patient's own FreeSurfer parcellation on the MNI grid, the input of 05_roi_discrimination), shown in
        original space. Chalfont has no FreeSurfer parcellation (a deviation stated in the Methods): the regional
        means of the enhanced estimator on the fixed template atlas (v2_features/chalfont/<ID>_roi.json).
        """
        pid = self.patient_id
        if DATASET == "inhouse":
            csv = Path(f"results_analysis/{pid}/{pid}_comprehensive_asymmetry_analysis.csv")
            if not csv.exists():
                print(f"[!] No Pipeline A table for {pid}: {csv}")
                print(f"    Run: python scripts/build_comprehensive_asymmetry.py {pid}")
                return None
            if not self._has_native_anatomy():
                return None
            df = _pd.read_csv(csv).rename(columns={"left_segid": "left", "right_segid": "right"})
            return (df, csv, self.freesurfer_files["Parcellation"],
                    [self.freesurfer_files["T1w"], "-n", "T1 (original space)"])
        roi = Path(f"v2_features/{DATASET}/{pid}_roi.json")
        if not roi.exists() or not FIXED_ATLAS:
            print(f"[!] No regional means for {pid}: {roi}")
            print(f"    Run: python scripts/v2/prep_subject_features.py --dataset {DATASET}")
            return None
        v2 = load_script("scripts/v2/analyze_v2.py")
        li = v2.li_vector(v2.load_roi(roi.parent, pid))
        df = _pd.DataFrame([(l, r, self._pair_name(l), li[n]) for l, r, n in v2.PAIRS],
                           columns=["left", "right", "region", "laterality_index"])
        return df, roi, FIXED_ATLAS, None

    def view_regional_li(self, display=(0.01, 0.15)):
        """Analysis #1 (Pipeline A): the regional LI of each of the 37 pairs, painted on the parcellation."""
        if not _HAVE_SCI:
            print("[!] numpy/nibabel/pandas needed (conda activate Asym).")
            return
        got = self._regional_table()
        if got is None:
            return
        df, source, parc, background = got
        pid = self.patient_id
        df = df.assign(abs_li=df["laterality_index"].abs()).sort_values("abs_li", ascending=False)
        print()
        print("=" * 70)
        print(f"  ANALYSIS #1 · REGIONAL LI = (L - R) / (L + R) for {pid}   ({source})")
        print(f"  {self._reference_line()}")
        print("  Negative = left lower. Largest |LI| first:")
        for r in df.head(10).itertuples():
            print(f"    {r.region:28s} LI {r.laterality_index:+.3f}   lower: {'L' if r.laterality_index < 0 else 'R'}")
        try:   # the aggregation scores of 05_roi_discrimination (Methods: nine scores per patient)
            mdt = load_script("05_roi_discrimination.py").EZDiscriminatorMDT()
            scores = {"signed_weighted_li": mdt.signed_weighted_li(df), "temporal_signed_li": mdt.temporal_signed_li(df),
                      "mesial_temporal_li": mdt.mesial_temporal_li(df), "mean_li": mdt.mean_li(df),
                      "anatomically_weighted": mdt.anatomically_weighted(df),
                      "temporal_focused_score": mdt.temporal_focused_score(df),
                      "dominant_side": mdt.dominant_side_score(df)}
            print("  Aggregation scores (05_; each score's orientation was chosen on the cohort, so a sign is not a call):")
            for k, v in scores.items():
                print(f"    {k:24s} {v:+.3f}")
        except Exception as e:                  # noqa: BLE001 - the map is still shown
            print(f"  (aggregation scores not available: {e})")
        print("=" * 70)
        lo, hi = display
        painted = paint_regions({(int(r.left), int(r.right)): float(r.laterality_index) for r in df.itertuples()},
                                parc, f"{pid}_{DATASET}_regional_li.nii.gz")
        self._launch(f"ANALYSIS #1 · REGIONAL LI (37 pairs) — {pid}",
                     [self._zlayer(painted, lo, hi, f"regional LI (|LI| >= {lo:g}; each region minus its homologue)"),
                      self._atlas_layer(parc)],
                     notes=["Each region carries LI of its pair, the right member with the opposite sign: blue = lower than "
                            "its homologue, red = higher (the zAI convention).",
                            f"|LI| < {lo:g} transparent; colour saturates at |LI| = {hi:g}.",
                            "In-house: own FreeSurfer parcellation, original (MNI) space." if DATASET == "inhouse" else
                            "Chalfont: no FreeSurfer; fixed template atlas in symmetric space."],
                     background=background)

    def regional_normative(self, feat):
        """z_LI per pair against the band's controls and its means over the mesial, temporal and all-37 sets.

        The computation of scripts/v2/analyze_v2.py (li_vector, zscore_frame, MESIAL / TEMPORAL / PAIRS), so the
        means equal the R_mesial / R_temporal / R_all37 columns of ez_analysis_mdt/v2/<dataset>/R_patients.csv.
        """
        import dataset_config
        v2 = load_script("scripts/v2/analyze_v2.py")
        ctrl = [c for c in dataset_config.configure(DATASET).control_ids(self.band) if (feat / f"{c}_roi.json").exists()]
        ctrl_li = _pd.DataFrame([v2.li_vector(v2.load_roi(feat, c)) for c in ctrl], index=ctrl)
        z = v2.zscore_frame(ctrl_li, _pd.Series(v2.li_vector(v2.load_roi(feat, self.patient_id))))
        sets = {"mesial": v2.MESIAL, "temporal": v2.TEMPORAL, "all 37": [n for _, _, n in v2.PAIRS]}
        return z, {k: float(_np.nanmean([z[n] for n in s])) for k, s in sets.items()}, ctrl

    def view_regional_normative(self, display=(1.0, 3.0)):
        """Enhanced estimator (i), post hoc: z_LI of each pair against the band's controls, and the calibrated call."""
        if not _HAVE_SCI:
            print("[!] numpy/nibabel/pandas needed (conda activate Asym).")
            return
        pid, feat = self.patient_id, Path(f"v2_features/{DATASET}")
        if not (feat / f"{pid}_roi.json").exists() or not self.band or self.space != "symmetric":
            print(f"[!] No regional features for {pid} ({feat}/{pid}_roi.json) or no control band.")
            print(f"    Run: python scripts/v2/prep_subject_features.py --dataset {DATASET}")
            return
        v2 = load_script("scripts/v2/analyze_v2.py")
        z, zbar, ctrl = self.regional_normative(feat)
        summ = Path(f"ez_analysis_mdt/v2/{DATASET}/summary.csv")
        thr = None
        if summ.exists():
            row = _pd.read_csv(summ).query("statistic == 'R_temporal'")
            thr = float(row["threshold"].iloc[0]) if len(row) else None
        print()
        print("=" * 70)
        print(f"  ENHANCED ESTIMATOR (i) · REGIONAL NORMATIVE z_LI for {pid} vs {len(ctrl)} controls ({self.band})")
        print(f"  {self._reference_line()}")
        for k, v in zbar.items():
            print(f"    mean z_LI, {k:8s} {v:+.2f}   sign -> {'L' if v < 0 else 'R'}")
        if thr is not None:
            t = zbar["temporal"]
            call = ("L" if t < 0 else "R") if abs(t) >= thr else "no call"
            print(f"  Calibrated call (temporal, |mean z_LI| >= {thr:.2f} = 95th percentile of leave-one-out controls, "
                  f"5 % FP): {call}")
        print("  Temporal pairs:")
        for l, r, n in v2.PAIRS:
            if n in v2.TEMPORAL:
                print(f"    {self._pair_name(l):28s} z_LI {z[n]:+.2f}")
        print("=" * 70)
        lo, hi = display
        painted = paint_regions({(l, r): float(z[n]) for l, r, n in v2.PAIRS}, self.overlay_parc,
                                f"{pid}_{DATASET}_regional_zli.nii.gz")
        self._launch(f"ENHANCED ESTIMATOR (i) · REGIONAL NORMATIVE z_LI — {pid}",
                     [self._zlayer(painted, lo, hi, f"regional z_LI vs controls (|z| >= {lo:g})"),
                      self._atlas_layer()],
                     notes=["z_LI = (LI - control mean) / control SD per pair (scripts/v2/analyze_v2.py); the right member "
                            "carries the opposite sign: blue = lower than its homologue relative to the controls.",
                            f"|z_LI| < {lo:g} transparent. The statistic is the mean over the set, not a regional threshold.",
                            "Post hoc (Methods: enhanced estimators); the same controls give the calibrated threshold."])

    # ------------------------------------------------------------------
    # #3 PERFUSION z vs CONTROLS (the direction rule's input; post hoc on its own)
    # ------------------------------------------------------------------

    def view_zscore_control_map(self):
        """#3 perfusion z against the patient's control band, |z| >= 1.96."""
        group = self._pick_zscore_group()
        if not group:
            return
        notes = ["Red-yellow = CBF above the controls (hyper), blue = below (hypo); |z| < 1.96 transparent.",
                 "One-sided against the controls: a global CBF offset between scanners shifts the whole map.",
                 "Shown inside the brain only (consensus parcellation); the map on disk also covers the scalp."]
        if DATASET == "inhouse":
            notes.append("In-house controls are HCP subjects scanned elsewhere, so read #3 with that offset in mind.")
        notes.append("The direction rule takes the median of this map over each zAI cluster and over its mirror "
                     f"(|median| >= 2 = deviating side): option {OPTION['view_direction_classes']}.")
        self._launch(f"#3 PERFUSION z vs CONTROLS ({group}) — {self.patient_id}",
                     [self._zlayer(self._in_brain(self.zscore_control_files[group]["zscore"]), 1.96, 6,
                                   f"#3 perfusion z vs {group} (|z| >= 1.96)"),
                      self._atlas_layer()],
                     notes=notes)

    DIRECTION_COLOURS = {c: fsl for c, (fsl, _) in DIRECTION_CLASSES.items()}

    def direction_class_maps(self):
        """Per-class masks of the 02_ clusters on their predicted-side member, the other members, and the table.

        See direction_classes(); a pair is drawn once, on the member in the predicted hemisphere
        (cluster_hemi == ez_side_pred), the member that counts and votes. Returns (masks, other, table, n_mismatch).
        """
        cls_vol, other, ref, df, mismatch = direction_classes(self.zai_dir, self.patient_id)
        masks = {c: save_view_map(f"{self.patient_id}_{DATASET}_direction_{c}.nii.gz", cls_vol == i, ref)
                 for i, c in enumerate(DIRECTION_CLASSES, 1) if (cls_vol == i).any()}
        other = save_view_map(f"{self.patient_id}_{DATASET}_direction_other_member.nii.gz", other, ref)
        pairs = df[df["cluster_hemi"] == df["ez_side_pred"]]
        table = (_pd.crosstab(pairs["direction_class"], pairs["ez_side_pred"])
                 .reindex(list(DIRECTION_CLASSES)).fillna(0).astype(int))
        return masks, other, table, mismatch

    def view_direction_classes(self):
        """Direction-of-abnormality inference (02_): every cluster pair coloured by its class and predicted side."""
        if not self._check_zai() or not _HAVE_SCI:
            return
        rep = self.zai_dir / f"{self.patient_id}_asymmetry_cluster_report.csv"
        if not rep.exists() or "direction_class" not in _pd.read_csv(rep, nrows=1).columns:
            print(f"[!] No direction columns in {rep}. Run: python 02_compute_zai.py --sym"
                  f"{' --dataset chalfont' if DATASET == 'chalfont' else ''}")
            return
        masks, other, table, mismatch = self.direction_class_maps()
        print()
        print("=" * 70)
        print(f"  DIRECTION CLASSES for {self.patient_id} (cluster pairs at |zAI| >= 1.96, >= 50 voxels; one member each)")
        print(f"  {self._reference_line()}")
        print(f"    {'class':8s} {'L':>5s} {'R':>5s}   colour")
        for c, row in table.iterrows():
            print(f"    {c:8s} {row.get('L', 0):5d} {row.get('R', 0):5d}   {self.DIRECTION_COLOURS[c]}")
        if mismatch:
            print(f"  [!] {mismatch} report rows did not match a cluster of the saved masks (not drawn).")
        layers = [self._mask_layer(path, self.DIRECTION_COLOURS[c], f"{c} (predicted-side member)")
                  for c, path in masks.items()]
        layers.append(self._mask_layer(other, "greyscale", "mirror members (not voting)", hidden=True, alpha=50))
        notes = ["Per cluster and its mirror, the median #3 z on each side: one side |z| >= 2 -> hypo or hyper on that "
                 "side; both -> mixed; neither -> subtle (side of the larger |z|). The EZ is the deviating side.",
                 "Colours: " + ", ".join(f"{c} = {col}" for c, col in self.DIRECTION_COLOURS.items())
                 + "; each pair is drawn on the member in the predicted hemisphere (the one that counts and votes)."]
        if self.band:
            zf = self.zscore_control_files[self.band]["zscore"]
            layers.insert(0, self._zlayer(self._in_brain(zf), 2, 6, "#3 perfusion z (|z| >= 2, the rule's cut)",
                                          hidden=True))
            if self._collect is None:
                off = float(_np.median(self._gm_values(self._in_brain(zf))))
                notes.append(f"Global offset: median #3 z in 37-pair gray matter {off:+.2f}. The class follows this offset "
                             "against the controls (deck: 'The hypo/hyper class tracked the global CBF offset').")
        layers.append(self._atlas_layer())
        self._launch(f"DIRECTION OF THE ABNORMALITY (02_ direction_class) — {self.patient_id}", layers, notes=notes,
                     centre=self._focus(self.zai_files["significant"]))

    def view_zscore_clusters(self):
        """Analysis #3 on its own (post hoc): hypo- and hyperperfusion clusters (01_: z beyond +-1.96, >= 10 voxels)."""
        group = self._pick_zscore_group()
        if not group:
            return
        f = self.zscore_control_files[group]
        notes = ["Red = hyperperfused, blue = hypoperfused clusters (>= 10 voxels, 01_build_control_normative).",
                 "Cluster tables: " + str(Path(f["zscore"]).with_name(f"{self.patient_id}_vs_{group}_cluster_report.csv"))]
        subj = Path(f"ez_analysis_mdt/perfz/{DATASET}/subjects.csv")
        if _HAVE_SCI and subj.exists():   # the post hoc #3 rule: side with more hypoperfused GM voxels (z <= -2)
            row = _pd.read_csv(subj)
            row = row[row["id"] == self.patient_id]
            if len(row):
                hl, hr = int(row["p_raw_hypo_L"].iloc[0]), int(row["p_raw_hypo_R"].iloc[0])
                side = "L" if hl > hr else ("R" if hr > hl else "no side (tie)")
                notes.append(f"#3 rule (Methods: direct perfusion z): hypoperfused 37-pair GM voxels (z <= -2) L {hl:,} vs "
                             f"R {hr:,} -> {side} ({subj}).")
        self._launch(f"ANALYSIS #3 · HYPO / HYPER CLUSTERS vs {group} — {self.patient_id}",
                     [self._mask_layer(self._in_brain(f["hyper_clusters"]), "red", "#3 hyper clusters (z > 1.96)"),
                      self._mask_layer(self._in_brain(f["hypo_clusters"]), "blue", "#3 hypo clusters (z < -1.96)"),
                      self._atlas_layer(hidden=False)],
                     notes=notes)

    # ------------------------------------------------------------------
    # #4 zAI MAP (and the island-cleaned display aid)
    # ------------------------------------------------------------------

    def _check_zai(self):
        """Verify zAI files exist; print remediation if not."""
        zai_path = self.zai_files["asymmetry_zscore"]
        if not zai_path.exists():
            print(f"[!] zAI map not found for {self.patient_id}.")
            print(f"    Expected: {zai_path}")
            print(f"    Run: python 02_compute_zai.py --sym{' --dataset chalfont' if DATASET == 'chalfont' else ''}")
            return False
        return True

    def _gm_values(self, path, labels=CLINICAL_ROI_LABELS):
        """Non-zero values of a symmetric-space map inside the given labels of the consensus parcellation."""
        d = _np.asanyarray(_nib.load(str(path)).dataobj, dtype=_np.float32)
        m = _np.isin(_np.asanyarray(_nib.load(str(self.overlay_parc)).dataobj).astype(_np.int32), list(labels))
        return d[m & (d != 0)]

    def _one_voxel(self, voxel):
        """AI, control mean and SD of AI, and zAI at one voxel: the arithmetic of Eq. zAI at the opened voxel."""
        cbf = AllLayersView(self.patient_id, DATASET).files["cbf"]
        g = Path(f"{RESULTS_ROOT}/asymmetry/groups/{self.band}")
        if voxel is None or not cbf or not self.band or not (g / "sd_asymmetry.nii.gz").exists():
            return None
        val = lambda f: float(_np.asanyarray(_nib.load(str(f)).dataobj)[tuple(voxel)])   # noqa: E731
        ai, mu, sd = (val(voxel_ai(cbf, self.overlay_parc, f"{self.patient_id}_{DATASET}")),
                      val(g / "mean_asymmetry.nii.gz"), val(g / "sd_asymmetry.nii.gz"))
        z = val(self.zai_files["asymmetry_zscore"]) if self.zai_files["asymmetry_zscore"].exists() else None
        return ai, mu, sd, z

    def view_voxel_ai(self, display=(10, 100)):
        """Analysis #2: the voxel asymmetry index of the patient alone, before any control is involved."""
        if not _HAVE_SCI or self.space != "symmetric":
            print("[!] Needs numpy/nibabel and the symmetric template (conda activate Asym; no --orig).")
            return
        cbf = AllLayersView(self.patient_id, DATASET).files["cbf"]
        if not cbf:
            print(f"[!] No CBF on the symmetric template for {self.patient_id} (symreg/sym_perf or "
                  "external_chalfont/sym_perf).")
            return
        ai = voxel_ai(cbf, self.overlay_parc, f"{self.patient_id}_{DATASET}")
        lo, hi = display
        notes = ["AI(v) = 200 (CBF(v) - CBF(mirror)) / (CBF(v) + CBF(mirror)) (02_compute_zai.compute_asymmetry_map), "
                 "zero unless both voxels have positive CBF; computed for display, 02_ does not save it.",
                 f"Red = voxel above its mirror, blue = below; |AI| < {lo:g} % transparent (the thesis ROI cut-off), "
                 f"saturates at {hi:g} %. Antisymmetric: every finding appears twice."]
        if self._collect is None:
            g = _np.abs(self._gm_values(ai))
            notes.append(f"37-pair gray matter: median |AI| {_np.median(g):.0f} %, {100 * (g >= lo).mean():.0f} % of voxels "
                         f"at |AI| >= {lo:g} %. No control is involved yet: option {OPTION['view_control_normative']} shows "
                         f"what the controls give, {OPTION['view_zai_asymmetry_map']} the z.")
        centre = self._focus(self.zai_files["significant"])
        if centre is not None and self._collect is None:
            notes.append(f"At the opened voxel: AI {self._one_voxel(centre)[0]:+.1f} %. Options "
                         f"{OPTION['view_control_normative']} and {OPTION['view_zai_asymmetry_map']} open on the same voxel.")
        self._launch(f"ANALYSIS #2 · VOXEL AI (patient alone) — {self.patient_id}",
                     [self._layer(self._in_brain(cbf), [*CBF_ARGS, "-a", "60"], "CBF on T_sym", hidden=True),
                      self._zlayer(ai, lo, hi, f"#2 voxel AI (%, |AI| >= {lo:g})"),
                      self._atlas_layer()],
                     notes=notes, centre=centre)

    def view_control_normative(self):
        """Control normative of the asymmetry index: per-voxel mean and SD over the band's controls (02_)."""
        if not _HAVE_SCI or not self.band:
            print("[!] Needs numpy/nibabel and a control band for this patient.")
            return
        g = Path(f"{RESULTS_ROOT}/asymmetry/groups/{self.band}")
        mean_ai, sd_ai = g / "mean_asymmetry.nii.gz", g / "sd_asymmetry.nii.gz"
        if not sd_ai.exists():
            print(f"[!] No control normative in {g}. Run: python 02_compute_zai.py --sym"
                  f"{' --dataset chalfont' if DATASET == 'chalfont' else ''}")
            return
        import json
        meta = g / "metadata.json"
        n = json.loads(meta.read_text()).get("n_subjects", "?") if meta.exists() else "?"
        notes = [f"{n} controls ({self.band}); per-voxel mean and SD of AI (Welford), SD floored at 5 % AI and "
                 "max-pooled with the mirror voxel (02_compute_zai.build_group_asymmetry_stats).",
                 f"zAI(v) = (AI(v) - mean(v)) / SD(v): option {OPTION['view_zai_asymmetry_map']}. Hidden: the control mean "
                 "AI and the SD of CBF (#3's normative; the mean CBF sits under the template)."]
        centre = self._focus(self.zai_files["significant"])
        if self._collect is None:
            sd = self._gm_values(sd_ai)
            notes.insert(1, f"Median control SD in 37-pair gray matter: {_np.median(sd):.0f} % AI, so a voxel needs "
                            f"|AI| of about {1.96 * _np.median(sd):.0f} % above the control mean to reach |zAI| = 1.96.")
            one = self._one_voxel(centre)
            if one:
                notes.append(f"At the opened voxel: control mean {one[1]:+.1f} %, SD {one[2]:.1f} % AI.")
        self._launch(f"CONTROL NORMATIVE OF AI ({self.band}) — {self.patient_id}",
                     [self._layer(self._in_brain(sd_ai), ["-cm", "hot", "-dr", "20", "100", "-cr", "1", "100000",
                                                          "-a", "80"], "control SD of AI (%)"),
                      self._zlayer(self._in_brain(mean_ai), 5, 30, "control mean AI (%, |mean| >= 5)", hidden=True),
                      self._layer(self._in_brain(f"{self.group_dir}/sd_perfusion.nii.gz"),
                                  ["-cm", "hot", "-dr", "5", "40", "-cr", "1", "100000", "-a", "80"],
                                  "control SD of CBF (#3 normative)", hidden=True),
                      self._atlas_layer()],
                     notes=notes, centre=centre)

    def view_zai_asymmetry_map(self):
        """#4 zAI, voxel vs mirror, shown from |zAI| >= 1.96 (the cluster-forming threshold), with its calibration."""
        if not self._check_zai():
            return
        zai = self.zai_files["asymmetry_zscore"]
        notes = ["zAI(v) = (AI(v) - control mean) / control SD; red-yellow = voxel above its mirror, blue = below; "
                 "|zAI| < 1.96 transparent.",
                 "Antisymmetric: every finding appears twice (red on one side, blue on the mirror).",
                 "To see every value, lower the display range of the zAI layer in FSLeyes."]
        if self._collect is None and _HAVE_SCI and self.space == "symmetric":
            # calibration as in the Methods (symreg/check_full_calibration.py): gray-matter voxels at |zAI| >= 1.96
            g = _np.abs(self._gm_values(zai, load_script("symreg/check_full_calibration.py").GM))
            cohort = "in-house patients 12.6 %, leave-one-out controls 6.9 %"
            csv = Path(f"{RESULTS_ROOT}/calibration/patient_gm_calibration.csv")
            if DATASET != "inhouse" and csv.exists():
                cohort = f"{DATASET} patients {_pd.read_csv(csv)['pct_abs_zai_ge_1_96'].mean():.1f} %"
            notes.append(f"Calibration: {100 * (g >= 1.96).mean():.1f} % of gray-matter voxels at |zAI| >= 1.96 "
                         f"(nominal 5 %; cohort mean: {cohort}).")
        centre = self._focus(self.zai_files["significant"])
        if self._collect is None and _HAVE_SCI:
            one = self._one_voxel(centre)
            if one and one[3] is not None:
                ai, mu, sd, z = one
                notes.append(f"At the opened voxel: zAI = (AI {ai:+.1f} - mean {mu:+.1f}) / SD {sd:.1f} = {(ai - mu) / sd:+.2f} "
                             f"(map value {z:+.2f}; clipped at +-20).")
        self._launch(f"#4 zAI MAP (voxel vs mirror, control-referenced) — {self.patient_id}",
                     [self._zlayer(zai, 1.96, 6, "zAI (|zAI| >= 1.96)"),
                      self._atlas_layer()],
                     notes=notes, centre=centre)

    def view_zai_cleaned_overlay(self):
        """Island-cleaned zAI (clean_overlay.py: |zAI| >= 3, >= 50 voxels, 37-ROI scope)."""
        if not self._check_zai():
            return
        import importlib.util as _ilu
        import sys as _sys
        co_path = Path(__file__).parent / "clean_overlay.py"
        if not co_path.exists():
            print("[!] clean_overlay.py not found; cannot generate cleaned overlay.")
            return
        try:
            spec = _ilu.spec_from_file_location("clean_overlay", str(co_path))
            co = _ilu.module_from_spec(spec)
            _sys.modules["clean_overlay"] = co
            spec.loader.exec_module(co)
            co.set_dataset(RESULTS_ROOT, self.overlay_parc if self.space == "symmetric" else FIXED_ATLAS)
        except Exception as e:
            print(f"[!] Could not load clean_overlay.py: {e}")
            return
        print(f"\n[*] ISLAND-CLEANED zAI OVERLAY — {self.patient_id}")
        print("    Thresholding |zAI|>=3, min_cluster=50, opening+closing.")
        report = co.process_patient(self.patient_id, threshold=3.0, min_cluster=50, space="zai",
                                    use_roi_mask=True)
        if not report:
            print("[!] Cleaning failed; see messages above.")
            return
        cleaned = report.get("cleaned_nifti")
        print(f"\n  Islands removed: {report['n_islands_removed']} ({report['voxels_removed']:,} vox)")
        print(f"  Montage: {report.get('montage_png')}")
        print(f"  QC:      {report.get('qc_png')}")
        if not report.get("surviving_voxels"):
            print("  [!] No cluster survives cleaning (|zAI| >= 3, >= 50 voxels after opening); nothing to show.")
            print(f"      Option {OPTION['view_zai_clinical']} shows the uncleaned clinical clusters.")
            return
        self._launch(f"ISLAND-CLEANED zAI — {self.patient_id}",
                     [self._zlayer(cleaned, 3, 8, "zAI cleaned (|zAI| >= 3, >= 50 vox)"),
                      self._atlas_layer()],
                     centre=self._peak(cleaned))


    # ------------------------------------------------------------------
    # THRESHOLDING: clusters at 1.96 (02_), clinical maps (03_)
    # ------------------------------------------------------------------

    def view_zai_significant(self):
        """Significant zAI clusters from 02_ (|zAI| >= 1.96, >= 50 voxels, both pair members)."""
        if not self._check_zai():
            return
        sig = self.zai_files["significant"]
        self._launch(f"SIGNIFICANT zAI CLUSTERS — {self.patient_id}",
                     [self._zlayer(sig, 1.96, 6, "zAI clusters (|zAI| >= 1.96, >= 50 vox)"),
                      self._zlayer(self.zai_files["asymmetry_zscore"], 1.96, 6, "zAI, all voxels", hidden=True),
                      self._mask_layer(self.zai_files["left_dominant"], "red", "sign mask: voxel > mirror", hidden=True),
                      self._mask_layer(self.zai_files["right_dominant"], "blue", "sign mask: voxel < mirror", hidden=True),
                      self._atlas_layer()],
                     notes=["Clusters formed at |zAI| >= 1.96 with >= 50 voxels (02_compute_zai, 18-connected); both "
                            "members shown.",
                            "Hidden: the sign masks (<ID>_asymmetry_left/right_dominant; despite the names not "
                            f"hemisphere-specific). The predicted side comes from the direction rule (option "
                            f"{OPTION['view_direction_classes']}).",
                            f"The clinical tier keeps a cluster whose PEAK reaches 4 (options {OPTION['view_clinical_report']}, "
                            f"{OPTION['view_zai_dominant_clusters']}); the clinical maps ({OPTION['view_zai_clinical']}) use a voxel threshold of 3."],
                     centre=self._peak(sig))

    def view_zai_clinical(self):
        """Clinical maps (03_): gray matter, 37-pair scope, |zAI| >= 3, >= 50 voxels, both members of each pair."""
        cdir, pid = self.clinical_dir, self.patient_id
        both = cdir / f"{pid}_clinical_zai_significant_clusters.nii.gz"
        sig = cdir / f"{pid}_clinical_zai_lateralized_significant.nii.gz"
        left = cdir / f"{pid}_clinical_zai_lateralized_left_dominant.nii.gz"
        right = cdir / f"{pid}_clinical_zai_lateralized_right_dominant.nii.gz"
        if not (both.exists() or sig.exists()):
            print(f"[!] Clinical zAI maps not found in {cdir}")
            print(f"    Run: python 03_clinical_maps.py --sym{' --dataset chalfont' if DATASET == 'chalfont' else ''}")
            return
        self._launch(f"CLINICAL zAI (gray matter, 37 region pairs) — {pid}",
                     [self._zlayer(both, 3, 8, "clinical zAI clusters, both members (|zAI| >= 3)"),
                      self._zlayer(sig, 3, 8, "clinical zAI, higher member only", hidden=True),
                      self._mask_layer(left, "red", "higher on L (L > R)", hidden=True),
                      self._mask_layer(right, "blue", "higher on R (R > L)", hidden=True),
                      self._atlas_layer()],
                     notes=["Both members of each clinical pair (|zAI| >= 3, >= 50 voxels): red = above its mirror, "
                            "blue = below.",
                            "The crosshair is on the member the rule assigns to the EZ side (hypo = the blue member).",
                            "Hidden: the same clusters on the higher member only, and split by side (L > R, R > L).",
                            f"Option {OPTION['view_clinical_report']} gives the patient-level call."],
                     centre=self._focus(both if both.exists() else sig))

    # ------------------------------------------------------------------
    # LATERALIZATION DECISION: top-N clinical-tier clusters, 37-region view, combined view
    # ------------------------------------------------------------------

    def view_zai_dominant_clusters(self, top_n=3, threshold=4.0):
        """Show ONLY the top-N clusters by peak |zAI| in FSLeyes.

        Motivation: the canonical zAI map for some patients (e.g. P013)
        contains 60K+-voxel clusters in non-EZ regions that visually drown
        out a smaller clinically relevant cluster. This view filters to the
        top-N highest-peak clusters and prints their anatomical regions
        + peak values to stdout for clinical context.

        Falls back gracefully if the cluster_labels NIfTI is missing
        (prints regions only).
        """
        if not self._check_zai():
            return
        if not _HAVE_SCI:
            print("[!] This option requires numpy / nibabel / pandas.")
            print("    Install with: pip install numpy nibabel pandas")
            return

        pid = self.patient_id

        # Prefer the gray-matter clinical cluster report (cleaner; matches
        # the patient report view); fall back to the asymmetry report if
        # the clinical one isn't present.
        clin_csv = self.clinical_dir / f"{pid}_clinical_zai_cluster_report.csv"
        asym_csv = self.zai_dir / f"{pid}_asymmetry_cluster_report.csv"

        if clin_csv.exists():
            csv_path = clin_csv
            peak_col, mean_col = "peak_zai", "mean_zai"
            tag = "clinical zAI (gray matter)"
        elif asym_csv.exists():
            csv_path = asym_csv
            peak_col, mean_col = "peak_z", "mean_z"
            tag = "asymmetry zAI (full brain)"
        else:
            print(f"[!] No cluster_report CSV found for {pid}.")
            print(f"    Expected: {clin_csv}")
            print(f"             or {asym_csv}")
            print(f"    Run: python 03_clinical_maps.py --patient {pid}")
            return

        try:
            df = _pd.read_csv(csv_path)
        except Exception as e:
            print(f"[!] Could not read {csv_path}: {e}")
            return

        if peak_col not in df.columns:
            print(f"[!] Expected column '{peak_col}' missing from {csv_path}.")
            print(f"    Found columns: {list(df.columns)}")
            return

        # Filter by absolute peak threshold and sort
        df["abs_peak"] = df[peak_col].abs()
        df_thr = df[df["abs_peak"] >= threshold].copy()
        if df_thr.empty:
            print(f"[!] No clusters with |peak zAI| >= {threshold:g} in {csv_path.name}.")
            print(f"    Lower the threshold or pick option {OPTION['view_zai_asymmetry_map']} / {OPTION['view_zai_significant']} to inspect the full map.")
            return

        df_top = df_thr.nlargest(top_n, "abs_peak").copy()

        # Stdout summary for clinical context
        print()
        print("=" * 70)
        print(f"  TOP-{top_n} DOMINANT zAI CLUSTERS for {pid}  (|zAI| >= {threshold:g})")
        print(f"  Source: {csv_path.name}  ({tag})")
        print(f"  {self._reference_line()}")
        print("  Legend: 'higher on L/R' = hemisphere with MORE perfusion than its")
        print("          mirror (zAI sign combined with cluster hemisphere). This is")
        print("          not the predicted EZ side: see dir/pred (direction_class /")
        print("          ez_side_pred = hemisphere deviating from controls) when present.")
        print("=" * 70)
        for _, row in df_top.iterrows():
            side = _cluster_physical_side(row, mean_col)
            extra = ""
            if "direction_class" in row.index and isinstance(row["direction_class"], str):
                extra += f"  dir={row['direction_class']}"
            if "ez_side_pred" in row.index and isinstance(row["ez_side_pred"], str):
                extra += f"  pred={row['ez_side_pred']}"
            print(
                f"  #{int(row['cluster_id']):3d}  higher on {side}  "
                f"peak={row[peak_col]:+6.2f}  size={int(row['size_voxels']):6d} vox  "
                f"region={row['primary_region']}{extra}"
            )
        print("=" * 70)
        print()

        # Build a binary mask of just those clusters using the cluster
        # labels NIfTI. Prefer clinical labels; fall back to asymmetry
        # labels (rarely present) — if neither, print regions and exit.
        clin_labels = self.clinical_dir / f"{pid}_clinical_zai_cluster_labels.nii.gz"
        asym_labels = self.zai_dir / f"{pid}_asymmetry_cluster_labels.nii.gz"

        labels_path = None
        if clin_labels.exists() and csv_path == clin_csv:
            labels_path = clin_labels
        elif asym_labels.exists() and csv_path == asym_csv:
            labels_path = asym_labels
        elif clin_labels.exists():
            labels_path = clin_labels
        elif asym_labels.exists():
            labels_path = asym_labels

        if labels_path is None:
            print("  [!] No cluster_labels NIfTI found — printed top-N regions only.")
            print(f"      To get a focal FSLeyes overlay, run option {OPTION['view_zai_clinical']} (clinical maps).")
            return

        # Build top-N binary mask
        try:
            labels_nii = _nib.load(str(labels_path))
            labels_data = labels_nii.get_fdata().astype(_np.int32)
        except Exception as e:
            print(f"  [!] Could not load cluster labels {labels_path}: {e}")
            return

        # cluster_id in the CSV indexes left-dominant clusters and right-
        # dominant clusters in separate id-spaces; the labels NIfTI may
        # use signed labels (positive = L-dom, negative = R-dom) or a
        # single id-space. Try both.
        top_ids = df_top["cluster_id"].astype(int).tolist()
        top_dirs = df_top["direction"].tolist() if "direction" in df_top.columns else [None] * len(top_ids)

        # Attempt 1: signed labels
        signed_ids = []
        for cid, direction in zip(top_ids, top_dirs):
            if direction == "right-dominant":
                signed_ids.append(-cid)
            else:
                signed_ids.append(cid)
        mask = _np.isin(labels_data, signed_ids)
        if mask.sum() == 0:
            # Attempt 2: unsigned labels
            mask = _np.isin(labels_data, top_ids)

        if mask.sum() == 0:
            print(f"  [!] None of the top-{top_n} cluster IDs were found in the labels NIfTI.")
            print(f"      Falling back to FSLeyes view of full clinical zAI (option {OPTION['view_zai_clinical']}).")
            self.view_zai_clinical()
            return

        # Save temp mask
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / f"{pid}_top{top_n}_zai_clusters.nii.gz"
        _nib.save(
            _nib.Nifti1Image(
                mask.astype(_np.uint8), labels_nii.affine, labels_nii.header
            ),
            str(tmp_path),
        )
        print(f"  [+] Top-{top_n} mask written: {tmp_path}  ({int(mask.sum())} voxels)")

        # Launch: template + top-N mask (yellow), opened on the strongest of those clusters
        self._launch(f"TOP-{top_n} zAI CLUSTERS — {pid}",
                     [self._zlayer(self.zai_files["asymmetry_zscore"], 1.96, 6, "zAI, all voxels (|zAI| >= 1.96)",
                                   hidden=True),
                      self._mask_layer(tmp_path, "yellow", f"top-{top_n} clusters", alpha=90),
                      self._atlas_layer()],
                     notes=[f"Yellow = the {top_n} clusters listed above (temporary mask {tmp_path})."],
                     centre=self._peak(self.zai_files["asymmetry_zscore"], within=mask))

    def view_zai_clinical_roi_mask(self, threshold=4.0, top_n=5):
        """Restrict zAI display to the 37 clinically-relevant
        Desikan-Killiany regions (3 subcortical pairs + 34 cortical pairs).

        Useful to suppress non-cortical / non-EZ
        regions (e.g. cerebellum, brainstem, ventricles) that FSLeyes
        would otherwise show as noisy zAI signal.

        Also prints the top-N regions (by peak |zAI|) limited to the
        37-ROI mask, so the reader can see the focal asymmetry hits
        without having to scroll through cluster reports.
        """
        if not self._check_zai():
            return
        if not _HAVE_SCI:
            print("[!] This option requires numpy / nibabel / pandas.")
            print("    Install with: pip install numpy nibabel pandas")
            return

        pid = self.patient_id
        parc_path = Path(self.overlay_parc)
        print(f"  Parcellation ({self.space} space): {parc_path}")
        if not parc_path.exists():
            print(f"[!] Parcellation not found: {parc_path}")
            print(f"    Cannot build 37-ROI clinical mask without aparc+aseg.")
            return

        ai_path = self.zai_files["asymmetry_zscore"]

        # Load parcellation + zAI map
        try:
            parc_nii = _nib.load(str(parc_path))
            parc_data = parc_nii.get_fdata().astype(_np.int32)
        except Exception as e:
            print(f"[!] Could not load parcellation {parc_path}: {e}")
            return
        try:
            ai_nii = _nib.load(str(ai_path))
            ai_data = ai_nii.get_fdata()
        except Exception as e:
            print(f"[!] Could not load zAI map {ai_path}: {e}")
            return

        if parc_data.shape != ai_data.shape:
            print(f"[!] Shape mismatch: parc={parc_data.shape}, zAI={ai_data.shape}")
            print("    The 37-ROI mask requires both volumes in the same space.")
            return

        # Build 37-ROI mask
        roi_mask = _np.isin(parc_data, CLINICAL_ROI_LABELS)
        n_roi_vox = int(roi_mask.sum())
        if n_roi_vox == 0:
            print("[!] 37-ROI mask is empty — parcellation labels not in expected ranges.")
            return

        # Apply mask
        ai_masked = _np.where(roi_mask, ai_data, 0.0)

        # Top-N regions by peak |zAI| within the 37-ROI mask

        print()
        print("=" * 70)
        print(f"  37-ROI CLINICAL MASK VIEW for {pid}")
        print(f"  ROIs: {len(CLINICAL_ROI_LABELS)} FreeSurfer labels "
              f"(3 subcortical + 34 cortical pairs = 37)")
        print(f"  Mask voxels: {n_roi_vox:,}  ({100.0 * n_roi_vox / roi_mask.size:.1f}% of volume)")
        print(f"  Display threshold: |zAI| >= {threshold:g}")
        print(f"  {self._reference_line()}")
        print("=" * 70)

        # Compute peak |zAI| per FreeSurfer label and rank
        peak_per_label = []
        for lbl in CLINICAL_ROI_LABELS:
            sel = (parc_data == lbl)
            if not sel.any():
                continue
            vals = ai_data[sel]
            vals = vals[_np.isfinite(vals)]
            if vals.size == 0:
                continue
            peak_idx = _np.argmax(_np.abs(vals))
            peak_v = float(vals[peak_idx])
            mean_v = float(_np.mean(vals[_np.abs(vals) >= threshold])) if (_np.abs(vals) >= threshold).any() else 0.0
            n_thr = int((_np.abs(vals) >= threshold).sum())
            peak_per_label.append((lbl, peak_v, mean_v, n_thr, sel.sum()))

        # Sort by absolute peak descending
        peak_per_label.sort(key=lambda x: abs(x[1]), reverse=True)

        # Inline FreeSurfer Desikan-Killiany label-name table
        # (only the 37 ROI pairs we actually display).
        _DK_CORTICAL_NAMES = [
            "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal",
            "corpuscallosum", "cuneus", "entorhinal", "fusiform",
            "inferiorparietal", "inferiortemporal", "isthmuscingulate",
            "lateraloccipital", "lateralorbitofrontal", "lingual",
            "medialorbitofrontal", "middletemporal", "parahippocampal",
            "paracentral", "parsopercularis", "parsorbitalis",
            "parstriangularis", "pericalcarine", "postcentral",
            "posteriorcingulate", "precentral", "precuneus",
            "rostralanteriorcingulate", "rostralmiddlefrontal",
            "superiorfrontal", "superiorparietal", "superiortemporal",
            "supramarginal", "frontalpole", "temporalpole",
            "transversetemporal", "insula",
        ]
        _FS_NAMES = {
            10: "L-Thalamus-Proper",   49: "R-Thalamus-Proper",
            17: "L-Hippocampus",       53: "R-Hippocampus",
            18: "L-Amygdala",          54: "R-Amygdala",
        }
        for i, nm in enumerate(_DK_CORTICAL_NAMES):
            # FreeSurfer convention: 1001..1035 = ctx-lh-<name>, 2001..2035 = ctx-rh-<name>
            _FS_NAMES[1001 + i] = f"ctx-lh-{nm}"
            _FS_NAMES[2001 + i] = f"ctx-rh-{nm}"

        print(f"  Top {top_n} ROIs by peak |zAI|:")
        shown = 0
        for lbl, peak_v, mean_v, n_thr, n_vox in peak_per_label:
            if shown >= top_n:
                break
            if lbl in (10, 17, 18) or 1001 <= lbl <= 1035:
                side = "LEFT"
            else:  # 49, 53, 54 or 2001-2035
                side = "RIGHT"
            # ROI hemisphere + zAI sign -> side with higher perfusion
            higher = side[0] if peak_v > 0 else ("R" if side == "LEFT" else "L")
            name = _FS_NAMES.get(lbl, f"label-{lbl}")
            tag = "(suprathr.)" if n_thr > 0 else "(subthr.)"
            print(
                f"    {side:5s}  peak={peak_v:+6.2f} (higher on {higher})  "
                f"#vox >= {threshold:g}: {n_thr:6d}  "
                f"{tag:12s}  {name} (label {lbl})"
            )
            shown += 1
        if shown == 0:
            print("    (no ROI had any non-zero zAI)")
        print("=" * 70)
        print()

        # Save masked zAI map
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / f"{pid}_zai_37roi_masked.nii.gz"
        _nib.save(
            _nib.Nifti1Image(ai_masked.astype(_np.float32), ai_nii.affine, ai_nii.header),
            str(tmp_path),
        )
        print(f"  [+] 37-ROI masked zAI written: {tmp_path}")

        self._launch(f"37-ROI zAI VIEW — {pid}",
                     [self._zlayer(tmp_path, threshold, 8, f"zAI in the 37 ROI pairs (|zAI| >= {threshold:g})"),
                      self._atlas_layer()],
                     notes=[f"zAI restricted to the 37 region pairs, shown from |zAI| >= {threshold:g} "
                            f"(temporary map {tmp_path})."],
                     centre=self._peak(tmp_path))

    def view_combined_presurgical(self):
        """ZAI map + clinical clusters + side masks + regions, opened on the clinical peak."""
        if not self._check_zai():
            return
        cdir, pid = self.clinical_dir, self.patient_id
        both = cdir / f"{pid}_clinical_zai_significant_clusters.nii.gz"
        left = cdir / f"{pid}_clinical_zai_lateralized_left_dominant.nii.gz"
        right = cdir / f"{pid}_clinical_zai_lateralized_right_dominant.nii.gz"
        group = self.band
        layers = [self._zlayer(self.zai_files["asymmetry_zscore"], 1.96, 6, "zAI (|zAI| >= 1.96)", alpha=45),
                  self._zlayer(both, 3, 8, "clinical zAI clusters, both members (|zAI| >= 3)"),
                  self._mask_layer(left, "red", "higher on L (L > R)", hidden=True),
                  self._mask_layer(right, "blue", "higher on R (R > L)", hidden=True)]
        if group:
            layers.insert(0, self._zlayer(self._in_brain(self.zscore_control_files[group]["zscore"]), 1.96, 6,
                                          f"#3 perfusion z vs {group}", hidden=True))
        layers.append(self._atlas_layer(hidden=False))
        self._launch(f"COMBINED PRESURGICAL VIEW — {pid}", layers,
                     notes=[self._reference_line(),
                            "Visible: zAI >= 1.96 (faded), clinical clusters (|zAI| >= 3, both members), region outlines.",
                            "Hidden: #3 perfusion z, control mean CBF, higher-on-L / higher-on-R masks.",
                            "The crosshair is on the member the rule assigns to the EZ side (hypo = the blue member)."],
                     centre=self._focus(both) or self._peak(self.zai_files["significant"]))

    # ------------------------------------------------------------------
    # UTILITIES: file list, FSLeyes commands
    # ------------------------------------------------------------------

    def show_file_list(self):
        """Show detailed canonical-file inventory for the current patient."""
        print("\n[*] CANONICAL FILE INVENTORY")
        print("=" * 70)
        self.check_files()

    def show_quick_commands(self):
        """The FSLeyes commands of the main map views and the all-layer view, in menu order, for copy/paste."""
        got, self._collect = [], []
        try:
            for name in ("view_zai_asymmetry_map", "view_zai_significant", "view_zai_clinical",
                         "view_zscore_control_map", "view_combined_presurgical"):
                n0 = len(self._collect)
                getattr(self, name)()
                got += [(OPTION[name], title, argv) for title, argv in self._collect[n0:]]
        finally:
            self._collect = None
        print("\n[*] FSLeyes COMMANDS for " + self.patient_id + " (run from the project root)")
        print("=" * 70)
        for num, title, argv in got:
            print(f"# option {num}: {title}\n{shlex.join(argv)}\n")
        if _HAVE_SCI:
            v = AllLayersView(self.patient_id, DATASET)
            voxel, where = v.focus()
            print(f"# ALL LAYERS (option {OPTION['view_all_layers']}: CBF, #3, #4, clinical, atlas; opened at {where})")
            print(shlex.join(v.fsleyes_command(voxel)) + "\n")
        print(f"# PNG of all layers for the IDE (option {OPTION['render_all_layers_png']})")
        print(f"python interactive_viewer.py --subject {self.patient_id} --png --open\n")

    # ------------------------------------------------------------------
    # PATIENT REPORT (06_clinical_interpretation)
    # ------------------------------------------------------------------
    #
    # Per-patient report. Imports the analyzer from
    # 06_clinical_interpretation.py, runs it for the current patient with
    # the canonical |zAI|>=4 / size>=100 thresholds, prints the report
    # to stdout, and (if FSLeyes is available + the cluster_labels NIfTI
    # is on disk) offers to launch the top-N cluster overlay (the top-3
    # view) so the reader can flip from the deterministic answer to
    # the spatial view in one keystroke.

    def view_clinical_report(self):
        """Run 06_clinical_interpretation.py for the current patient."""
        pid = self.patient_id

        # Lazy import — only needed for this option, and the script lives at
        # the repo root, so the module name is "06_clinical_interpretation"
        # but Python can't import a module that starts with a digit. Use
        # importlib.util to load it by file path.
        import importlib.util

        script_path = Path("06_clinical_interpretation.py").resolve()
        if not script_path.exists():
            print(f"[!] Script not found: {script_path}")
            print("    Cannot generate the clinical report.")
            return

        try:
            spec = importlib.util.spec_from_file_location(
                "_clin_interp", str(script_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[!] Could not import 06_clinical_interpretation.py: {e}")
            return
        if DATASET != "inhouse":   # point 06_ at the external outputs and reference sides
            import dataset_config
            mod._apply_dataset_config(dataset_config.configure(DATASET))

        ci = mod.ClinicalInterpreter(pid)
        if not ci.has_data():
            print(f"[!] No zAI cluster report on disk for {pid} "
                  f"(expected at {ci.cluster_csv}).")
            print(f"    Run: python 02_compute_zai.py --sym{' --dataset chalfont' if DATASET == 'chalfont' else ''}")
            return

        report = ci.run()
        if report is None:
            print(f"[!] {pid}: cluster report exists but is empty.")
            return

        print()
        print(ci.render_text(report))

        # Offer to launch the spatial view (top-N dominant clusters).
        try:
            ans = input("\nLaunch FSLeyes for top-3 dominant clusters? [y/N]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans == "y":
            self.view_zai_dominant_clusters(top_n=3, threshold=4.0)

    # ------------------------------------------------------------------
    # REGISTRATION CHAIN (ASL -> T1 -> symmetric template)
    # ------------------------------------------------------------------

    def _reg(self):
        """RegistrationView of the current patient, or None (reported) when inputs are missing."""
        if not _HAVE_SCI:
            print("[!] numpy/nibabel needed for the registration views (conda activate Asym).")
            return None
        rv = RegistrationView(self.patient_id, DATASET)
        if rv.missing():
            print("[!] Registration files missing: " + ", ".join(rv.missing()))
            return None
        return rv

    def _reg_qc_notes(self, rv):
        try:
            return ["QC: " + "   ".join(f"{k} {v}" for k, v in rv.qc().items())]
        except Exception as e:                  # noqa: BLE001 - QC is informative only
            return [f"QC not available: {e}"]

    def _template_bg(self, name="T_sym (symmetric template)"):
        return [SYM_T1, "-dr", "0", "0.95", "-n", name]

    def view_reg_template(self):
        """How T_sym was built: the mean of 30 HCP T1s, averaged with its own mirror image."""
        if not _HAVE_SCI:
            print("[!] numpy/nibabel needed (conda activate Asym).")
            return
        self._launch("TEMPLATE T_sym = (mean T1 + its mirror) / 2  (symreg/build_sym_template.py)",
                     [self._layer(SYM_T1_AVG, ["-dr", "0", "1"], "mean of 30 HCP T1s (before symmetrisation)",
                                  hidden=True),
                      self._layer(RegistrationView.template_asymmetry(),
                                  ["-cm", "hot", "-dr", "10", "40", "-cr", "10", "1000", "-a", "80"],
                                  "asymmetry removed: |mean - mirror| / local mean (%, >= 10)"),
                      self._atlas_layer()],
                     notes=["30 HCP control T1 brains (10 per band), each scaled to its 98th percentile, averaged on the "
                            "0.8-mm MNI grid; T_sym = (mean + flip(mean)) / 2 is flip-symmetric by construction.",
                            "Hot layer: where the plain mean differed from its mirror by >= 10 % of local intensity "
                            "(sulci, ventricles, petalia) - removed in T_sym. Tick 'mean of 30 HCP T1s' to compare."],
                     background=self._template_bg())

    def view_reg_inputs(self):
        """Registration step 0: the inputs as they enter registration, each on its own grid (world coordinates)."""
        rv = self._reg()
        if not rv:
            return
        if DATASET == "chalfont":
            layers = [self._layer(rv.cbf_native, [*CBF_ARGS, "-a", "60"], "CBF (TC-Flow), ASL space, not yet registered"),
                      self._layer(rv.asl_ref, ["-cm", "greyscale", "-a", "60"], "ASL reference (PD_N4), ASL space",
                                  hidden=True),
                      self._atlas_layer(rv.labels_native, hidden=True)]
            notes = ["T1 = 1-mm MPRAGE x GIF brain mask; the ASL images are shown where the scanner coordinates put them.",
                     "Any offset between the ASL reference and the T1 here is what step 1 corrects."]
        else:
            layers = [self._layer(rv.cbf_native, [*CBF_ARGS, "-a", "60"], "CBF (HCP-ASL, MNI grid)"),
                      self._atlas_layer(rv.labels_native, hidden=True)]
            notes = ["T1 brain and CBF arrive on the MNI 0.8-mm grid from the HCP pipelines (ASL -> T1 -> MNI done "
                     "by HCP-ASL, Kirk et al. 2025); only steps 2-4 run in this project."]
        self._launch(f"REGISTRATION · inputs — {self.patient_id}", layers, notes=notes,
                     background=[str(rv.t1), "-n", "input T1 (own grid)"])

    def view_reg_asl_to_t1(self):
        """Registration step 1: ASL reference and CBF resampled into the T1 with the saved rigid matrix."""
        rv = self._reg()
        if not rv:
            return
        if DATASET != "chalfont":
            print("\n[*] Step 1 (ASL -> T1) for in-house subjects was done by HCP-ASL before this project:")
            print(f"    the CBF arrives on the MNI grid already aligned with the T1. Option {OPTION['view_reg_inputs']} shows it on the T1;")
            print(f"    steps 2-4 (options {OPTION['view_reg_affine']}-{OPTION['view_reg_output']}) are the ones run here (symreg/reg_subject.sh).")
            return
        try:
            ref, cbf, native = rv.asl_in_t1("asl_ref"), rv.asl_in_t1("cbf"), rv.asl_in_t1("cbf_native")
        except Exception as e:                  # noqa: BLE001
            print(f"[!] {e}")
            return
        method = rv.step1_method()
        self._launch(f"REGISTRATION · step 1, ASL -> T1 (rigid) — {self.patient_id}",
                     [self._layer(cbf, [*CBF_ARGS, "-a", "60"], "CBF in T1 space (spline, inside the T1 brain + 2 mm)"),
                      self._layer(native, [*CBF_ARGS, "-a", "60"], "CBF, native voxels (nearest neighbour, unmasked)",
                                  hidden=True),
                      self._layer(ref, ["-cm", "greyscale", "-a", "60"], "ASL reference (PD) in T1 space", hidden=True),
                      self._mask_layer(rv.t1_mask, "green", "T1 brain mask (GIF)", hidden=True, alpha=35),
                      self._atlas_layer(rv.labels_native, hidden=True)],
                     notes=[f"{RegistrationView.STEP1_TEXT[method]} -> {rv.asl2t1.name} (saved by {rv.script}).",
                            "The CBF is shown as the pipeline resamples it (spline), masked to the T1 brain + 2 mm. The "
                            "hidden 'native voxels' layer is the same matrix without smoothing: the 1.9 x 1.9 x 4 mm "
                            "slices, which the ASL acquired obliquely to this T1. Smearing along the slice direction "
                            "in the smooth layer is interpolation of those slices, not misregistration.",
                            "Check: perfusion fills the cortex and spares the ventricles; the ASL slab (36-40 slices) "
                            "ends in a straight edge below the cerebellum and sometimes at the vertex.",
                            *self._reg_qc_notes(rv)],
                     background=[str(rv.t1), "-n", "T1 (own grid)"])

    def view_reg_affine(self):
        """Registration step 2: T1 after the affine registration to T_sym, with the template outline."""
        rv = self._reg()
        if not rv:
            return
        try:
            t1a = rv.t1_affine()
        except Exception as e:                  # noqa: BLE001
            print(f"[!] {e}")
            return
        why = ("Chalfont T1s start in scanner space (1-mm grid, head ~30 deg pitched, tens of mm off the template), so a "
               "rigid 6-dof fit with flirt's rotation search finds the pose first and the 12-dof fit refines it."
               if DATASET == "chalfont" else
               "In-house T1s are already on the template's MNI grid (centre within ~2 mm, rotation ~1 deg), so a "
               "12-dof refinement started at the identity (-nosearch) is enough.")
        self._launch(f"REGISTRATION · step 2, T1 -> T_sym affine — {self.patient_id}",
                     [self._layer(t1a, ["-dr", "0", str(_peak_scale(t1a))], "T1 after the affine (flirt)"),
                      self._layer(RegistrationView.template_outline(), ["-ot", "label", "-o", "-w", "2"],
                                  "template outline (brain, white matter)")],
                     notes=[f"{rv.affine_desc}; matrix recomputed with the same command, cached in {rv.cache}.", why,
                            "Toggle 'T1 after the affine' to compare with T_sym underneath; the outline is the template's."],
                     background=self._template_bg())

    def view_reg_nonlinear(self):
        """Registration step 3: T1 after fnirt on T_sym, template outline and the left-right mirror check."""
        rv = self._reg()
        if not rv:
            return
        try:
            t1w, mir = rv.t1_warped(), rv.t1_warped_mirror()
        except Exception as e:                  # noqa: BLE001
            print(f"[!] {e}")
            return
        self._launch(f"REGISTRATION · step 3, T1 -> T_sym non-linear (fnirt) — {self.patient_id}",
                     [self._layer(t1w, ["-dr", "0", str(_peak_scale(t1w))], "T1 after fnirt"),
                      self._layer(mir, ["-dr", "0", str(_peak_scale(t1w)), "-cm", "red-yellow", "-a", "50"],
                                  "same T1 mirrored left-right (mirror check)", hidden=True),
                      self._layer(RegistrationView.template_outline(), ["-ot", "label", "-o", "-w", "2"],
                                  "template outline (brain, white matter)")],
                     notes=["fnirt --subsamp 8,4,2 --miter 5,5,3 --infwhm 8,4,2 --reffwhm 6,2,0 --warpres 10,10,10 "
                            "--regmod bending_energy (identical for every subject); warp from " + rv.warp.name + ".",
                            "Mirror check: tick the mirrored layer; where it disagrees with the T1 the mirror voxel is "
                            "not homologous (residual individual folding).", *self._reg_qc_notes(rv)],
                     background=self._template_bg())

    def view_reg_output(self):
        """Registration step 4: the CBF resampled once to T_sym, with the consensus regions."""
        rv = self._reg()
        if not rv:
            return
        self._launch(f"REGISTRATION · step 4, CBF and labels on T_sym — {self.patient_id}",
                     [self._layer(self._in_brain(rv.cbf_sym), [*CBF_ARGS, "-a", "65"], "CBF on T_sym (ml/100 g/min, >= 10)"),
                      self._atlas_layer(hidden=False)],
                     notes=[("applywarp --premat asl2t1.mat --warp: the rigid and the non-linear transform composed, "
                             "one spline resampling, float32, nothing clamped." if DATASET == "chalfont" else
                             "applywarp --warp: CBF spline, parcellation nearest neighbour."),
                            "Region outlines: the consensus parcellation used for gray matter and the 37 pairs.",
                            "Output grid 227 x 272 x 227 at 0.8 mm: the input of 01_ / 02_ / 03_."],
                     background=self._template_bg())

    def render_reg_png(self):
        """The whole chain of the current patient in one PNG (for the IDE)."""
        if not _HAVE_SCI:
            print("[!] numpy/nibabel needed (conda activate Asym).")
            return
        try:
            run_registration_png(self.patient_id, DATASET, open_png=True)
        except Exception as e:                  # noqa: BLE001
            print(f"[!] {e}")

    # ------------------------------------------------------------------
    # MENU & MAIN LOOP
    # ------------------------------------------------------------------

    def show_menu(self):
        """Display the main visualization menu."""
        avail = self._detect_available_data()

        print(f"\nINTERACTIVE OVERLAY VIEWER — {DATASET} cohort ({RESULTS_ROOT}/)")
        print("=" * 70)
        print(f"Current patient: {self.patient_id}" + (f"   control band: {self.band}" if self.band else ""))
        print(self._reference_line())
        print(f"Available data: {self._availability_str()}")
        print(f"Underlay: {self.space} space ({self.overlay_t1})")
        print("Colours: red-yellow = above controls (#3) / above the mirror voxel (#4); blue = below.")
        print(f"Predicted EZ side = hemisphere deviating from controls (option {OPTION['view_clinical_report']}, "
              "ez_side_pred).")
        print("=" * 70)

        fill = dict(zai="" if avail["zai"] else "  (no zAI for this patient)",
                    clin="" if avail["clinical"] else "  (no clinical maps)",
                    rawz="" if avail["raw_zscore"] else "  (no perfusion z)",
                    anat="" if avail["anatomy"] else "  (in-house only: no native anatomy)",
                    step1="flirt 6 dof, normmi" if DATASET == "chalfont" else "done by HCP-ASL; shows a note",
                    affine="flirt 6 dof + 12 dof" if DATASET == "chalfont" else "flirt 12 dof",
                    band=self.band or "no band")
        print()
        print("Patient Selection:")
        print("   S. Switch Patient")
        for title, items in MENU:
            print()
            print(title + ":")
            for name, label in items:
                print(f"  {OPTION[name]:>2}. {label.format(**fill)}")
        print("   0. Exit")
        print("   (* prints a table in the terminal before opening FSLeyes)")
        print()

        return input(f"Enter choice (0-{len(OPTION)}, S): ").strip().lower()

    def view_all_layers(self):
        """Every layer of the current subject in FSLeyes, centred on the peak cluster."""
        if not _HAVE_SCI:
            print("[!] numpy/nibabel needed for this option (conda activate Asym).")
            return
        run_all_layers(self.patient_id, DATASET)

    def render_all_layers_png(self):
        """The same layers as a PNG under visual_analysis/subject_views/<dataset>/."""
        if not _HAVE_SCI:
            print("[!] numpy/nibabel needed for this option (conda activate Asym).")
            return
        run_all_layers(self.patient_id, DATASET, png=True, open_png=True)

    def _miss_analysis(self):
        ma = getattr(self, "_miss_cache", None)
        if ma is None or ma.dataset != DATASET:
            ma = self._miss_cache = MissAnalysis(DATASET)
        return ma

    def view_miss_overview(self):
        """Every method's call against the reference for this cohort; writes the tables, figures and report."""
        if not _HAVE_SCI:
            print("[!] numpy/nibabel/pandas needed (conda activate Asym).")
            return
        self._miss_cache = run_misses(DATASET)
        print(f"    Per patient: option {OPTION['view_miss_case']}.")

    def view_miss_case(self):
        """The current patient: reference and its file, each clinical source, each method, why the rule differs."""
        if not _HAVE_SCI:
            print("[!] numpy/nibabel/pandas needed (conda activate Asym).")
            return
        ma = self._miss_analysis()
        if self.patient_id not in ma.c.index:
            print(f"[!] {self.patient_id} is not in the analysed {DATASET} cohort (controls are not scored).")
            return
        print(ma.case_text(self.patient_id))
        out = ma.case_panel(self.patient_id, ma.figures / "cases" / f"{self.patient_id}.png")
        print(f"    Case panel: {out.relative_to(BASE_DIR)}")
        open_file(out)

    def main_loop(self):
        """Main interactive loop."""

        print("Interactive Overlay Viewer for canonical zAI / clinical maps")
        print("=" * 70)

        # First, select patient
        if self.select_patient() is None:
            print("No patient selected. Exiting.")
            return

        # Show file inventory once at start
        self.check_files()

        dispatch = {num: getattr(self, name) for name, num in OPTION.items()}
        no_pause_choices = {"0", "s"} | {OPTION[name] for name in PRINT_ONLY}

        while True:
            choice = self.show_menu()

            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "s":
                if self.select_patient() is None:
                    print("Goodbye!")
                    break
                self.check_files()
                continue
            elif choice in dispatch:
                dispatch[choice]()
            else:
                print(f"[!] Invalid choice. Please enter 0-{len(OPTION)} or S.")

            if choice not in no_pause_choices:
                input("\nPress Enter to return to menu...")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Interactive FSLeyes launcher for Pipeline B zAI maps. Without --subject/--all "
                    "it opens the menu; with them it shows all layers of one subject (or renders "
                    "PNGs for a whole dataset) and exits.")
    parser.add_argument(
        "--orig", action="store_true",
        help="Use original-space Dataset/<PID>/ T1 + aparc+aseg as the zAI "
             "underlay instead of the symmetric template (default: symmetric "
             "when symreg files exist). Not voxel-aligned with zAI maps.")
    parser.add_argument(
        "--dataset", choices=["inhouse", "chalfont"], default=None,
        help="inhouse (default): results_zscore/ and ez_ground_truth.py; "
             "chalfont: external_chalfont/results_zscore/, reference sides from "
             "external_chalfont/metadata/subjects.csv, fixed template atlas as "
             "the parcellation underlay. With --subject it is taken from the ID "
             "(P0xx in-house, C/D/N Chalfont) unless given.")
    cli = parser.add_argument_group(f"all layers without the menu (options {OPTION['view_all_layers']}-{OPTION['render_all_layers_png']})")
    cli.add_argument("--subject", metavar="ID", help="show all layers of this subject and exit")
    cli.add_argument("--all", action="store_true", help="PNG for every patient of --dataset")
    cli.add_argument("--png", action="store_true", help="write a PNG instead of opening FSLeyes")
    cli.add_argument("--open", action="store_true", help="open the PNG (VS Code if available)")
    cli.add_argument("--print", action="store_true", help="print the FSLeyes command and exit")
    cli.add_argument("--at", nargs=3, type=int, metavar=("I", "J", "K"),
                     help="centre voxel instead of the peak cluster")
    cli.add_argument("--out", type=Path, default=ALL_LAYER_PNG_DIR, help="PNG folder (default %(default)s)")
    cli.add_argument("--misses", action="store_true",
                     help="misdetected cases: tables, figures and report for --dataset (both cohorts when not given); "
                          "with --subject: that patient's summary and case panel")
    cli.add_argument("--registration", action="store_true",
                     help=f"with --subject: PNG of the registration chain (menu option {OPTION['render_reg_png']}); with --all: for every patient")
    args = parser.parse_args()
    args.out = args.out.resolve()
    os.chdir(BASE_DIR)   # every path below is relative to the project root
    if args.misses:
        if not _HAVE_SCI:
            parser.error("numpy/nibabel/pandas needed (conda activate Asym)")
        if args.subject:
            ds = args.dataset or dataset_of(args.subject)
            ma = MissAnalysis(ds)
            print(ma.case_text(args.subject))
            out = ma.case_panel(args.subject, ma.figures / "cases" / f"{args.subject}.png")
            print(f"    Case panel: {out.relative_to(BASE_DIR)}")
            if args.open:
                open_file(out)
        else:
            for ds in ([args.dataset] if args.dataset else ["inhouse", "chalfont"]):
                run_misses(ds)
        return
    if args.subject or args.all:
        if not _HAVE_SCI:
            parser.error("numpy/nibabel needed (conda activate Asym)")
        if args.registration:
            ds = args.dataset or (dataset_of(args.subject) if args.subject else "inhouse")
            for sid in (all_layer_patient_ids(ds) if args.all else [args.subject]):
                run_registration_png(sid, ds, out_dir=args.out, open_png=args.open and not args.all)
        elif args.all:
            ds = args.dataset or "inhouse"
            for sid in all_layer_patient_ids(ds):
                run_all_layers(sid, ds, png=True, at=args.at, out_dir=args.out, quiet=True)
        else:
            run_all_layers(args.subject, args.dataset, png=args.png, open_png=args.open,
                           print_only=args.print, at=args.at, out_dir=args.out)
        return
    configure_dataset(args.dataset or "inhouse")
    try:
        viewer = InteractiveOverlayViewer(use_sym=not args.orig)
        viewer.main_loop()
    except KeyboardInterrupt:
        print("\nViewer stopped by user. Goodbye!")


if __name__ == "__main__":
    main()
