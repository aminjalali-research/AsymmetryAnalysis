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
Menu items 1-2 always show original-space raw anatomy.

Reading zAI: the zAI map is antisymmetric, zAI(v) = -zAI(mirror v).
Red (positive) = this voxel is MORE perfused than its mirror voxel; blue
(negative) = LESS perfused than its mirror. Positive zAI means L>R only for
voxels in the left hemisphere. The predicted EZ side comes from
direction-of-abnormality inference (cluster report columns direction_class /
ez_side_pred: the hemisphere that deviates from controls), not from the sign
of zAI alone.

The viewer does not read results/, results_voxel/ or other legacy result
directories.

Menu: 25 numbered options (1-25) plus S (switch patient) and 0 (exit):
  1-2   raw anatomy (original space; in-house only)
  3-5   #3 perfusion z vs controls (3 map |z| >= 1.96, 4 hypo/hyper clusters,
        5 clusters + region outlines)
  6-12  #4 zAI (6 map |zAI| >= 1.96, 7 sign masks, 8 significant clusters,
        9 clinical GM |zAI| >= 3, 10 top-3 clusters, 11 37-ROI view,
        12 combined presurgical view)
  13-16 utilities (file list, FSLeyes commands, patient report,
        island-cleaned overlay)
  17-18 all layers of one subject (CBF, #3 perfusion z, #4 zAI, clinical
        clusters, atlas) centred on the peak cluster: 17 FSLeyes, 18 PNG
  19-25 registration chain of the current subject: 19 template build, 20 inputs,
        21 ASL -> T1, 22 T1 -> T_sym affine, 23 fnirt + mirror check + QC,
        24 CBF and labels on T_sym, 25 the chain as one PNG

Every FSLeyes view goes through InteractiveOverlayViewer._launch: the
symmetric template is the underlay (control mean CBF as a hidden layer),
signed maps are shown from the stated threshold (|value| below it is
transparent), focal views open on their peak voxel (9, 12, 17 and 18 on the
strongest clinical-tier cluster of the 06_ report, on its predicted side), missing layers are
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

# Lazy imports for the new clinical-mask / top-N options. These are only
# required when the user picks the new menu options (10 / 11). Pulled in
# at call-time so the legacy menu options keep zero-dep.
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
# All layers of one subject (menu options 17-18 and the --subject / --all
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
    call (option 15) weighs all clusters and can differ from this cluster's side.

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
# Registration chain of one subject (menu options 19-25, --registration).
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

    def asl_in_t1(self, image="asl_ref"):
        """Step 1 output: the ASL reference (or the CBF) resampled into the T1 with the saved rigid matrix."""
        src = self.asl_ref if image == "asl_ref" else self.cbf_native
        return self._cached(f"{image}_in_t1.nii.gz", lambda o: self._fsl(
            "flirt", "-in", src, "-ref", self.t1, "-applyxfm", "-init", self.asl2t1, "-out", o))

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
                            "L/R coverage": row["coverage_lr_ratio"], "flag": row.get("flag") or "none"})
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
            cols.append(("1 · ASL reference -> T1\n(flirt 6 dof, normmi)", t1n, canon(self.asl_in_t1())[0],
                         centre, False))
        else:
            cols.append(("1 · CBF on the input T1\n(ASL -> T1 -> MNI by HCP-ASL)", t1n, canon(self.cbf_native)[0],
                         centre, False))
        cols += [("2 · T1 -> T_sym, affine\n(" + ("flirt 6 + 12 dof" if self.dataset == "chalfont"
                                                else "flirt 12 dof") + "; template outline)",
                  canon(self.t1_affine())[0], None, at_t, True),
                 ("3 · T1 -> T_sym, non-linear\n(fnirt; template outline)", canon(self.t1_warped())[0], None, at_t, True),
                 ("4 · CBF on T_sym\n(applywarp, one resampling)", tmpl, canon(self.cbf_sym)[0], at_t, False)]
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
    """PNG of one subject's registration chain (menu option 25, --subject ID --registration)."""
    rv = RegistrationView(subject_id, dataset)
    if rv.missing():
        print("[!] cannot draw the registration chain; missing: " + ", ".join(rv.missing()))
        return None
    out = rv.render_png(Path(out_dir) / rv.dataset / f"{subject_id}_registration.png")
    print(out)
    if open_png:
        open_file(out)
    return out


class InteractiveOverlayViewer:
    """Interactive viewer for canonical pipeline outputs."""

    def __init__(self, patient_id=None, use_sym=True):
        self.base_dir = Path(".")
        self.patient_id = patient_id
        self.use_sym = use_sym
        self._missing = []      # files a layer builder could not find (reported by _launch)
        self._collect = None    # list -> _launch records commands instead of running them (option 14)
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
        print("\n  CBF registered to the template (options 17-18):")
        print(f"     [{'+' if cbf else '-'}] {Path(cbf).relative_to(BASE_DIR) if cbf else 'not found'}")

        if DATASET != "inhouse" and not Path(self.freesurfer_files["T1w"]).exists():
            print("\n  Raw anatomy, ORIGINAL space: in-house only (no Dataset/<ID>/ for this cohort)")
        else:
            print("\n  Raw anatomy, ORIGINAL space (Dataset/; menu items 1-2 only):")
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
        if self._collect is not None:          # option 14: record, do not run
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
        print("    Chalfont: option 17 shows the template, the patient's CBF and every map; 13 lists the files.")
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
    # OPTION 1: Basic Anatomy
    # ------------------------------------------------------------------

    def view_basic_anatomy(self):
        """Option 1: T1 + FreeSurfer parcellation in ORIGINAL space (in-house only; not aligned with zAI)."""
        if not self._has_native_anatomy():
            return
        f = self.freesurfer_files
        self._launch(f"RAW ANATOMY, original space — {self.patient_id}",
                     [self._atlas_layer(f["Parcellation"], hidden=False)],
                     notes=["Original space: NOT voxel-aligned with the symmetric-space z / zAI maps (options 3-18)."],
                     background=[f["T1w"], "-n", "T1 (original space)"])

    # ------------------------------------------------------------------
    # OPTION 2: Perfusion Comparison
    # ------------------------------------------------------------------

    def view_perfusion_comparison(self):
        """Option 2: the patient's CBF on the T1, ORIGINAL space (in-house only; not aligned with zAI)."""
        if not self._has_native_anatomy():
            return
        f = self.freesurfer_files
        self._launch(f"RAW PERFUSION, original space — {self.patient_id}",
                     [self._layer(f["Perfusion_Resampled"], [*CBF_ARGS, "-a", "60"], "CBF (ml/100 g/min)"),
                      self._layer(self.mni_perfusion, ["-cm", "cool", "-dr", "0", "80", "-a", "40"],
                                  "CBF, legacy Dataset MNI copy", hidden=True)],
                     notes=["Original space: NOT voxel-aligned with the symmetric-space maps.",
                            "Option 17 shows the same CBF after registration to the symmetric template."],
                     background=[f["T1w"], "-n", "T1 (original space)"])

    # ------------------------------------------------------------------
    # OPTION 3: Raw Voxel z-Score Map (Pipeline B Step 1-2)
    # ------------------------------------------------------------------

    def view_zscore_control_map(self):
        """Option 3: #3 perfusion z against the patient's control band, |z| >= 1.96."""
        group = self._pick_zscore_group()
        if not group:
            return
        notes = ["Red-yellow = CBF above the controls (hyper), blue = below (hypo); |z| < 1.96 transparent.",
                 "One-sided against the controls: a global CBF offset between scanners shifts the whole map.",
                 "Shown inside the brain only (consensus parcellation); the map on disk also covers the scalp."]
        if DATASET == "inhouse":
            notes.append("In-house controls are HCP subjects scanned elsewhere, so read #3 with that offset in mind.")
        self._launch(f"#3 PERFUSION z vs CONTROLS ({group}) — {self.patient_id}",
                     [self._zlayer(self._in_brain(self.zscore_control_files[group]["zscore"]), 1.96, 6,
                                   f"#3 perfusion z vs {group} (|z| >= 1.96)"),
                      self._atlas_layer()],
                     notes=notes)

    # ------------------------------------------------------------------
    # OPTION 4: Significant z-score Clusters Only
    # ------------------------------------------------------------------

    def view_zscore_clusters(self, with_regions=False):
        """Option 4: #3 hypo- and hyperperfusion clusters (01_: z beyond +-1.96, >= 10 voxels)."""
        group = self._pick_zscore_group()
        if not group:
            return
        f = self.zscore_control_files[group]
        self._launch(f"#3 HYPO / HYPER CLUSTERS vs {group} — {self.patient_id}",
                     [self._mask_layer(self._in_brain(f["hyper_clusters"]), "red", "#3 hyper clusters (z > 1.96)"),
                      self._mask_layer(self._in_brain(f["hypo_clusters"]), "blue", "#3 hypo clusters (z < -1.96)"),
                      self._atlas_layer(hidden=not with_regions)],
                     notes=["Red = hyperperfused, blue = hypoperfused clusters (>= 10 voxels, 01_build_control_normative).",
                            "Cluster tables: " + str(Path(f["zscore"]).with_name(
                                f"{self.patient_id}_vs_{group}_cluster_report.csv"))])

    # ------------------------------------------------------------------
    # OPTION 5: z-score Clusters + Brain Region Labels
    # ------------------------------------------------------------------

    def view_zscore_with_parcellation(self):
        """Option 5: option 4 with the region outlines switched on."""
        self.view_zscore_clusters(with_regions=True)

    # ------------------------------------------------------------------
    # OPTION 6: zAI Asymmetry Map (PRIMARY CLINICAL PRODUCT)
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

    def view_zai_asymmetry_map(self):
        """Option 6: #4 zAI, voxel vs mirror, shown from |zAI| >= 1.96 (the cluster-forming threshold)."""
        if not self._check_zai():
            return
        self._launch(f"#4 zAI MAP (voxel vs mirror, control-referenced) — {self.patient_id}",
                     [self._zlayer(self.zai_files["asymmetry_zscore"], 1.96, 6, "zAI (|zAI| >= 1.96)"),
                      self._atlas_layer()],
                     notes=["Red-yellow = voxel above its mirror, blue = below; |zAI| < 1.96 transparent.",
                            "Antisymmetric: every finding appears twice (red on one side, blue on the mirror).",
                            "To see every value, lower the display range of the zAI layer in FSLeyes."])

    def view_zai_cleaned_overlay(self):
        """Option 16: island-cleaned zAI (clean_overlay.py: |zAI| >= 3, >= 50 voxels, 37-ROI scope)."""
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
            print("      Option 9 shows the uncleaned clinical clusters.")
            return
        self._launch(f"ISLAND-CLEANED zAI — {self.patient_id}",
                     [self._zlayer(cleaned, 3, 8, "zAI cleaned (|zAI| >= 3, >= 50 vox)"),
                      self._atlas_layer()],
                     centre=self._peak(cleaned))

    # ------------------------------------------------------------------
    # OPTION 7: zAI sign maps (positive vs negative, full brain)
    # ------------------------------------------------------------------

    def view_zai_dominance(self):
        """Option 7: zAI sign masks from 02_ (positive / negative members of the significant clusters).

        <PID>_asymmetry_left_dominant / _right_dominant are binary masks of the
        positive and negative voxels of <PID>_asymmetry_significant. Despite the
        legacy names they are NOT hemisphere-specific: every finding appears in
        both (red on the higher member, blue on its mirror). Option 9 shows the
        one-sided lateralized clinical maps.
        """
        if not self._check_zai():
            return
        self._launch(f"zAI SIGN MAPS (both members of each pair) — {self.patient_id}",
                     [self._mask_layer(self.zai_files["left_dominant"], "red", "zAI positive (voxel > mirror)"),
                      self._mask_layer(self.zai_files["right_dominant"], "blue", "zAI negative (voxel < mirror)"),
                      self._atlas_layer()],
                     notes=["Red = voxel higher than its mirror, blue = lower (|zAI| >= 1.96 clusters, >= 50 voxels).",
                            "Not the predicted EZ side: see option 15 (direction_class / ez_side_pred)."])

    # ------------------------------------------------------------------
    # OPTION 8: Significant zAI Clusters
    # ------------------------------------------------------------------

    def view_zai_significant(self):
        """Option 8: significant zAI clusters from 02_ (|zAI| >= 1.96, >= 50 voxels, both pair members)."""
        if not self._check_zai():
            return
        sig = self.zai_files["significant"]
        self._launch(f"SIGNIFICANT zAI CLUSTERS — {self.patient_id}",
                     [self._zlayer(sig, 1.96, 6, "zAI clusters (|zAI| >= 1.96, >= 50 vox)"),
                      self._zlayer(self.zai_files["asymmetry_zscore"], 1.96, 6, "zAI, all voxels", hidden=True),
                      self._atlas_layer()],
                     notes=["Clusters formed at |zAI| >= 1.96 with >= 50 voxels (02_compute_zai); both members shown.",
                            "The clinical tier keeps a cluster whose PEAK reaches 4: options 9, 10 and 15."],
                     centre=self._peak(sig))

    # ------------------------------------------------------------------
    # OPTION 9: Clinical zAI (gray matter only, lateralized) — patient report view
    # ------------------------------------------------------------------

    def view_zai_clinical(self):
        """Option 9: clinical zAI (03_): gray matter, 37-ROI scope, |zAI| >= 3, >= 50 voxels, higher member only."""
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
                            "Option 15 gives the patient-level call."],
                     centre=self._focus(both if both.exists() else sig))

    # ------------------------------------------------------------------
    # OPTION 10: Top-N Dominant Clusters (focal)
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
            print(f"    Lower the threshold or pick option 6 / 8 to inspect the full map.")
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
            print(f"      To get a focal FSLeyes overlay, run option 9 (clinical zAI).")
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
            print(f"      Falling back to FSLeyes view of full clinical zAI (option 9).")
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

    # ------------------------------------------------------------------
    # OPTION 11: 37-ROI Clinical Mask View
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # OPTION 12: Combined Presurgical View (recommended for review)
    # ------------------------------------------------------------------

    def view_combined_presurgical(self):
        """Option 12: zAI map + clinical clusters + side masks + regions, opened on the clinical peak."""
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
    # OPTION 13: Show File List
    # ------------------------------------------------------------------

    def show_file_list(self):
        """Show detailed canonical-file inventory for the current patient."""
        print("\n[*] CANONICAL FILE INVENTORY")
        print("=" * 70)
        self.check_files()

    # ------------------------------------------------------------------
    # OPTION 14: Show Quick FSLeyes Commands (canonical zAI paths)
    # ------------------------------------------------------------------

    def show_quick_commands(self):
        """Option 14: the FSLeyes commands options 3, 6, 8, 9, 12 and 17 would run, for copy/paste."""
        self._collect = []
        try:
            for fn in (self.view_zscore_control_map, self.view_zai_asymmetry_map, self.view_zai_significant,
                       self.view_zai_clinical, self.view_combined_presurgical):
                fn()
        finally:
            got, self._collect = self._collect, None
        print("\n[*] FSLeyes COMMANDS for " + self.patient_id + " (run from the project root)")
        print("=" * 70)
        for title, argv in got:
            print(f"# {title}\n{shlex.join(argv)}\n")
        if _HAVE_SCI:
            v = AllLayersView(self.patient_id, DATASET)
            voxel, where = v.focus()
            print(f"# ALL LAYERS (option 17: CBF, #3, #4, clinical, atlas; opened at {where})")
            print(shlex.join(v.fsleyes_command(voxel)) + "\n")
        print("# PNG of all layers for the IDE (option 18)")
        print(f"python interactive_viewer.py --subject {self.patient_id} --png --open\n")

    # ------------------------------------------------------------------
    # OPTION 15: Clinical Report (replaces former Custom Overlay Builder)
    # ------------------------------------------------------------------
    #
    # Per-patient report. Imports the analyzer from
    # 06_clinical_interpretation.py, runs it for the current patient with
    # the canonical |zAI|>=4 / size>=100 thresholds, prints the report
    # to stdout, and (if FSLeyes is available + the cluster_labels NIfTI
    # is on disk) offers to launch the top-N cluster overlay (option 10
    # logic) so the reader can flip from the deterministic answer to
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
    # OPTIONS 19-25: REGISTRATION CHAIN (ASL -> T1 -> symmetric template)
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
        """Option 19: how T_sym was built: the mean of 30 HCP T1s, averaged with its own mirror image."""
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
        """Option 20 (step 0): the inputs as they enter registration, each on its own grid (world coordinates)."""
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
        """Option 21 (step 1): ASL reference and CBF resampled into the T1 with the saved rigid matrix."""
        rv = self._reg()
        if not rv:
            return
        if DATASET != "chalfont":
            print("\n[*] Step 1 (ASL -> T1) for in-house subjects was done by HCP-ASL before this project:")
            print("    the CBF arrives on the MNI grid already aligned with the T1. Option 20 shows it on the T1;")
            print("    steps 2-4 (options 22-24) are the ones run here (symreg/reg_subject.sh).")
            return
        try:
            ref, cbf = rv.asl_in_t1("asl_ref"), rv.asl_in_t1("cbf")
        except Exception as e:                  # noqa: BLE001
            print(f"[!] {e}")
            return
        self._launch(f"REGISTRATION · step 1, ASL -> T1 (rigid) — {self.patient_id}",
                     [self._layer(cbf, [*CBF_ARGS, "-a", "60"], "CBF in T1 space"),
                      self._layer(ref, ["-cm", "greyscale", "-a", "60"], "ASL reference in T1 space", hidden=True),
                      self._atlas_layer(rv.labels_native, hidden=True)],
                     notes=["bet -f 0.3 on the PD reference, then flirt 6 dof, normalized mutual information, "
                            f"+-30 deg search -> {rv.asl2t1.name} (saved by {rv.script}).",
                            "Check: perfusion should fill the cortex and spare the ventricles; tick the ASL reference "
                            "(greyscale) to compare its edges with the T1."],
                     background=[str(rv.t1), "-n", "T1 (own grid)"])

    def view_reg_affine(self):
        """Option 22 (step 2): T1 after the affine registration to T_sym, with the template outline."""
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
        """Option 23 (step 3): T1 after fnirt on T_sym, template outline and the left-right mirror check."""
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
        """Option 24 (step 4): the CBF resampled once to T_sym, with the consensus regions."""
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
        """Option 25: the whole chain of the current patient in one PNG (for the IDE)."""
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
        print("Predicted EZ side = hemisphere deviating from controls (option 15, ez_side_pred).")
        print("=" * 70)

        zai_tag = "" if avail["zai"] else "  (no zAI for this patient)"
        clin_tag = "" if avail["clinical"] else "  (no clinical maps)"
        rawz_tag = "" if avail["raw_zscore"] else "  (no perfusion z)"
        anat_tag = "" if avail["anatomy"] else "  (in-house only: no native anatomy)"

        print()
        print("Patient Selection:")
        print("   S. Switch Patient")
        print()
        print("Anatomy, ORIGINAL space (not aligned with the maps below):")
        print(f"   1. T1 + FreeSurfer parcellation{anat_tag}")
        print(f"   2. CBF on the T1{anat_tag}")
        print()
        print("#3 Perfusion z vs controls (one-sided; supplementary):")
        print(f"   3. Perfusion z map (|z| >= 1.96){rawz_tag}")
        print(f"   4. Hypo / hyper clusters (z beyond 1.96, >= 10 voxels){rawz_tag}")
        print(f"   5. Clusters + region outlines{rawz_tag}")
        print()
        print("#4 zAI, voxel vs its mirror (PRIMARY):")
        print(f"   6. zAI map (|zAI| >= 1.96 shown; both members of each pair){zai_tag}")
        print(f"   7. zAI sign masks (red = voxel > mirror, blue = voxel < mirror){zai_tag}")
        print(f"   8. Significant zAI clusters (|zAI| >= 1.96, >= 50 voxels){zai_tag}")
        print(f"   9. Clinical zAI (gray matter, |zAI| >= 3, higher member, lateralized){clin_tag}")
        print(f"  10. Top-3 clusters by peak |zAI| >= 4 * (table + overlay){zai_tag}")
        print(f"  11. 37-ROI view, |zAI| >= 4 * (top regions table + overlay){zai_tag}")
        print(f"  12. Combined presurgical view (RECOMMENDED){zai_tag}")
        print()
        print("Utilities:")
        print("  13. File list")
        print("  14. FSLeyes commands for this patient (copy / paste)")
        print(f"  15. Patient report * (rule call, clinical-tier clusters, predicted side){zai_tag}")
        print(f"  16. Island-cleaned zAI (|zAI| >= 3, >= 50 voxels) + montage{zai_tag}")
        print()
        print("All layers, one subject (CBF + #3 perfusion z + #4 zAI + clinical clusters, at the peak):")
        print("  17. All layers in FSLeyes (template + clinical on; tick the others)")
        print("  18. All layers as a PNG (for the IDE; opens in VS Code when `code` is on PATH)")
        print()
        step1 = "flirt 6 dof, normmi" if DATASET == "chalfont" else "done by HCP-ASL; shows a note"
        affine = "flirt 6 dof + 12 dof" if DATASET == "chalfont" else "flirt 12 dof"
        print("Registration, ASL -> T1 -> symmetric template (current patient):")
        print("  19. Template: mean of 30 HCP T1s vs T_sym (the asymmetry removed)")
        print("  20. Step 0 · inputs before registration (T1, ASL reference / CBF, labels)")
        print(f"  21. Step 1 · ASL -> T1, rigid ({step1})")
        print(f"  22. Step 2 · T1 -> T_sym, affine ({affine}; first run ~30-45 s, cached)")
        print("  23. Step 3 · T1 -> T_sym, non-linear (fnirt) + mirror check + QC")
        print("  24. Step 4 · CBF and labels on T_sym (applywarp, one resampling)")
        print("  25. Registration chain as a PNG (all steps in one figure, for the IDE)")
        print("   0. Exit")
        print("   (* prints a table in the terminal before opening FSLeyes)")
        print()

        return input("Enter choice (0-25, S): ").strip().lower()

    def view_all_layers(self):
        """Option 17: every layer of the current subject in FSLeyes, centred on the peak cluster."""
        if not _HAVE_SCI:
            print("[!] numpy/nibabel needed for this option (conda activate Asym).")
            return
        run_all_layers(self.patient_id, DATASET)

    def render_all_layers_png(self):
        """Option 18: the same layers as a PNG under visual_analysis/subject_views/<dataset>/."""
        if not _HAVE_SCI:
            print("[!] numpy/nibabel needed for this option (conda activate Asym).")
            return
        run_all_layers(self.patient_id, DATASET, png=True, open_png=True)

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

        dispatch = {
            "1": self.view_basic_anatomy,
            "2": self.view_perfusion_comparison,
            "3": self.view_zscore_control_map,
            "4": self.view_zscore_clusters,
            "5": self.view_zscore_with_parcellation,
            "6": self.view_zai_asymmetry_map,
            "7": self.view_zai_dominance,
            "8": self.view_zai_significant,
            "9": self.view_zai_clinical,
            "10": self.view_zai_dominant_clusters,
            "11": self.view_zai_clinical_roi_mask,
            "12": self.view_combined_presurgical,
            "13": self.show_file_list,
            "14": self.show_quick_commands,
            "15": self.view_clinical_report,
            "16": self.view_zai_cleaned_overlay,
            "17": self.view_all_layers,
            "18": self.render_all_layers_png,
            "19": self.view_reg_template,
            "20": self.view_reg_inputs,
            "21": self.view_reg_asl_to_t1,
            "22": self.view_reg_affine,
            "23": self.view_reg_nonlinear,
            "24": self.view_reg_output,
            "25": self.render_reg_png,
        }

        # Options that print to stdout (no FSLeyes wait needed)
        no_pause_choices = {"0", "13", "14", "15", "18", "25", "s"}

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
                print("[!] Invalid choice. Please enter 0-25 or S.")

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
    cli = parser.add_argument_group("all layers without the menu (options 17-18)")
    cli.add_argument("--subject", metavar="ID", help="show all layers of this subject and exit")
    cli.add_argument("--all", action="store_true", help="PNG for every patient of --dataset")
    cli.add_argument("--png", action="store_true", help="write a PNG instead of opening FSLeyes")
    cli.add_argument("--open", action="store_true", help="open the PNG (VS Code if available)")
    cli.add_argument("--print", action="store_true", help="print the FSLeyes command and exit")
    cli.add_argument("--at", nargs=3, type=int, metavar=("I", "J", "K"),
                     help="centre voxel instead of the peak cluster")
    cli.add_argument("--out", type=Path, default=ALL_LAYER_PNG_DIR, help="PNG folder (default %(default)s)")
    cli.add_argument("--registration", action="store_true",
                     help="with --subject: PNG of the registration chain (menu option 25); with --all: for every patient")
    args = parser.parse_args()
    args.out = args.out.resolve()
    os.chdir(BASE_DIR)   # every path below is relative to the project root
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
