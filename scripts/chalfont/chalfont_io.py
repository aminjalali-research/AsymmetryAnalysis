#!/usr/bin/env python3
"""Shared helpers for the Chalfont external-validation staging/registration step.

Layout contract (see CHALFONT_PLAN.md): source data is read-only under
SOURCE_ROOT/{controls,discrete,negative}/<ID>/; project outputs live under
external_chalfont/ (staged/, sym_perf/, qc/, metadata/, logs/, atlas/).

Several source files are plain (uncompressed) NIfTI-1 with a misleading .gz
extension; `load_any_nifti` handles both cases by checking the gzip magic bytes.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/home/amin/multimodal/chalfont/data14/data")
GROUPS = ("controls", "discrete", "negative")
EXT = PROJECT / "external_chalfont"
STAGED = EXT / "staged"
SYM_PERF = EXT / "sym_perf"
QC = EXT / "qc"
META = EXT / "metadata"
LOGS = EXT / "logs"
ATLAS = EXT / "atlas"
TEMPLATE = PROJECT / "symreg" / "templates" / "T1_sym.nii.gz"
TEMPLATE_2MM = ATLAS / "T1_sym_2mm.nii.gz"

# GIF / Neuromorphometrics parcellation: 0 = non-brain outer tissue,
# 1-3 = non-brain low/mid/high, 4 = non-ventricular CSF, >= 5 brain structures.
GIF_FIRST_BRAIN_LABEL = 5

STAGED_FILES = ("T1.nii.gz", "T1_brain.nii.gz", "brain_mask.nii.gz", "asl_ref.nii.gz",
                "flow.nii.gz", "tcflow.nii.gz", "gif_parc.nii.gz", "gif_seg.nii.gz")


def is_gzip(path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def load_any_nifti(path) -> nib.Nifti1Image:
    """Load a NIfTI-1 file whether or not its .gz extension is truthful."""
    path = str(path)
    if is_gzip(path):
        return nib.load(path)
    # Uncompressed single-file NIfTI with a .gz name: bypass nibabel's extension logic.
    fm = {"image": nib.FileHolder(fileobj=open(path, "rb")),
          "header": nib.FileHolder(fileobj=open(path, "rb"))}
    return nib.Nifti1Image.from_file_map(fm)


def save_gz(img: nib.Nifti1Image, path, dtype=None) -> None:
    """Write a properly gzipped NIfTI (optionally casting the data)."""
    data = np.asanyarray(img.dataobj)
    if dtype is not None:
        data = data.astype(dtype)
    out = nib.Nifti1Image(data, img.affine)
    out.header.set_zooms(img.header.get_zooms()[: data.ndim])
    out.header.set_xyzt_units("mm")
    nib.save(out, str(path))


def _prefix(path) -> str:
    """Scan-ID prefix of a Chalfont file name (e.g. '14643' in 14643_500_...)."""
    return os.path.basename(str(path)).split("_")[0]


def find_subjects() -> list[dict]:
    """Enumerate every subject directory and locate its source files.

    Returns one dict per subject with keys: id, group, dir, t1, parc, seg, flow,
    tcflow, pd_n4, raw, complete (bool), reason (str; empty when complete),
    t1_candidates (list).
    """
    subjects = []
    for group in GROUPS:
        gdir = SOURCE_ROOT / group
        if not gdir.is_dir():
            continue
        for sid in sorted(os.listdir(gdir)):
            d = gdir / sid
            if not d.is_dir():
                continue
            t1s = sorted(glob.glob(f"{d}/clinical/*_1mm_Cor_MPRAGE.nii.gz"))
            parcs = sorted(glob.glob(f"{d}/clinical/GIF/*_NeuroMorph_Parcellation.nii.gz"))
            segs = sorted(glob.glob(f"{d}/clinical/GIF/*_NeuroMorph_Segmentation.nii.gz"))
            flows = sorted(glob.glob(f"{d}/eASL/*_eASL_Flow.nii.gz"))
            tcs = sorted(glob.glob(f"{d}/eASL/*_eASL_Transit-corrected_Flow.nii.gz"))
            raws = sorted(glob.glob(f"{d}/eASL/*_eASL_Raw.nii.gz"))
            pd_n4 = d / "eASL" / "PD_N4.nii.gz"
            missing = [k for k, v in (("MPRAGE", t1s), ("Flow", flows), ("TC-Flow", tcs)) if not v]
            # Choose the MPRAGE: the one the GIF parcellation was computed from,
            # else the one from the eASL session, else the first.
            t1 = None
            if t1s:
                t1 = t1s[0]
                if parcs:
                    m = [t for t in t1s if _prefix(t) == _prefix(parcs[0])]
                    if m:
                        t1 = m[0]
                elif tcs:
                    m = [t for t in t1s if _prefix(t) == _prefix(tcs[0])]
                    if m:
                        t1 = m[0]
            parc = next((p for p in parcs if t1 and _prefix(p) == _prefix(t1)), parcs[0] if parcs else None)
            seg = next((p for p in segs if t1 and _prefix(p) == _prefix(t1)), segs[0] if segs else None)
            subjects.append(dict(
                id=sid, group=group, dir=str(d), t1=t1, t1_candidates=t1s, parc=parc, seg=seg,
                flow=flows[0] if flows else None, tcflow=tcs[0] if tcs else None,
                pd_n4=str(pd_n4) if pd_n4.exists() else None, raw=raws[0] if raws else None,
                complete=not missing, reason="" if not missing else "missing " + "+".join(missing),
            ))
    return subjects


def complete_ids() -> list[str]:
    return [s["id"] for s in find_subjects() if s["complete"]]


# ---------------------------------------------------------------------------
# Exact flip0-symmetric 2 mm grid for QC (the 0.8 mm template is symmetric under
# np.flip(axis=0); its mirror centre is index 113 -> world x = -0.4 mm).
# ---------------------------------------------------------------------------
SYM_2MM_SHAPE = (91, 109, 91)


def sym_2mm_affine(template_affine: np.ndarray) -> np.ndarray:
    aff = np.diag([-2.0, 2.0, 2.0, 1.0])
    aff[0, 3] = -0.4 + 45 * 2.0           # x centres symmetric about -0.4 mm
    aff[1, 3] = template_affine[1, 3]
    aff[2, 3] = template_affine[2, 3]
    return aff


def resample_to_sym_2mm(img: nib.Nifti1Image, order: int = 1) -> nib.Nifti1Image:
    """Resample a template-space image onto the exact-symmetric 2 mm QC grid."""
    from scipy.ndimage import map_coordinates
    tmpl = nib.load(str(TEMPLATE))
    aff = sym_2mm_affine(tmpl.affine)
    n = SYM_2MM_SHAPE
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in n], indexing="ij")
    world = aff @ np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)])
    vox = np.linalg.inv(img.affine) @ world
    data = np.asanyarray(img.dataobj).astype(np.float32)
    out = map_coordinates(data, vox[:3], order=order, mode="constant", cval=0.0)
    return nib.Nifti1Image(out.reshape(n).astype(np.float32), aff)


def ensure_template_2mm() -> nib.Nifti1Image:
    if TEMPLATE_2MM.exists():
        return nib.load(str(TEMPLATE_2MM))
    ATLAS.mkdir(parents=True, exist_ok=True)
    out = resample_to_sym_2mm(nib.load(str(TEMPLATE)), order=1)
    nib.save(out, str(TEMPLATE_2MM))
    return out


def template_mask(tmpl_data: np.ndarray, frac: float = 0.1) -> np.ndarray:
    p98 = np.percentile(tmpl_data[tmpl_data > 0], 98)
    return tmpl_data > frac * p98


def flip_ai_stats(x: np.ndarray, mask: np.ndarray) -> dict:
    """Mirror asymmetry index of an image: 100 (x - flip0 x) / mean, in mask & both > 0."""
    f = np.flip(x, axis=0)
    m = (x + f) / 2.0
    v = mask & (x > 0) & (f > 0)
    ai = 100.0 * (x[v] - f[v]) / m[v]
    return dict(mean_abs_ai=float(np.mean(np.abs(ai))), median_abs_ai=float(np.median(np.abs(ai))),
                n_vox=int(v.sum()))


if __name__ == "__main__":
    # CLI: chalfont_io.py resample2mm <in.nii.gz> <out.nii.gz> [order]
    if len(sys.argv) >= 4 and sys.argv[1] == "resample2mm":
        order = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        nib.save(resample_to_sym_2mm(nib.load(sys.argv[2]), order=order), sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "complete_ids":
        print("\n".join(complete_ids()))
    else:
        for s in find_subjects():
            print(s["group"], s["id"], "complete" if s["complete"] else s["reason"])
