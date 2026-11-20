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


def rigid_gap(mat_a, mat_b, moving_image) -> tuple[float, float]:
    """Rotation (deg) and displacement (mm) of the moving image's centre between two FSL matrices (flirt -omat act
    on the scaled-voxel coordinates of the moving image, so the centre is dims x pixdims / 2)."""
    A, B = np.loadtxt(mat_a), np.loadtxt(mat_b)
    R = B[:3, :3] @ np.linalg.inv(A[:3, :3])
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    img = nib.load(str(moving_image))
    centre = np.array([*(np.array(img.shape[:3]) * np.array(img.header.get_zooms()[:3]) / 2), 1.0])
    return rot, float(np.linalg.norm(((B - A) @ centre)[:3]))


def asl_t1_gm_corr(mat, staged_dir) -> float:
    """Step-1 fit score for one ASL -> T1 matrix: correlation of the TC-Flow CBF (spline-resampled into the T1) with the
    GIF cortex indicator, inside the T1 brain and the ASL slab. NaN without a GIF parcellation."""
    import os
    import subprocess
    import tempfile
    from scipy import ndimage
    st = Path(staged_dir)
    if not (st / "gif_parc.nii.gz").exists():
        return float("nan")
    env = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"}
    with tempfile.TemporaryDirectory() as td:
        outs = {}
        for name, interp in (("tcflow", "spline"), ("asl_ref", "trilinear")):
            subprocess.run(["flirt", "-in", str(st / f"{name}.nii.gz"), "-ref", str(st / "T1_brain.nii.gz"), "-applyxfm",
                            "-init", str(mat), "-interp", interp, "-out", f"{td}/{name}.nii.gz"], check=True, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            outs[name] = nib.load(f"{td}/{name}.nii.gz").get_fdata(dtype=np.float32)
    parc = np.asanyarray(nib.load(str(st / "gif_parc.nii.gz")).dataobj)
    mask = nib.load(str(st / "brain_mask.nii.gz")).get_fdata() > 0
    fov = ndimage.binary_erosion(outs["asl_ref"] > 0, iterations=3)
    inb = mask & fov & (outs["tcflow"] != 0)
    if inb.sum() < 100:
        return float("nan")
    return float(np.corrcoef(outs["tcflow"][inb], (parc[inb] >= 100).astype(float))[0, 1])


def _rigid_about(centre, rx=0.0, ry=0.0, rz=0.0, tx=0.0, ty=0.0, tz=0.0) -> np.ndarray:
    """Rigid transform (degrees, mm) about `centre` in the moving image's scaled-voxel space (FSL convention)."""
    def rot(axis, deg):
        a = np.radians(deg); c, s = np.cos(a), np.sin(a); R = np.eye(4)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        R[i, i] = c; R[j, j] = c; R[i, j] = -s; R[j, i] = s
        return R
    T = np.eye(4); T[:3, 3] = centre; Ti = np.eye(4); Ti[:3, 3] = -centre
    P = np.eye(4); P[:3, 3] = (tx, ty, tz)
    return P @ T @ rot(0, rx) @ rot(1, ry) @ rot(2, rz) @ Ti


def asl_grid_corr(M: np.ndarray, staged_dir, workdir) -> float:
    """Step-1 fit score on the ASL grid (cheap, one flirt call): the T1 cortex indicator and brain mask are carried
    into ASL space with the inverse of M; correlation of the native TC-Flow CBF with the cortex indicator inside the
    projected brain. `workdir` must hold ctx_brain.nii.gz (4D: cortex indicator, brain) made by step1polish."""
    import os
    import subprocess
    st, wd = Path(staged_dir), Path(workdir)
    np.savetxt(wd / "inv.mat", np.linalg.inv(M), fmt="%.8f")
    subprocess.run(["flirt", "-in", str(wd / "ctx_brain.nii.gz"), "-ref", str(st / "asl_ref.nii.gz"), "-applyxfm",
                    "-init", str(wd / "inv.mat"), "-interp", "trilinear", "-out", str(wd / "proj.nii.gz")],
                   check=True, capture_output=True, env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"})
    proj = np.asanyarray(nib.load(str(wd / "proj.nii.gz")).dataobj).astype(np.float32)
    cbf = np.asanyarray(nib.load(str(st / "tcflow.nii.gz")).dataobj).astype(np.float32)
    m = (proj[..., 1] > 0.5) & (cbf > 0)
    return float(np.corrcoef(cbf[m], proj[..., 0][m])[0, 1]) if m.sum() > 500 else float("nan")


def step1polish(mat, staged_dir, max_deg=3.0, max_mm=3.0, steps=((1.0, 1.0), (0.5, 0.5), (0.25, 0.25)), min_gain=0.005):
    """AUDIT ONLY, NOT USED BY THE PIPELINE (rejected 2026-11-19: it raises its own CBF-cortex objective but makes the
    independent checks worse -- CBF outside the brain worse in 42/80 polished subjects, in the ventricles in 61/80;
    external_chalfont/qc/asl2t1_polish_independent_check.csv).
    Coordinate descent of the six rigid parameters (about the ASL volume centre) that maximises asl_grid_corr,
    bounded to +-max_deg / +-max_mm from the input matrix. Returns (M_best, gain, c0, c1, n_evaluations); the
    input matrix is returned unchanged when the gain is below min_gain or there is no GIF parcellation."""
    import tempfile
    st = Path(staged_dir)
    M0 = np.loadtxt(mat)
    if not (st / "gif_parc.nii.gz").exists():
        return M0, 0.0, float("nan"), float("nan"), 0
    asl = nib.load(str(st / "asl_ref.nii.gz"))
    centre = np.array(asl.shape[:3]) * np.array(asl.header.get_zooms()[:3]) / 2
    with tempfile.TemporaryDirectory(prefix="polish_") as td:
        wd = Path(td)
        parc = nib.load(str(st / "gif_parc.nii.gz")); p = np.asanyarray(parc.dataobj)
        nib.save(nib.Nifti1Image(np.stack([(p >= 100), (p > 0)], -1).astype(np.float32), parc.affine), str(wd / "ctx_brain.nii.gz"))
        x = np.zeros(6); n = 0
        best = c0 = asl_grid_corr(M0, st, wd); n += 1
        for sd, sm in steps:
            improved = True
            while improved:
                improved = False
                for k in range(6):
                    for sgn in (+1, -1):
                        y = x.copy(); y[k] += sgn * (sd if k < 3 else sm)
                        if np.any(np.abs(y[:3]) > max_deg + 1e-9) or np.any(np.abs(y[3:]) > max_mm + 1e-9):
                            continue
                        c = asl_grid_corr(M0 @ _rigid_about(centre, *y), st, wd); n += 1
                        if c > best + 1e-6:
                            best, x, improved = c, y, True
        M1 = M0 @ _rigid_about(centre, *x)
    gain = best - c0
    return (M1 if gain >= min_gain else M0), gain, c0, best, n


if __name__ == "__main__":
    # CLI: chalfont_io.py resample2mm <in.nii.gz> <out.nii.gz> [order]
    #      chalfont_io.py step1polish <asl2t1.mat> <staged dir> <max_deg> <max_mm>   (audit only; rejected, see step1polish)
    #          coordinate descent of the rigid fit on the ASL-grid CBF-cortex correlation; overwrites the matrix when
    #          the gain is >= 0.005 (the original is kept as <mat>.prepolish). Prints the scores and the move.
    if len(sys.argv) == 6 and sys.argv[1] == "step1polish":
        mat, st, max_deg, max_mm = sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
        M1, gain, c0, c1, n = step1polish(mat, st, max_deg, max_mm)
        if gain >= 0.005:
            # independent check in T1 space (the step1pick metric): the polished matrix must not score lower there
            cand = mat + ".candidate"
            np.savetxt(cand, M1, fmt="%.10f")
            t0, t1 = asl_t1_gm_corr(mat, st), asl_t1_gm_corr(cand, st)
            rot, mm = rigid_gap(mat, cand, Path(st) / "asl_ref.nii.gz")
            if np.isnan(t0) or np.isnan(t1) or t1 >= t0:
                Path(mat + ".prepolish").write_text(Path(mat).read_text())
                Path(cand).replace(mat)
                print(f"step1polish: ASL-grid correlation {c0:.4f} -> {c1:.4f} (+{gain:.4f}), T1-space {t0:.4f} -> {t1:.4f}, "
                      f"moved {rot:.2f} deg / {mm:.2f} mm, {n} evaluations; matrix updated")
            else:
                Path(cand).unlink()
                print(f"step1polish: ASL-grid correlation {c0:.4f} -> {c1:.4f} (+{gain:.4f}) but T1-space {t0:.4f} -> {t1:.4f}; "
                      f"matrix kept")
        else:
            print(f"step1polish: ASL-grid correlation {c0:.4f}, best nudge +{gain:.4f} < 0.005, {n} evaluations; matrix kept")
        sys.exit(0)
    #      chalfont_io.py rigidgap <a.mat> <b.mat> <moving.nii.gz> <max_deg> <max_mm>  (exit 1 when exceeded)
    #      chalfont_io.py step1pick <init.mat> <bbr.mat> <staged dir> <max_deg> <max_mm>
    #          exit 0 = accept the BBR refinement: within the limits AND its CBF-cortex correlation is not lower than
    #          the initialisation's (minus 0.01); exit 1 = keep the initialisation. Prints both scores and the gap.
    if len(sys.argv) == 7 and sys.argv[1] == "step1pick":
        init, bbr, st, max_deg, max_mm = sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6])
        rot, mm = rigid_gap(init, bbr, Path(st) / "asl_ref.nii.gz")
        c0, c1 = asl_t1_gm_corr(init, st), asl_t1_gm_corr(bbr, st)
        ok = rot <= max_deg and mm <= max_mm and (np.isnan(c0) or np.isnan(c1) or c1 >= c0 - 0.01)
        print(f"step1pick: BBR moved {rot:.2f} deg, {mm:.2f} mm; CBF-cortex correlation init {c0:.3f} -> bbr {c1:.3f}; "
              f"{'accepted' if ok else 'rejected'}")
        sys.exit(0 if ok else 1)
    if len(sys.argv) >= 4 and sys.argv[1] == "resample2mm":
        order = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        nib.save(resample_to_sym_2mm(nib.load(sys.argv[2]), order=order), sys.argv[3])
    elif len(sys.argv) == 7 and sys.argv[1] == "rigidgap":
        rot, mm = rigid_gap(sys.argv[2], sys.argv[3], sys.argv[4])
        print(f"rigid gap: {rot:.2f} deg, {mm:.2f} mm at the moving image's centre")
        sys.exit(0 if rot <= float(sys.argv[5]) and mm <= float(sys.argv[6]) else 1)
    elif len(sys.argv) >= 2 and sys.argv[1] == "complete_ids":
        print("\n".join(complete_ids()))
    else:
        for s in find_subjects():
            print(s["group"], s["id"], "complete" if s["complete"] else s["reason"])
