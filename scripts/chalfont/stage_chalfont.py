#!/usr/bin/env python3
"""Stage Chalfont subjects into external_chalfont/staged/<ID>/ (properly gzipped).

Per complete subject (MPRAGE + eASL Flow + Transit-corrected Flow):
  T1.nii.gz          MPRAGE (int16, scanner space)
  T1_brain.nii.gz    T1 * brain_mask
  brain_mask.nii.gz  GIF parcellation >= 5 (brain structures incl. ventricles,
                     excludes non-brain 0-3 and non-ventricular CSF 4), largest
                     connected component, 2-voxel binary closing (keeps sulcal CSF),
                     hole-filled; `bet` fallback when GIF is missing
  asl_ref.nii.gz     eASL/PD_N4.nii.gz (asl_ref_source=pd_n4), or volume index 4 (PD/M0,
                     saturates at 32766) of the 4D eASL_Raw when absent (raw_pd); none = failed
  flow.nii.gz, tcflow.nii.gz  CBF maps as float32 (negatives kept)
  gif_parc.nii.gz    GIF parcellation (int16), gif_seg.nii.gz GIF 4D tissue maps (float32)

Writes external_chalfont/metadata/staging_manifest.csv.
Usage: stage_chalfont.py [--ids C001 D001 ...] [--force] [-j N]
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chalfont_io as cio  # noqa: E402


def gif_brain_mask(parc_img: nib.Nifti1Image) -> np.ndarray:
    parc = np.asanyarray(parc_img.dataobj)
    mask = parc >= cio.GIF_FIRST_BRAIN_LABEL
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        mask = lab == (1 + int(np.argmax(sizes)))
    # Close sulci (~2 mm) so sulcal CSF is kept, as in the HCP brain masks the
    # symmetric template was built from; then fill enclosed holes.
    mask = ndimage.binary_closing(mask, iterations=2, border_value=0)
    mask = ndimage.binary_fill_holes(mask)
    return mask.astype(np.uint8)


def bet_brain_mask(t1_path: Path, out_dir: Path) -> np.ndarray:
    prefix = out_dir / "_bet_tmp"
    subprocess.run(["bet", str(t1_path), str(prefix), "-R", "-f", "0.5", "-m"], check=True,
                   env={**__import__("os").environ, "FSLOUTPUTTYPE": "NIFTI_GZ"})
    mask = np.asanyarray(nib.load(f"{prefix}_mask.nii.gz").dataobj).astype(np.uint8)
    for f in out_dir.glob("_bet_tmp*"):
        f.unlink()
    return mask


def mask_qc_png(t1: np.ndarray, mask: np.ndarray, sid: str, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    vmax = np.percentile(t1[mask > 0], 99) if mask.any() else t1.max()
    cuts = [t1.shape[0] // 2, t1.shape[1] // 2, t1.shape[2] // 2]
    sl = [(t1[cuts[0]], mask[cuts[0]]), (t1[:, cuts[1]], mask[:, cuts[1]]), (t1[:, :, cuts[2]], mask[:, :, cuts[2]])]
    for ax, (a, m), name in zip(axes, sl, ["i-cut", "j-cut", "k-cut"]):
        ax.imshow(a.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        if m.any():
            ax.contour(m.T, levels=[0.5], colors="r", linewidths=0.6)
        ax.set_title(f"{sid} {name}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def stage_one(sub: dict, force: bool = False) -> dict:
    sid, group = sub["id"], sub["group"]
    row = dict(id=sid, group=group, status="", reason=sub["reason"], brain_volume_ml="",
               t1_source="", asl_ref_source="", mask_source="", n_t1_candidates=len(sub["t1_candidates"]))
    if not sub["complete"]:
        row["status"] = "incomplete"
        return row
    out = cio.STAGED / sid
    out.mkdir(parents=True, exist_ok=True)
    row["t1_source"] = Path(sub["t1"]).name
    if not force and all((out / f).exists() for f in cio.STAGED_FILES if f != "gif_seg.nii.gz" or sub["seg"]):
        row["status"] = "staged"
        row["reason"] = ""
        row["asl_ref_source"] = "pd_n4" if sub["pd_n4"] else ("raw_pd" if sub["raw"] else "none")
        row["mask_source"] = "gif" if (out / "gif_parc.nii.gz").exists() else "bet"
        m = np.asanyarray(nib.load(str(out / "brain_mask.nii.gz")).dataobj)
        z = nib.load(str(out / "brain_mask.nii.gz")).header.get_zooms()
        row["brain_volume_ml"] = round(float(m.sum()) * float(np.prod(z[:3])) / 1000.0, 1)
        return row
    try:
        t1 = cio.load_any_nifti(sub["t1"])
        t1_data = np.asanyarray(t1.dataobj).astype(np.int16)
        t1_img = nib.Nifti1Image(t1_data, t1.affine)
        t1_img.header.set_zooms(t1.header.get_zooms()[:3])
        nib.save(t1_img, str(out / "T1.nii.gz"))

        # Brain mask
        mask = None
        if sub["parc"]:
            parc = cio.load_any_nifti(sub["parc"])
            if parc.shape[:3] == t1.shape[:3]:
                cio.save_gz(parc, out / "gif_parc.nii.gz", dtype=np.int16)
                mask = gif_brain_mask(parc)
                row["mask_source"] = "gif"
            else:
                row["reason"] = f"gif shape {parc.shape} != T1 {t1.shape}; bet fallback"
        if sub["seg"] and sub["parc"]:
            seg = cio.load_any_nifti(sub["seg"])
            if seg.shape[:3] == t1.shape[:3]:
                cio.save_gz(seg, out / "gif_seg.nii.gz", dtype=np.float32)
        if mask is None:
            mask = bet_brain_mask(out / "T1.nii.gz", out)
            row["mask_source"] = "bet"
            if not row["reason"]:
                row["reason"] = "no GIF parcellation; bet fallback"
        vox_ml = float(np.prod(t1.header.get_zooms()[:3])) / 1000.0
        row["brain_volume_ml"] = round(float(mask.sum()) * vox_ml, 1)
        m_img = nib.Nifti1Image(mask.astype(np.uint8), t1.affine)
        m_img.header.set_zooms(t1.header.get_zooms()[:3])
        nib.save(m_img, str(out / "brain_mask.nii.gz"))
        tb = nib.Nifti1Image((t1_data * (mask > 0)).astype(np.int16), t1.affine)
        tb.header.set_zooms(t1.header.get_zooms()[:3])
        nib.save(tb, str(out / "T1_brain.nii.gz"))
        (cio.QC / "mask").mkdir(parents=True, exist_ok=True)
        mask_qc_png(t1_data.astype(np.float32), mask, sid, cio.QC / "mask" / f"{sid}.png")

        # ASL images
        if sub["pd_n4"]:
            ref = cio.load_any_nifti(sub["pd_n4"])
            cio.save_gz(ref, out / "asl_ref.nii.gz", dtype=np.float32)
            row["asl_ref_source"] = "pd_n4"
        elif sub["raw"]:
            raw = cio.load_any_nifti(sub["raw"])
            rd = np.asanyarray(raw.dataobj)
            if rd.ndim != 4 or rd.shape[3] < 5:
                raise RuntimeError(f"unexpected eASL_Raw shape {rd.shape}")
            ref = nib.Nifti1Image(rd[..., 4].astype(np.float32), raw.affine)
            ref.header.set_zooms(raw.header.get_zooms()[:3])
            nib.save(ref, str(out / "asl_ref.nii.gz"))
            row["asl_ref_source"] = "raw_pd"
        else:
            row["asl_ref_source"] = "none"
            raise RuntimeError("no PD_N4 and no eASL_Raw for the ASL reference")
        for key, name in (("flow", "flow.nii.gz"), ("tcflow", "tcflow.nii.gz")):
            im = cio.load_any_nifti(sub[key])
            if im.shape[:3] != ref.shape[:3]:
                raise RuntimeError(f"{key} shape {im.shape} != asl_ref {ref.shape}")
            cio.save_gz(im, out / name, dtype=np.float32)
        row["status"] = "staged"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["reason"] = f"{type(exc).__name__}: {exc}"
    return row


def _worker(args):
    return stage_one(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-j", type=int, default=6)
    a = ap.parse_args()
    cio.META.mkdir(parents=True, exist_ok=True)
    subs = cio.find_subjects()
    if a.ids:
        subs = [s for s in subs if s["id"] in set(a.ids)]
    with Pool(a.j) as pool:
        rows = pool.map(_worker, [(s, a.force) for s in subs])
    manifest = cio.META / "staging_manifest.csv"
    # Merge with an existing manifest when staging a subset.
    old = {}
    if manifest.exists() and a.ids:
        with open(manifest) as fh:
            old = {r["id"]: r for r in csv.DictReader(fh)}
    for r in rows:
        old[r["id"]] = r
    cols = ["id", "group", "status", "reason", "brain_volume_ml", "t1_source", "asl_ref_source",
            "mask_source", "n_t1_candidates"]
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for sid in sorted(old):
            w.writerow({c: old[sid].get(c, "") for c in cols})
    n = {}
    for r in rows:
        n[r["status"]] = n.get(r["status"], 0) + 1
    print("staging summary:", n, "->", manifest)
    for r in rows:
        if r["status"] == "failed":
            print("FAILED", r["id"], r["reason"])


if __name__ == "__main__":
    main()
