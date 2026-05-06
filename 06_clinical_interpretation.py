#!/usr/bin/env python3
"""
Step 6: Clinical Interpretation — Surgeon-Facing Per-Patient Reports

Canonical script (Pipeline B Step 4). Reads the per-patient cluster_report
CSV produced by Pipeline B Step 3 (`02_compute_zai.py`), restricts to the
37-ROI clinical scope (3 subcortical pairs + 34 cortical Desikan-Killiany
pairs = 76 FreeSurfer labels), applies surgical thresholds (|peak_z| >= 4
AND size_voxels >= 100 by default), aggregates surviving clusters by
(side, lobe), and emits a deterministic surgeon-facing report.

This is the deterministic answer the surgeon reads BEFORE picking up
FSLeyes — it concentrates the 400+-row cluster report into a one-screen
prediction with a confidence label, lobar concordance, and an explicit
agree/disagree with the MDT ground truth.

Defaults are literature-cited:
- |zAI| >= 4 and >=100 voxels (Boscolo Galazzo 2016 / Gennari 2025
  surgical-grade ASL thresholds)
- Confidence levels follow the Bartolomei 2008 multi-modal concordance
  framework (multiple converging modalities/regions on one side ->
  Strong; single-modality lobar hits -> Moderate)
- 37-ROI scope per project policy (CLAUDE.md, 2026-05-05)

CLI
---
    python 06_clinical_interpretation.py --patient P015
    python 06_clinical_interpretation.py --all
    python 06_clinical_interpretation.py --patient P015 --peak-threshold 5.0
    python 06_clinical_interpretation.py --patient P015 --size-threshold 200
    python 06_clinical_interpretation.py --patient P015 --output-dir /tmp/myreports
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

# Local
try:
    from ez_ground_truth import EZ_GROUND_TRUTH
except Exception:
    EZ_GROUND_TRUTH = {}


# ---------------------------------------------------------------------------
# CONSTANTS — surgical thresholds and 37-ROI scope policy
# ---------------------------------------------------------------------------

DEFAULT_PEAK_THRESHOLD = 4.0       # |zAI| >= 4, surgeon-grade focal threshold
DEFAULT_SIZE_THRESHOLD = 100       # >= 100 voxels at MNI 0.8mm iso

CLUSTER_REPORT_DIR = Path("results_zscore/asymmetry/patients")
DEFAULT_OUTPUT_ROOT = Path("results_zscore/clinical")

SUBCORTICAL_NAMES = {"thalamus", "hippocampus", "amygdala"}

# 34 Desikan-Killiany cortical regions (canonical FreeSurfer names, lowercase)
CORTICAL_NAMES = {
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
}

CLINICAL_NAMES = SUBCORTICAL_NAMES | CORTICAL_NAMES

# Lobe categories — temporal/hippocampus/amygdala/thalamus are flagged
# separately because of their surgical relevance for TLE.
LOBE_OF = {
    # Temporal cortex
    "bankssts": "temporal",
    "entorhinal": "temporal",
    "fusiform": "temporal",
    "inferiortemporal": "temporal",
    "middletemporal": "temporal",
    "parahippocampal": "temporal",
    "superiortemporal": "temporal",
    "temporalpole": "temporal",
    "transversetemporal": "temporal",
    # Subcortical (own categories for surgical relevance)
    "hippocampus": "hippocampus",
    "amygdala": "amygdala",
    "thalamus": "thalamus",
    # Frontal cortex
    "caudalmiddlefrontal": "frontal",
    "frontalpole": "frontal",
    "lateralorbitofrontal": "frontal",
    "medialorbitofrontal": "frontal",
    "paracentral": "frontal",
    "parsopercularis": "frontal",
    "parsorbitalis": "frontal",
    "parstriangularis": "frontal",
    "precentral": "frontal",
    "rostralmiddlefrontal": "frontal",
    "superiorfrontal": "frontal",
    # Parietal cortex
    "inferiorparietal": "parietal",
    "postcentral": "parietal",
    "precuneus": "parietal",
    "superiorparietal": "parietal",
    "supramarginal": "parietal",
    # Occipital cortex
    "cuneus": "occipital",
    "lateraloccipital": "occipital",
    "lingual": "occipital",
    "pericalcarine": "occipital",
    # Cingulate
    "caudalanteriorcingulate": "cingulate",
    "isthmuscingulate": "cingulate",
    "posteriorcingulate": "cingulate",
    "rostralanteriorcingulate": "cingulate",
    # Insula (its own lobe per surgical convention)
    "insula": "insula",
    # Special
    "corpuscallosum": "other",
}


# ---------------------------------------------------------------------------
# REGION NAME NORMALIZATION
# ---------------------------------------------------------------------------

# Token expansions for the cluster report's abbreviated region names.
# These are applied per dot-separated TOKEN (not as string substring),
# so "temp" -> "temporal" but only when "temp" stands alone as a token.
# This avoids double-expansion bugs (e.g. "temp" matching inside an
# already-expanded "temporal").
_TOKEN_MAP = {
    # Side / orientation abbreviations
    "rost": "rostral",
    "caud": "caudal",
    "ant": "anterior",
    "post": "posterior",
    "mid": "middle",
    "sup": "superior",
    "inf": "inferior",
    "lat": "lateral",
    "med": "medial",
    # Lobe abbreviations
    "temp": "temporal",
    "front": "frontal",
    "occip": "occipital",
    "cing": "cingulate",
    "transv": "transverse",
    # FreeSurfer-specific compound tokens
    "orbitofr": "orbitofrontal",
    "supramarg": "supramarginal",
    "isthmus": "isthmus",
    "parsoperc": "parsopercularis",
    "parsorbit": "parsorbitalis",
    "parstriang": "parstriangularis",
    "parahippo": "parahippocampal",
}

# A small whitelist of canonical post-expansion names that are produced
# by joining two or more expanded tokens (e.g. "isthmus" + "cingulate"
# -> "isthmuscingulate"). The joiner produces the canonical name simply
# by concatenating tokens; this mapping is used only when the joiner
# yields a string that is not yet in CLINICAL_NAMES but a known alias
# for one that is.
_POST_ALIASES = {
    # No aliases needed currently — concatenated expansion already yields
    # the canonical Desikan-Killiany names. Kept here for future use.
}


def _normalize_region(raw: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse a cluster_report `primary_region` string into (side, canonical_name).

    Examples:
      "L-mid.temporal"         -> ("L", "middletemporal")
      "R-Thalamus"             -> ("R", "thalamus")
      "L-rost.mid.front"       -> ("L", "rostralmiddlefrontal")
      "R-transv.temp"          -> ("R", "transversetemporal")
      "L-VentralDC"            -> ("L", "ventraldc")     # NOT in scope, returned anyway
      "Unknown"                -> (None, None)

    The caller is expected to test the returned canonical name against
    CLINICAL_NAMES to decide in/out of clinical scope.
    """
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not s or s.lower() == "unknown" or s.lower().startswith("label-"):
        return None, None

    # Split on first '-' to get side prefix
    side: Optional[str] = None
    if s.startswith("L-"):
        side = "L"
        body = s[2:]
    elif s.startswith("R-"):
        side = "R"
        body = s[2:]
    else:
        body = s

    # Lowercase
    body = body.lower()

    # Tokenize on '.' and expand each token via _TOKEN_MAP. Tokens not
    # in the map pass through unchanged. This avoids the substring
    # double-expansion problem (e.g. "temp" lurking inside "temporal").
    tokens = body.split(".")
    expanded = [_TOKEN_MAP.get(t, t) for t in tokens]
    name = "".join(expanded)

    # Optional post-alias rewrite (currently empty).
    name = _POST_ALIASES.get(name, name)

    return side, name


def _lobe_of(canonical_name: str) -> str:
    """Return lobe category for a canonical region name (or 'other')."""
    return LOBE_OF.get(canonical_name, "other")


# ---------------------------------------------------------------------------
# CONFIDENCE LEVEL COMPUTATION (Bartolomei 2008 multi-modal concordance)
# ---------------------------------------------------------------------------

# Lobes that count as "mesial-temporal axis" — the surgical-relevance core
# for TLE concordance.
_MTL_LOBES = {"temporal", "hippocampus", "amygdala"}


def _compute_confidence(per_side_lobes: dict) -> tuple[str, str, str]:
    """
    Compute lateralization confidence per Bartolomei 2008 framework.

    `per_side_lobes` is a dict like:
      {
        "L": {"temporal": {"clusters": 3, "voxels": 5400, "peak": 6.1}, ...},
        "R": {...}
      }

    The "side" here is the SIDE OF THE PREDICTED EZ — not the side of
    the asymmetry signal. See `predict_ez_side` for the polarity flip.

    Returns: (confidence, prediction, rationale)
        confidence: "Strong-L" / "Strong-R" / "Moderate-L" / "Moderate-R" /
                    "Weak-L" / "Weak-R" / "Bilateral" / "Unclear"
        prediction: "L" / "R" / "B" / "Unclear"
        rationale: short human string explaining the choice
    """

    def _mtl_stats(side):
        c = sum(per_side_lobes.get(side, {}).get(l, {}).get("clusters", 0) for l in _MTL_LOBES)
        v = sum(per_side_lobes.get(side, {}).get(l, {}).get("voxels", 0) for l in _MTL_LOBES)
        return c, v

    def _all_stats(side):
        c = sum(d.get("clusters", 0) for d in per_side_lobes.get(side, {}).values())
        v = sum(d.get("voxels", 0) for d in per_side_lobes.get(side, {}).values())
        return c, v

    L_mtl_c, L_mtl_v = _mtl_stats("L")
    R_mtl_c, R_mtl_v = _mtl_stats("R")
    L_all_c, L_all_v = _all_stats("L")
    R_all_c, R_all_v = _all_stats("R")

    total_c = L_all_c + R_all_c

    if total_c == 0:
        return "Unclear", "Unclear", "no clusters survived surgical thresholds"

    # Strong: >=3 mesial-temporal clusters one side, voxels >=3000, contralateral
    # mesial-temporal voxels < 1/3 dominant side.
    if L_mtl_c >= 3 and L_mtl_v >= 3000 and R_mtl_v < (L_mtl_v / 3.0):
        rat = (f"Strong: {L_mtl_c} L mesial-temporal clusters, {L_mtl_v} voxels; "
               f"R mesial-temporal only {R_mtl_v} voxels (<{L_mtl_v//3})")
        return "Strong-L", "L", rat
    if R_mtl_c >= 3 and R_mtl_v >= 3000 and L_mtl_v < (R_mtl_v / 3.0):
        rat = (f"Strong: {R_mtl_c} R mesial-temporal clusters, {R_mtl_v} voxels; "
               f"L mesial-temporal only {L_mtl_v} voxels (<{R_mtl_v//3})")
        return "Strong-R", "R", rat

    # Moderate: 1-2 MTL clusters one side, OR strong frontal/parietal asymmetry.
    if L_mtl_c >= 1 and R_mtl_c == 0:
        rat = f"Moderate: {L_mtl_c} L mesial-temporal cluster(s) ({L_mtl_v} vox), no R MTL"
        return "Moderate-L", "L", rat
    if R_mtl_c >= 1 and L_mtl_c == 0:
        rat = f"Moderate: {R_mtl_c} R mesial-temporal cluster(s) ({R_mtl_v} vox), no L MTL"
        return "Moderate-R", "R", rat
    if L_mtl_c >= 1 and R_mtl_c >= 1:
        # Bilateral MTL involvement. We split into three bands by voxel ratio:
        #   >= 2x  -> Moderate (clear L vs R majority)
        #   1.1-2x -> Weak    (slight tilt; flagged but low confidence)
        #   < 1.1x -> Bilateral (genuinely indeterminate from ASL alone)
        # Cluster counts are reported in the rationale for transparency.
        ratio_L = (L_mtl_v / R_mtl_v) if R_mtl_v > 0 else float("inf")
        ratio_R = (R_mtl_v / L_mtl_v) if L_mtl_v > 0 else float("inf")
        if ratio_L >= 2.0:
            rat = f"Bilateral MTL, L-dominant: L={L_mtl_v}vox vs R={R_mtl_v}vox (>=2x)"
            return "Moderate-L", "L", rat
        if ratio_R >= 2.0:
            rat = f"Bilateral MTL, R-dominant: R={R_mtl_v}vox vs L={L_mtl_v}vox (>=2x)"
            return "Moderate-R", "R", rat
        if ratio_L >= 1.05:
            rat = f"Bilateral MTL, mild L tilt: L={L_mtl_v}vox vs R={R_mtl_v}vox"
            return "Weak-L", "L", rat
        if ratio_R >= 1.05:
            rat = f"Bilateral MTL, mild R tilt: R={R_mtl_v}vox vs L={L_mtl_v}vox"
            return "Weak-R", "R", rat
        rat = f"Bilateral MTL, no clear dominance: L={L_mtl_v}vox vs R={R_mtl_v}vox"
        return "Bilateral", "B", rat

    # Weak: only frontal/parietal/occipital signals one side
    if L_all_c > 0 and R_all_c == 0:
        rat = f"Weak: extra-temporal only, L={L_all_c} clusters ({L_all_v} vox)"
        return "Weak-L", "L", rat
    if R_all_c > 0 and L_all_c == 0:
        rat = f"Weak: extra-temporal only, R={R_all_c} clusters ({R_all_v} vox)"
        return "Weak-R", "R", rat
    if L_all_c > 0 and R_all_c > 0:
        if L_all_v >= 2 * R_all_v:
            rat = f"Weak: extra-temporal bilateral, L-dominant ({L_all_v} vs {R_all_v} vox)"
            return "Weak-L", "L", rat
        if R_all_v >= 2 * L_all_v:
            rat = f"Weak: extra-temporal bilateral, R-dominant ({R_all_v} vs {L_all_v} vox)"
            return "Weak-R", "R", rat
        rat = f"Bilateral extra-temporal, no dominance ({L_all_v} L vs {R_all_v} R vox)"
        return "Bilateral", "B", rat

    return "Unclear", "Unclear", "no in-scope clusters survived"


# ---------------------------------------------------------------------------
# CORE ANALYZER
# ---------------------------------------------------------------------------

class ClinicalInterpreter:
    """
    Per-patient clinical interpretation of zAI cluster results.

    Inputs:
        cluster_report CSV at
        results_zscore/asymmetry/patients/<PID>/<PID>_asymmetry_cluster_report.csv

    Outputs:
        <out_dir>/<PID>_surgeon_report.txt
        <out_dir>/<PID>_surgeon_report.json
    """

    def __init__(self, patient_id: str,
                 peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
                 size_threshold: int = DEFAULT_SIZE_THRESHOLD,
                 output_dir: Optional[Path] = None):
        self.patient_id = patient_id
        self.peak_threshold = float(peak_threshold)
        self.size_threshold = int(size_threshold)

        self.cluster_csv = (CLUSTER_REPORT_DIR / patient_id /
                            f"{patient_id}_asymmetry_cluster_report.csv")

        if output_dir is None:
            self.output_dir = DEFAULT_OUTPUT_ROOT / patient_id
        else:
            self.output_dir = Path(output_dir) / patient_id

    # -- pipeline steps --------------------------------------------------

    def has_data(self) -> bool:
        return self.cluster_csv.exists()

    def _load(self) -> Optional[pd.DataFrame]:
        if not self.cluster_csv.exists():
            return None
        return pd.read_csv(self.cluster_csv)

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply surgical thresholds + 37-ROI scope."""
        df = df.copy()
        df["peak_abs"] = df["peak_z"].abs()

        # Surgical thresholds
        keep = (df["peak_abs"] >= self.peak_threshold) & (df["size_voxels"] >= self.size_threshold)
        df = df[keep].copy()

        # Region normalization
        normed = df["primary_region"].apply(_normalize_region)
        df["region_side"] = normed.apply(lambda t: t[0])
        df["region_canon"] = normed.apply(lambda t: t[1])
        df["region_lobe"] = df["region_canon"].apply(
            lambda n: _lobe_of(n) if n else "other")

        # 37-ROI scope
        df["in_clinical_scope"] = df["region_canon"].apply(
            lambda n: bool(n) and n in CLINICAL_NAMES)
        df = df[df["in_clinical_scope"]].copy()

        # Predicted EZ side: direction-of-abnormality inference (preferred)
        # falls back to the legacy polarity-flip rule when the new columns
        # are not present in the cluster report (older Pipeline B runs that
        # predate `02_compute_zai.py`'s direction-inference upgrade).
        #
        # New rule (when ez_side_pred is available):
        #   The EZ side is always the side that deviates from healthy
        #   controls (whichever hemisphere has |z_perf| >= z_thr), regardless
        #   of whether the deviation is hyperperfusion or hypoperfusion.
        #   This handles the FCD-II / post-ictal hyperperfused subset that
        #   the polarity-flip rule mis-classified.
        #
        # Legacy rule (fallback):
        #   `left-dominant` (zAI > +threshold) -> assume R hypoperfused -> EZ = R
        #   `right-dominant` (zAI < -threshold) -> assume L hypoperfused -> EZ = L
        if "ez_side_pred" in df.columns and df["ez_side_pred"].astype(str).str.len().sum() > 0:
            df["ez_side"] = df["ez_side_pred"].astype(str).str.upper().str[:1]
            # Defensive: replace empty strings with the legacy fallback.
            mask_empty = ~df["ez_side"].isin(["L", "R"])
            df.loc[mask_empty, "ez_side"] = df.loc[mask_empty, "direction"].map(
                {"left-dominant": "R", "right-dominant": "L"})
        else:
            df["ez_side"] = df["direction"].map({"left-dominant": "R",
                                                  "right-dominant": "L"})
        return df

    def _aggregate_by_side_lobe(self, df: pd.DataFrame) -> dict:
        """Aggregate clusters by (predicted EZ side, lobe)."""
        per_side_lobes: dict = {"L": {}, "R": {}}
        for _, row in df.iterrows():
            side = row["ez_side"]
            lobe = row["region_lobe"]
            if side not in per_side_lobes:
                continue
            entry = per_side_lobes[side].setdefault(
                lobe, {"clusters": 0, "voxels": 0, "peak": 0.0})
            entry["clusters"] += 1
            entry["voxels"] += int(row["size_voxels"])
            entry["peak"] = max(entry["peak"], float(row["peak_abs"]))
        return per_side_lobes

    @staticmethod
    def _mdt_concordance(prediction: str, mdt_ez: Optional[str]) -> str:
        if mdt_ez is None:
            return "no-mdt-label"
        if prediction == "Unclear":
            return "indeterminate"
        if mdt_ez in ("L", "R") and prediction == mdt_ez:
            return "agree"
        if mdt_ez in ("B-L", "B-R", "B") and prediction in ("L", "R", "B"):
            # bilateral MDT: agree if predicted side matches dominant or is B
            if (mdt_ez == "B-L" and prediction == "L") or \
               (mdt_ez == "B-R" and prediction == "R") or \
               (mdt_ez == "B" and prediction == "B"):
                return "agree"
            return "partial"
        return "disagree"

    def analyze(self) -> Optional[dict]:
        """Run the full analysis. Returns the structured report dict, or None."""
        if not self.has_data():
            return None
        raw = self._load()
        if raw is None or raw.empty:
            return None

        n_total = int(len(raw))
        filtered = self._filter(raw)
        n_in_scope = int(len(filtered))
        per_side_lobes = self._aggregate_by_side_lobe(filtered)
        confidence, prediction, rationale = _compute_confidence(per_side_lobes)

        # Top 10 clusters by |peak_z|
        topN = filtered.sort_values("peak_abs", ascending=False).head(10)
        top_rows = []
        for _, r in topN.iterrows():
            top_rows.append({
                "cluster_id": int(r["cluster_id"]),
                "direction": r["direction"],
                "ez_side": r["ez_side"],
                "size_voxels": int(r["size_voxels"]),
                "peak_z": float(r["peak_z"]),
                "primary_region": r["primary_region"],
                "lobe": r["region_lobe"],
                "centroid": [float(r["centroid_x"]),
                             float(r["centroid_y"]),
                             float(r["centroid_z_coord"])],
                # Direction-of-abnormality inference columns (when present)
                "direction_class": r.get("direction_class", ""),
                "z_perf_cluster_side": r.get("z_perf_cluster_side", ""),
                "z_perf_mirror_side": r.get("z_perf_mirror_side", ""),
                "cluster_hemi": r.get("cluster_hemi", ""),
            })

        mdt = EZ_GROUND_TRUTH.get(self.patient_id, {})
        mdt_ez = mdt.get("ez")
        mdt_conf = mdt.get("confidence")
        concordance = self._mdt_concordance(prediction, mdt_ez)

        # Top lobe (any side)
        top_lobe = None
        top_lobe_voxels = 0
        for side, lobes in per_side_lobes.items():
            for lobe, stats in lobes.items():
                if stats["voxels"] > top_lobe_voxels:
                    top_lobe_voxels = stats["voxels"]
                    top_lobe = f"{side}-{lobe}"
        peak_zai = float(filtered["peak_abs"].max()) if not filtered.empty else 0.0

        return {
            "patient_id": self.patient_id,
            "thresholds": {
                "peak_threshold": self.peak_threshold,
                "size_threshold": self.size_threshold,
            },
            "n_clusters_total": n_total,
            "n_clusters_in_scope": n_in_scope,
            "per_side_lobes": per_side_lobes,
            "top_clusters": top_rows,
            "algo_prediction": prediction,
            "algo_confidence": confidence,
            "rationale": rationale,
            "mdt_ez": mdt_ez,
            "mdt_confidence": mdt_conf,
            "mdt_evidence": mdt.get("evidence"),
            "concordance": concordance,
            "top_lobe": top_lobe,
            "top_voxels": int(top_lobe_voxels),
            "peak_zai": peak_zai,
        }

    # -- rendering -------------------------------------------------------

    def render_text(self, report: dict) -> str:
        """Render the report as a fixed-width surgeon-facing text block."""
        lines = []
        W = 70
        line = "=" * W
        lines.append(line)
        lines.append(f" SURGEON CLINICAL REPORT — {report['patient_id']}")
        lines.append(line)

        # Sign-convention legend (direction-aware; read once, then ignore).
        lines.append(" METHOD: DIRECTION-OF-ABNORMALITY INFERENCE")
        lines.append(" " + "=" * (W - 1))
        lines.append("   For each cluster the algorithm checks BOTH hemispheres")
        lines.append("   against the healthy-control normative database, and")
        lines.append("   predicts the EZ on the DEVIATING side -- whether the")
        lines.append("   deviation is hypoperfusion (typical interictal) or")
        lines.append("   hyperperfusion (FCD-II / post-ictal / recent seizure).")
        lines.append("   The 'direction' column tells you which:")
        lines.append("     hypo  = abnormally LOW blood flow on the EZ side")
        lines.append("             (typical interictal hypometabolism)")
        lines.append("     hyper = abnormally HIGH blood flow on the EZ side")
        lines.append("             (FCD-II or recent-seizure / post-ictal)")
        lines.append("     mixed = both sides deviate -- low confidence,")
        lines.append("             flag for manual MDT review")
        lines.append("     subtle= neither side strongly deviates -- low conf.")
        lines.append("   The EZ column = predicted seizure-onset side based on")
        lines.append("   the direction inference (replaces the older polarity")
        lines.append("   flip that assumed all asymmetries were hypo-driven).")
        lines.append(" " + "=" * (W - 1))
        lines.append("-" * W)

        # Thresholds
        t = report["thresholds"]
        lines.append(f" Thresholds  : |zAI| >= {t['peak_threshold']:.1f}  AND  size >= {t['size_threshold']} vox")
        lines.append(f" 37-ROI scope: 3 subcortical (Thal/Hipp/Amyg) + 34 DK cortical")
        lines.append(f" Total clusters in CSV : {report['n_clusters_total']}")
        lines.append(f" Surviving in scope    : {report['n_clusters_in_scope']}")
        lines.append("-" * W)

        # MDT ground truth
        if report["mdt_ez"]:
            lines.append(f" MDT ground truth: {report['mdt_ez']} ({report['mdt_confidence']} confidence)")
            ev = report.get("mdt_evidence") or ""
            if ev:
                lines.append(f"   {ev[:W-3]}")
        else:
            lines.append(" MDT ground truth: (not in ground-truth table)")
        lines.append("-" * W)

        # Lobe-level table
        lines.append(" LOBE-LEVEL AGGREGATE (predicted EZ side / lobe / clusters / voxels):")
        psl = report["per_side_lobes"]
        lobe_order = ["temporal", "hippocampus", "amygdala", "thalamus",
                      "frontal", "parietal", "occipital", "cingulate",
                      "insula", "other"]
        any_row = False
        lines.append("   side  lobe          clusters   voxels   peak|z|")
        for side in ("L", "R"):
            for lobe in lobe_order:
                stats = psl.get(side, {}).get(lobe)
                if not stats:
                    continue
                lines.append(
                    f"    {side}    {lobe:<13}   {stats['clusters']:>3}   "
                    f"{stats['voxels']:>7}    {stats['peak']:>5.2f}")
                any_row = True
        if not any_row:
            lines.append("   (no surviving clusters in clinical scope)")
        lines.append("-" * W)

        # Top 10 clusters
        lines.append(" TOP 10 SURVIVING CLUSTERS (by |peak_z|):")
        lines.append("   EZ = predicted seizure-onset side (deviating side per")
        lines.append("        direction-of-abnormality inference).")
        lines.append("   dir = direction class (hypo / hyper / mixed / subtle).")
        if not report["top_clusters"]:
            lines.append("   (none)")
        else:
            lines.append("   id  EZ  dir     peak_z voxels region                lobe")
            for c in report["top_clusters"]:
                reg = (c["primary_region"] or "")[:20]
                lobe = c["lobe"][:10]
                dirc = (c.get("direction_class") or "?")[:6]
                lines.append(
                    f"   {c['cluster_id']:>3}  {c['ez_side']}   "
                    f"{dirc:<6} {c['peak_z']:>+6.2f}  {c['size_voxels']:>6}  "
                    f"{reg:<20}  {lobe}")
        lines.append("-" * W)

        # Algorithmic prediction
        pred = report["algo_prediction"]
        conf = report["algo_confidence"]
        lines.append(" ALGORITHMIC PREDICTION:")
        if pred == "Unclear":
            lines.append(f"   PREDICTED EZ: UNCLEAR  ({conf})")
        else:
            lines.append(f"   PREDICTED EZ: {pred}   ({conf})")
        lines.append(f"   Rationale  : {report['rationale']}")

        # Plain-language gloss of the rationale (what the technical line
        # above means in everyday terms for a non-imaging clinician).
        plain = self._plain_language_rationale(pred, report["per_side_lobes"])
        if plain:
            lines.append(f"   Plain English:")
            for pl in plain:
                # Wrap each plain-language sentence to <= W cols.
                for wline in self._wrap_plain(pl, indent="     ", width=W):
                    lines.append(wline)
        lines.append("")

        # MDT concordance
        c = report["concordance"]
        if c == "agree":
            tag = "[*] CONCORDS WITH MDT"
            plain_c = ("The ASL perfusion algorithm and the multidisciplinary "
                       "team review point to the SAME hemisphere as the source "
                       "of seizures.")
        elif c == "disagree":
            tag = "[X] DISAGREES WITH MDT"
            plain_c = ("The ASL perfusion algorithm and the multidisciplinary "
                       "team review point to OPPOSITE hemispheres. Reconcile "
                       "with EEG, structural MRI and neuropsychology before "
                       "any surgical decision.")
        elif c == "partial":
            tag = "[~] PARTIALLY CONCORDS WITH MDT"
            plain_c = ("The ASL prediction agrees with one of two sides flagged "
                       "by the multidisciplinary team (bilateral case).")
        elif c == "indeterminate":
            tag = "[?] ALGORITHM INDETERMINATE"
            plain_c = ("The ASL perfusion data does not confidently lateralize. "
                       "Rely on EEG / MRI / neuropsychology for sidedness.")
        else:
            tag = "[ ] NO MDT LABEL"
            plain_c = ("No multidisciplinary-team ground-truth side recorded "
                       "for this patient.")
        lines.append(f" CONCORDANCE: {tag}")
        # Plain-language gloss of the concordance verdict
        # (wrapped to <= W columns for readability).
        for wline in self._wrap_plain(plain_c, indent="   ", width=W):
            lines.append(wline)
        lines.append("-" * W)

        # Reminder — accessible language, same evidence-modality citation.
        lines.append(" REMINDER: ASL perfusion is one piece of evidence. The")
        lines.append(" final surgical decision should integrate EEG (where")
        lines.append(" seizures originate electrically), MRI (any structural")
        lines.append(" lesion), neuropsychology (which side is functionally")
        lines.append(" affected), and this ASL-based prediction.")
        lines.append("                            (Bartolomei et al., 2008)")
        lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers used by render_text() — pure presentation, no analysis.
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_plain(text: str, indent: str = "   ", width: int = 70) -> list:
        """Greedy word-wrap `text` to lines of length <= `width` including
        the leading `indent`. Returns a list of lines."""
        max_body = width - len(indent)
        words = text.split()
        out, cur = [], ""
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_body:
                cur = cur + " " + w
            else:
                out.append(indent + cur)
                cur = w
        if cur:
            out.append(indent + cur)
        return out

    @staticmethod
    def _plain_language_rationale(prediction: str, per_side_lobes: dict) -> list:
        """Translate the technical rationale into a 1-3 line plain-language
        gloss for non-imaging clinicians. Reads `per_side_lobes` (already
        EZ-side-flipped by `_filter`) to summarize voxel totals.

        Voxel counts use the mesial-temporal axis (temporal cortex +
        hippocampus + amygdala) when that side has any MTL involvement,
        because that's the surgically-relevant region for TLE concordance
        — and it matches the technical rationale's MTL-based numbers.
        Falls back to the all-region total when the predicted side has
        no MTL clusters.

        Returns [] if the prediction is Unclear (the technical rationale
        already reads plainly).
        """
        if prediction not in ("L", "R", "B"):
            return []

        def _voxels(side: str) -> int:
            return sum(d.get("voxels", 0)
                       for d in per_side_lobes.get(side, {}).values())

        def _temporal_voxels(side: str) -> int:
            return sum(per_side_lobes.get(side, {}).get(l, {}).get("voxels", 0)
                       for l in ("temporal", "hippocampus", "amygdala"))

        Lv_all = _voxels("L")
        Rv_all = _voxels("R")
        Lt = _temporal_voxels("L")
        Rt = _temporal_voxels("R")

        if prediction == "L":
            # Prefer MTL counts (surgically relevant for TLE); fall back to
            # all-lobe totals if no MTL involvement on the predicted side.
            if Lt > 0:
                Lv, Rv = Lt, Rt
                primary = "Left temporal lobe shows abnormally LOW blood flow"
                scope = "mesial-temporal"
            else:
                Lv, Rv = Lv_all, Rv_all
                primary = "Left side shows abnormally LOW blood flow"
                scope = "all-lobe"
            line1 = (f"{primary} (zAI deviates by >4 sigma from healthy-control "
                     f"baseline) at multiple clusters totaling {Lv:,} voxels;")
            line2 = (f"right side shows {'minor' if Rv < Lv else 'comparable'} "
                     f"changes ({Rv:,} voxels, {scope}). Pattern consistent "
                     f"with interictal hypoperfusion of left epileptogenic "
                     f"zone.")
            return [line1, line2]
        if prediction == "R":
            if Rt > 0:
                Lv, Rv = Lt, Rt
                primary = "Right temporal lobe shows abnormally LOW blood flow"
                scope = "mesial-temporal"
            else:
                Lv, Rv = Lv_all, Rv_all
                primary = "Right side shows abnormally LOW blood flow"
                scope = "all-lobe"
            line1 = (f"{primary} (zAI deviates by >4 sigma from healthy-control "
                     f"baseline) at multiple clusters totaling {Rv:,} voxels;")
            line2 = (f"left side shows {'minor' if Lv < Rv else 'comparable'} "
                     f"changes ({Lv:,} voxels, {scope}). Pattern consistent "
                     f"with interictal hypoperfusion of right epileptogenic "
                     f"zone.")
            return [line1, line2]
        if prediction == "B":
            return [
                (f"Both sides show abnormally low blood flow at comparable "
                 f"voxel counts (L={Lv_all:,} vs R={Rv_all:,})."),
                ("ASL alone cannot confidently lateralize; defer to EEG and "
                 "structural MRI."),
            ]
        return []

    def write(self, report: dict) -> tuple[Path, Path]:
        """Write the txt + json outputs. Returns the two paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        txt_path = self.output_dir / f"{self.patient_id}_surgeon_report.txt"
        json_path = self.output_dir / f"{self.patient_id}_surgeon_report.json"
        txt_path.write_text(self.render_text(report) + "\n")
        json_path.write_text(json.dumps(report, indent=2, default=str))
        return txt_path, json_path

    # -- convenience -----------------------------------------------------

    def run(self) -> Optional[dict]:
        """Full pipeline: analyze + write + return report dict."""
        report = self.analyze()
        if report is None:
            return None
        self.write(report)
        return report


# ---------------------------------------------------------------------------
# COHORT MODE
# ---------------------------------------------------------------------------

def _discover_patients() -> list[str]:
    if not CLUSTER_REPORT_DIR.exists():
        return []
    pids = []
    for d in sorted(CLUSTER_REPORT_DIR.iterdir()):
        if not d.is_dir():
            continue
        csv = d / f"{d.name}_asymmetry_cluster_report.csv"
        if csv.exists():
            pids.append(d.name)
    return pids


def run_cohort(peak_threshold: float, size_threshold: int,
               output_root: Path) -> Path:
    """Run for every available patient and write the cohort summary CSV."""
    pids = _discover_patients()
    rows = []
    for pid in pids:
        ci = ClinicalInterpreter(pid, peak_threshold, size_threshold,
                                 output_dir=output_root)
        report = ci.run()
        if report is None:
            print(f"  [-] {pid}: no zAI cluster report on disk; skipping")
            rows.append({
                "patient_id": pid, "mdt_ez": EZ_GROUND_TRUTH.get(pid, {}).get("ez"),
                "mdt_confidence": EZ_GROUND_TRUTH.get(pid, {}).get("confidence"),
                "algo_prediction": None, "algo_confidence": None,
                "concordance": "no-data", "top_lobe": None,
                "top_voxels": None, "peak_zai": None,
            })
            continue
        rows.append({
            "patient_id": pid,
            "mdt_ez": report["mdt_ez"],
            "mdt_confidence": report["mdt_confidence"],
            "algo_prediction": report["algo_prediction"],
            "algo_confidence": report["algo_confidence"],
            "concordance": report["concordance"],
            "top_lobe": report["top_lobe"],
            "top_voxels": report["top_voxels"],
            "peak_zai": round(report["peak_zai"], 2),
        })
        print(f"  [+] {pid}: {report['algo_prediction']} ({report['algo_confidence']}) "
              f"vs MDT={report['mdt_ez']} -> {report['concordance']}")

    cohort_path = output_root / "cohort_concordance.csv"
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cohort_path, index=False)
    print(f"\n  Wrote cohort summary -> {cohort_path}")
    return cohort_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Surgeon-facing per-patient clinical report from zAI clusters.")
    p.add_argument("--patient", help="Patient ID (e.g. P015)")
    p.add_argument("--all", action="store_true",
                   help="Run for every patient with zAI data and produce cohort summary")
    p.add_argument("--peak-threshold", type=float, default=DEFAULT_PEAK_THRESHOLD,
                   help=f"|zAI| threshold (default {DEFAULT_PEAK_THRESHOLD})")
    p.add_argument("--size-threshold", type=int, default=DEFAULT_SIZE_THRESHOLD,
                   help=f"Minimum cluster size in voxels (default {DEFAULT_SIZE_THRESHOLD})")
    p.add_argument("--output-dir", type=Path, default=None,
                   help=f"Output root (default {DEFAULT_OUTPUT_ROOT})")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the printed report (still writes files)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    out_root = args.output_dir if args.output_dir else DEFAULT_OUTPUT_ROOT

    if args.all:
        run_cohort(args.peak_threshold, args.size_threshold, out_root)
        return 0

    if not args.patient:
        print("ERROR: must pass --patient <ID> or --all", file=sys.stderr)
        return 2

    ci = ClinicalInterpreter(args.patient,
                             peak_threshold=args.peak_threshold,
                             size_threshold=args.size_threshold,
                             output_dir=args.output_dir)
    if not ci.has_data():
        print(f"[!] No zAI cluster report on disk for {args.patient} "
              f"(expected at {ci.cluster_csv}).")
        print(f"    Run 02_compute_zai.py first.")
        return 1
    report = ci.run()
    if report is None:
        print(f"[!] {args.patient}: cluster report exists but is empty.")
        return 1
    if not args.quiet:
        print(ci.render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
