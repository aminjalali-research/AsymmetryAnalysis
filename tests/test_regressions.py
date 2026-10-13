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
"""

import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path

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
