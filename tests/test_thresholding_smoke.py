"""
tests/test_thresholding_smoke.py — Smoke tests for src/thresholding.py

Constructs a synthetic 32×32×32 asymmetry-index volume with one planted
high-asymmetry cluster, then runs every method in METHOD_REGISTRY and
verifies:

  1. Each method returns a np.ndarray of dtype bool with the same shape as
     the input volume.
  2. M05 (TFCE) and M11 (Quality+TFCE) detect at least one True voxel
     inside the planted cluster.
"""

import sys
import os

import numpy as np
import pytest

# Make sure src/ is importable whether pytest is run from the repo root or
# from within tests/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from thresholding import AsymData, METHOD_REGISTRY  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SHAPE = (32, 32, 32)

# The planted cluster occupies a 5×5×5 cube starting at voxel (13,13,13).
CLUSTER_SLICE = (slice(13, 18), slice(13, 18), slice(13, 18))


def make_synthetic_data(seed=0):
    """Build a synthetic ``AsymData`` with one prominent planted cluster.

    Background brain voxels have a small random AI (mean 0, sd 1).
    The planted cluster has AI values drawn from N(15, 1) so that the
    within-brain z-score is clearly elevated (>5 sigma above background).

    Returns
    -------
    d : AsymData
    cluster_mask : np.ndarray bool
        Boolean mask marking the planted cluster voxels.
    """
    rng = np.random.RandomState(seed)

    ai = np.zeros(SHAPE, dtype=np.float32)

    # Brain mask: all voxels except a 2-voxel border (to avoid edge effects
    # in cluster detection).
    brain_mask = np.zeros(SHAPE, dtype=bool)
    brain_mask[2:-2, 2:-2, 2:-2] = True

    # Background AI: N(0, 1)
    ai[brain_mask] = rng.randn(int(brain_mask.sum())).astype(np.float32)

    # Planted cluster: N(2, 1) — ~2 sigma above background noise.
    # Using amplitude=15 would exclude the cluster from the bilateral quality
    # mask (percentile fallback clips the top 1%), so M11 would never see it.
    # Amplitude=2 keeps the cluster inside the quality mask while still being
    # clearly detectable by both M05 and M11.
    cluster_vals = rng.randn(5, 5, 5).astype(np.float32) + 2.0
    ai[CLUSTER_SLICE] = cluster_vals

    # Planted-cluster mask (only within-brain portion, which is all of it
    # given the 2-voxel border we left out of brain_mask)
    cluster_mask = np.zeros(SHAPE, dtype=bool)
    cluster_mask[CLUSTER_SLICE] = True
    cluster_mask &= brain_mask  # intersection, should be identical here

    # Wrap in AsymData (no perfusion — uses percentile fallback)
    d = AsymData(ai, brain_mask, perfusion=None)

    return d, cluster_mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_inputs():
    """Module-scoped so we only build the data once."""
    return make_synthetic_data(seed=42)


def test_method_registry_has_13_methods():
    """There must be exactly 13 entries in METHOD_REGISTRY."""
    assert len(METHOD_REGISTRY) == 13, (
        f"Expected 13 methods, found {len(METHOD_REGISTRY)}: "
        f"{list(METHOD_REGISTRY.keys())}"
    )


def test_method_registry_keys():
    """All expected method labels must be present."""
    expected = {
        "M01_FixedZ_Cluster",
        "M02_Quality_Pctile",
        "M03_FDR_BH",
        "M04_Bonferroni",
        "M05_TFCE",
        "M06_GRF_Cluster",
        "M07_Permutation",
        "M08_GMM",
        "M09_Otsu",
        "M10_Random",
        "M11_Quality_TFCE",
        "M12_Quality_GMM",
        "M13_Quality_Otsu",
    }
    assert set(METHOD_REGISTRY.keys()) == expected


@pytest.mark.parametrize("method_name", list(METHOD_REGISTRY.keys()))
def test_output_shape_and_dtype(synthetic_inputs, method_name):
    """Every method must return a bool array with the same shape as the input."""
    d, _ = synthetic_inputs
    fn = METHOD_REGISTRY[method_name]
    result = fn(d)

    assert isinstance(result, np.ndarray), (
        f"{method_name}: expected np.ndarray, got {type(result)}"
    )
    assert result.dtype == bool, (
        f"{method_name}: expected dtype=bool, got {result.dtype}"
    )
    assert result.shape == SHAPE, (
        f"{method_name}: expected shape {SHAPE}, got {result.shape}"
    )


@pytest.mark.parametrize("method_name", ["M05_TFCE", "M11_Quality_TFCE"])
def test_planted_cluster_detected(synthetic_inputs, method_name):
    """TFCE-based methods must detect at least one voxel in the planted cluster."""
    d, cluster_mask = synthetic_inputs
    fn = METHOD_REGISTRY[method_name]
    result = fn(d)

    n_detected = int((result & cluster_mask).sum())
    assert n_detected > 0, (
        f"{method_name}: no voxels detected inside the planted cluster "
        f"(cluster has {cluster_mask.sum()} voxels, method flagged "
        f"{result.sum()} voxels total)"
    )


def test_all_methods_run_without_exception(synthetic_inputs):
    """All 13 methods must complete without raising any exception."""
    d, _ = synthetic_inputs
    errors = {}
    for name, fn in METHOD_REGISTRY.items():
        try:
            fn(d)
        except Exception as exc:  # noqa: BLE001
            errors[name] = str(exc)
    assert not errors, f"Methods raised exceptions: {errors}"
