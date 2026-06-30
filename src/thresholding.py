"""
src/thresholding.py — 13 Thresholding/Clustering Methods for Asymmetry Maps

Factored out from ``control_zscore_asymmetry_clustering.py`` during the
2026-05-04 cleanup phase.  The algorithms are copied verbatim; only the
import block and the module-level path constants from the original script
have been removed (they belong to the pipeline, not to this library module).

## The 13 methods

| Key                   | Function               | Description                        |
|-----------------------|------------------------|------------------------------------|
| M01_FixedZ_Cluster    | m1_fixed_z_cluster     | Fixed z-threshold + cluster filter |
| M02_Quality_Pctile    | m2_quality_percentile  | Quality mask + percentile cutoff   |
| M03_FDR_BH            | m3_fdr                 | Benjamini-Hochberg FDR             |
| M04_Bonferroni        | m4_bonferroni          | Bonferroni correction              |
| M05_TFCE              | m5_tfce                | Threshold-Free Cluster Enhancement |
| M06_GRF_Cluster       | m6_grf                 | Gaussian Random Field              |
| M07_Permutation       | m7_permutation         | Permutation / null distribution    |
| M08_GMM               | m8_gmm                 | Gaussian Mixture Model             |
| M09_Otsu              | m9_otsu                | Otsu's method                      |
| M10_Random            | m10_random             | Random baseline                    |
| M11_Quality_TFCE      | m11_quality_tfce       | Quality mask + TFCE (best)         |
| M12_Quality_GMM       | m12_quality_gmm        | Quality mask + GMM                 |
| M13_Quality_Otsu      | m13_quality_otsu       | Quality mask + Otsu                |

## Input contract — ``AsymData``

Each method accepts a single ``AsymData`` instance with the following fields:

    ai          : np.ndarray, shape (X,Y,Z) — signed asymmetry index map
    abs_ai      : np.ndarray, shape (X,Y,Z) — abs(ai)
    brain_mask  : np.ndarray bool, shape (X,Y,Z) — valid-brain voxels
    n_brain     : int — count of brain-mask voxels
    zscore      : np.ndarray, shape (X,Y,Z) — z-score of ai within brain
    abs_z       : np.ndarray, shape (X,Y,Z) — abs(zscore)
    perfusion   : np.ndarray or None — raw perfusion volume (optional)

    bilateral_quality_mask(min_frac) -> bool array — voxels with adequate
        bilateral perfusion (falls back to a percentile filter when perfusion
        is None).

## Returns

Every method returns a ``np.ndarray`` of dtype ``bool`` with the same shape
as ``d.brain_mask``, where ``True`` marks "significant" voxels.

## Registry

``METHOD_REGISTRY`` is the canonical dict ``{label: callable}`` for iterating
over all 13 methods.
"""

import math
import warnings

import numpy as np
from scipy import ndimage, stats
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

M1_ZSCORE_THRESHOLD = 1.64
M1_MIN_CLUSTER = 50
M2_PERCENTILE = 90
M2_MIN_CLUSTER = 10
M2_MIN_PERF_FRAC = 0.10
M3_Q_VALUE = 0.05
M4_ALPHA = 0.05
M5_H_POWER = 2.0
M5_E_POWER = 0.5
M5_DH = 0.1
M5_MIN_CLUSTER = 10
M6_VOXEL_PTHRESH = 0.001
M6_CLUSTER_ALPHA = 0.05
M7_N_PERMUTATIONS = 200   # cluster-level null at the 95th pctile; 200 perms give
                          # adequate resolution and keep the 17-patient sweep tractable
                          # (1000 perms x 17 patients x full-volume labelling = hours).
M7_CLUSTER_FORMING_Z = 1.64
M7_ALPHA = 0.05
M8_N_COMPONENTS = 3
M8_MIN_CLUSTER = 10
M9_MIN_CLUSTER = 10
M10_COVERAGE_PCT = 3.0
CLUSTER_CONNECTIVITY = 2


# ============================================================================
# INPUT CONTRACT
# ============================================================================

class AsymData:
    """Wrapper that bundles all inputs required by the 13 thresholding methods.

    Parameters
    ----------
    ai_map : np.ndarray, shape (X, Y, Z)
        Signed asymmetry-index volume (any of the 11 AI methods).
    brain_mask : np.ndarray bool, shape (X, Y, Z)
        Voxels that belong to the brain (or gray-matter-restricted brain).
    perfusion : np.ndarray or None
        Raw perfusion volume in the same space; used by the bilateral quality
        mask.  When ``None`` a percentile-based fallback is used instead.
    """

    def __init__(self, ai_map, brain_mask, perfusion=None):
        self.ai = ai_map
        self.abs_ai = np.abs(ai_map)
        self.brain_mask = brain_mask
        self.n_brain = int(brain_mask.sum())
        self.perfusion = perfusion

        # z-score of AI within brain
        brain_vals = ai_map[brain_mask]
        mu, sigma = brain_vals.mean(), brain_vals.std()
        self.zscore = np.zeros_like(ai_map)
        if sigma > 1e-6:
            self.zscore[brain_mask] = (ai_map[brain_mask] - mu) / sigma
        self.abs_z = np.abs(self.zscore)

    def bilateral_quality_mask(self, min_frac=M2_MIN_PERF_FRAC):
        """Return a bool mask of voxels with adequate bilateral perfusion.

        When a perfusion volume is available, excludes voxels where either
        the voxel itself or its left-right mirror falls below
        ``min_frac * median(CBF)``.  Falls back to a 99th-percentile
        outlier-exclusion mask when no perfusion is provided.
        """
        if self.perfusion is not None:
            perf = self.perfusion
            perf_flip = np.flip(perf, axis=0)
            brain_perf = perf[self.brain_mask]
            median_cbf = np.median(brain_perf[brain_perf > 0]) if (brain_perf > 0).any() else 1.0
            min_cbf = min_frac * median_cbf
            return self.brain_mask & (perf > min_cbf) & (perf_flip > min_cbf)
        else:
            return self.brain_mask & (self.abs_ai < np.percentile(self.abs_ai[self.brain_mask], 99))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_structure_3d():
    """Return the 3-D binary structure used for connected-component labeling."""
    return ndimage.generate_binary_structure(3, CLUSTER_CONNECTIVITY)


def cluster_filter(mask, min_size):
    """Remove connected components smaller than ``min_size`` voxels."""
    if min_size <= 0:
        return mask
    struct = get_structure_3d()
    labeled, n = ndimage.label(mask, structure=struct)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    for i, s in enumerate(sizes, 1):
        if s < min_size:
            labeled[labeled == i] = 0
    return labeled > 0


def cluster_sizes(mask):
    """Return a list of cluster sizes (voxel counts), sorted descending."""
    struct = get_structure_3d()
    labeled, n = ndimage.label(mask, structure=struct)
    if n == 0:
        return []
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return sorted(sizes, reverse=True)


def estimate_smoothness_fwhm(data, brain_mask):
    """Estimate per-axis smoothness (FWHM in voxels) from finite differences."""
    masked = data.copy()
    masked[~brain_mask] = 0
    fwhm = np.zeros(3)
    for axis in range(3):
        d = np.diff(masked, axis=axis)
        var_d = np.var(d[d != 0]) if np.any(d != 0) else 1.0
        var_data = np.var(masked[brain_mask]) if brain_mask.any() else 1.0
        if var_data > 0 and var_d > 0:
            r1 = max(1.0 - var_d / (2.0 * var_data), 0.001)
            fwhm[axis] = np.sqrt(-2.0 * np.log(2.0) / np.log(r1)) if r1 < 1.0 else 20.0
        else:
            fwhm[axis] = 4.0
    return fwhm


# ============================================================================
# THE 13 THRESHOLDING METHODS
# ============================================================================

def m1_fixed_z_cluster(d):
    """M1 — Fixed z-threshold + cluster-size filter."""
    mask = d.abs_z >= M1_ZSCORE_THRESHOLD
    mask &= d.brain_mask
    return cluster_filter(mask, M1_MIN_CLUSTER)


def m2_quality_percentile(d):
    """M2 — Bilateral quality mask + percentile cutoff."""
    quality = d.bilateral_quality_mask()
    vals = d.abs_ai[quality]
    if len(vals) == 0:
        return np.zeros_like(d.brain_mask)
    cutoff = max(np.percentile(vals, M2_PERCENTILE), 0.01)
    mask = quality & (d.abs_ai >= cutoff)
    return cluster_filter(mask, M2_MIN_CLUSTER)


def m3_fdr(d):
    """M3 — Benjamini-Hochberg FDR correction."""
    # Restrict to voxels where the asymmetry statistic is actually defined
    # (non-zero). The map is zero outside the patient-valid gray matter; if
    # those zeros are included, the half-normal noise estimate below collapses
    # (median -> 0, sigma_noise -> 0) and the method spuriously returns nothing.
    quality = d.bilateral_quality_mask() & (d.abs_ai > 0)
    vals = d.abs_ai[quality]
    n = len(vals)
    if n < 20:
        return np.zeros_like(d.brain_mask)
    median_ai = np.median(vals)
    noise_vals = vals[vals <= median_ai]
    if len(noise_vals) < 10:
        return np.zeros_like(d.brain_mask)
    sigma_noise = np.sqrt(np.pi / 2) * np.mean(noise_vals)
    if sigma_noise <= 0:
        return np.zeros_like(d.brain_mask)
    pvals = 2 * stats.norm.sf(vals / sigma_noise)
    pvals = np.clip(pvals, 1e-300, 1.0)
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    bh_critical = M3_Q_VALUE * np.arange(1, n + 1) / n
    reject = sorted_p <= bh_critical
    if not reject.any():
        return np.zeros_like(d.brain_mask)
    max_k = np.max(np.where(reject)[0])
    threshold_p = sorted_p[max_k]
    pvals_vol = np.ones_like(d.ai)
    quality_idx = np.where(quality)
    pvals_vol[quality_idx] = pvals
    mask = quality & (pvals_vol <= threshold_p)
    return mask


def m4_bonferroni(d):
    """M4 — Bonferroni correction."""
    # Non-zero restriction: see m3_fdr — including zero (no-data) voxels
    # collapses the half-normal noise estimate and forces a zero result.
    quality = d.bilateral_quality_mask() & (d.abs_ai > 0)
    vals = d.abs_ai[quality]
    n = len(vals)
    if n < 20:
        return np.zeros_like(d.brain_mask)
    median_ai = np.median(vals)
    noise_vals = vals[vals <= median_ai]
    if len(noise_vals) < 10:
        return np.zeros_like(d.brain_mask)
    sigma_noise = np.sqrt(np.pi / 2) * np.mean(noise_vals)
    if sigma_noise <= 0:
        return np.zeros_like(d.brain_mask)
    pvals = 2 * stats.norm.sf(vals / sigma_noise)
    pvals = np.clip(pvals, 1e-300, 1.0)
    bonf = M4_ALPHA / n
    mask = np.zeros_like(d.brain_mask)
    quality_idx = np.where(quality)
    surviving = pvals <= bonf
    mask[quality_idx[0][surviving], quality_idx[1][surviving], quality_idx[2][surviving]] = True
    return mask


def m5_tfce(d):
    """M5 — Threshold-Free Cluster Enhancement (TFCE)."""
    z = d.abs_z.copy()
    z[~d.brain_mask] = 0
    tfce = np.zeros_like(z)
    struct = get_structure_3d()
    max_z = z.max()
    if max_z <= 0:
        return np.zeros_like(d.brain_mask)
    heights = np.arange(M5_DH, max_z + M5_DH, M5_DH)
    for h in heights:
        suprathresh = z >= h
        labeled, n_cl = ndimage.label(suprathresh, structure=struct)
        if n_cl == 0:
            continue
        cl_sizes_arr = ndimage.sum(suprathresh, labeled, range(1, n_cl + 1))
        extent_map = np.zeros_like(z)
        for cid, cs in enumerate(cl_sizes_arr, 1):
            extent_map[labeled == cid] = cs
        tfce += (extent_map ** M5_E_POWER) * (h ** M5_H_POWER) * M5_DH
    brain_tfce = tfce[d.brain_mask]
    if len(brain_tfce) == 0 or brain_tfce.max() == 0:
        return np.zeros_like(d.brain_mask)
    tfce_cutoff = np.percentile(brain_tfce[brain_tfce > 0], 95) if (brain_tfce > 0).any() else 0
    mask = d.brain_mask & (tfce >= tfce_cutoff)
    return cluster_filter(mask, M5_MIN_CLUSTER)


def m6_grf(d):
    """M6 — Gaussian Random Field theory cluster correction."""
    quality = d.bilateral_quality_mask()
    vals = d.abs_ai[quality]
    if len(vals) == 0:
        return np.zeros_like(d.brain_mask)
    cdt = np.percentile(vals, 85)
    cdt = max(cdt, 0.01)
    suprathresh = quality & (d.abs_ai >= cdt)
    if not suprathresh.any():
        return np.zeros_like(d.brain_mask)
    fwhm = estimate_smoothness_fwhm(d.ai, quality)
    resel_vol = max(np.prod(fwhm), 1.0)
    n_quality = int(quality.sum())
    n_resels = n_quality / resel_vol
    cdt_z = (cdt - np.mean(vals)) / max(np.std(vals), 1e-6)
    cdt_z = max(cdt_z, 0.5)
    D = 3
    beta = (math.gamma(D / 2 + 1) * (4 * np.log(2)) ** (D / 2)) / resel_vol
    if beta > 0:
        k_min = int(np.ceil((-np.log(M6_CLUSTER_ALPHA) / beta) ** (D / 2)))
    else:
        k_min = 50
    k_min = max(k_min, 10)
    k_min = min(k_min, 2000)
    return cluster_filter(suprathresh, k_min)


def m7_permutation(d):
    """M7 — Permutation test / null-distribution cluster correction."""
    # Non-zero restriction: the cluster-defining threshold (85th percentile)
    # and the permutation null are otherwise dominated by zero (no-data)
    # voxels, which collapses the observed-vs-null contrast.
    quality = d.bilateral_quality_mask() & (d.abs_ai > 0)
    vals = d.abs_ai[quality]
    n_quality = len(vals)
    if n_quality == 0:
        return np.zeros_like(d.brain_mask)
    cdt = max(np.percentile(vals, 85), 0.01)
    obs_mask = quality & (d.abs_ai >= cdt)
    obs_sizes_list = cluster_sizes(obs_mask)
    if not obs_sizes_list:
        return np.zeros_like(d.brain_mask)
    rng = np.random.RandomState(42)
    quality_idx = np.where(quality)
    null_max = []
    for _ in range(M7_N_PERMUTATIONS):
        perm_vals = vals.copy()
        rng.shuffle(perm_vals)
        perm_vol = np.zeros_like(d.ai)
        perm_vol[quality_idx] = perm_vals
        perm_mask = quality & (perm_vol >= cdt)
        ps = cluster_sizes(perm_mask)
        null_max.append(ps[0] if ps else 0)
    crit = int(np.percentile(null_max, 100 * (1 - M7_ALPHA)))
    crit = max(crit, 1)
    return cluster_filter(obs_mask, crit)


def m8_gmm(d):
    """M8 — Gaussian Mixture Model component separation."""
    vals = d.abs_ai[d.brain_mask].reshape(-1, 1)
    if len(vals) < M8_N_COMPONENTS * 10:
        return np.zeros_like(d.brain_mask)
    gmm = GaussianMixture(n_components=M8_N_COMPONENTS, covariance_type="full",
                           random_state=42, n_init=5, max_iter=200)
    gmm.fit(vals)
    signal_comp = np.argmax(gmm.means_.flatten())
    posteriors = gmm.predict_proba(vals)
    signal_prob = posteriors[:, signal_comp]
    mask = np.zeros_like(d.brain_mask)
    brain_idx = np.where(d.brain_mask)
    surviving = signal_prob > 0.5
    mask[brain_idx[0][surviving], brain_idx[1][surviving], brain_idx[2][surviving]] = True
    return cluster_filter(mask, M8_MIN_CLUSTER)


def m9_otsu(d):
    """M9 — Otsu's global thresholding."""
    vals = d.abs_ai[d.brain_mask]
    if len(vals) == 0:
        return np.zeros_like(d.brain_mask)
    n_bins = 256
    hist, bin_edges = np.histogram(vals, bins=n_bins, range=(0, vals.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return np.zeros_like(d.brain_mask)
    best_thresh, best_var = 0, 0
    cum_sum, cum_mean = 0, 0
    global_mean = np.sum(hist * bin_centers) / total
    for i in range(n_bins):
        cum_sum += hist[i]
        if cum_sum == 0:
            continue
        bg_w = cum_sum / total
        fg_w = 1.0 - bg_w
        if fg_w == 0:
            break
        cum_mean += hist[i] * bin_centers[i]
        bg_m = cum_mean / cum_sum
        fg_m = (global_mean * total - cum_mean) / (total - cum_sum)
        bv = bg_w * fg_w * (bg_m - fg_m) ** 2
        if bv > best_var:
            best_var = bv
            best_thresh = bin_centers[i]
    mask = d.brain_mask & (d.abs_ai >= best_thresh)
    return cluster_filter(mask, M9_MIN_CLUSTER)


def m10_random(d):
    """M10 — Random baseline (covers ~3 % of brain voxels)."""
    rng = np.random.RandomState(42)
    brain_idx = np.where(d.brain_mask)
    n_select = max(1, int(d.n_brain * M10_COVERAGE_PCT / 100))
    selected = rng.choice(len(brain_idx[0]), size=n_select, replace=False)
    mask = np.zeros_like(d.brain_mask)
    mask[brain_idx[0][selected], brain_idx[1][selected], brain_idx[2][selected]] = True
    return mask


def m11_quality_tfce(d):
    """M11 — Quality mask + TFCE (recommended best method)."""
    quality = d.bilateral_quality_mask()
    z = d.abs_ai.copy()
    z[~quality] = 0
    tfce = np.zeros_like(z)
    struct = get_structure_3d()
    max_z = z.max()
    if max_z <= 0:
        return np.zeros_like(d.brain_mask)
    dh = max_z / 200
    heights = np.arange(dh, max_z + dh, dh)
    for h in heights:
        suprathresh = z >= h
        labeled, n_cl = ndimage.label(suprathresh, structure=struct)
        if n_cl == 0:
            continue
        cl_sizes_arr = ndimage.sum(suprathresh, labeled, range(1, n_cl + 1))
        extent_map = np.zeros_like(z)
        for cid, cs in enumerate(cl_sizes_arr, 1):
            extent_map[labeled == cid] = cs
        tfce += (extent_map ** M5_E_POWER) * (h ** M5_H_POWER) * dh
    brain_tfce = tfce[quality]
    if len(brain_tfce) == 0 or brain_tfce.max() == 0:
        return np.zeros_like(d.brain_mask)
    tfce_cutoff = np.percentile(brain_tfce[brain_tfce > 0], 95) if (brain_tfce > 0).any() else 0
    mask = quality & (tfce >= tfce_cutoff)
    return cluster_filter(mask, M5_MIN_CLUSTER)


def m12_quality_gmm(d):
    """M12 — Bilateral quality mask + Gaussian Mixture Model."""
    quality = d.bilateral_quality_mask()
    vals = d.abs_ai[quality].reshape(-1, 1)
    if len(vals) < M8_N_COMPONENTS * 10:
        return np.zeros_like(d.brain_mask)
    gmm = GaussianMixture(n_components=M8_N_COMPONENTS, covariance_type="full",
                           random_state=42, n_init=5, max_iter=200)
    gmm.fit(vals)
    signal_comp = np.argmax(gmm.means_.flatten())
    posteriors = gmm.predict_proba(vals)
    signal_prob = posteriors[:, signal_comp]
    mask = np.zeros_like(d.brain_mask)
    quality_idx = np.where(quality)
    surviving = signal_prob > 0.5
    mask[quality_idx[0][surviving], quality_idx[1][surviving], quality_idx[2][surviving]] = True
    return cluster_filter(mask, M8_MIN_CLUSTER)


def m13_quality_otsu(d):
    """M13 — Bilateral quality mask + Otsu thresholding."""
    quality = d.bilateral_quality_mask()
    vals = d.abs_ai[quality]
    if len(vals) == 0:
        return np.zeros_like(d.brain_mask)
    n_bins = 256
    hist, bin_edges = np.histogram(vals, bins=n_bins, range=(0, vals.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return np.zeros_like(d.brain_mask)
    best_thresh, best_var = 0, 0
    cum_sum, cum_mean = 0, 0
    global_mean = np.sum(hist * bin_centers) / total
    for i in range(n_bins):
        cum_sum += hist[i]
        if cum_sum == 0:
            continue
        bg_w = cum_sum / total
        fg_w = 1.0 - bg_w
        if fg_w == 0:
            break
        cum_mean += hist[i] * bin_centers[i]
        bg_m = cum_mean / cum_sum
        fg_m = (global_mean * total - cum_mean) / (total - cum_sum)
        bv = bg_w * fg_w * (bg_m - fg_m) ** 2
        if bv > best_var:
            best_var = bv
            best_thresh = bin_centers[i]
    mask = quality & (d.abs_ai >= best_thresh)
    return cluster_filter(mask, M9_MIN_CLUSTER)


# ============================================================================
# METHOD REGISTRY
# ============================================================================

METHOD_REGISTRY = {
    "M01_FixedZ_Cluster": m1_fixed_z_cluster,
    "M02_Quality_Pctile": m2_quality_percentile,
    "M03_FDR_BH":         m3_fdr,
    "M04_Bonferroni":     m4_bonferroni,
    "M05_TFCE":           m5_tfce,
    "M06_GRF_Cluster":    m6_grf,
    "M07_Permutation":    m7_permutation,
    "M08_GMM":            m8_gmm,
    "M09_Otsu":           m9_otsu,
    "M10_Random":         m10_random,
    "M11_Quality_TFCE":   m11_quality_tfce,
    "M12_Quality_GMM":    m12_quality_gmm,
    "M13_Quality_Otsu":   m13_quality_otsu,
}
