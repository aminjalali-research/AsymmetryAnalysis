"""
Asymmetry Calculator Module
===========================

Reference implementation of 15 hemispheric perfusion asymmetry indices used in
the AsymmetryAnalysis framework for ASL-based epileptogenic zone lateralization.

All formulas are taken directly from primary statistical literature (citations
inline below).  This module is the single canonical home for asymmetry
definitions; the manuscript Methods §"Asymmetry Metrics" and Supplementary
Table S1 are kept consistent with the implementations here.

Categories
----------
1. Ratio-based (5):
       laterality_index, absolute_asymmetry_index, percent_difference,
       log_ratio, coefficient_of_asymmetry
2. Baselines (2):
       simple_difference, ratio
3. Standardized effect sizes (3):
       cohen_d_asymmetry, hedges_g_asymmetry, glass_delta_asymmetry
4. Distribution-based (2):
       zscore_asymmetry, robust_zscore_asymmetry
5. Advanced (3):
       cv_ratio, hyperperfusion_ratio, normalized_difference

Total: 15 indices.  See `audits/2026-05-04-formula-fix.md` for the rationale
behind each formula choice and the indices that were dropped (Min-Max
Normalized, Signed Squared, mutual_information_asymmetry,
hemispheric_dominance_score, normalized_asymmetry as defined as
`(L-R)/max(L,R)`, relative_asymmetry_index).

Per-ROI vs cross-ROI conventions
--------------------------------
Static methods on `AsymmetryCalculator` operate on a *single* bilateral pair
(scalar L, R, and optionally per-voxel SDs/Ns).  The `calculate_all_indices`
classmethod operates on a `bilateral_df` (37 region-pairs per patient) and
additionally derives the cross-ROI distribution-based indices
(`zscore_asymmetry`, `robust_zscore_asymmetry`) which require knowledge of
all ROI Laterality-Index values.

History
-------
* 2026-05-04  Formula audit & fix per
  `docs/superpowers/audits/2026-05-04-formula-fix.md`.  All effect-size,
  distribution-based, and advanced formulas were rewritten to match
  primary-literature definitions.  Previously-implemented but
  non-literature indices (``relative_asymmetry_index``,
  ``mutual_information_asymmetry``, ``hemispheric_dominance_score``,
  ``normalized_asymmetry``) were removed.  Five indices required by the
  manuscript that had no implementation (``hedges_g_asymmetry``,
  ``robust_zscore_asymmetry``, ``cv_ratio``, ``hyperperfusion_ratio``,
  ``normalized_difference``) were added.  Two supplementary baselines
  (``simple_difference``, ``ratio``) were added for completeness.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Small-sample correction factor used by Hedges' g.
# ---------------------------------------------------------------------------
def _hedges_correction(df_total):
    """Hedges & Olkin (1985) exact small-sample correction factor J(df).

    J(df) = 1 - 3 / (4 * df - 1)

    where df = n_L + n_R - 2.  J → 1 as df → ∞.

    Reference: Hedges, L. V. (1981). Distribution theory for Glass's
    estimator of effect size and related estimators. *Journal of
    Educational Statistics*, 6(2), 107-128.  See also Hedges & Olkin
    (1985), *Statistical Methods for Meta-Analysis*, Academic Press,
    Chapter 5.
    """
    if df_total <= 1:
        return np.nan
    return 1.0 - 3.0 / (4.0 * df_total - 1.0)


class AsymmetryCalculator:
    """Compute 15 hemispheric asymmetry indices.

    Each `static` method takes the mean perfusion of the left (L) and right
    (R) homologue regions, plus optional per-voxel SDs / counts when an
    effect-size formula needs them.  The `calculate_all_indices` method
    materialises the full 15-column DataFrame from a `bilateral_df`.
    """

    # ======================================================================
    # 1. Ratio-based indices
    # ======================================================================

    @staticmethod
    def laterality_index(left, right):
        """Laterality Index (LI).

        LI = (L - R) / (L + R),  range [-1, +1].

        Reference: Springer, J. A. et al. (1999). Language dominance in
        neurologically normal and epilepsy subjects: a functional MRI
        study. *Brain*, 122(11), 2033-2046.
        """
        denom = left + right
        return (left - right) / denom if denom != 0 else 0.0

    @staticmethod
    def absolute_asymmetry_index(left, right):
        """Absolute Asymmetry Index (AAI).

        AAI = |L - R| / (L + R),  range [0, 1].  Magnitude only.

        Common ASL/PET asymmetry metric; see Pendse et al. (2010).
        """
        denom = left + right
        return abs(left - right) / denom if denom != 0 else 0.0

    @staticmethod
    def percent_difference(left, right):
        """Percent Difference (PD).

        PD = 100 * (L - R) / [0.5 (L + R)],  range (-200%, +200%).

        This is the canonical "asymmetry index" used in clinical FDG-PET
        and ASL-CBF studies (Van Bogaert et al., 2000).  It is the same
        signed quantity as LI rescaled by 200.
        """
        mean_lr = 0.5 * (left + right)
        return 100.0 * (left - right) / mean_lr if mean_lr != 0 else 0.0

    @staticmethod
    def log_ratio(left, right):
        """Log Ratio (LR).

        LR = ln(L / R),  range (-∞, +∞).

        Standard log-transform of the per-region ratio; symmetric around
        zero and approximately additive.
        """
        if right == 0 or left <= 0:
            return np.nan
        return float(np.log(left / right))

    @staticmethod
    def coefficient_of_asymmetry(left, right):
        """Coefficient of Asymmetry (CoA).

        CoA = (L - R) / sqrt(L * R),  geometric-mean normalisation.

        Reference: Galaburda, A. M., LeMay, M., Kemper, T. L., &
        Geschwind, N. (1978).  Right-left asymmetries in the brain.
        *Science*, 199(4331), 852-856.  Also widely used in
        morphometric asymmetry literature (Plessen et al., 2014).
        """
        gmean = np.sqrt(left * right) if (left > 0 and right > 0) else 0.0
        return (left - right) / gmean if gmean != 0 else 0.0

    # ======================================================================
    # 2. Baseline indices (supplementary)
    # ======================================================================

    @staticmethod
    def simple_difference(left, right):
        """Simple Difference: L - R.  Unbounded baseline."""
        return float(left - right)

    @staticmethod
    def ratio(left, right):
        """Ratio: L / R.  Range (0, +∞).  Returns NaN when R == 0."""
        if right == 0:
            return np.nan
        return float(left / right)

    # ======================================================================
    # 3. Standardised effect sizes
    # ======================================================================

    @staticmethod
    def cohen_d_asymmetry(left, right, left_std=None, right_std=None,
                          left_n=None, right_n=None):
        """Cohen's d for left vs right hemisphere voxel distributions.

        Two-sample independent-groups d (Cohen, 1988, eq. 2.5.1):

            d = (mean_L - mean_R) / s_pooled

        where the *unbiased* pooled SD is

            s_pooled = sqrt( ((n_L - 1) * s_L^2 + (n_R - 1) * s_R^2)
                             / (n_L + n_R - 2) ).

        When `left_n` and `right_n` are not supplied the equal-n
        approximation `s_pooled = sqrt((s_L^2 + s_R^2) / 2)` is used.
        When SDs are not supplied at all, returns NaN — Cohen's d is
        undefined without dispersion estimates and should not be faked.

        Reference: Cohen, J. (1988). *Statistical Power Analysis for
        the Behavioral Sciences* (2nd ed.). Hillsdale, NJ: Lawrence
        Erlbaum, p. 20.
        """
        if left_std is None or right_std is None:
            return np.nan
        if left_n is not None and right_n is not None and (left_n + right_n) > 2:
            num = (left_n - 1) * left_std ** 2 + (right_n - 1) * right_std ** 2
            denom = left_n + right_n - 2
            s_pooled = np.sqrt(num / denom) if denom > 0 else 0.0
        else:
            s_pooled = np.sqrt((left_std ** 2 + right_std ** 2) / 2.0)
        return (left - right) / s_pooled if s_pooled != 0 else 0.0

    @staticmethod
    def hedges_g_asymmetry(left, right, left_std=None, right_std=None,
                           left_n=None, right_n=None):
        """Hedges' g — small-sample-bias-corrected Cohen's d.

            g = J(df) * d
            J(df) = 1 - 3 / (4 * df - 1),  df = n_L + n_R - 2.

        Returns NaN when SDs are missing.  When n_L / n_R are not
        supplied we cannot compute J exactly; in that case J is taken
        as 1 (i.e. g == d) and a `RuntimeWarning` is implicit.

        Reference: Hedges, L. V. (1981). Distribution theory for
        Glass's estimator of effect size and related estimators.
        *Journal of Educational Statistics*, 6(2), 107-128.  See also
        Hedges & Olkin (1985), *Statistical Methods for Meta-Analysis*,
        Academic Press.
        """
        d = AsymmetryCalculator.cohen_d_asymmetry(
            left, right, left_std, right_std, left_n, right_n
        )
        if np.isnan(d):
            return np.nan
        if left_n is None or right_n is None:
            return d  # J undefined → fall back to uncorrected d
        df_total = left_n + right_n - 2
        J = _hedges_correction(df_total)
        return d * J if not np.isnan(J) else np.nan

    @staticmethod
    def glass_delta_asymmetry(left, right, control_std=None, right_std=None):
        """Glass's Δ — effect size using control-group SD.

            Δ = (mean_L - mean_R) / s_control

        For laterality, the right hemisphere acts as the control group
        by convention (Glass, 1976).  Pass `control_std=right_std`
        explicitly, or use the `right_std` keyword.  Returns NaN when
        no control SD is supplied — Glass's Δ requires a reference SD
        and cannot be back-derived from means alone.

        Reference: Glass, G. V. (1976). Primary, secondary, and
        meta-analysis of research. *Educational Researcher*, 5(10),
        3-8.
        """
        s_ref = control_std if control_std is not None else right_std
        if s_ref is None:
            return np.nan
        return (left - right) / s_ref if s_ref != 0 else 0.0

    # ======================================================================
    # 4. Distribution-based indices  (require the full LI vector)
    # ======================================================================

    @staticmethod
    def zscore_asymmetry_vector(li_values):
        """Per-region z-score of the LI distribution.

            z_i = (LI_i - mean(LI)) / sd(LI)

        Operates on the full per-patient LI vector (shape (n_regions,))
        and returns a vector of the same shape.

        Reference: Standard parametric z-score; see manuscript
        Methods §Distribution-Based Metrics.
        """
        li = np.asarray(li_values, dtype=float)
        mu = np.nanmean(li)
        sd = np.nanstd(li, ddof=1)
        if sd == 0 or np.isnan(sd):
            return np.zeros_like(li)
        return (li - mu) / sd

    @staticmethod
    def robust_zscore_asymmetry_vector(li_values):
        """Robust per-region z-score using median / MAD.

            z_robust_i = (LI_i - median(LI)) / (1.4826 * MAD)
            MAD        = median(|LI - median(LI)|)

        The constant 1.4826 ≈ 1 / Φ⁻¹(3/4) makes the scaled MAD a
        consistent estimator of σ for normally distributed data
        (Huber, 1981).

        Reference: Huber, P. J. (1981). *Robust Statistics*. Wiley.
        Leys et al. (2013).  Detecting outliers: do not use SD around
        the mean, use absolute deviation around the median.  *J. Exp.
        Soc. Psychol.*, 49(4), 764-766.
        """
        li = np.asarray(li_values, dtype=float)
        med = np.nanmedian(li)
        mad = np.nanmedian(np.abs(li - med))
        scale = 1.4826 * mad
        if scale == 0 or np.isnan(scale):
            return np.zeros_like(li)
        return (li - med) / scale

    # ======================================================================
    # 5. Advanced indices
    # ======================================================================

    @staticmethod
    def cv_ratio(left, right, left_std=None, right_std=None):
        """Coefficient-of-variation ratio.

            CVR = CV_L / CV_R  =  (s_L / |mean_L|) / (s_R / |mean_R|)

        Range (0, +∞).  Compares relative dispersion across hemispheres.
        Requires both per-voxel SDs.  Returns NaN otherwise.

        Reference: Standard descriptive-statistics ratio.  See e.g.
        Reed et al. (2002). *Use of coefficient of variation in
        assessing variability of quantitative assays.*  Clin Diagn
        Lab Immunol, 9(6), 1235-1239.
        """
        if left_std is None or right_std is None:
            return np.nan
        if left == 0 or right == 0:
            return np.nan
        cv_l = left_std / abs(left)
        cv_r = right_std / abs(right)
        return cv_l / cv_r if cv_r != 0 else np.nan

    @staticmethod
    def hyperperfusion_ratio(left, right):
        """Hyperperfusion Ratio.

            HR = max(L, R) / min(L, R),  range [1, +∞).

        Magnitude-only fold-change; ignores directionality.  Useful for
        thresholding "how many-fold different are the two sides".

        Reference: Common formulation in clinical perfusion
        asymmetry reports; see e.g. Wintermark et al. (2003) on
        CBF asymmetry indices.
        """
        if left <= 0 or right <= 0:
            return np.nan
        hi, lo = max(left, right), min(left, right)
        return hi / lo if lo != 0 else np.nan

    @staticmethod
    def normalized_difference(left, right):
        """Normalised Difference.

            ND = (L - R) / sqrt(L^2 + R^2),  range [-1, +1].

        Magnitude-normalised signed asymmetry; bounded and symmetric
        about zero.  Equivalent to cosine-distance-based asymmetry.

        Reference: Used in shape and intensity asymmetry literature;
        see e.g. Toga & Thompson (2003).  Mapping brain asymmetry.
        *Nat Rev Neurosci*, 4(1), 37-48.
        """
        denom = np.sqrt(left ** 2 + right ** 2)
        return (left - right) / denom if denom != 0 else 0.0

    # ======================================================================
    # Aggregator
    # ======================================================================

    def calculate_all_indices(self, bilateral_df):
        """Materialise all 15 asymmetry indices for a patient.

        Parameters
        ----------
        bilateral_df : pandas.DataFrame
            One row per bilateral region pair.  Must contain
            ``left_mean_perfusion`` and ``right_mean_perfusion``.
            ``left_std``, ``right_std``, ``left_nvoxels``,
            ``right_nvoxels`` are required for the effect-size and
            CV-ratio columns; if absent those columns will contain
            NaN.

        Returns
        -------
        pandas.DataFrame
            Copy of input with 15 new columns, one per index.
        """
        df = bilateral_df.copy()

        L = df["left_mean_perfusion"].astype(float)
        R = df["right_mean_perfusion"].astype(float)

        has_std = "left_std" in df.columns and "right_std" in df.columns
        has_n = "left_nvoxels" in df.columns and "right_nvoxels" in df.columns

        sL = df["left_std"].astype(float) if has_std else None
        sR = df["right_std"].astype(float) if has_std else None
        nL = df["left_nvoxels"].astype(float) if has_n else None
        nR = df["right_nvoxels"].astype(float) if has_n else None

        # ---- Ratio-based (5) ----
        df["laterality_index"] = [
            self.laterality_index(l, r) for l, r in zip(L, R)
        ]
        df["absolute_asymmetry_index"] = [
            self.absolute_asymmetry_index(l, r) for l, r in zip(L, R)
        ]
        df["percent_difference"] = [
            self.percent_difference(l, r) for l, r in zip(L, R)
        ]
        df["log_ratio"] = [self.log_ratio(l, r) for l, r in zip(L, R)]
        df["coefficient_of_asymmetry"] = [
            self.coefficient_of_asymmetry(l, r) for l, r in zip(L, R)
        ]

        # ---- Baselines (2) ----
        df["simple_difference"] = [
            self.simple_difference(l, r) for l, r in zip(L, R)
        ]
        df["ratio"] = [self.ratio(l, r) for l, r in zip(L, R)]

        # ---- Effect sizes (3) ----
        if has_std:
            sL_arr = sL.values
            sR_arr = sR.values
            nL_arr = nL.values if has_n else [None] * len(df)
            nR_arr = nR.values if has_n else [None] * len(df)
            df["cohen_d_asymmetry"] = [
                self.cohen_d_asymmetry(l, r, sl, sr, nl, nr)
                for l, r, sl, sr, nl, nr in zip(L, R, sL_arr, sR_arr, nL_arr, nR_arr)
            ]
            df["hedges_g_asymmetry"] = [
                self.hedges_g_asymmetry(l, r, sl, sr, nl, nr)
                for l, r, sl, sr, nl, nr in zip(L, R, sL_arr, sR_arr, nL_arr, nR_arr)
            ]
            df["glass_delta_asymmetry"] = [
                self.glass_delta_asymmetry(l, r, control_std=sr)
                for l, r, sr in zip(L, R, sR_arr)
            ]
        else:
            df["cohen_d_asymmetry"] = np.nan
            df["hedges_g_asymmetry"] = np.nan
            df["glass_delta_asymmetry"] = np.nan

        # ---- Distribution-based (2) ----
        # Operate on the per-patient LI vector.
        li_vec = df["laterality_index"].values
        df["zscore_asymmetry"] = self.zscore_asymmetry_vector(li_vec)
        df["robust_zscore_asymmetry"] = self.robust_zscore_asymmetry_vector(
            li_vec
        )

        # ---- Advanced (3) ----
        if has_std:
            sL_arr = sL.values
            sR_arr = sR.values
            df["cv_ratio"] = [
                self.cv_ratio(l, r, sl, sr)
                for l, r, sl, sr in zip(L, R, sL_arr, sR_arr)
            ]
        else:
            df["cv_ratio"] = np.nan
        df["hyperperfusion_ratio"] = [
            self.hyperperfusion_ratio(l, r) for l, r in zip(L, R)
        ]
        df["normalized_difference"] = [
            self.normalized_difference(l, r) for l, r in zip(L, R)
        ]

        # Store the LI distribution stats used for zscore_asymmetry, for
        # downstream traceability.
        df.attrs["li_mean"] = float(np.nanmean(li_vec))
        df.attrs["li_sd"] = float(np.nanstd(li_vec, ddof=1))
        df.attrs["li_median"] = float(np.nanmedian(li_vec))
        df.attrs["li_mad"] = float(
            np.nanmedian(np.abs(li_vec - np.nanmedian(li_vec)))
        )

        return df

    # ----------------------------------------------------------------------
    # Discoverability
    # ----------------------------------------------------------------------
    INDEX_NAMES = (
        # ratio-based (5)
        "laterality_index",
        "absolute_asymmetry_index",
        "percent_difference",
        "log_ratio",
        "coefficient_of_asymmetry",
        # baselines (2)
        "simple_difference",
        "ratio",
        # effect sizes (3)
        "cohen_d_asymmetry",
        "hedges_g_asymmetry",
        "glass_delta_asymmetry",
        # distribution-based (2)
        "zscore_asymmetry",
        "robust_zscore_asymmetry",
        # advanced (3)
        "cv_ratio",
        "hyperperfusion_ratio",
        "normalized_difference",
    )
