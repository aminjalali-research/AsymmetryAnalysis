"""
Asymmetry Calculator Module

This module provides multiple methods for calculating brain asymmetry indices
from bilateral perfusion measurements.
"""

import pandas as pd
import numpy as np


class AsymmetryCalculator:
    """Calculate multiple asymmetry indices for brain regions"""

    @staticmethod
    def laterality_index(left, right):
        """Standard Laterality Index: (L-R)/(L+R)"""
        return (left - right) / (left + right) if (left + right) != 0 else 0

    @staticmethod
    def asymmetry_index(left, right):
        """Asymmetry Index: (L-R)/((L+R)/2) * 100"""
        mean_lr = (left + right) / 2
        return (left - right) / mean_lr * 100 if mean_lr != 0 else 0

    @staticmethod
    def percentage_difference(left, right):
        """Percentage difference: (L-R)/L * 100"""
        return (left - right) / left * 100 if left != 0 else 0

    @staticmethod
    def normalized_asymmetry(left, right):
        """Normalized asymmetry: (L-R)/max(L,R)"""
        max_lr = max(left, right)
        return (left - right) / max_lr if max_lr != 0 else 0

    @staticmethod
    def log_ratio(left, right):
        """Log ratio: log(L/R)"""
        return np.log(left / right) if right != 0 and left > 0 else np.nan

    @staticmethod
    def absolute_asymmetry_index(left, right):
        """Absolute Asymmetry Index: |L-R|/(L+R) - magnitude without direction"""
        return abs(left - right) / (left + right) if (left + right) != 0 else 0

    @staticmethod
    def relative_asymmetry_index(left, right):
        """Relative Asymmetry Index: (L-R)/min(L,R) - sensitive to smaller values"""
        min_lr = min(left, right)
        return (left - right) / min_lr if min_lr != 0 else 0

    @staticmethod
    def coefficient_of_asymmetry(left, right):
        """Coefficient of Asymmetry: (L-R)/sqrt(L*R) - geometric mean normalization"""
        geometric_mean = np.sqrt(left * right)
        return (left - right) / geometric_mean if geometric_mean != 0 else 0

    @staticmethod
    def z_score_asymmetry(left, right):
        """Z-score based asymmetry: (L-R)/sqrt((L+R)/2) - accounts for noise"""
        pooled_std = np.sqrt((left + right) / 2)
        return (left - right) / pooled_std if pooled_std != 0 else 0

    @staticmethod
    def signed_log_ratio(left, right):
        """Signed Log Ratio: sign(L-R) * log(max(L,R)/min(L,R)) - symmetric in log space"""
        if left == 0 or right == 0:
            return np.nan
        max_val = max(left, right)
        min_val = min(left, right)
        sign = 1 if left > right else -1 if left < right else 0
        return sign * np.log(max_val / min_val) if min_val != 0 else np.nan

    @staticmethod
    def cohen_d_asymmetry(left, right, left_std=None, right_std=None):
        """Cohen's d for asymmetry: effect size measure (requires standard deviations)"""
        if left_std is None or right_std is None:
            # Use simple difference if no std provided
            pooled_std = np.sqrt((left + right) / 2)
            return (left - right) / pooled_std if pooled_std != 0 else 0
        else:
            # True Cohen's d with actual standard deviations
            pooled_std = np.sqrt((left_std**2 + right_std**2) / 2)
            return (left - right) / pooled_std if pooled_std != 0 else 0

    @staticmethod
    def fold_change_asymmetry(left, right):
        """Fold Change Asymmetry: log2(L/R) - common in biological data"""
        return np.log2(left / right) if right != 0 and left > 0 else np.nan

    @staticmethod
    def mutual_information_asymmetry(left, right):
        """Mutual Information based asymmetry: considers joint distribution"""
        # Simplified version - for full MI, would need distributions
        total = left + right
        if total == 0:
            return 0
        p_left = left / total
        p_right = right / total
        if p_left == 0 or p_right == 0:
            return 1.0  # Maximum asymmetry
        return abs(p_left * np.log2(p_left) + p_right * np.log2(p_right))

    @staticmethod
    def volume_corrected_asymmetry(left, right, left_vol, right_vol):
        """Volume-corrected asymmetry: accounts for structural differences"""
        if left_vol == 0 or right_vol == 0:
            return 0
        # Normalize perfusion by volume first
        left_normalized = left / left_vol
        right_normalized = right / right_vol
        # Then calculate asymmetry
        return (
            (left_normalized - right_normalized) / (left_normalized + right_normalized)
            if (left_normalized + right_normalized) != 0
            else 0
        )

    @staticmethod
    def robust_median_asymmetry(left, right):
        """Robust asymmetry using median-based calculation"""
        median_lr = np.median([left, right])
        return (left - right) / (2 * median_lr) if median_lr != 0 else 0

    @staticmethod
    def interquartile_ratio(left, right):
        """Interquartile-based asymmetry for robust analysis"""
        values = np.array([left, right])
        if len(values) < 2:
            return 0
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        return (left - right) / iqr if iqr != 0 else 0

    @staticmethod
    def skewness_difference(left, right):
        """Asymmetry based on distribution skewness difference"""
        # Simplified for single values - would need distributions for true skewness
        return np.tanh((left - right) / (left + right)) if (left + right) != 0 else 0

    @staticmethod
    def kurtosis_difference(left, right):
        """Asymmetry based on distribution kurtosis difference"""
        # Simplified version using normalized difference
        mean_val = (left + right) / 2
        return (
            ((left - mean_val) ** 4 - (right - mean_val) ** 4) / (mean_val**4)
            if mean_val != 0
            else 0
        )

    @staticmethod
    def entropy_difference(left, right):
        """Information-theoretic asymmetry measure"""
        total = left + right
        if total == 0:
            return 0
        p_left = left / total
        p_right = right / total

        # Handle edge cases
        if p_left == 0:
            entropy_left = 0
        else:
            entropy_left = -p_left * np.log2(p_left)

        if p_right == 0:
            entropy_right = 0
        else:
            entropy_right = -p_right * np.log2(p_right)

        return entropy_left - entropy_right

    @staticmethod
    def mahalanobis_distance(left, right):
        """Mahalanobis-inspired asymmetry distance"""
        # Simplified version for two values
        diff = left - right
        pooled_var = (left + right) / 2
        return diff / np.sqrt(pooled_var) if pooled_var > 0 else 0

    def calculate_all_indices(self, bilateral_df):
        """Calculate all 16 asymmetry indices for comprehensive analysis"""
        result_df = bilateral_df.copy()

        # Original 5 methods
        result_df["laterality_index"] = bilateral_df.apply(
            lambda row: self.laterality_index(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["asymmetry_index"] = bilateral_df.apply(
            lambda row: self.asymmetry_index(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["percentage_difference"] = bilateral_df.apply(
            lambda row: self.percentage_difference(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["normalized_asymmetry"] = bilateral_df.apply(
            lambda row: self.normalized_asymmetry(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["log_ratio"] = bilateral_df.apply(
            lambda row: self.log_ratio(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        # Advanced statistical methods (6-11)
        result_df["absolute_asymmetry_index"] = bilateral_df.apply(
            lambda row: self.absolute_asymmetry_index(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["relative_asymmetry_index"] = bilateral_df.apply(
            lambda row: self.relative_asymmetry_index(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["coefficient_of_asymmetry"] = bilateral_df.apply(
            lambda row: self.coefficient_of_asymmetry(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["z_score_asymmetry"] = bilateral_df.apply(
            lambda row: self.z_score_asymmetry(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["signed_log_ratio"] = bilateral_df.apply(
            lambda row: self.signed_log_ratio(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        # Sophisticated measures (12-16)
        result_df["cohen_d_asymmetry"] = bilateral_df.apply(
            lambda row: self.cohen_d_asymmetry(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["fold_change_asymmetry"] = bilateral_df.apply(
            lambda row: self.fold_change_asymmetry(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["mutual_information_asymmetry"] = bilateral_df.apply(
            lambda row: self.mutual_information_asymmetry(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        # Volume-corrected and new sophisticated methods
        if (
            "left_volume" in bilateral_df.columns
            and "right_volume" in bilateral_df.columns
        ):
            result_df["volume_corrected_asymmetry"] = bilateral_df.apply(
                lambda row: self.volume_corrected_asymmetry(
                    row["left_mean_perfusion"],
                    row["right_mean_perfusion"],
                    row["left_volume"],
                    row["right_volume"],
                ),
                axis=1,
            )
        else:
            result_df["volume_corrected_asymmetry"] = np.nan

        # New sophisticated methods (16 total)
        result_df["robust_median_asymmetry"] = bilateral_df.apply(
            lambda row: self.robust_median_asymmetry(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["interquartile_ratio"] = bilateral_df.apply(
            lambda row: self.interquartile_ratio(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["skewness_difference"] = bilateral_df.apply(
            lambda row: self.skewness_difference(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["kurtosis_difference"] = bilateral_df.apply(
            lambda row: self.kurtosis_difference(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["entropy_difference"] = bilateral_df.apply(
            lambda row: self.entropy_difference(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        result_df["mahalanobis_distance"] = bilateral_df.apply(
            lambda row: self.mahalanobis_distance(
                row["left_mean_perfusion"], row["right_mean_perfusion"]
            ),
            axis=1,
        )

        return result_df
