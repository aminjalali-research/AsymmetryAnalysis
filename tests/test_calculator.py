"""
tests/test_calculator.py — Unit tests for src/calculator.py (15 indices)

Constructs synthetic L/R perfusion pairs (with optional per-voxel SDs and
voxel counts) and checks that every index in `AsymmetryCalculator.INDEX_NAMES`

  1. Returns a finite numeric value for normal-range inputs.
  2. Has the correct sign for L > R, L < R, and L == R inputs.
  3. Matches its literature formula on hand-checkable closed forms.

These tests are deliberately minimal — full numerical agreement against
published reference values lives in the audit doc.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calculator import AsymmetryCalculator  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def calc():
    return AsymmetryCalculator()


@pytest.fixture
def synthetic_bilateral_df():
    """Three regions: L>R, L<R, L==R, with realistic perfusion / SD / N."""
    return pd.DataFrame(
        {
            "region": ["asym_left", "asym_right", "symmetric"],
            "left_mean_perfusion": [60.0, 40.0, 50.0],
            "right_mean_perfusion": [40.0, 60.0, 50.0],
            "left_std": [8.0, 7.0, 7.5],
            "right_std": [7.5, 8.0, 7.5],
            "left_nvoxels": [1200, 1100, 1300],
            "right_nvoxels": [1250, 1150, 1280],
        }
    )


# ---------------------------------------------------------------------------
# Per-formula closed-form checks
# ---------------------------------------------------------------------------
class TestRatioBasedIndices:
    def test_laterality_index(self, calc):
        assert calc.laterality_index(60, 40) == pytest.approx(0.2)
        assert calc.laterality_index(40, 60) == pytest.approx(-0.2)
        assert calc.laterality_index(50, 50) == 0.0
        assert calc.laterality_index(0, 0) == 0.0

    def test_absolute_asymmetry_index(self, calc):
        assert calc.absolute_asymmetry_index(60, 40) == pytest.approx(0.2)
        assert calc.absolute_asymmetry_index(40, 60) == pytest.approx(0.2)
        assert calc.absolute_asymmetry_index(50, 50) == 0.0

    def test_percent_difference(self, calc):
        # PD = 100*(60-40)/(50) = 40
        assert calc.percent_difference(60, 40) == pytest.approx(40.0)
        assert calc.percent_difference(40, 60) == pytest.approx(-40.0)
        assert calc.percent_difference(50, 50) == 0.0

    def test_log_ratio(self, calc):
        assert calc.log_ratio(100, 50) == pytest.approx(math.log(2))
        assert calc.log_ratio(50, 50) == pytest.approx(0.0)
        assert math.isnan(calc.log_ratio(0, 50))
        assert math.isnan(calc.log_ratio(50, 0))

    def test_coefficient_of_asymmetry(self, calc):
        # CoA = (60-40)/sqrt(60*40) = 20/sqrt(2400)
        assert calc.coefficient_of_asymmetry(60, 40) == pytest.approx(
            20.0 / math.sqrt(2400.0)
        )
        assert calc.coefficient_of_asymmetry(50, 50) == 0.0


class TestBaselines:
    def test_simple_difference(self, calc):
        assert calc.simple_difference(60, 40) == 20.0
        assert calc.simple_difference(40, 60) == -20.0

    def test_ratio(self, calc):
        assert calc.ratio(60, 40) == pytest.approx(1.5)
        assert math.isnan(calc.ratio(60, 0))


class TestEffectSizes:
    def test_cohen_d_with_stds_and_n(self, calc):
        # n_L = n_R = 100, s_L = s_R = 10
        # s_pooled = sqrt(((99 * 100) + (99 * 100)) / 198) = sqrt(100) = 10
        # d = (60 - 40) / 10 = 2.0
        d = calc.cohen_d_asymmetry(60, 40, 10, 10, 100, 100)
        assert d == pytest.approx(2.0)

    def test_cohen_d_equal_n_approx(self, calc):
        # No n provided → equal-n approximation: s_pooled = sqrt((100+100)/2) = 10
        d = calc.cohen_d_asymmetry(60, 40, 10, 10)
        assert d == pytest.approx(2.0)

    def test_cohen_d_missing_sd_returns_nan(self, calc):
        assert math.isnan(calc.cohen_d_asymmetry(60, 40))

    def test_hedges_g_smaller_than_cohen_d(self, calc):
        # Hedges' g < Cohen's d for finite n
        d = calc.cohen_d_asymmetry(60, 40, 10, 10, 100, 100)
        g = calc.hedges_g_asymmetry(60, 40, 10, 10, 100, 100)
        assert g < d
        # J(df=198) = 1 - 3/(4*198 - 1) = 1 - 3/791
        J_expected = 1.0 - 3.0 / (4 * 198 - 1)
        assert g == pytest.approx(d * J_expected)

    def test_hedges_g_missing_n_falls_back_to_d(self, calc):
        d = calc.cohen_d_asymmetry(60, 40, 10, 10)
        g = calc.hedges_g_asymmetry(60, 40, 10, 10)
        assert g == pytest.approx(d)

    def test_glass_delta_with_right_std(self, calc):
        # Δ = (60 - 40) / s_R = 20 / 10 = 2.0
        delta = calc.glass_delta_asymmetry(60, 40, control_std=10)
        assert delta == pytest.approx(2.0)
        # right_std kwarg should also work
        assert calc.glass_delta_asymmetry(60, 40, right_std=10) == pytest.approx(
            2.0
        )

    def test_glass_delta_missing_std_returns_nan(self, calc):
        assert math.isnan(calc.glass_delta_asymmetry(60, 40))


class TestDistributionBased:
    def test_zscore_vector(self, calc):
        li = np.array([0.1, -0.1, 0.0, 0.2, -0.2])
        z = calc.zscore_asymmetry_vector(li)
        assert z.shape == li.shape
        assert abs(np.nanmean(z)) < 1e-10  # zero-mean
        assert np.nanstd(z, ddof=1) == pytest.approx(1.0)

    def test_robust_zscore_vector(self, calc):
        # Inject an outlier; robust z should down-weight it
        li = np.array([0.1, -0.1, 0.0, 0.2, -0.2, 100.0])
        z_robust = calc.robust_zscore_asymmetry_vector(li)
        z_classic = calc.zscore_asymmetry_vector(li)
        # Classic z of outlier is ~2 (because variance is huge);
        # robust z should be much larger, signalling outlier status.
        assert abs(z_robust[-1]) > abs(z_classic[-1])


class TestAdvancedIndices:
    def test_cv_ratio(self, calc):
        # CV_L = 10/100 = 0.1; CV_R = 5/100 = 0.05; ratio = 2
        assert calc.cv_ratio(100, 100, 10, 5) == pytest.approx(2.0)
        assert math.isnan(calc.cv_ratio(100, 100))  # missing SDs

    def test_hyperperfusion_ratio(self, calc):
        assert calc.hyperperfusion_ratio(60, 40) == pytest.approx(1.5)
        assert calc.hyperperfusion_ratio(40, 60) == pytest.approx(1.5)
        # symmetry → ratio = 1
        assert calc.hyperperfusion_ratio(50, 50) == pytest.approx(1.0)
        assert math.isnan(calc.hyperperfusion_ratio(-1, 50))

    def test_normalized_difference_bounded(self, calc):
        nd = calc.normalized_difference(60, 40)
        # ND = 20 / sqrt(3600 + 1600) = 20 / sqrt(5200)
        assert nd == pytest.approx(20.0 / math.sqrt(5200.0))
        assert -1 <= nd <= 1
        # symmetry → 0
        assert calc.normalized_difference(50, 50) == 0.0


# ---------------------------------------------------------------------------
# Aggregator integration test
# ---------------------------------------------------------------------------
class TestCalculateAllIndices:
    def test_15_columns_added(self, calc, synthetic_bilateral_df):
        result = calc.calculate_all_indices(synthetic_bilateral_df)
        for name in AsymmetryCalculator.INDEX_NAMES:
            assert name in result.columns, f"missing column {name!r}"
        assert len(AsymmetryCalculator.INDEX_NAMES) == 15

    def test_signs_match_expected(self, calc, synthetic_bilateral_df):
        result = calc.calculate_all_indices(synthetic_bilateral_df)
        # asym_left: L > R → LI > 0, AAI > 0, PD > 0, log_ratio > 0,
        # simple_diff > 0, ratio > 1
        row = result.iloc[0]
        assert row["laterality_index"] > 0
        assert row["absolute_asymmetry_index"] > 0
        assert row["percent_difference"] > 0
        assert row["log_ratio"] > 0
        assert row["simple_difference"] > 0
        assert row["ratio"] > 1
        assert row["hyperperfusion_ratio"] > 1
        assert row["normalized_difference"] > 0

        # asym_right: L < R → LI < 0, AAI > 0 (magnitude), PD < 0
        row = result.iloc[1]
        assert row["laterality_index"] < 0
        assert row["absolute_asymmetry_index"] > 0
        assert row["percent_difference"] < 0
        assert row["log_ratio"] < 0
        assert row["normalized_difference"] < 0

        # symmetric: most indices = 0; HR = 1; ratio = 1
        row = result.iloc[2]
        assert row["laterality_index"] == pytest.approx(0.0)
        assert row["absolute_asymmetry_index"] == pytest.approx(0.0)
        assert row["percent_difference"] == pytest.approx(0.0)
        assert row["log_ratio"] == pytest.approx(0.0)
        assert row["simple_difference"] == pytest.approx(0.0)
        assert row["ratio"] == pytest.approx(1.0)
        assert row["hyperperfusion_ratio"] == pytest.approx(1.0)
        assert row["normalized_difference"] == pytest.approx(0.0)

    def test_effect_sizes_finite_when_stds_present(
        self, calc, synthetic_bilateral_df
    ):
        result = calc.calculate_all_indices(synthetic_bilateral_df)
        for col in ("cohen_d_asymmetry", "hedges_g_asymmetry",
                    "glass_delta_asymmetry"):
            assert result[col].notna().all(), f"{col} should be finite"

    def test_effect_sizes_nan_when_stds_missing(self, calc):
        df = pd.DataFrame(
            {
                "region": ["a"],
                "left_mean_perfusion": [60.0],
                "right_mean_perfusion": [40.0],
            }
        )
        result = calc.calculate_all_indices(df)
        assert math.isnan(result["cohen_d_asymmetry"].iloc[0])
        assert math.isnan(result["hedges_g_asymmetry"].iloc[0])
        assert math.isnan(result["glass_delta_asymmetry"].iloc[0])
        assert math.isnan(result["cv_ratio"].iloc[0])
        # but these don't need stds:
        assert not math.isnan(result["laterality_index"].iloc[0])
        assert not math.isnan(result["hyperperfusion_ratio"].iloc[0])

    def test_zscore_is_zero_mean_unit_sd(
        self, calc, synthetic_bilateral_df
    ):
        result = calc.calculate_all_indices(synthetic_bilateral_df)
        z = result["zscore_asymmetry"].values
        assert abs(np.nanmean(z)) < 1e-10
        # ddof=1 → sd of length-3 vector should be 1
        assert np.nanstd(z, ddof=1) == pytest.approx(1.0, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
