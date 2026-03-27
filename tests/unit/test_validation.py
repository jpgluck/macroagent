import pytest
import numpy as np
import pandas as pd
from utils.validation import (
    _mape,
    _mae,
    _rmse,
    _compute_growth_rates,
    _validate_company_df,
)


class TestMAP:
    """Test _mape (Mean Absolute Percentage Error)."""

    def test_mape_perfect_prediction(self):
        """Perfect predictions should yield MAPE of 0."""
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([100.0, 200.0, 300.0])
        assert _mape(actual, predicted) == 0.0

    def test_mape_known_values(self):
        """Test MAPE with known values: actual=[100, 200], predicted=[110, 190] -> MAPE=7.5%."""
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 190.0])
        # MAPE = mean(|100-110|/100, |200-190|/200) * 100 = mean(0.1, 0.05) * 100 = 7.5
        result = _mape(actual, predicted)
        assert abs(result - 7.5) < 1e-9

    def test_mape_all_over_predictions(self):
        """Test MAPE when all predictions are 10% high."""
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([110.0, 220.0, 330.0])
        # All errors are 10%
        assert abs(_mape(actual, predicted) - 10.0) < 1e-9

    def test_mape_all_under_predictions(self):
        """Test MAPE when all predictions are 20% low."""
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([80.0, 160.0, 240.0])
        # All errors are 20%
        assert abs(_mape(actual, predicted) - 20.0) < 1e-9


class TestMAE:
    """Test _mae (Mean Absolute Error)."""

    def test_mae_perfect_prediction(self):
        """Perfect predictions should yield MAE of 0."""
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([100.0, 200.0, 300.0])
        assert _mae(actual, predicted) == 0.0

    def test_mae_known_values(self):
        """Test MAE with known values: actual=[100, 200], predicted=[110, 190] -> MAE=10."""
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 190.0])
        # MAE = mean(|100-110|, |200-190|) = mean(10, 10) = 10
        assert abs(_mae(actual, predicted) - 10.0) < 1e-9

    def test_mae_symmetric_errors(self):
        """Test MAE with symmetric errors."""
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([95.0, 205.0, 295.0])
        # Errors: 5, 5, 5; MAE = 5
        assert abs(_mae(actual, predicted) - 5.0) < 1e-9


class TestRMSE:
    """Test _rmse (Root Mean Squared Error)."""

    def test_rmse_perfect_prediction(self):
        """Perfect predictions should yield RMSE of 0."""
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([100.0, 200.0, 300.0])
        assert _rmse(actual, predicted) == 0.0

    def test_rmse_known_values(self):
        """Test RMSE with known values: actual=[100, 200], predicted=[110, 190] -> RMSE=sqrt(100)=10."""
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 190.0])
        # Squared errors: 100, 100; MSE = 100; RMSE = 10
        assert abs(_rmse(actual, predicted) - 10.0) < 1e-9

    def test_rmse_single_value(self):
        """Test RMSE with a single prediction error."""
        actual = np.array([100.0])
        predicted = np.array([106.0])
        # RMSE = 6
        assert abs(_rmse(actual, predicted) - 6.0) < 1e-9

    def test_rmse_greater_than_mae(self):
        """RMSE should be >= MAE for non-uniform errors."""
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([90.0, 210.0, 310.0])
        # Errors: 10, 10, 10; MAE = 10, RMSE = 10
        assert _rmse(actual, predicted) >= _mae(actual, predicted)

    def test_rmse_greater_than_mae_with_variance(self):
        """RMSE > MAE when errors have variance."""
        actual = np.array([100.0, 200.0])
        predicted = np.array([90.0, 220.0])
        # Errors: 10, 20; MAE = 15; MSE = (100 + 400) / 2 = 250; RMSE = sqrt(250) ≈ 15.81
        mae = _mae(actual, predicted)
        rmse = _rmse(actual, predicted)
        assert rmse > mae


class TestComputeGrowthRates:
    """Test _compute_growth_rates function."""

    def test_growth_rates_with_sufficient_data(self):
        """Test with 36+ months of data → all 3 windows populated."""
        # 48 months of data (Jan 2020 - Dec 2023)
        dates = pd.date_range("2020-01-01", periods=48, freq="MS")
        values = np.arange(1.0, 49.0)
        df = pd.DataFrame({"ds": dates, "y": values})

        rates = _compute_growth_rates(df)

        # All windows should be populated (not None)
        assert rates["Lifetime"] is not None
        assert rates["3-Year"] is not None
        assert rates["5-Year"] is not None
        assert isinstance(rates["Lifetime"], float)
        assert isinstance(rates["3-Year"], float)
        assert isinstance(rates["5-Year"], float)

    def test_growth_rates_short_data(self):
        """Test with 13 months → only Lifetime, others None."""
        dates = pd.date_range("2023-01-01", periods=13, freq="MS")
        values = np.arange(1.0, 14.0)
        df = pd.DataFrame({"ds": dates, "y": values})

        rates = _compute_growth_rates(df)

        # Only Lifetime should have a value
        assert rates["Lifetime"] is not None
        assert rates["3-Year"] is None
        assert rates["5-Year"] is None

    def test_growth_rates_minimum_data(self):
        """Test with exactly 12 months → no YoY data available."""
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        values = np.arange(1.0, 13.0)
        df = pd.DataFrame({"ds": dates, "y": values})

        rates = _compute_growth_rates(df)

        # pct_change(periods=12) requires at least 13 points; with 12 we get 0 YoY points
        # So Lifetime should be 0.0 (empty yoy.median() defaults to 0.0)
        assert rates["Lifetime"] == 0.0
        assert rates["3-Year"] is None
        assert rates["5-Year"] is None

    def test_growth_rates_constant_values(self):
        """Test with constant values → all growth rates should be 0."""
        dates = pd.date_range("2020-01-01", periods=48, freq="MS")
        values = np.ones(48) * 100.0
        df = pd.DataFrame({"ds": dates, "y": values})

        rates = _compute_growth_rates(df)

        # All rates should be 0 (no growth)
        assert abs(rates["Lifetime"] - 0.0) < 1e-9
        assert abs(rates["3-Year"] - 0.0) < 1e-9
        assert abs(rates["5-Year"] - 0.0) < 1e-9

    def test_growth_rates_linear_growth(self):
        """Test with linearly increasing values."""
        dates = pd.date_range("2020-01-01", periods=48, freq="MS")
        values = np.arange(100.0, 148.0)  # 100 to 147
        df = pd.DataFrame({"ds": dates, "y": values})

        rates = _compute_growth_rates(df)

        # All windows should have positive growth rates
        assert rates["Lifetime"] > 0
        assert rates["3-Year"] > 0
        assert rates["5-Year"] > 0

    def test_growth_rates_return_type(self):
        """Test that return value is always a dict with correct keys."""
        dates = pd.date_range("2023-01-01", periods=24, freq="MS")
        values = np.arange(1.0, 25.0)
        df = pd.DataFrame({"ds": dates, "y": values})

        rates = _compute_growth_rates(df)

        assert isinstance(rates, dict)
        assert "Lifetime" in rates
        assert "3-Year" in rates
        assert "5-Year" in rates
        assert len(rates) == 3


class TestValidateCompanyDF:
    """Test _validate_company_df function."""

    def test_valid_csv_basic(self):
        """Test validation passes with columns 'Date' and 'Demand', 24 rows."""
        dates = pd.date_range("2022-01-01", periods=24, freq="MS")
        df = pd.DataFrame({
            "Date": dates,
            "Demand": np.arange(100.0, 124.0),
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is True
        assert error == ""
        assert cleaned is not None
        assert list(cleaned.columns) == ["ds", "y"]
        assert len(cleaned) == 24

    def test_valid_csv_revenue_column(self):
        """Test validation with 'Revenue' column."""
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        df = pd.DataFrame({
            "Date": dates,
            "Revenue": np.ones(12) * 500.0,
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is True
        assert cleaned is not None
        assert "y" in cleaned.columns

    def test_valid_csv_y_column(self):
        """Test validation with 'y' column."""
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        df = pd.DataFrame({
            "Date": dates,
            "y": np.arange(10.0, 22.0),
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is True
        assert cleaned is not None

    def test_missing_date_column(self):
        """Test validation fails when date column is missing."""
        df = pd.DataFrame({
            "Time": pd.date_range("2022-01-01", periods=12, freq="MS"),
            "Demand": np.ones(12) * 100.0,
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is False
        assert "date" in error.lower()
        assert cleaned is None

    def test_missing_demand_column(self):
        """Test validation fails when demand column is missing."""
        df = pd.DataFrame({
            "Date": pd.date_range("2022-01-01", periods=12, freq="MS"),
            "Sales": np.ones(12) * 100.0,
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is False
        assert "demand" in error.lower() or "revenue" in error.lower()
        assert cleaned is None

    def test_too_few_rows(self):
        """Test validation fails with fewer than 12 rows."""
        dates = pd.date_range("2022-01-01", periods=5, freq="MS")
        df = pd.DataFrame({
            "Date": dates,
            "Demand": np.ones(5) * 100.0,
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is False
        assert "12" in error
        assert cleaned is None

    def test_negative_values(self):
        """Test validation fails with negative demand values."""
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        df = pd.DataFrame({
            "Date": dates,
            "Demand": np.array([-10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100], dtype=float),
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is False
        assert "positive" in error.lower()
        assert cleaned is None

    def test_zero_values(self):
        """Test validation fails with zero demand values."""
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        df = pd.DataFrame({
            "Date": dates,
            "Demand": np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100], dtype=float),
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is False
        assert "positive" in error.lower()
        assert cleaned is None

    def test_nan_values_in_demand(self):
        """Test validation fails with NaN in demand column."""
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        demands = np.array([100, 200, np.nan, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200], dtype=float)
        df = pd.DataFrame({
            "Date": dates,
            "Demand": demands,
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is False
        assert "positive" in error.lower() or "blanks" in error.lower()
        assert cleaned is None

    def test_nan_values_in_date(self):
        """Test validation fails with NaN in date column."""
        dates = pd.to_datetime([
            "2022-01-01", "2022-02-01", pd.NaT, "2022-04-01",
            "2022-05-01", "2022-06-01", "2022-07-01", "2022-08-01",
            "2022-09-01", "2022-10-01", "2022-11-01", "2022-12-01"
        ])
        df = pd.DataFrame({
            "Date": dates,
            "Demand": np.arange(100.0, 112.0),
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is False
        assert "date" in error.lower()
        assert cleaned is None

    def test_year_only_dates(self):
        """Test validation with year-only dates (e.g., '2020', '2021')."""
        df = pd.DataFrame({
            "Date": ["2020", "2021", "2022", "2023"],
            "Demand": [100.0, 200.0, 300.0, 400.0],
        })

        # With only 4 rows, this should fail the minimum requirement
        ok, error, cleaned = _validate_company_df(df)
        assert ok is False
        assert "12" in error

        # Now with sufficient rows
        dates_str = ["2011", "2012", "2013", "2014", "2015", "2016",
                     "2017", "2018", "2019", "2020", "2021", "2022"]
        df = pd.DataFrame({
            "Date": dates_str,
            "Demand": np.arange(100.0, 112.0),
        })

        ok, error, cleaned = _validate_company_df(df)
        assert ok is True
        assert cleaned is not None

    def test_sorts_by_date(self):
        """Test that output is sorted by date (ascending)."""
        # Intentionally shuffle dates
        dates = [
            "2022-03-01", "2022-01-01", "2022-02-01",
            "2022-06-01", "2022-04-01", "2022-05-01",
            "2022-09-01", "2022-07-01", "2022-08-01",
            "2022-12-01", "2022-10-01", "2022-11-01",
        ]
        df = pd.DataFrame({
            "Date": pd.to_datetime(dates),
            "Demand": np.arange(100.0, 112.0),
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is True
        assert cleaned is not None
        # Check that dates are in ascending order
        assert (cleaned["ds"].diff().dropna() >= pd.Timedelta(0)).all()

    def test_case_insensitive_column_matching(self):
        """Test that column matching is case-insensitive."""
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        df = pd.DataFrame({
            "DATE": dates,  # uppercase
            "DEMAND": np.arange(100.0, 112.0),  # uppercase
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is True
        assert cleaned is not None

    def test_mixed_case_column_matching(self):
        """Test with mixed-case column names."""
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        df = pd.DataFrame({
            "OrderDate": dates,
            "RevenueForecast": np.arange(100.0, 112.0),
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is True
        assert cleaned is not None

    def test_extra_columns_ignored(self):
        """Test that extra columns are ignored."""
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        df = pd.DataFrame({
            "Date": dates,
            "Demand": np.arange(100.0, 112.0),
            "Category": ["A"] * 12,
            "Region": ["North"] * 12,
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is True
        assert cleaned is not None
        assert len(cleaned.columns) == 2
        assert list(cleaned.columns) == ["ds", "y"]

    def test_exactly_12_rows(self):
        """Test with exactly 12 rows (minimum allowed)."""
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")
        df = pd.DataFrame({
            "Date": dates,
            "Demand": np.arange(100.0, 112.0),
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is True
        assert cleaned is not None
        assert len(cleaned) == 12

    def test_large_dataset(self):
        """Test with a large dataset (5 years of monthly data)."""
        dates = pd.date_range("2019-01-01", periods=60, freq="MS")
        df = pd.DataFrame({
            "Date": dates,
            "Demand": np.random.uniform(100, 1000, 60),
        })

        ok, error, cleaned = _validate_company_df(df)

        assert ok is True
        assert cleaned is not None
        assert len(cleaned) == 60

    def test_column_name_priority(self):
        """Test column detection priority: 'demand' > 'revenue' > 'y'."""
        dates = pd.date_range("2022-01-01", periods=12, freq="MS")

        # Test 'y' is matched (exact lowercase match of column name)
        df = pd.DataFrame({
            "Date": dates,
            "y": np.arange(100.0, 112.0),
        })
        ok, error, cleaned = _validate_company_df(df)
        assert ok is True
        assert (cleaned["y"] == df["y"]).all()

    def test_parsing_error(self):
        """Test that parsing errors are caught."""
        # Create a DataFrame that might cause parsing issues
        df = pd.DataFrame({
            "Date": ["not", "a", "date", "really", "bad", "input",
                     "more", "invalid", "stuff", "here", "no", "way"],
            "Demand": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        })

        ok, error, cleaned = _validate_company_df(df)

        # Dates will parse but become NaT, which should fail validation
        assert ok is False
        assert cleaned is None
