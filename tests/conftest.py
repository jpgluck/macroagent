"""
Shared pytest fixtures for MacroAgent Forecaster tests.

Provides realistic test data, mock objects, and DataFrames for testing
validation, forecasting, backtesting, and charting modules.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta


# ==============================================================================
# Fixture 1: sample_company_df
# ==============================================================================

@pytest.fixture(scope="function")
def sample_company_df():
    """
    DataFrame with 36 rows of monthly data.
    Columns: 'ds' (datetime, monthly from 2021-01-01), 'y' (positive floats).
    Includes realistic demand pattern: ~1000-2000 with upward trend and seasonality.
    Sorted by ds.
    """
    # Start date
    start_date = pd.Timestamp("2021-01-01")
    dates = [start_date + pd.DateOffset(months=i) for i in range(36)]

    # Generate demand with trend + seasonality
    np.random.seed(42)
    trend = np.linspace(1000, 1500, 36)
    seasonality = 200 * np.sin(np.arange(36) * 2 * np.pi / 12)
    noise = np.random.normal(0, 50, 36)
    y_values = trend + seasonality + noise

    # Ensure all positive
    y_values = np.abs(y_values) + 1000

    df = pd.DataFrame({
        "ds": dates,
        "y": y_values,
    })
    df = df.sort_values("ds").reset_index(drop=True)
    return df


# ==============================================================================
# Fixture 2: raw_csv_df
# ==============================================================================

@pytest.fixture(scope="function")
def raw_csv_df(sample_company_df):
    """
    DataFrame simulating raw CSV upload.
    Columns: 'Date' and 'Demand' (matching validation's substring detection).
    Same 36 rows as sample_company_df but with original column names.
    """
    df = sample_company_df.copy()
    df.columns = ["Date", "Demand"]
    return df


# ==============================================================================
# Fixture 3: mock_indicators
# ==============================================================================

@pytest.fixture(scope="function")
def mock_indicators(sample_company_df):
    """
    Dict of {name: pd.Series} for 3 indicators.
    Each Series: 36 monthly values indexed by month-start timestamps.
    Aligned with sample_company_df dates. Realistic ranges:
    - GDP_growth: 1-4%
    - inflation: 2-8%
    - unemployment: 3-6%
    """
    dates = pd.DatetimeIndex(sample_company_df["ds"])

    np.random.seed(42)

    # GDP growth: 1-4%, slight trend
    gdp_vals = 2.0 + 1.0 * np.sin(np.arange(36) * 2 * np.pi / 12) + np.random.normal(0, 0.3, 36)
    gdp_vals = np.clip(gdp_vals, 1.0, 4.0)

    # Inflation: 2-8%, more volatile
    inflation_vals = 3.0 + 2.0 * np.sin(np.arange(36) * 2 * np.pi / 12 + 1) + np.random.normal(0, 0.5, 36)
    inflation_vals = np.clip(inflation_vals, 2.0, 8.0)

    # Unemployment: 3-6%, inverse seasonality
    unemployment_vals = 4.5 + 1.0 * np.cos(np.arange(36) * 2 * np.pi / 12) + np.random.normal(0, 0.2, 36)
    unemployment_vals = np.clip(unemployment_vals, 3.0, 6.0)

    return {
        "GDP_growth": pd.Series(gdp_vals, index=dates, name="GDP_growth"),
        "inflation": pd.Series(inflation_vals, index=dates, name="inflation"),
        "unemployment": pd.Series(unemployment_vals, index=dates, name="unemployment"),
    }


# ==============================================================================
# Fixture 4: merged_df
# ==============================================================================

@pytest.fixture(scope="function")
def merged_df(sample_company_df, mock_indicators):
    """
    DataFrame combining sample_company_df with mock_indicators columns.
    Mimics output of FredHelper.align_all_indicators().
    Columns: ds, y, GDP_growth, inflation, unemployment.
    """
    df = sample_company_df.copy()

    # Convert indicators dict to DataFrame for merging
    indicators_df = pd.DataFrame(mock_indicators)
    indicators_df.index.name = "ds"
    indicators_df = indicators_df.reset_index()

    # Inner join on date
    merged = pd.merge(df, indicators_df, on="ds", how="inner")
    merged = merged.sort_values("ds").reset_index(drop=True)
    return merged


# ==============================================================================
# Fixture 5: sample_forecast_df
# ==============================================================================

@pytest.fixture(scope="function")
def sample_forecast_df(sample_company_df, mock_indicators):
    """
    DataFrame mimicking Prophet forecast output.
    48 rows (36 historical + 12 future).
    Columns: ds, yhat, yhat_lower, yhat_upper, trend, yearly,
    plus GDP_growth, inflation, unemployment.
    """
    # Generate 48 dates (36 historical + 12 future)
    start_date = pd.Timestamp("2021-01-01")
    forecast_dates = [start_date + pd.DateOffset(months=i) for i in range(48)]

    np.random.seed(42)

    # Generate forecast with trend and seasonality
    trend = np.linspace(1000, 1800, 48)
    yearly = 150 * np.sin(np.arange(48) * 2 * np.pi / 12)
    noise = np.random.normal(0, 100, 48)
    yhat_vals = trend + yearly + noise
    yhat_vals = np.abs(yhat_vals) + 1200

    # Confidence intervals
    yhat_lower = yhat_vals * 0.90
    yhat_upper = yhat_vals * 1.10

    df = pd.DataFrame({
        "ds": forecast_dates,
        "yhat": yhat_vals,
        "yhat_lower": yhat_lower,
        "yhat_upper": yhat_upper,
        "trend": trend,
        "yearly": yearly,
    })

    # Extend mock indicators to 48 months
    for name, series in mock_indicators.items():
        extended_vals = np.tile(series.values, 2)[:48]
        df[name] = extended_vals

    return df


# ==============================================================================
# Fixture 6: sample_backtest_results_df
# ==============================================================================

@pytest.fixture(scope="function")
def sample_backtest_results_df():
    """
    DataFrame with backtest results.
    Columns: ds, y, yhat_macro, yhat_vanilla, yhat_naive.
    12 rows representing hold-out period.
    """
    start_date = pd.Timestamp("2023-01-01")
    dates = [start_date + pd.DateOffset(months=i) for i in range(12)]

    np.random.seed(42)

    # Actual values
    y_vals = 1500 + 100 * np.arange(12) + 150 * np.sin(np.arange(12) * 2 * np.pi / 12)

    # Predictions (slightly different from actuals)
    yhat_macro = y_vals + np.random.normal(0, 80, 12)
    yhat_vanilla = y_vals + np.random.normal(0, 120, 12)
    yhat_naive = y_vals + np.random.normal(0, 100, 12)

    df = pd.DataFrame({
        "ds": dates,
        "y": y_vals,
        "yhat_macro": yhat_macro,
        "yhat_vanilla": yhat_vanilla,
        "yhat_naive": yhat_naive,
    })
    return df


# ==============================================================================
# Fixture 7: mock_fred_helper
# ==============================================================================

@pytest.fixture(scope="function")
def mock_fred_helper(merged_df):
    """
    MagicMock of FredHelper with realistic behavior.

    Methods mocked:
    - rank_indicators: returns dict with all_rankings, selected, selected_names,
      r_squared, coefficients
    - prepare_future_regressors: returns input future_df with scenario values filled

    Class attributes:
    - INDICATOR_CATALOGUE
    - SCENARIO_DEFAULTS
    """
    from utils.fred_helper import FredHelper

    helper = MagicMock(spec=FredHelper)

    # Set class attributes
    helper.INDICATOR_CATALOGUE = FredHelper.INDICATOR_CATALOGUE
    helper.SCENARIO_DEFAULTS = FredHelper.SCENARIO_DEFAULTS

    # Mock rank_indicators
    def mock_rank_indicators(df, top_n=3):
        """Return realistic ranking output."""
        return {
            "all_rankings": [
                {"name": "GDP_growth", "label": "Real GDP Growth Rate", "corr": 0.65},
                {"name": "inflation", "label": "CPI Inflation (YoY %)", "corr": -0.52},
                {"name": "unemployment", "label": "Unemployment Rate", "corr": -0.48},
                {"name": "fed_funds", "label": "Federal Funds Rate", "corr": -0.35},
                {"name": "consumer_sentiment", "label": "Consumer Sentiment (UMich)", "corr": 0.42},
            ],
            "selected": [
                {"name": "GDP_growth", "label": "Real GDP Growth Rate", "corr": 0.65},
                {"name": "inflation", "label": "CPI Inflation (YoY %)", "corr": -0.52},
                {"name": "unemployment", "label": "Unemployment Rate", "corr": -0.48},
            ],
            "selected_names": ["GDP_growth", "inflation", "unemployment"],
            "r_squared": 0.45,
            "coefficients": {
                "GDP_growth": 45.2,
                "inflation": -22.8,
                "unemployment": -31.5,
            },
        }

    helper.rank_indicators = MagicMock(side_effect=mock_rank_indicators)

    # Mock prepare_future_regressors
    def mock_prepare_future_regressors(future_df, scenario_values=None, **kwargs):
        """Fill NaN scenario values with provided values."""
        future = future_df.copy()
        if scenario_values:
            for name, value in scenario_values.items():
                if name in future.columns:
                    future[name] = future[name].fillna(value)
                else:
                    future[name] = value
        return future

    helper.prepare_future_regressors = MagicMock(side_effect=mock_prepare_future_regressors)

    return helper


# ==============================================================================
# Fixture 8: all_rankings_list
# ==============================================================================

@pytest.fixture(scope="function")
def all_rankings_list():
    """
    List of dicts like [{"name": "GDP_growth", "label": "...", "corr": 0.65}, ...]
    for 5+ indicators, sorted by |corr| descending.
    """
    return [
        {"name": "GDP_growth", "label": "Real GDP Growth Rate", "corr": 0.6543},
        {"name": "inflation", "label": "CPI Inflation (YoY %)", "corr": -0.5234},
        {"name": "unemployment", "label": "Unemployment Rate", "corr": -0.4821},
        {"name": "consumer_sentiment", "label": "Consumer Sentiment (UMich)", "corr": 0.4156},
        {"name": "fed_funds", "label": "Federal Funds Rate", "corr": -0.3542},
        {"name": "industrial_prod", "label": "Industrial Production (YoY %)", "corr": 0.2987},
    ]
