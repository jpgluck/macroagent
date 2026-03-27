"""
Integration tests for utils/forecasting.py — run_forecast().

Mocks Prophet to avoid slow training for most tests.
One real-Prophet test is marked @pytest.mark.slow.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(periods=36):
    return pd.date_range("2021-01-01", periods=periods, freq="MS")


def _make_merged(periods=36, regressor_names=None):
    """Build a minimal merged DataFrame with ds, y, and optional regressor columns."""
    regressor_names = regressor_names or []
    dates = _make_dates(periods)
    np.random.seed(0)
    df = pd.DataFrame({"ds": dates, "y": np.random.uniform(1000, 2000, periods)})
    for name in regressor_names:
        df[name] = np.random.uniform(0, 5, periods)
    return df


def _make_mock_prophet_cls(periods=36, regressor_names=None):
    """
    Return a mock class for Prophet.

    The mock *instance* (returned by calling the class) has:
    - fit()                   → no-op, returns instance
    - add_regressor()         → no-op
    - make_future_dataframe() → DataFrame with 'ds' column (periods + 12 rows)
    - predict()               → DataFrame with ds, yhat, yhat_lower, yhat_upper
    """
    regressor_names = regressor_names or []
    total = periods + 12
    future_dates = _make_dates(total)

    future_df = pd.DataFrame({"ds": future_dates})
    for name in regressor_names:
        future_df[name] = np.nan  # historical rows will be filled by merge

    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": np.random.uniform(1000, 2000, total),
        "yhat_lower": np.random.uniform(800, 1000, total),
        "yhat_upper": np.random.uniform(2000, 2200, total),
    })

    mock_instance = MagicMock()
    mock_instance.fit.return_value = mock_instance
    mock_instance.add_regressor.return_value = None
    mock_instance.make_future_dataframe.return_value = future_df.copy()
    mock_instance.predict.return_value = forecast_df.copy()

    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls, mock_instance, forecast_df


def _make_mock_helper(future_passthrough=True):
    """
    MagicMock for FredHelper.
    prepare_future_regressors returns its first argument unchanged when
    future_passthrough=True.
    """
    helper = MagicMock()
    if future_passthrough:
        helper.prepare_future_regressors.side_effect = lambda future_df, **kw: future_df
    return helper


# ---------------------------------------------------------------------------
# Test 1 — run_forecast returns (forecast_df, model, train_df) tuple
# ---------------------------------------------------------------------------

def test_run_forecast_returns_tuple():
    """run_forecast must return a 3-tuple: (forecast_df, model, train_df)."""
    sel_names = ["GDP_growth"]
    merged = _make_merged(regressor_names=sel_names)
    mock_cls, mock_instance, expected_forecast = _make_mock_prophet_cls(
        regressor_names=sel_names
    )
    helper = _make_mock_helper()

    with patch("utils.forecasting.Prophet", mock_cls):
        from utils.forecasting import run_forecast
        result = run_forecast(
            merged=merged,
            sel_names=sel_names,
            scenario_vals={"GDP_growth": 2.5},
            trend_flexibility=0.05,
            helper=helper,
        )

    assert isinstance(result, tuple), "run_forecast must return a tuple"
    assert len(result) == 3, "Tuple must have exactly 3 elements"

    forecast_df, model, train_df = result
    assert isinstance(forecast_df, pd.DataFrame), "First element must be a DataFrame"
    assert model is mock_instance, "Second element must be the Prophet model instance"
    assert isinstance(train_df, pd.DataFrame), "Third element must be a DataFrame"


# ---------------------------------------------------------------------------
# Test 2 — add_regressor called for each name in sel_names
# ---------------------------------------------------------------------------

def test_run_forecast_adds_regressors():
    """model.add_regressor must be called once per regressor, in order."""
    sel_names = ["GDP_growth", "inflation"]
    merged = _make_merged(regressor_names=sel_names)
    mock_cls, mock_instance, _ = _make_mock_prophet_cls(regressor_names=sel_names)
    helper = _make_mock_helper()

    with patch("utils.forecasting.Prophet", mock_cls):
        from utils.forecasting import run_forecast
        run_forecast(
            merged=merged,
            sel_names=sel_names,
            scenario_vals={},
            trend_flexibility=0.05,
            helper=helper,
        )

    assert mock_instance.add_regressor.call_count == 2, (
        "add_regressor should be called once per indicator"
    )
    calls = mock_instance.add_regressor.call_args_list
    assert calls[0] == call("GDP_growth"), "First add_regressor call should be GDP_growth"
    assert calls[1] == call("inflation"), "Second add_regressor call should be inflation"


# ---------------------------------------------------------------------------
# Test 3 — Prophet instantiated with the supplied changepoint_prior_scale
# ---------------------------------------------------------------------------

def test_run_forecast_uses_trend_flexibility():
    """Prophet must be constructed with changepoint_prior_scale=trend_flexibility."""
    trend_flexibility = 0.35
    sel_names = ["unemployment"]
    merged = _make_merged(regressor_names=sel_names)
    mock_cls, _, _ = _make_mock_prophet_cls(regressor_names=sel_names)
    helper = _make_mock_helper()

    with patch("utils.forecasting.Prophet", mock_cls):
        from utils.forecasting import run_forecast
        run_forecast(
            merged=merged,
            sel_names=sel_names,
            scenario_vals={},
            trend_flexibility=trend_flexibility,
            helper=helper,
        )

    mock_cls.assert_called_once()
    _, kwargs = mock_cls.call_args
    assert "changepoint_prior_scale" in kwargs, (
        "Prophet must be called with changepoint_prior_scale keyword"
    )
    assert kwargs["changepoint_prior_scale"] == trend_flexibility, (
        f"Expected changepoint_prior_scale={trend_flexibility}, "
        f"got {kwargs['changepoint_prior_scale']}"
    )


# ---------------------------------------------------------------------------
# Test 4 — helper.prepare_future_regressors called with scenario_values
# ---------------------------------------------------------------------------

def test_run_forecast_calls_prepare_future_regressors():
    """helper.prepare_future_regressors must be called with the supplied scenario_vals."""
    sel_names = ["GDP_growth", "inflation"]
    scenario_vals = {"GDP_growth": 3.1, "inflation": 4.5}
    merged = _make_merged(regressor_names=sel_names)
    mock_cls, _, _ = _make_mock_prophet_cls(regressor_names=sel_names)
    helper = _make_mock_helper()

    with patch("utils.forecasting.Prophet", mock_cls):
        from utils.forecasting import run_forecast
        run_forecast(
            merged=merged,
            sel_names=sel_names,
            scenario_vals=scenario_vals,
            trend_flexibility=0.05,
            helper=helper,
        )

    helper.prepare_future_regressors.assert_called_once()
    _, kwargs = helper.prepare_future_regressors.call_args
    assert "scenario_values" in kwargs, (
        "prepare_future_regressors must be called with scenario_values keyword"
    )
    assert kwargs["scenario_values"] == scenario_vals, (
        "scenario_values passed to prepare_future_regressors must match the input"
    )


# ---------------------------------------------------------------------------
# Test 5 — train_df has exactly ["ds", "y"] + sel_names columns
# ---------------------------------------------------------------------------

def test_run_forecast_train_df_columns():
    """train_df returned by run_forecast must have exactly ds, y, and regressor columns."""
    sel_names = ["GDP_growth", "inflation", "unemployment"]
    merged = _make_merged(regressor_names=sel_names)
    # Add an extra column that should NOT appear in train_df
    merged["extra_col"] = 99.0
    mock_cls, _, _ = _make_mock_prophet_cls(regressor_names=sel_names)
    helper = _make_mock_helper()

    with patch("utils.forecasting.Prophet", mock_cls):
        from utils.forecasting import run_forecast
        _, _, train_df = run_forecast(
            merged=merged,
            sel_names=sel_names,
            scenario_vals={},
            trend_flexibility=0.05,
            helper=helper,
        )

    expected_cols = ["ds", "y"] + sel_names
    assert list(train_df.columns) == expected_cols, (
        f"train_df columns should be {expected_cols}, got {list(train_df.columns)}"
    )
    assert "extra_col" not in train_df.columns, (
        "train_df must not include columns beyond ds, y, and sel_names"
    )


# ---------------------------------------------------------------------------
# Test 6 — real Prophet training (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_forecast_real_prophet():
    """
    Train an actual Prophet model (no mocking).

    Uses 36 months of data with 1 regressor.
    Asserts forecast has 'yhat' column and 48 rows (36 historical + 12 future).
    """
    # Import here so the module is only touched in this test
    try:
        from prophet import Prophet as _Prophet  # noqa: F401
    except ImportError:
        pytest.skip("prophet not installed")

    sel_names = ["GDP_growth"]
    dates = pd.date_range("2021-01-01", periods=36, freq="MS")
    np.random.seed(42)

    merged = pd.DataFrame({
        "ds": dates,
        "y": 1000 + 300 * np.sin(np.arange(36) * 2 * np.pi / 12)
              + np.linspace(0, 200, 36)
              + np.random.normal(0, 30, 36),
        "GDP_growth": 2.5 + np.random.normal(0, 0.3, 36),
    })

    helper = MagicMock()
    # Real passthrough: return future_df unchanged so Prophet can predict
    helper.prepare_future_regressors.side_effect = (
        lambda future_df, scenario_values=None, **kw: future_df.fillna(
            {name: scenario_values.get(name, 2.5) for name in sel_names}
            if scenario_values
            else {}
        )
    )

    from utils.forecasting import run_forecast

    forecast_df, model, train_df = run_forecast(
        merged=merged,
        sel_names=sel_names,
        scenario_vals={"GDP_growth": 2.5},
        trend_flexibility=0.05,
        helper=helper,
    )

    assert "yhat" in forecast_df.columns, "forecast_df must contain a 'yhat' column"
    assert len(forecast_df) == 48, (
        f"forecast_df must have 48 rows (36 historical + 12 future), got {len(forecast_df)}"
    )
