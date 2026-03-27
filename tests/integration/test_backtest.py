"""
Integration tests for utils/backtest.py :: _run_backtest

All tests mock Prophet to avoid slow model training while still exercising the
full control-flow of the backtest engine (fold splitting, indicator re-ranking,
metric aggregation, naive baseline, etc.).
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_merged_df(n_months: int, start: str = "2018-01-01") -> pd.DataFrame:
    """Return a merged DataFrame with ds/y/GDP_growth/inflation/unemployment."""
    dates = pd.date_range(start, periods=n_months, freq="MS")
    np.random.seed(0)
    trend = np.linspace(1000.0, 1500.0, n_months)
    seasonality = 150.0 * np.sin(np.arange(n_months) * 2 * np.pi / 12)
    noise = np.random.normal(0, 40, n_months)
    y_vals = trend + seasonality + noise + 500  # ensure positive

    gdp = 2.0 + np.random.normal(0, 0.3, n_months)
    inflation = 3.0 + np.random.normal(0, 0.5, n_months)
    unemployment = 4.5 + np.random.normal(0, 0.2, n_months)

    return pd.DataFrame({
        "ds": dates,
        "y": y_vals,
        "GDP_growth": gdp,
        "inflation": inflation,
        "unemployment": unemployment,
    })


def _make_helper_mock(selected_names=None):
    """Return a MagicMock helper whose rank_indicators returns a realistic dict."""
    from utils.fred_helper import FredHelper

    if selected_names is None:
        selected_names = ["GDP_growth", "inflation"]

    helper = MagicMock()
    helper.INDICATOR_CATALOGUE = FredHelper.INDICATOR_CATALOGUE
    helper.SCENARIO_DEFAULTS = FredHelper.SCENARIO_DEFAULTS

    ranking_result = {
        "selected_names": selected_names,
        "all_rankings": [
            {"name": n, "label": FredHelper.INDICATOR_CATALOGUE[n][1], "corr": 0.5}
            for n in selected_names
        ],
        "selected": [
            {"name": n, "label": FredHelper.INDICATOR_CATALOGUE[n][1], "corr": 0.5}
            for n in selected_names
        ],
        "r_squared": 0.4,
        "coefficients": {n: 0.5 for n in selected_names},
    }
    helper.rank_indicators = MagicMock(return_value=ranking_result)
    return helper


def _make_prophet_mock(merged_df: pd.DataFrame, horizon: int):
    """
    Build a Prophet class mock whose instances return plausible forecast DataFrames.

    The mock predict() returns a DataFrame containing every date in merged_df
    plus `horizon` extra months, with yhat = actual y (or a constant for future
    rows) to guarantee non-NaN metrics.
    """
    all_dates = pd.date_range(
        merged_df["ds"].min(),
        periods=len(merged_df) + horizon,
        freq="MS",
    )

    def make_future_df(periods, freq="MS"):
        # return a DataFrame with the ds column the fold actually needs
        future = pd.DataFrame({"ds": all_dates})
        return future

    def mock_predict(future_df):
        n = len(future_df)
        # Re-index y values so predictions are close to actuals for test rows
        y_map = dict(zip(merged_df["ds"], merged_df["y"]))
        yhat = np.array([
            y_map.get(d, merged_df["y"].mean())
            for d in future_df["ds"]
        ], dtype=float)
        return pd.DataFrame({
            "ds": future_df["ds"].values,
            "yhat": yhat,
            "yhat_lower": yhat * 0.9,
            "yhat_upper": yhat * 1.1,
        })

    prophet_instance = MagicMock()
    prophet_instance.fit = MagicMock(return_value=None)
    prophet_instance.make_future_dataframe = MagicMock(side_effect=make_future_df)
    prophet_instance.predict = MagicMock(side_effect=mock_predict)
    prophet_instance.add_regressor = MagicMock()

    prophet_cls = MagicMock(return_value=prophet_instance)
    return prophet_cls, prophet_instance


# ---------------------------------------------------------------------------
# Test 1 — insufficient data returns error
# ---------------------------------------------------------------------------

class TestInsufficientDataReturnsError:
    """_run_backtest should return (None, error_str) when data is too short."""

    def test_20_months_returns_none_and_error(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=20)
        helper = _make_helper_mock()

        result, error = _run_backtest(
            merged=merged,
            helper=helper,
            horizon=12,
            n_folds=2,
            trend_flex=0.05,
            growth_rate_override=None,
        )

        assert result is None, "Expected None result for insufficient data"
        assert isinstance(error, str), "Expected a string error message"
        assert len(error) > 0, "Error message should not be empty"

    def test_error_message_mentions_months(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=20)
        helper = _make_helper_mock()

        _, error = _run_backtest(
            merged=merged,
            helper=helper,
            horizon=12,
            n_folds=2,
            trend_flex=0.05,
            growth_rate_override=None,
        )

        # The error should be informative enough to mention data size
        assert error is not None
        assert any(
            token in error.lower()
            for token in ["month", "data", "horizon", "enough"]
        ), f"Error message not informative: {error!r}"


# ---------------------------------------------------------------------------
# Test 2 — successful backtest structure
# ---------------------------------------------------------------------------

class TestSuccessfulBacktestStructure:
    """With sufficient data and mocked Prophet, _run_backtest should return
    a well-formed result dict."""

    EXPECTED_KEYS = {
        "results_df",
        "train_df",
        "fold_metrics",
        "fold_indicators",
        "macro_mape",
        "macro_mae",
        "macro_rmse",
        "vanilla_mape",
        "vanilla_mae",
        "vanilla_rmse",
        "naive_mape",
        "naive_mae",
        "naive_rmse",
        "n_test_months",
        "n_folds",
    }

    def test_returns_dict_and_none_error(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, error = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=2,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        assert error is None, f"Expected no error, got: {error!r}"
        assert isinstance(result, dict), "Expected a dict result"

    def test_all_expected_keys_present(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, _ = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=2,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        missing = self.EXPECTED_KEYS - set(result.keys())
        assert not missing, f"Result dict missing keys: {missing}"

    def test_results_df_is_dataframe(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, _ = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=2,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        assert isinstance(result["results_df"], pd.DataFrame)
        assert isinstance(result["train_df"], pd.DataFrame)

    def test_results_df_has_required_columns(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, _ = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=2,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        cols = set(result["results_df"].columns)
        for required in ("ds", "y", "yhat_macro", "yhat_vanilla", "yhat_naive"):
            assert required in cols, f"results_df missing column: {required!r}"

    def test_averaged_metrics_are_floats(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, _ = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=2,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        for key in ("macro_mape", "vanilla_mape", "naive_mape",
                    "macro_mae", "vanilla_mae", "naive_mae",
                    "macro_rmse", "vanilla_rmse", "naive_rmse"):
            val = result[key]
            assert isinstance(val, float), f"{key!r} should be float, got {type(val)}"
            assert val >= 0, f"{key!r} should be non-negative, got {val}"


# ---------------------------------------------------------------------------
# Test 3 — folds use pre-cutoff data only
# ---------------------------------------------------------------------------

class TestFoldsUsePreCutoffData:
    """rank_indicators must only receive rows with ds <= cutoff_date for each fold."""

    def test_rank_indicators_called_with_pre_cutoff_slices(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        captured_dfs = []

        def capturing_rank_indicators(df, top_n=3):
            captured_dfs.append(df.copy())
            return {
                "selected_names": ["GDP_growth", "inflation"],
                "all_rankings": [
                    {"name": "GDP_growth", "label": "Real GDP Growth Rate", "corr": 0.5},
                    {"name": "inflation", "label": "CPI Inflation (YoY %)", "corr": -0.4},
                ],
                "selected": [
                    {"name": "GDP_growth", "label": "Real GDP Growth Rate", "corr": 0.5},
                    {"name": "inflation", "label": "CPI Inflation (YoY %)", "corr": -0.4},
                ],
                "r_squared": 0.4,
                "coefficients": {"GDP_growth": 0.5, "inflation": -0.3},
            }

        helper.rank_indicators = MagicMock(side_effect=capturing_rank_indicators)

        max_date = merged["ds"].max()

        with patch("utils.backtest.Prophet", prophet_cls):
            result, error = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=2,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        assert error is None, f"Unexpected error: {error!r}"
        assert len(captured_dfs) > 0, "rank_indicators was never called"

        for fold_df in captured_dfs:
            # Every date passed to rank_indicators must be <= max cutoff
            # (i.e., strictly before the end of the full series minus at least one horizon)
            fold_max = fold_df["ds"].max()
            assert fold_max < max_date, (
                f"rank_indicators received data up to {fold_max}, "
                f"which is not strictly before the series end {max_date}. "
                "Data leakage detected."
            )

    def test_rank_indicators_call_count_matches_folds(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, error = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=2,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        assert error is None
        # rank_indicators should be called once per fold
        n_actual_folds = result["n_folds"]
        assert helper.rank_indicators.call_count == n_actual_folds, (
            f"rank_indicators called {helper.rank_indicators.call_count} times "
            f"but n_folds={n_actual_folds}"
        )


# ---------------------------------------------------------------------------
# Test 4 — naive baseline uses growth_rate_override
# ---------------------------------------------------------------------------

class TestNaiveBaselineUsesGrowthOverride:
    """When growth_rate_override is provided, naive predictions must use it
    instead of the training data's median YoY growth."""

    def _run_with_override(self, override: float):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, error = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=1,
                trend_flex=0.05,
                growth_rate_override=override,
            )
        assert error is None, f"Unexpected error: {error!r}"
        return result

    def test_naive_predictions_differ_for_different_overrides(self):
        """Two different growth overrides must produce different naive forecasts."""
        result_low = self._run_with_override(0.0)
        result_high = self._run_with_override(0.5)

        naive_low = result_low["results_df"]["yhat_naive"].values
        naive_high = result_high["results_df"]["yhat_naive"].values

        assert not np.allclose(naive_low, naive_high), (
            "Naive predictions with override=0.0 and override=0.5 should differ, "
            "but they are identical."
        )

    def test_naive_zero_growth_equals_last_year_actuals(self):
        """With growth_rate_override=0.0, naive yhat should equal the corresponding
        training actuals from 12 months prior (within floating-point tolerance)."""
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, error = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=1,
                trend_flex=0.05,
                growth_rate_override=0.0,
            )

        assert error is None
        results_df = result["results_df"]

        # Build the training cutoff for the single fold
        max_date = merged["ds"].max()
        cutoff = max_date - pd.DateOffset(months=12)

        train = merged[merged["ds"] <= cutoff].copy()
        naive_source = train.tail(12)[["ds", "y"]].reset_index(drop=True)
        test_rows = results_df.sort_values("ds").reset_index(drop=True)

        n = min(len(naive_source), len(test_rows))
        expected_naive = naive_source["y"].values[:n] * 1.0  # g=0 → ×1
        actual_naive = test_rows["yhat_naive"].values[:n]

        np.testing.assert_allclose(
            actual_naive, expected_naive, rtol=1e-6,
            err_msg="With growth_rate_override=0.0, naive yhat should equal prior-year actuals",
        )

    def test_positive_growth_override_inflates_naive(self):
        """With a positive override the naive predictions should be larger than
        with a zero override."""
        result_zero = self._run_with_override(0.0)
        result_pos = self._run_with_override(0.1)

        mean_zero = result_zero["results_df"]["yhat_naive"].mean()
        mean_pos = result_pos["results_df"]["yhat_naive"].mean()

        assert mean_pos > mean_zero, (
            f"Expected naive mean with override=0.1 ({mean_pos:.2f}) > "
            f"override=0.0 ({mean_zero:.2f})"
        )


# ---------------------------------------------------------------------------
# Test 5 — fold_metrics structure
# ---------------------------------------------------------------------------

class TestFoldMetricsStructure:
    """Every entry in fold_metrics must have the complete set of required keys."""

    REQUIRED_FOLD_KEYS = {
        "fold",
        "cutoff",
        "n_months",
        "macro_mape",
        "macro_mae",
        "macro_rmse",
        "vanilla_mape",
        "vanilla_mae",
        "vanilla_rmse",
        "naive_mape",
        "naive_mae",
        "naive_rmse",
    }

    def _get_fold_metrics(self, n_folds: int = 2):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, error = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=n_folds,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        assert error is None, f"Unexpected error: {error!r}"
        return result["fold_metrics"]

    def test_fold_metrics_is_list(self):
        fold_metrics = self._get_fold_metrics()
        assert isinstance(fold_metrics, list)
        assert len(fold_metrics) > 0

    def test_each_fold_has_all_required_keys(self):
        fold_metrics = self._get_fold_metrics()
        for i, fm in enumerate(fold_metrics):
            missing = self.REQUIRED_FOLD_KEYS - set(fm.keys())
            assert not missing, (
                f"fold_metrics[{i}] missing keys: {missing}. Got: {set(fm.keys())}"
            )

    def test_fold_index_starts_at_one(self):
        fold_metrics = self._get_fold_metrics()
        for i, fm in enumerate(fold_metrics):
            assert fm["fold"] == i + 1, (
                f"Expected fold={i + 1}, got fold={fm['fold']}"
            )

    def test_cutoff_is_string(self):
        """cutoff should be a human-readable string like 'Jan 2024'."""
        fold_metrics = self._get_fold_metrics()
        for fm in fold_metrics:
            assert isinstance(fm["cutoff"], str), (
                f"Expected cutoff to be str, got {type(fm['cutoff'])}"
            )
            assert len(fm["cutoff"]) > 0

    def test_n_months_is_positive_int(self):
        fold_metrics = self._get_fold_metrics()
        for fm in fold_metrics:
            assert isinstance(fm["n_months"], int), (
                f"Expected n_months int, got {type(fm['n_months'])}"
            )
            assert fm["n_months"] >= 2

    def test_metric_values_are_non_negative_floats(self):
        fold_metrics = self._get_fold_metrics()
        metric_keys = [
            "macro_mape", "macro_mae", "macro_rmse",
            "vanilla_mape", "vanilla_mae", "vanilla_rmse",
            "naive_mape", "naive_mae", "naive_rmse",
        ]
        for fm in fold_metrics:
            for key in metric_keys:
                val = fm[key]
                assert isinstance(val, float), (
                    f"{key!r} should be float, got {type(val)}"
                )
                assert val >= 0, f"{key!r} should be >= 0, got {val}"

    def test_fold_indicators_length_matches_fold_metrics(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, _ = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=2,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        assert len(result["fold_metrics"]) == len(result["fold_indicators"]), (
            "fold_metrics and fold_indicators must have the same length"
        )


# ---------------------------------------------------------------------------
# Test 6 — n_folds capped by available data
# ---------------------------------------------------------------------------

class TestNFoldsCapped:
    """Requesting more folds than the data can support should silently cap
    actual_folds to (total_months - min_train_months) // horizon."""

    def test_folds_capped_at_data_limit(self):
        from utils.backtest import _run_backtest

        # 48 months, horizon=12 → max_possible = (48-24)//12 = 2
        merged = _make_merged_df(n_months=48)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, error = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=10,   # asks for 10, but only 2 are possible
                trend_flex=0.05,
                growth_rate_override=None,
            )

        assert error is None, f"Unexpected error: {error!r}"
        assert result["n_folds"] <= 2, (
            f"Expected at most 2 folds for 48 months / horizon=12, "
            f"got {result['n_folds']}"
        )

    def test_folds_never_exceed_requested(self):
        """Even with abundant data, n_folds should never exceed the requested value."""
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=120)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, error = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=3,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        assert error is None
        assert result["n_folds"] <= 3, (
            f"n_folds should not exceed requested value 3, got {result['n_folds']}"
        )

    def test_n_folds_in_result_equals_len_fold_metrics(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=48)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, _ = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=10,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        assert result["n_folds"] == len(result["fold_metrics"]), (
            "n_folds in result dict should equal len(fold_metrics)"
        )

    def test_n_test_months_equals_sum_of_fold_n_months(self):
        from utils.backtest import _run_backtest

        merged = _make_merged_df(n_months=60)
        helper = _make_helper_mock()
        prophet_cls, _ = _make_prophet_mock(merged, horizon=12)

        with patch("utils.backtest.Prophet", prophet_cls):
            result, _ = _run_backtest(
                merged=merged,
                helper=helper,
                horizon=12,
                n_folds=2,
                trend_flex=0.05,
                growth_rate_override=None,
            )

        expected_total = sum(fm["n_months"] for fm in result["fold_metrics"])
        assert result["n_test_months"] == expected_total, (
            f"n_test_months={result['n_test_months']} != "
            f"sum(fold n_months)={expected_total}"
        )
