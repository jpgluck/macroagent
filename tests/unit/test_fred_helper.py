"""
tests/unit/test_fred_helper.py
-------------------------------
Unit tests for utils/fred_helper.py.

All FRED API calls are mocked via unittest.mock so no live network access
or API key is required.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from utils.fred_helper import FredHelper, _handle_fred_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monthly_series(n: int = 36, start: str = "2018-01-01", value: float = 100.0) -> pd.Series:
    """Return a pd.Series with a monthly DatetimeIndex and constant value."""
    idx = pd.date_range(start=start, periods=n, freq="MS")
    return pd.Series(np.full(n, value, dtype=float), index=idx)


def _make_daily_series(n: int = 730, start: str = "2020-01-01", value: float = 4.5) -> pd.Series:
    idx = pd.date_range(start=start, periods=n, freq="D")
    return pd.Series(np.full(n, value, dtype=float), index=idx)


def _make_quarterly_series(n: int = 20, start: str = "2015-01-01", value: float = 2.5) -> pd.Series:
    idx = pd.date_range(start=start, periods=n, freq="QS")
    return pd.Series(np.full(n, value, dtype=float), index=idx)


def _make_fred_helper(mock_fred_cls) -> FredHelper:
    """Instantiate FredHelper with a mocked Fred constructor."""
    return FredHelper(api_key="test-key-123")


# ---------------------------------------------------------------------------
# _handle_fred_error
# ---------------------------------------------------------------------------

class TestHandleFredError:

    def test_handle_fred_error_bad_api_key(self):
        """Exception message containing 'bad request' → RuntimeError about invalid API key."""
        exc = Exception("bad request: invalid credentials")
        with pytest.raises(RuntimeError, match="Invalid FRED API key"):
            _handle_fred_error(exc, "CPI Inflation")

    def test_handle_fred_error_bad_api_key_400(self):
        """Exception message containing '400' → RuntimeError about invalid API key."""
        exc = Exception("HTTP 400 error occurred")
        with pytest.raises(RuntimeError, match="Invalid FRED API key"):
            _handle_fred_error(exc, "GDP")

    def test_handle_fred_error_rate_limit(self):
        """Exception message containing '429' → RuntimeError mentioning rate limit."""
        exc = Exception("HTTP 429: too many requests")
        with pytest.raises(RuntimeError, match="rate limit"):
            _handle_fred_error(exc, "Unemployment Rate")

    def test_handle_fred_error_rate_limit_by_word(self):
        """Exception message containing 'rate' → RuntimeError mentioning rate limit."""
        exc = Exception("rate exceeded for series")
        with pytest.raises(RuntimeError, match="rate limit"):
            _handle_fred_error(exc, "Fed Funds")

    def test_handle_fred_error_connection(self):
        """Exception message containing 'connection' → RuntimeError about Network error."""
        exc = Exception("connection refused by host")
        with pytest.raises(RuntimeError, match="Network error"):
            _handle_fred_error(exc, "10-Year Treasury")

    def test_handle_fred_error_timeout(self):
        """Exception message containing 'timeout' → RuntimeError about Network error."""
        exc = Exception("timeout waiting for response")
        with pytest.raises(RuntimeError, match="Network error"):
            _handle_fred_error(exc, "Housing Starts")

    def test_handle_fred_error_unknown(self):
        """Generic exception → RuntimeError mentioning 'Unexpected error'."""
        exc = Exception("some completely unknown error XYZ")
        with pytest.raises(RuntimeError, match="Unexpected error"):
            _handle_fred_error(exc, "S&P 500")

    def test_handle_fred_error_always_raises(self):
        """_handle_fred_error always raises RuntimeError, never returns."""
        exc = Exception("any error")
        with pytest.raises(RuntimeError):
            _handle_fred_error(exc, "series")


# ---------------------------------------------------------------------------
# FredHelper.fetch_indicator
# ---------------------------------------------------------------------------

class TestFetchIndicator:

    @patch("utils.fred_helper.Fred")
    def test_fetch_indicator_level(self, mock_fred_cls):
        """Monthly level indicator resampled to month-start (MS) index."""
        raw = _make_monthly_series(n=24, start="2020-01-01", value=4.0)
        mock_fred_cls.return_value.get_series.return_value = raw

        helper = _make_fred_helper(mock_fred_cls)
        result = helper.fetch_indicator("unemployment")

        assert isinstance(result, pd.Series)
        # All index entries should be month-start
        assert all(ts.day == 1 for ts in result.index)
        assert result.name == "unemployment"
        assert len(result) > 0

    @patch("utils.fred_helper.Fred")
    def test_fetch_indicator_yoy_pct(self, mock_fred_cls):
        """inflation uses yoy_pct transform — verify YoY % is computed from 24-month series."""
        # Use linearly growing values so YoY % is well-defined
        n = 24
        idx = pd.date_range(start="2019-01-01", periods=n, freq="MS")
        values = np.linspace(200.0, 220.0, n)  # gradual increase
        raw = pd.Series(values, index=idx)
        mock_fred_cls.return_value.get_series.return_value = raw

        helper = _make_fred_helper(mock_fred_cls)
        result = helper.fetch_indicator("inflation")

        assert isinstance(result, pd.Series)
        assert result.name == "inflation"
        # After YoY pct_change(12), we lose the first 12 rows
        assert len(result) > 0
        # Values should be percentage changes (~small positive %), not raw index values
        assert result.abs().max() < 50.0  # not raw CPI levels (~200)

    @patch("utils.fred_helper.Fred")
    def test_fetch_indicator_quarterly(self, mock_fred_cls):
        """GDP_growth is quarterly level — verify it is forward-filled to monthly MS."""
        raw = _make_quarterly_series(n=20, start="2015-01-01", value=2.5)
        mock_fred_cls.return_value.get_series.return_value = raw

        helper = _make_fred_helper(mock_fred_cls)
        result = helper.fetch_indicator("GDP_growth")

        assert isinstance(result, pd.Series)
        assert result.name == "GDP_growth"
        # Monthly series should have more rows than the original 20 quarterly rows
        assert len(result) > 20
        # All values should be 2.5 (forward-filled constant)
        assert (result == 2.5).all()

    @patch("utils.fred_helper.Fred")
    def test_fetch_indicator_daily(self, mock_fred_cls):
        """treasury_10y is daily level — verify resampled to monthly last value."""
        raw = _make_daily_series(n=730, start="2020-01-01", value=1.75)
        mock_fred_cls.return_value.get_series.return_value = raw

        helper = _make_fred_helper(mock_fred_cls)
        result = helper.fetch_indicator("treasury_10y")

        assert isinstance(result, pd.Series)
        assert result.name == "treasury_10y"
        # Should collapse 730 daily rows to ~24 monthly rows
        assert len(result) <= 25
        assert all(ts.day == 1 for ts in result.index)
        # Values should equal the constant daily value
        assert (result == 1.75).all()

    @patch("utils.fred_helper.Fred")
    def test_fetch_indicator_error(self, mock_fred_cls):
        """When get_series raises, fetch_indicator should raise RuntimeError via _handle_fred_error."""
        mock_fred_cls.return_value.get_series.side_effect = Exception("bad request")

        helper = _make_fred_helper(mock_fred_cls)
        with pytest.raises(RuntimeError, match="Invalid FRED API key"):
            helper.fetch_indicator("unemployment")

    @patch("utils.fred_helper.Fred")
    def test_fetch_indicator_sp500_yoy_pct_daily(self, mock_fred_cls):
        """sp500 is daily yoy_pct — verify transform and monthly resampling."""
        n = 500
        idx = pd.date_range(start="2020-01-01", periods=n, freq="D")
        values = np.linspace(3000.0, 4000.0, n)
        raw = pd.Series(values, index=idx)
        mock_fred_cls.return_value.get_series.return_value = raw

        helper = _make_fred_helper(mock_fred_cls)
        result = helper.fetch_indicator("sp500")

        assert isinstance(result, pd.Series)
        assert result.name == "sp500"
        assert len(result) > 0
        assert all(ts.day == 1 for ts in result.index)


# ---------------------------------------------------------------------------
# FredHelper.fetch_all_indicators
# ---------------------------------------------------------------------------

class TestFetchAllIndicators:

    @patch("utils.fred_helper.Fred")
    def test_fetch_all_indicators_skips_failures(self, mock_fred_cls):
        """fetch_all_indicators should silently skip failing series and return only successes."""
        raw_monthly = _make_monthly_series(n=36, start="2018-01-01", value=5.0)
        call_count = {"n": 0}

        def get_series_side_effect(series_id):
            call_count["n"] += 1
            # Fail on the 3rd call
            if call_count["n"] == 3:
                raise Exception("connection error")
            return raw_monthly

        mock_fred_cls.return_value.get_series.side_effect = get_series_side_effect

        helper = _make_fred_helper(mock_fred_cls)
        results = helper.fetch_all_indicators()

        assert isinstance(results, dict)
        # 10 total catalogue entries, 1 failed → 9 successes
        assert len(results) == 9
        for series in results.values():
            assert isinstance(series, pd.Series)

    @patch("utils.fred_helper.Fred")
    def test_fetch_all_indicators_all_fail(self, mock_fred_cls):
        """If every series fails, returns an empty dict without raising."""
        mock_fred_cls.return_value.get_series.side_effect = Exception("network error")

        helper = _make_fred_helper(mock_fred_cls)
        results = helper.fetch_all_indicators()

        assert results == {}

    @patch("utils.fred_helper.Fred")
    def test_fetch_all_indicators_returns_all_on_success(self, mock_fred_cls):
        """When all fetches succeed, returns all 10 catalogue entries."""
        raw = _make_monthly_series(n=36, start="2018-01-01", value=3.0)
        mock_fred_cls.return_value.get_series.return_value = raw

        helper = _make_fred_helper(mock_fred_cls)
        results = helper.fetch_all_indicators()

        assert len(results) == len(FredHelper.INDICATOR_CATALOGUE)
        for name in FredHelper.INDICATOR_CATALOGUE:
            assert name in results


# ---------------------------------------------------------------------------
# FredHelper.align_all_indicators
# ---------------------------------------------------------------------------

class TestAlignAllIndicators:

    @patch("utils.fred_helper.Fred")
    def test_align_basic(self, mock_fred_cls):
        """Basic inner join on month-start dates produces correct columns."""
        helper = _make_fred_helper(mock_fred_cls)

        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        company_df = pd.DataFrame({"ds": dates, "y": np.arange(1, 25, dtype=float)})

        ind1 = pd.Series(np.ones(24) * 3.0, index=dates, name="unemployment")
        ind2 = pd.Series(np.ones(24) * 5.0, index=dates, name="fed_funds")

        result = helper.align_all_indicators(company_df, {"unemployment": ind1, "fed_funds": ind2})

        assert isinstance(result, pd.DataFrame)
        assert "ds" in result.columns
        assert "y" in result.columns
        assert "unemployment" in result.columns
        assert "fed_funds" in result.columns
        assert len(result) == 24
        assert (result["unemployment"] == 3.0).all()
        assert (result["fed_funds"] == 5.0).all()

    @patch("utils.fred_helper.Fred")
    def test_align_date_normalization(self, mock_fred_cls):
        """Company dates not on month-start are normalized to MS before joining."""
        helper = _make_fred_helper(mock_fred_cls)

        # Non-month-start dates (mid-month)
        dates_mid = pd.date_range("2020-01-15", periods=12, freq="ME")
        company_df = pd.DataFrame({"ds": dates_mid, "y": np.ones(12) * 10.0})

        # Indicator indexed on proper month-start
        ms_dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        ind = pd.Series(np.ones(12) * 2.5, index=ms_dates, name="fed_funds")

        result = helper.align_all_indicators(company_df, {"fed_funds": ind})

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 12
        # After normalization, all ds values should be on the 1st of the month
        assert all(ts.day == 1 for ts in result["ds"])

    @patch("utils.fred_helper.Fred")
    def test_align_partial_overlap(self, mock_fred_cls):
        """Inner join retains only rows present in both company data and indicators."""
        helper = _make_fred_helper(mock_fred_cls)

        company_dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        company_df = pd.DataFrame({"ds": company_dates, "y": np.ones(24)})

        # Indicator only covers the last 12 months of the 24
        ind_dates = pd.date_range("2021-01-01", periods=12, freq="MS")
        ind = pd.Series(np.ones(12) * 7.0, index=ind_dates, name="treasury_10y")

        result = helper.align_all_indicators(company_df, {"treasury_10y": ind})

        assert len(result) == 12
        assert (result["treasury_10y"] == 7.0).all()

    @patch("utils.fred_helper.Fred")
    def test_align_result_sorted_by_date(self, mock_fred_cls):
        """Result DataFrame is sorted ascending by ds."""
        helper = _make_fred_helper(mock_fred_cls)

        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        # Shuffle the company data
        shuffled_dates = dates[::-1]
        company_df = pd.DataFrame({"ds": shuffled_dates, "y": np.arange(12, dtype=float)})

        ind = pd.Series(np.ones(12), index=dates, name="unemployment")
        result = helper.align_all_indicators(company_df, {"unemployment": ind})

        assert list(result["ds"]) == sorted(result["ds"])


# ---------------------------------------------------------------------------
# FredHelper.rank_indicators
# ---------------------------------------------------------------------------

def _make_merged_df(n_months: int = 36, n_indicators: int = 3, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic merged_df with ds, y, and indicator columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_months, freq="MS")
    y = 1000.0 + np.cumsum(rng.normal(0, 10, n_months))

    df = pd.DataFrame({"ds": dates, "y": y})
    indicator_names = [f"ind_{i}" for i in range(n_indicators)]
    for name in indicator_names:
        df[name] = rng.normal(0, 1, n_months)

    return df


class TestRankIndicators:

    @patch("utils.fred_helper.Fred")
    def test_rank_basic(self, mock_fred_cls):
        """rank_indicators returns dict with expected keys and correct types."""
        helper = _make_fred_helper(mock_fred_cls)
        merged_df = _make_merged_df(n_months=36, n_indicators=3)

        result = helper.rank_indicators(merged_df, top_n=3)

        assert isinstance(result, dict)
        assert "all_rankings" in result
        assert "selected" in result
        assert "selected_names" in result
        assert "r_squared" in result
        assert "coefficients" in result

        assert isinstance(result["all_rankings"], list)
        assert isinstance(result["selected_names"], list)
        assert isinstance(result["r_squared"], float)
        assert isinstance(result["coefficients"], dict)

    @patch("utils.fred_helper.Fred")
    def test_rank_sorted_by_abs_corr(self, mock_fred_cls):
        """all_rankings list is sorted by |corr| descending."""
        helper = _make_fred_helper(mock_fred_cls)
        merged_df = _make_merged_df(n_months=36, n_indicators=3)

        result = helper.rank_indicators(merged_df, top_n=3)
        rankings = result["all_rankings"]

        abs_corrs = [abs(r["corr"]) for r in rankings]
        assert abs_corrs == sorted(abs_corrs, reverse=True)

    @patch("utils.fred_helper.Fred")
    def test_rank_top_n(self, mock_fred_cls):
        """top_n=2 selects exactly 2 indicators."""
        helper = _make_fred_helper(mock_fred_cls)
        merged_df = _make_merged_df(n_months=36, n_indicators=4)

        result = helper.rank_indicators(merged_df, top_n=2)

        assert len(result["selected"]) == 2
        assert len(result["selected_names"]) == 2

    @patch("utils.fred_helper.Fred")
    def test_rank_r_squared_is_float_between_0_and_1(self, mock_fred_cls):
        """OLS R² should be between 0 and 1."""
        helper = _make_fred_helper(mock_fred_cls)
        merged_df = _make_merged_df(n_months=36, n_indicators=3)

        result = helper.rank_indicators(merged_df, top_n=3)

        assert 0.0 <= result["r_squared"] <= 1.0

    @patch("utils.fred_helper.Fred")
    def test_rank_insufficient_data(self, mock_fred_cls):
        """Indicators with fewer than 12 non-null paired rows are excluded from rankings."""
        helper = _make_fred_helper(mock_fred_cls)

        # 36-row merged_df with one sparse indicator (only 8 real values)
        merged_df = _make_merged_df(n_months=36, n_indicators=2)
        sparse = np.full(36, np.nan)
        sparse[:8] = np.arange(8, dtype=float)  # only 8 non-null
        merged_df["sparse_ind"] = sparse

        result = helper.rank_indicators(merged_df, top_n=3)

        ranked_names = [r["name"] for r in result["all_rankings"]]
        assert "sparse_ind" not in ranked_names

    @patch("utils.fred_helper.Fred")
    def test_rank_selected_names_subset_of_all_rankings(self, mock_fred_cls):
        """selected_names must all appear in all_rankings."""
        helper = _make_fred_helper(mock_fred_cls)
        merged_df = _make_merged_df(n_months=36, n_indicators=5)

        result = helper.rank_indicators(merged_df, top_n=3)

        all_names = {r["name"] for r in result["all_rankings"]}
        for name in result["selected_names"]:
            assert name in all_names

    @patch("utils.fred_helper.Fred")
    def test_rank_coefficients_keys_match_selected(self, mock_fred_cls):
        """coefficients dict keys match selected_names."""
        helper = _make_fred_helper(mock_fred_cls)
        merged_df = _make_merged_df(n_months=36, n_indicators=3)

        result = helper.rank_indicators(merged_df, top_n=3)

        assert set(result["coefficients"].keys()) == set(result["selected_names"])

    @patch("utils.fred_helper.Fred")
    def test_rank_uses_catalogue_label_when_available(self, mock_fred_cls):
        """rank_indicators uses INDICATOR_CATALOGUE label for known indicator names."""
        helper = _make_fred_helper(mock_fred_cls)

        # Build a merged_df using real catalogue names
        n = 36
        dates = pd.date_range("2018-01-01", periods=n, freq="MS")
        rng = np.random.default_rng(0)
        y = 1000.0 + np.cumsum(rng.normal(0, 10, n))
        df = pd.DataFrame({
            "ds": dates,
            "y": y,
            "unemployment": rng.normal(5, 1, n),
            "fed_funds": rng.normal(2, 0.5, n),
        })

        result = helper.rank_indicators(df, top_n=2)

        for entry in result["all_rankings"]:
            if entry["name"] == "unemployment":
                assert entry["label"] == "Unemployment Rate"
            if entry["name"] == "fed_funds":
                assert entry["label"] == "Federal Funds Rate"


# ---------------------------------------------------------------------------
# FredHelper.prepare_future_regressors
# ---------------------------------------------------------------------------

class TestPrepareFutureRegressors:

    @patch("utils.fred_helper.Fred")
    def test_prepare_with_scenario_values(self, mock_fred_cls):
        """Passing scenario_values fills the named column in future_df."""
        helper = _make_fred_helper(mock_fred_cls)

        dates = pd.date_range("2024-01-01", periods=12, freq="MS")
        future_df = pd.DataFrame({"ds": dates})

        result = helper.prepare_future_regressors(future_df, scenario_values={"GDP_growth": 3.0})

        assert "GDP_growth" in result.columns
        assert (result["GDP_growth"] == 3.0).all()

    @patch("utils.fred_helper.Fred")
    def test_prepare_with_legacy_params(self, mock_fred_cls):
        """Legacy gdp_scenario and inf_scenario params fill GDP_growth and inflation columns."""
        helper = _make_fred_helper(mock_fred_cls)

        dates = pd.date_range("2024-01-01", periods=12, freq="MS")
        future_df = pd.DataFrame({"ds": dates})

        result = helper.prepare_future_regressors(
            future_df, gdp_scenario=2.0, inf_scenario=3.0
        )

        assert "GDP_growth" in result.columns
        assert "inflation" in result.columns
        assert (result["GDP_growth"] == 2.0).all()
        assert (result["inflation"] == 3.0).all()

    @patch("utils.fred_helper.Fred")
    def test_prepare_fillna_preserves_existing(self, mock_fred_cls):
        """Existing non-NaN values in a column are preserved; only NaN rows are filled."""
        helper = _make_fred_helper(mock_fred_cls)

        dates = pd.date_range("2024-01-01", periods=6, freq="MS")
        # First 3 rows have real historical values, last 3 are NaN (forecast horizon)
        existing = pd.Series([1.5, 2.0, 2.5, np.nan, np.nan, np.nan], index=range(6))
        future_df = pd.DataFrame({"ds": dates, "unemployment": existing})

        result = helper.prepare_future_regressors(
            future_df, scenario_values={"unemployment": 5.0}
        )

        assert result["unemployment"].iloc[0] == 1.5
        assert result["unemployment"].iloc[1] == 2.0
        assert result["unemployment"].iloc[2] == 2.5
        assert result["unemployment"].iloc[3] == 5.0
        assert result["unemployment"].iloc[4] == 5.0
        assert result["unemployment"].iloc[5] == 5.0

    @patch("utils.fred_helper.Fred")
    def test_prepare_does_not_mutate_input(self, mock_fred_cls):
        """prepare_future_regressors returns a copy; original future_df is unchanged."""
        helper = _make_fred_helper(mock_fred_cls)

        dates = pd.date_range("2024-01-01", periods=6, freq="MS")
        future_df = pd.DataFrame({"ds": dates})
        original_columns = list(future_df.columns)

        helper.prepare_future_regressors(future_df, scenario_values={"GDP_growth": 2.0})

        assert list(future_df.columns) == original_columns

    @patch("utils.fred_helper.Fred")
    def test_prepare_multiple_scenario_values(self, mock_fred_cls):
        """Multiple indicators in scenario_values are all injected."""
        helper = _make_fred_helper(mock_fred_cls)

        dates = pd.date_range("2024-01-01", periods=12, freq="MS")
        future_df = pd.DataFrame({"ds": dates})

        scenarios = {
            "GDP_growth": 3.0,
            "inflation": 2.5,
            "unemployment": 4.0,
        }
        result = helper.prepare_future_regressors(future_df, scenario_values=scenarios)

        for name, val in scenarios.items():
            assert name in result.columns
            assert (result[name] == val).all()

    @patch("utils.fred_helper.Fred")
    def test_prepare_no_scenario_values_returns_unchanged(self, mock_fred_cls):
        """Calling with no scenarios and no legacy params returns df with no new columns."""
        helper = _make_fred_helper(mock_fred_cls)

        dates = pd.date_range("2024-01-01", periods=6, freq="MS")
        future_df = pd.DataFrame({"ds": dates, "y": np.ones(6)})

        result = helper.prepare_future_regressors(future_df)

        assert list(result.columns) == list(future_df.columns)

    @patch("utils.fred_helper.Fred")
    def test_prepare_legacy_only_gdp(self, mock_fred_cls):
        """Only gdp_scenario provided (no inf_scenario) → only GDP_growth column added."""
        helper = _make_fred_helper(mock_fred_cls)

        dates = pd.date_range("2024-01-01", periods=6, freq="MS")
        future_df = pd.DataFrame({"ds": dates})

        result = helper.prepare_future_regressors(future_df, gdp_scenario=1.5)

        assert "GDP_growth" in result.columns
        assert "inflation" not in result.columns
