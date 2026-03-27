"""Unit tests for utils/charts.py chart-building functions."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from utils.charts import (
    _build_forecast_chart,
    _build_component_chart,
    _build_correlation_bar,
    _build_backtest_chart,
    _scatter_chart,
)


class TestBuildForecastChart:
    """Tests for _build_forecast_chart()."""

    @pytest.fixture
    def sample_forecast_data(self):
        """Create sample forecast DataFrame with required columns."""
        dates = pd.date_range("2024-01-01", periods=12, freq="MS")
        return pd.DataFrame({
            "ds": dates,
            "yhat": [100 + i * 5 for i in range(12)],
            "yhat_lower": [95 + i * 5 for i in range(12)],
            "yhat_upper": [105 + i * 5 for i in range(12)],
        })

    @pytest.fixture
    def sample_train_data(self):
        """Create sample historical training DataFrame."""
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        return pd.DataFrame({
            "ds": dates,
            "y": [100 + i * 2 for i in range(12)],
        })

    def test_forecast_chart_returns_figure(self, sample_forecast_data, sample_train_data):
        """Test that the function returns a Plotly Figure."""
        fig = _build_forecast_chart(sample_forecast_data, sample_train_data)
        assert isinstance(fig, go.Figure)

    def test_forecast_chart_has_traces(self, sample_forecast_data, sample_train_data):
        """Test that the figure contains traces."""
        fig = _build_forecast_chart(sample_forecast_data, sample_train_data)
        assert len(fig.data) > 0

    def test_forecast_chart_without_growth_rate(self, sample_forecast_data, sample_train_data):
        """Test forecast chart without growth_rate (should have 3 main traces + shape)."""
        fig = _build_forecast_chart(
            sample_forecast_data,
            sample_train_data,
            growth_rate=None
        )
        # Should have: uncertainty band, forecast line, historical data
        assert len(fig.data) == 3
        assert fig.data[0].name == "80% Confidence Band"
        assert fig.data[1].name == "Macro-Adjusted Forecast"
        assert fig.data[2].name == "Historical Demand"

    def test_forecast_chart_with_growth_rate(self, sample_forecast_data, sample_train_data):
        """Test forecast chart with growth_rate parameter adds trendline trace."""
        fig = _build_forecast_chart(
            sample_forecast_data,
            sample_train_data,
            growth_rate=0.1
        )
        # Should have: uncertainty band, forecast line, historical data, growth trendline
        assert len(fig.data) == 4
        assert "Growth Trendline" in fig.data[3].name
        assert "10.0%/yr" in fig.data[3].name

    def test_forecast_chart_with_negative_growth_rate(self, sample_forecast_data, sample_train_data):
        """Test forecast chart with negative growth_rate."""
        fig = _build_forecast_chart(
            sample_forecast_data,
            sample_train_data,
            growth_rate=-0.05
        )
        assert len(fig.data) == 4
        assert "-5.0%/yr" in fig.data[3].name

    def test_forecast_chart_layout(self, sample_forecast_data, sample_train_data):
        """Test that the figure has proper layout configuration."""
        fig = _build_forecast_chart(sample_forecast_data, sample_train_data)
        assert fig.layout.title.text == "12-Month Macro-Adjusted Demand Forecast"
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Demand"
        assert fig.layout.height == 480


class TestBuildComponentChart:
    """Tests for _build_component_chart()."""

    @pytest.fixture
    def sample_component_data(self):
        """Create sample component forecast data."""
        dates = pd.date_range("2024-01-01", periods=15, freq="MS")
        return pd.DataFrame({
            "ds": dates,
            "trend": [100 + i * 2 for i in range(15)],
            "yearly": [2 + i * 0.5 for i in range(15)],
            "GDP_growth": [1 + i * 0.1 for i in range(15)],
            "inflation": [0.5 + i * 0.05 for i in range(15)],
        })

    def test_component_chart_returns_figure(self, sample_component_data):
        """Test that the function returns a Plotly Figure."""
        fig = _build_component_chart(sample_component_data)
        assert isinstance(fig, go.Figure)

    def test_component_chart_has_traces(self, sample_component_data):
        """Test that the figure contains traces."""
        fig = _build_component_chart(sample_component_data)
        assert len(fig.data) > 0

    def test_component_chart_default_indicators(self, sample_component_data):
        """Test component chart with default indicators (GDP_growth, inflation)."""
        fig = _build_component_chart(sample_component_data, selected_names=None)
        # Should have: trend, yearly, GDP_growth, inflation (4 traces)
        assert len(fig.data) == 4
        names = [trace.name for trace in fig.data]
        assert "Trend" in names
        assert "Yearly" in names
        assert "Gdp Growth" in names
        assert "Inflation" in names

    def test_component_chart_with_custom_indicators(self, sample_component_data):
        """Test component chart with custom selected indicators."""
        fig = _build_component_chart(
            sample_component_data,
            selected_names=["GDP_growth"]
        )
        # Should have: trend, yearly, GDP_growth (3 traces)
        assert len(fig.data) == 3
        names = [trace.name for trace in fig.data]
        assert "Trend" in names
        assert "Yearly" in names
        assert "Gdp Growth" in names

    def test_component_chart_missing_columns_ignored(self, sample_component_data):
        """Test that missing columns are silently ignored."""
        fig = _build_component_chart(
            sample_component_data,
            selected_names=["nonexistent_indicator"]
        )
        # Should have: trend, yearly (only base columns)
        assert len(fig.data) == 2

    def test_component_chart_layout(self, sample_component_data):
        """Test that the figure has proper layout configuration."""
        fig = _build_component_chart(sample_component_data)
        assert fig.layout.title.text == "Forecast Components (Trend + Seasonality + Macro)"
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Component Contribution"
        assert fig.layout.height == 380


class TestBuildCorrelationBar:
    """Tests for _build_correlation_bar()."""

    @pytest.fixture
    def sample_rankings(self):
        """Create sample indicator rankings."""
        return [
            {"name": "GDP_growth", "label": "GDP Growth", "corr": 0.85},
            {"name": "inflation", "label": "Inflation", "corr": -0.42},
            {"name": "unemployment", "label": "Unemployment", "corr": -0.65},
            {"name": "fed_funds", "label": "Fed Funds Rate", "corr": 0.15},
            {"name": "treasury_10yr", "label": "10-Yr Treasury Yield", "corr": 0.38},
        ]

    def test_correlation_bar_returns_figure(self, sample_rankings):
        """Test that the function returns a Plotly Figure."""
        fig = _build_correlation_bar(sample_rankings, ["GDP_growth"])
        assert isinstance(fig, go.Figure)

    def test_correlation_bar_has_traces(self, sample_rankings):
        """Test that the figure contains a bar trace."""
        fig = _build_correlation_bar(sample_rankings, ["GDP_growth"])
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Bar)

    def test_correlation_bar_selected_colors(self, sample_rankings):
        """Test that selected indicators are colored differently from unselected."""
        fig = _build_correlation_bar(sample_rankings, ["GDP_growth", "inflation"])
        colors = fig.data[0].marker.color
        # First item (GDP_growth, corr > 0) should be blue
        assert colors[0] == "#1E88E5"
        # Second item (inflation, corr < 0) should be red
        assert colors[1] == "#E53935"
        # Remaining items should be gray
        assert colors[2] == "#BDBDBD"
        assert colors[3] == "#BDBDBD"
        assert colors[4] == "#BDBDBD"

    def test_correlation_bar_unselected_all_gray(self, sample_rankings):
        """Test that all bars are gray when none are selected."""
        fig = _build_correlation_bar(sample_rankings, [])
        colors = fig.data[0].marker.color
        assert all(c == "#BDBDBD" for c in colors)

    def test_correlation_bar_text_labels(self, sample_rankings):
        """Test that bar text shows correlation values."""
        fig = _build_correlation_bar(sample_rankings, ["GDP_growth"])
        text_labels = fig.data[0].text
        assert len(text_labels) == len(sample_rankings)
        assert "+0.850" in text_labels[0]
        assert "-0.420" in text_labels[1]

    def test_correlation_bar_layout(self, sample_rankings):
        """Test that the figure has proper layout configuration."""
        fig = _build_correlation_bar(sample_rankings, ["GDP_growth"])
        assert "All Evaluated Indicators" in fig.layout.title.text
        assert fig.layout.yaxis.title.text == "Pearson r"
        assert fig.layout.height == 420


class TestBuildBacktestChart:
    """Tests for _build_backtest_chart()."""

    @pytest.fixture
    def sample_backtest_data(self):
        """Create sample backtest results DataFrame."""
        # Hold-out period: last 6 months
        holdout_dates = pd.date_range("2023-07-01", periods=6, freq="MS")
        return pd.DataFrame({
            "ds": holdout_dates,
            "y": [100, 102, 104, 106, 108, 110],
            "yhat_macro": [101, 103, 105, 107, 109, 111],
            "yhat_vanilla": [99, 101, 103, 105, 107, 109],
            "yhat_naive": [100, 101, 102, 103, 104, 105],
        })

    @pytest.fixture
    def sample_full_train_data(self):
        """Create sample full training data (before and after hold-out)."""
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        return pd.DataFrame({
            "ds": dates,
            "y": [90 + i * 1.5 for i in range(12)],
        })

    def test_backtest_chart_returns_figure(self, sample_backtest_data, sample_full_train_data):
        """Test that the function returns a Plotly Figure."""
        fig = _build_backtest_chart(sample_backtest_data, sample_full_train_data)
        assert isinstance(fig, go.Figure)

    def test_backtest_chart_has_traces(self, sample_backtest_data, sample_full_train_data):
        """Test that the figure contains traces."""
        fig = _build_backtest_chart(sample_backtest_data, sample_full_train_data)
        assert len(fig.data) > 0

    def test_backtest_chart_with_naive_baseline(self, sample_backtest_data, sample_full_train_data):
        """Test backtest chart includes naive baseline when present."""
        fig = _build_backtest_chart(sample_backtest_data, sample_full_train_data)
        # Should have: history, actuals, macro-augmented, vanilla, naive
        assert len(fig.data) == 5
        names = [trace.name for trace in fig.data]
        assert "Historical (Training)" in names
        assert "Actual (Hold-Out)" in names
        assert "Macro-Augmented Prophet" in names
        assert "Vanilla Prophet (no macro)" in names
        assert any("Growth-Adjusted Naive" in name for name in names)

    def test_backtest_chart_without_naive_baseline(self, sample_backtest_data, sample_full_train_data):
        """Test backtest chart without naive baseline when not present."""
        results_no_naive = sample_backtest_data.drop(columns=["yhat_naive"])
        fig = _build_backtest_chart(results_no_naive, sample_full_train_data)
        # Should have: history, actuals, macro-augmented, vanilla (4 traces)
        assert len(fig.data) == 4

    def test_backtest_chart_layout(self, sample_backtest_data, sample_full_train_data):
        """Test that the figure has proper layout configuration."""
        fig = _build_backtest_chart(sample_backtest_data, sample_full_train_data)
        assert fig.layout.title.text == "Backtest: Actual vs. Predicted (Last 12 Months Hold-Out)"
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Demand"
        assert fig.layout.height == 480


class TestScatterChart:
    """Tests for _scatter_chart()."""

    @pytest.fixture
    def sample_merged_data(self):
        """Create sample merged data with at least 13 rows for YoY calculation."""
        dates = pd.date_range("2022-01-01", periods=20, freq="MS")
        return pd.DataFrame({
            "ds": dates,
            "y": [100 + i * 1.5 for i in range(20)],
            "GDP_growth": [1.5 + i * 0.1 for i in range(20)],
            "inflation": [2.0 + i * 0.05 for i in range(20)],
        })

    def test_scatter_chart_returns_figure(self, sample_merged_data):
        """Test that the function returns a Plotly Figure."""
        fig = _scatter_chart(sample_merged_data, "GDP_growth", "GDP Growth")
        assert isinstance(fig, go.Figure)

    def test_scatter_chart_has_traces(self, sample_merged_data):
        """Test that the figure contains traces (scatter + trendline)."""
        fig = _scatter_chart(sample_merged_data, "GDP_growth", "GDP Growth")
        assert len(fig.data) > 0

    def test_scatter_chart_with_sufficient_data(self, sample_merged_data):
        """Test scatter chart with sufficient rows (>= 13) produces non-empty chart."""
        fig = _scatter_chart(sample_merged_data, "GDP_growth", "GDP Growth")
        # Should have scatter points + trendline
        assert len(fig.data) >= 2

    def test_scatter_chart_minimum_data_requirement(self):
        """Test scatter chart needs at least 13 rows to have non-empty YoY data."""
        # Create exactly 13 rows
        dates = pd.date_range("2023-01-01", periods=13, freq="MS")
        df = pd.DataFrame({
            "ds": dates,
            "y": [100 + i for i in range(13)],
            "indicator": [1.5 + i * 0.1 for i in range(13)],
        })
        fig = _scatter_chart(df, "indicator", "Indicator")
        # Should have at least scatter points
        assert len(fig.data) > 0

    def test_scatter_chart_with_less_than_13_rows(self):
        """Test scatter chart with < 13 rows produces empty chart (all NaN for YoY)."""
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        df = pd.DataFrame({
            "ds": dates,
            "y": [100 + i for i in range(12)],
            "indicator": [1.5 + i * 0.1 for i in range(12)],
        })
        fig = _scatter_chart(df, "indicator", "Indicator")
        # Should still return a Figure, but with no data points
        assert isinstance(fig, go.Figure)

    def test_scatter_chart_different_x_columns(self, sample_merged_data):
        """Test scatter chart works with different x column."""
        fig_gdp = _scatter_chart(sample_merged_data, "GDP_growth", "GDP Growth")
        fig_inflation = _scatter_chart(sample_merged_data, "inflation", "Inflation")

        assert isinstance(fig_gdp, go.Figure)
        assert isinstance(fig_inflation, go.Figure)
        # Titles should reflect the different columns
        assert "GDP Growth" in fig_gdp.layout.title.text
        assert "Inflation" in fig_inflation.layout.title.text

    def test_scatter_chart_layout(self, sample_merged_data):
        """Test that the figure has proper layout configuration."""
        fig = _scatter_chart(sample_merged_data, "GDP_growth", "GDP Growth")
        assert "GDP Growth" in fig.layout.title.text
        assert "Demand Growth (YoY %)" in fig.layout.yaxis.title.text

    def test_scatter_chart_marker_styling(self, sample_merged_data):
        """Test that scatter points have expected styling."""
        fig = _scatter_chart(sample_merged_data, "GDP_growth", "GDP Growth")
        # Find scatter trace (not trendline)
        scatter_trace = None
        for trace in fig.data:
            if trace.mode == "markers":
                scatter_trace = trace
                break

        assert scatter_trace is not None
        assert scatter_trace.marker.color == "#1E88E5"
        assert scatter_trace.marker.size == 7
        assert scatter_trace.marker.opacity == 0.75
