"""
app.py  –  MacroAgent Forecaster
=================================
A fully open-source, free, educational Streamlit web application that lets
any user (e.g., a law firm or supply-chain team) upload their historical
demand data and run "what-if" forecasts augmented with real GDP growth and
inflation data from the Federal Reserve FRED API.

LEGAL / COMPLIANCE NOTICE
--------------------------
  This application accesses the FRED API for EDUCATIONAL AND PERSONAL USE ONLY
  in accordance with the Federal Reserve Bank of St. Louis Terms of Use.
  See: https://fred.stlouisfed.org/about/terms

  • Users must supply their own free FRED API key.
  • Retrieved data must NOT be used for commercial purposes, AI training,
    or redistribution.
  • This app stores NO data server-side; all processing is ephemeral.
  • This is NOT investment advice.
  • Built as a Rutgers BAIT / Supply Chain Management student project.

Run with:
    streamlit run app.py
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import io
import traceback

# ---------------------------------------------------------------------------
# Third-party imports  (all from requirements.txt)
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from utils.fred_helper import FredHelper

# ===========================================================================
# PAGE CONFIG  (must be the very first Streamlit call)
# ===========================================================================
st.set_page_config(
    page_title="MacroAgent Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# SESSION STATE  – persist data across widget interactions / reruns
# ===========================================================================
_STATE_DEFAULTS = {
    "company_df":          None,   # uploaded & validated demand data (Prophet format)
    "fred_helper":         None,   # authenticated FredHelper instance
    "merged_df":           None,   # company data merged with all FRED indicators
    "corr_stats":          None,   # legacy — kept for compat, no longer primary
    "forecast_df":         None,   # Prophet forecast result
    "model":               None,   # trained Prophet model
    "gdp_scenario":        2.0,    # legacy fallback
    "inf_scenario":        5.0,    # legacy fallback
    "backtest_metrics":    None,   # dict of backtest results and error metrics
    "all_indicators":      None,   # dict {name: pd.Series} from fetch_all_indicators
    "indicator_ranking":   None,   # dict from rank_indicators()
    "selected_indicators": None,   # list of selected indicator names
    "scenario_values":     None,   # dict {name: float} from sidebar sliders
    "growth_rate_basis":   "Lifetime",  # selected growth rate window
    "growth_rate_value":   None,        # computed or custom growth rate (float)
    "trend_flexibility":   0.05,        # Prophet changepoint_prior_scale
    "backtest_horizon":    "1 Year",    # selected backtest horizon
    "backtest_metrics_5y": None,        # 5-year backtest results
}
for _key, _val in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# Cross-validation configuration (used by backtest)
N_FOLDS  = 3
HORIZON  = 12  # hold-out months per fold


# ===========================================================================
# HELPER UTILITIES
# ===========================================================================

def _disclaimer_banner(location: str = "top"):
    """Render the mandatory FRED compliance / educational-use disclaimer."""
    msg = (
        "**FOR EDUCATIONAL AND PERSONAL USE ONLY.**  "
        "This app complies with Federal Reserve FRED Terms of Use "
        "([read them here](https://fred.stlouisfed.org/about/terms)).  "
        "You must supply your own **free** API key.  "
        "Data may **not** be used for commercial purposes, AI training, or redistribution.  "
        "**Not investment advice.**  "
        "Built as a Rutgers BAIT / SCM student project."
    )
    st.error(msg, icon="⚠️")


def _mape(actual, predicted):
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def _mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _compute_growth_rates(df: pd.DataFrame) -> dict:
    """
    Compute median YoY growth rates over different time windows from company data.
    df must have 'ds' (datetime) and 'y' (numeric) columns, sorted by ds.
    Returns dict: {"3-Year": float_or_None, "5-Year": float_or_None, "Lifetime": float}
    """
    ts = df.set_index("ds")["y"]
    yoy = ts.pct_change(periods=12).dropna()

    rates = {}
    # Lifetime
    rates["Lifetime"] = float(yoy.median()) if len(yoy) > 0 else 0.0

    # 3-Year: last 36 months of YoY data
    cutoff_3y = ts.index.max() - pd.DateOffset(years=3)
    yoy_3y = yoy[yoy.index >= cutoff_3y]
    rates["3-Year"] = float(yoy_3y.median()) if len(yoy_3y) >= 12 else None

    # 5-Year: last 60 months of YoY data
    cutoff_5y = ts.index.max() - pd.DateOffset(years=5)
    yoy_5y = yoy[yoy.index >= cutoff_5y]
    rates["5-Year"] = float(yoy_5y.median()) if len(yoy_5y) >= 12 else None

    return rates


def _validate_company_df(df: pd.DataFrame) -> tuple[bool, str, pd.DataFrame | None]:
    """
    Validate the uploaded CSV.

    Returns (ok, error_message, cleaned_df).
    Cleaned df has columns 'ds' (datetime) and 'y' (float).
    """
    # --- find date column ---------------------------------------------------
    date_col = next(
        (c for c in df.columns if "date" in c.lower()), None
    )
    if date_col is None:
        return False, "Could not find a column containing 'date'. Please rename it.", None

    # --- find demand column -------------------------------------------------
    demand_col = next(
        (c for c in df.columns
         if "demand" in c.lower() or "revenue" in c.lower() or "y" == c.lower()), None
    )
    if demand_col is None:
        return False, "Could not find a column containing 'demand' or 'revenue'. Please rename it.", None

    # --- parse & clean ------------------------------------------------------
    try:
        df = df[[date_col, demand_col]].copy()
        df.columns = ["ds", "y"]
        df.loc[:, "ds"] = pd.to_datetime(df["ds"].astype(str))
        df.loc[:, "y"] = pd.to_numeric(df["y"], errors="coerce")
    except Exception as exc:
        return False, f"Error parsing data: {exc}", None

    # --- business rules -----------------------------------------------------
    if len(df) < 12:
        return False, f"Need at least 12 rows; got {len(df)}. Please upload more history.", None

    if df["ds"].isna().any():
        return False, "Some date values could not be parsed. Please use ISO format (YYYY-MM-DD).", None

    if df["y"].isna().any() or (df["y"] <= 0).any():
        return False, "All demand values must be positive numbers (> 0) with no blanks.", None

    df = df.sort_values("ds").reset_index(drop=True)
    return True, "", df


def _build_forecast_chart(forecast: pd.DataFrame, train_df: pd.DataFrame, growth_rate: float = None) -> go.Figure:
    """Return an interactive Plotly figure with historical data, forecast, and uncertainty bands."""
    fig = go.Figure()

    # Uncertainty band (yhat_lower / yhat_upper)
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(30, 136, 229, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="80% Confidence Band",
    ))

    # Prophet forecast line
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name="Macro-Adjusted Forecast",
        line=dict(color="#1E88E5", width=2.5),
    ))

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=train_df["ds"],
        y=train_df["y"],
        mode="markers+lines",
        name="Historical Demand",
        line=dict(color="#43A047", width=1.5, dash="dot"),
        marker=dict(size=5),
    ))

    # Growth trendline reference
    if growth_rate is not None and len(train_df) > 0:
        last_actual = train_df["y"].iloc[-1]
        last_date = train_df["ds"].max()
        trend_dates = pd.date_range(last_date, periods=13, freq="MS")[1:]  # 12 future months
        trend_values = [last_actual * (1 + growth_rate) ** (i / 12) for i in range(1, 13)]
        fig.add_trace(go.Scatter(
            x=trend_dates,
            y=trend_values,
            mode="lines",
            name=f"Growth Trendline ({growth_rate * 100:+.1f}%/yr)",
            line=dict(color="#9E9E9E", width=1.5, dash="dot"),
        ))

    # Vertical line at today / forecast start
    cutoff = train_df["ds"].max()
    cutoff_str = cutoff.isoformat()
    fig.add_shape(
        type="line",
        x0=cutoff_str, x1=cutoff_str,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(width=1.5, dash="dash", color="#E53935"),
    )
    fig.add_annotation(
        x=cutoff_str, y=1,
        xref="x", yref="paper",
        text="Forecast Start",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(color="#E53935"),
    )

    fig.update_layout(
        title="12-Month Macro-Adjusted Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Demand",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=480,
    )
    return fig


def _build_component_chart(forecast: pd.DataFrame, selected_names: list[str] = None) -> go.Figure:
    """Build a stacked area chart showing Prophet's decomposed components."""
    base_cols = ["trend", "yearly"]
    regressor_cols = selected_names if selected_names else ["GDP_growth", "inflation"]
    cols_wanted = [c for c in base_cols + regressor_cols if c in forecast.columns]

    fig = go.Figure()
    colors = ["#1E88E5", "#43A047", "#FB8C00", "#E53935", "#8E24AA"]
    for col, color in zip(cols_wanted, colors):
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast[col],
            mode="lines",
            stackgroup="one",
            name=col.replace("_", " ").title(),
            line=dict(color=color),
        ))

    fig.update_layout(
        title="Forecast Components (Trend + Seasonality + Macro)",
        xaxis_title="Date",
        yaxis_title="Component Contribution",
        template="plotly_white",
        height=380,
    )
    return fig


def _build_correlation_bar(all_rankings: list, selected_names: list) -> go.Figure:
    """Bar chart of all evaluated indicator correlations with demand growth.
    Selected indicators are highlighted in blue; unselected in gray."""
    selected_set = set(selected_names)
    labels = [f"Demand Growth ↔ {r['label']}" for r in all_rankings]
    values = [r["corr"] for r in all_rankings]
    colors = []
    for r in all_rankings:
        if r["name"] in selected_set:
            colors.append("#1E88E5" if r["corr"] >= 0 else "#E53935")
        else:
            colors.append("#BDBDBD")

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_width=1, line_color="#888")
    fig.update_layout(
        title="All Evaluated Indicators — Correlation with Demand Growth (YoY, Detrended)<br>"
              "<sup>Blue/red = agent-selected  ·  Gray = evaluated but not selected</sup>",
        yaxis_title="Pearson r",
        yaxis=dict(range=[-1.1, 1.1], tickformat=".2f"),
        xaxis_title=None,
        xaxis_tickangle=-30,
        template="plotly_white",
        height=420,
        showlegend=False,
        margin=dict(b=120),
    )
    return fig


def _build_backtest_chart(results_df: pd.DataFrame, train_df: pd.DataFrame) -> go.Figure:
    """Chart overlaying actuals, macro-augmented forecast, and vanilla forecast for the hold-out period."""
    fig = go.Figure()

    # Training history (before hold-out)
    cutoff = results_df["ds"].min()
    history = train_df[train_df["ds"] < cutoff]
    fig.add_trace(go.Scatter(
        x=history["ds"], y=history["y"],
        mode="lines",
        name="Historical (Training)",
        line=dict(color="#43A047", width=1.5),
    ))

    # Actuals in hold-out period
    fig.add_trace(go.Scatter(
        x=results_df["ds"], y=results_df["y"],
        mode="markers+lines",
        name="Actual (Hold-Out)",
        line=dict(color="#43A047", width=2, dash="dot"),
        marker=dict(size=7),
    ))

    # Macro-augmented Prophet forecast
    fig.add_trace(go.Scatter(
        x=results_df["ds"], y=results_df["yhat_macro"],
        mode="lines",
        name="Macro-Augmented Prophet",
        line=dict(color="#1E88E5", width=2.5),
    ))

    # Vanilla Prophet forecast
    fig.add_trace(go.Scatter(
        x=results_df["ds"], y=results_df["yhat_vanilla"],
        mode="lines",
        name="Vanilla Prophet (no macro)",
        line=dict(color="#FB8C00", width=2, dash="dash"),
    ))

    # Naive baseline
    if "yhat_naive" in results_df.columns:
        fig.add_trace(go.Scatter(
            x=results_df["ds"], y=results_df["yhat_naive"],
            mode="lines",
            name="Growth-Adjusted Naive (last year × trend)",
            line=dict(color="#9E9E9E", width=1.5, dash="dot"),
        ))

    # Vertical line at hold-out start
    cutoff_str = cutoff.isoformat()
    fig.add_shape(
        type="line",
        x0=cutoff_str, x1=cutoff_str,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(width=1.5, dash="dash", color="#E53935"),
    )
    fig.add_annotation(
        x=cutoff_str, y=1,
        xref="x", yref="paper",
        text="Hold-Out Start",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(color="#E53935"),
    )

    fig.update_layout(
        title="Backtest: Actual vs. Predicted (Last 12 Months Hold-Out)",
        xaxis_title="Date",
        yaxis_title="Demand",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=480,
    )
    return fig


def _scatter_chart(merged_df: pd.DataFrame, x_col: str, x_label: str) -> go.Figure:
    """Scatter of YoY demand growth rate vs. a macro indicator with an OLS trend line.
    Uses detrended demand (YoY % change) so startup-style growth doesn't produce
    a misleading vertical blob of dots at a single GDP/inflation value."""
    df = merged_df.copy()
    df["demand_yoy"] = df["y"].pct_change(periods=12) * 100
    df = df.dropna(subset=["demand_yoy"])
    fig = px.scatter(
        df, x=x_col, y="demand_yoy",
        trendline="ols",
        labels={x_col: x_label, "demand_yoy": "Demand Growth (YoY %)"},
        template="plotly_white",
        title=f"Demand Growth Rate vs. {x_label}",
    )
    fig.update_traces(marker=dict(size=7, color="#1E88E5", opacity=0.75),
                      selector=dict(mode="markers"))
    return fig


def _run_backtest(merged, helper, horizon, n_folds, trend_flex, growth_rate_override):
    """Run n_folds expanding-window CV with given horizon.

    Returns a tuple: (dict_with_results, None) on success,
    or (None, "error message string") if insufficient data or all folds fail.

    The result dict has keys: results_df, train_df, fold_metrics, fold_indicators,
    and averaged macro/vanilla/naive MAPE/MAE/RMSE, n_test_months, n_folds.
    """
    min_train_months = 24

    max_date      = merged["ds"].max()
    earliest_date = merged["ds"].min()
    total_months  = int(round((max_date - earliest_date).days / 30.44))

    # Compute how many folds are possible
    max_possible_folds = max(0, (total_months - min_train_months) // horizon)
    actual_folds = min(n_folds, max_possible_folds)

    if actual_folds == 0:
        return (
            None,
            f"Not enough data for backtest with a {horizon}-month horizon. "
            f"Need at least {min_train_months + horizon} months; dataset has ~{total_months} months.",
        )

    cutoff_dates = sorted([
        max_date - pd.DateOffset(months=horizon * (i + 1))
        for i in range(actual_folds)
    ])  # earliest first

    # Drop folds that don't leave at least min_train_months of training data
    cutoff_dates = [
        c for c in cutoff_dates
        if (c - earliest_date).days > min_train_months * 30
    ]

    if len(cutoff_dates) == 0:
        return (
            None,
            f"No valid fold cutoffs found for {horizon}-month horizon. "
            "Upload more historical data.",
        )

    all_fold_results = []
    fold_metrics     = []
    fold_indicators  = []
    train_split      = None  # will hold last fold's train split for chart

    for fold_idx, cutoff_date in enumerate(cutoff_dates):
        # Re-select indicators using ONLY pre-cutoff data (no data leakage)
        tmp_train    = merged[merged["ds"] <= cutoff_date]
        bt_ranking   = helper.rank_indicators(tmp_train, top_n=3)
        bt_sel_names = bt_ranking["selected_names"]

        if not bt_sel_names:
            continue  # skip fold if no indicators found

        # Build fold DataFrames
        full_df = merged[["ds", "y"] + bt_sel_names].copy()
        train_split = full_df[full_df["ds"] <= cutoff_date].copy()
        test_split  = full_df[
            (full_df["ds"] > cutoff_date) &
            (full_df["ds"] <= cutoff_date + pd.DateOffset(months=horizon))
        ].copy()

        if len(test_split) < 2:
            continue

        # --- Macro-augmented Prophet -----------------------------------------
        macro_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
            changepoint_prior_scale=trend_flex,
        )
        for name in bt_sel_names:
            macro_model.add_regressor(name)
        macro_model.fit(train_split)

        future_macro = macro_model.make_future_dataframe(
            periods=len(test_split), freq="MS"
        )
        future_macro = future_macro.merge(
            full_df[["ds"] + bt_sel_names], on="ds", how="left"
        )
        for name in bt_sel_names:
            future_macro[name] = future_macro[name].fillna(
                train_split[name].mean()
            )
        forecast_macro = macro_model.predict(future_macro)

        # --- Vanilla Prophet (no regressors) -----------------------------------
        vanilla_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
            changepoint_prior_scale=trend_flex,
        )
        vanilla_model.fit(train_split[["ds", "y"]])
        future_vanilla = vanilla_model.make_future_dataframe(
            periods=len(test_split), freq="MS"
        )
        forecast_vanilla = vanilla_model.predict(future_vanilla)

        # --- Naive baseline (seasonal naive, growth-adjusted) ------------------
        if growth_rate_override is not None:
            g = growth_rate_override
        else:
            yoy_growth = (
                train_split.set_index("ds")["y"]
                .pct_change(periods=12)
                .dropna()
            )
            g = yoy_growth.median()

        if horizon > 12:
            # Multi-year: tile the last 12 months of training and apply
            # compound growth scaled by how many years ahead each point is.
            seasonal_pattern = train_split.tail(12)["y"].values
            n_repeats = int(np.ceil(len(test_split) / 12))
            tiled = np.tile(seasonal_pattern, n_repeats)[:len(test_split)]
            years_ahead = np.ceil(np.arange(1, len(test_split) + 1) / 12)
            naive_vals = tiled * (1 + g) ** years_ahead
        else:
            # 1-year: simple last-12-months × (1 + g)
            naive_source = train_split.tail(horizon)[["ds", "y"]].copy()
            naive_vals = naive_source["y"].values[:len(test_split)] * (1 + g)

        naive_preds = test_split[["ds"]].copy()
        naive_preds["yhat_naive"] = naive_vals

        # --- Align predictions to hold-out dates -------------------------------
        macro_preds = pd.merge_asof(
            test_split[["ds"]].sort_values("ds"),
            forecast_macro[["ds", "yhat"]].sort_values("ds"),
            on="ds", direction="nearest",
            tolerance=pd.Timedelta(days=15),
        ).rename(columns={"yhat": "yhat_macro"})

        vanilla_preds = pd.merge_asof(
            test_split[["ds"]].sort_values("ds"),
            forecast_vanilla[["ds", "yhat"]].sort_values("ds"),
            on="ds", direction="nearest",
            tolerance=pd.Timedelta(days=15),
        ).rename(columns={"yhat": "yhat_vanilla"})

        results = (
            test_split[["ds", "y"]]
            .merge(macro_preds,   on="ds")
            .merge(vanilla_preds, on="ds")
            .merge(naive_preds,   on="ds")
        )

        if len(results) < 2:
            continue

        a   = results["y"].values
        mp  = results["yhat_macro"].values
        vp  = results["yhat_vanilla"].values
        np_ = results["yhat_naive"].values

        all_fold_results.append(results)
        fold_metrics.append({
            "fold":          fold_idx + 1,
            "cutoff":        cutoff_date.strftime("%b %Y"),
            "n_months":      len(results),
            "macro_mape":    _mape(a, mp),
            "macro_mae":     _mae(a, mp),
            "macro_rmse":    _rmse(a, mp),
            "vanilla_mape":  _mape(a, vp),
            "vanilla_mae":   _mae(a, vp),
            "vanilla_rmse":  _rmse(a, vp),
            "naive_mape":    _mape(a, np_),
            "naive_mae":     _mae(a, np_),
            "naive_rmse":    _rmse(a, np_),
        })
        fold_indicators.append(bt_sel_names)

    if not fold_metrics:
        return (
            None,
            "All folds failed — could not align predictions to hold-out dates. "
            "Try uploading data with dates on the 1st of each month.",
        )

    combined_results = pd.concat(all_fold_results, ignore_index=True)

    avg_metrics = {
        "macro_mape":    np.mean([f["macro_mape"]   for f in fold_metrics]),
        "macro_mae":     np.mean([f["macro_mae"]    for f in fold_metrics]),
        "macro_rmse":    np.mean([f["macro_rmse"]   for f in fold_metrics]),
        "vanilla_mape":  np.mean([f["vanilla_mape"] for f in fold_metrics]),
        "vanilla_mae":   np.mean([f["vanilla_mae"]  for f in fold_metrics]),
        "vanilla_rmse":  np.mean([f["vanilla_rmse"] for f in fold_metrics]),
        "naive_mape":    np.mean([f["naive_mape"]   for f in fold_metrics]),
        "naive_mae":     np.mean([f["naive_mae"]    for f in fold_metrics]),
        "naive_rmse":    np.mean([f["naive_rmse"]   for f in fold_metrics]),
        "n_test_months": sum(f["n_months"] for f in fold_metrics),
        "n_folds":       len(fold_metrics),
    }

    bt_result = {
        "results_df":      combined_results,
        "train_df":        train_split,
        "fold_metrics":    fold_metrics,
        "fold_indicators": fold_indicators,
        **avg_metrics,
    }
    return (bt_result, None)


def _display_backtest_results(bt, label="1-Year"):
    """Render backtest metrics, charts, and insights for a completed backtest."""
    results_df  = bt["results_df"]
    train_df_bt = bt["train_df"]

    # Rank models by MAPE (lower is better)
    model_mapes = [
        ("Macro-Augmented Prophet", bt["macro_mape"]),
        ("Vanilla Prophet",        bt["vanilla_mape"]),
        ("Growth-Adjusted Naive",   bt["naive_mape"]),
    ]
    model_mapes.sort(key=lambda x: x[1])
    best_name, best_mape   = model_mapes[0]
    worst_name, worst_mape = model_mapes[-1]

    st.success(
        f"✅ {label} — {bt['n_folds']}-fold cross-validation complete over "
        f"**{bt['n_test_months']} total hold-out months**.  "
        f"Best model: **{best_name}** ({best_mape:.1f}% MAPE).  "
        f"Worst: {worst_name} ({worst_mape:.1f}% MAPE)."
    )

    # --- Error metrics table -------------------------------------------------
    st.subheader(f"{label} Accuracy Metrics: Three-Way Comparison")
    st.caption(
        f"Averaged across {bt['n_folds']} expanding-window folds "
        f"({bt['n_test_months']} total hold-out months). "
        "Each fold re-selects indicators using only pre-cutoff data (no data leakage).  "
        "The **Growth-Adjusted Naive** applies the training data's median YoY growth rate to last year's actuals — "
        "a much harder baseline to beat for high-growth companies. If a model can't beat this, it isn't adding value beyond trend."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "MAPE — Macro Prophet",
        f"{bt['macro_mape']:.1f}%",
        help="Mean Absolute Percentage Error. Lower is better.",
    )
    m2.metric(
        "MAPE — Vanilla Prophet",
        f"{bt['vanilla_mape']:.1f}%",
        help="Prophet without GDP/inflation regressors.",
    )
    m3.metric(
        "MAPE — Growth-Adj. Naive",
        f"{bt['naive_mape']:.1f}%",
        help="Seasonal naive: repeats the last 12 months of training data. The floor any real model should beat.",
    )

    # Full comparison table
    metrics_table = pd.DataFrame({
        "Metric": ["MAPE (%)", "MAE", "RMSE"],
        "Macro-Augmented Prophet": [
            f"{bt['macro_mape']:.2f}%",
            f"{bt['macro_mae']:,.0f}",
            f"{bt['macro_rmse']:,.0f}",
        ],
        "Vanilla Prophet": [
            f"{bt['vanilla_mape']:.2f}%",
            f"{bt['vanilla_mae']:,.0f}",
            f"{bt['vanilla_rmse']:,.0f}",
        ],
        "Growth-Adjusted Naive": [
            f"{bt['naive_mape']:.2f}%",
            f"{bt['naive_mae']:,.0f}",
            f"{bt['naive_rmse']:,.0f}",
        ],
    })
    st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    # --- Interpretation ------------------------------------------------------
    macro_vs_naive   = bt["naive_mape"] - bt["macro_mape"]
    vanilla_vs_naive = bt["naive_mape"] - bt["vanilla_mape"]
    macro_vs_vanilla = bt["vanilla_mape"] - bt["macro_mape"]

    insights = []
    if macro_vs_naive > 0:
        insights.append(
            f"Macro Prophet beats the growth-adjusted naive by **{macro_vs_naive:.1f} pp** MAPE — "
            f"Prophet is adding real predictive value beyond simple trend extrapolation."
        )
    else:
        insights.append(
            f"Macro Prophet underperforms the growth-adjusted naive by **{abs(macro_vs_naive):.1f} pp** MAPE — "
            f"the model is not beating a simple 'last year × growth rate' forecast. "
            f"The data may be too short, too noisy, or the trend may dominate any macro signal."
        )

    _sel_labels = ", ".join(
        FredHelper.INDICATOR_CATALOGUE[n][1]
        for n in (st.session_state.get("selected_indicators") or [])
    ) or "the selected macro indicators"
    if macro_vs_vanilla > 0:
        insights.append(
            f"Adding **{_sel_labels}** improved accuracy by **{macro_vs_vanilla:.1f} pp** MAPE "
            f"over vanilla Prophet — the agent-selected indicators carry real signal for this dataset."
        )
    elif macro_vs_vanilla < -0.5:
        insights.append(
            f"The selected macro indicators did **not** improve accuracy (vanilla was "
            f"{abs(macro_vs_vanilla):.1f} pp better) — "
            f"this company's demand may be driven more by company-specific factors than macro conditions."
        )
    else:
        insights.append(
            "The agent-selected macro indicators made negligible difference vs. vanilla Prophet for this dataset."
        )

    st.info("**Agent Insight:**  \n" + "  \n".join(insights))

    # --- Per-Fold Breakdown --------------------------------------------------
    with st.expander(f"Per-Fold Breakdown ({bt['n_folds']} folds)"):
        fold_table = pd.DataFrame(bt["fold_metrics"])
        fold_table = fold_table.rename(columns={
            "fold": "Fold", "cutoff": "Cutoff", "n_months": "Months",
            "macro_mape": "Macro MAPE%", "vanilla_mape": "Vanilla MAPE%",
            "naive_mape": "Growth-Adj Naive MAPE%",
        })
        display_cols = ["Fold", "Cutoff", "Months", "Macro MAPE%", "Vanilla MAPE%", "Naive MAPE%"]
        st.dataframe(
            fold_table[[c for c in display_cols if c in fold_table.columns]],
            use_container_width=True, hide_index=True,
        )
        st.caption("Indicators selected per fold (pre-cutoff data only — no leakage):")
        for i, (fm, fi) in enumerate(zip(bt["fold_metrics"], bt["fold_indicators"])):
            labels = [FredHelper.INDICATOR_CATALOGUE[n][1] for n in fi]
            st.caption(f"Fold {fm['fold']} (cutoff {fm['cutoff']}): {', '.join(labels)}")

    # --- Backtest chart ------------------------------------------------------
    st.subheader(f"{label} Forecast vs. Actuals (Hold-Out Period)")
    st.plotly_chart(
        _build_backtest_chart(results_df, train_df_bt),
        use_container_width=True,
    )

    # --- Month-by-month breakdown --------------------------------------------
    with st.expander("Month-by-Month Breakdown"):
        breakdown = results_df.copy()
        breakdown["Error (Macro)"] = (breakdown["yhat_macro"] - breakdown["y"]).map("{:+,.0f}".format)
        breakdown["Error (Vanilla)"] = (breakdown["yhat_vanilla"] - breakdown["y"]).map("{:+,.0f}".format)
        breakdown["Error (Growth-Adj Naive)"] = (breakdown["yhat_naive"] - breakdown["y"]).map("{:+,.0f}".format)
        breakdown["APE (Macro)"] = (
            ((breakdown["yhat_macro"] - breakdown["y"]) / breakdown["y"]).abs() * 100
        ).map("{:.1f}%".format)
        breakdown["APE (Vanilla)"] = (
            ((breakdown["yhat_vanilla"] - breakdown["y"]) / breakdown["y"]).abs() * 100
        ).map("{:.1f}%".format)
        breakdown["APE (Growth-Adj Naive)"] = (
            ((breakdown["yhat_naive"] - breakdown["y"]) / breakdown["y"]).abs() * 100
        ).map("{:.1f}%".format)
        breakdown["ds"] = breakdown["ds"].dt.strftime("%b %Y")
        st.dataframe(
            breakdown.rename(columns={
                "ds": "Month", "y": "Actual",
                "yhat_macro": "Macro Forecast",
                "yhat_vanilla": "Vanilla Forecast",
                "yhat_naive": "Growth-Adj Naive",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ===========================================================================
# TOP DISCLAIMER BANNER
# ===========================================================================

_disclaimer_banner("top")

# ===========================================================================
# TITLE & INTRO
# ===========================================================================

st.title("📈 MacroAgent Forecaster")
st.markdown(
    """
    **An AI Economic Agent for Macro-Informed Demand Forecasting**

    Upload your historical demand data, connect to the Federal Reserve's free FRED API,
    and run *what-if* scenarios to see how GDP growth and inflation may shape
    your future demand — all without spending a dollar.

    > *Zero cost · Open source · Powered by Prophet + FRED · Built for education*
    """
)

# ===========================================================================
# SIDEBAR  – all user inputs
# ===========================================================================

with st.sidebar:
    st.header("⚙️ Agent Controls")
    st.caption(
        "Work through each step in order. "
        "Your data is processed entirely in-browser — nothing is stored on any server."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Step 1: Upload data
    # -----------------------------------------------------------------------
    st.subheader("Step 1 · Upload Demand Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV with columns: **date**, **demand**",
        type=["csv"],
        help="Need a sample? Download data/sample_demand.csv from the GitHub repo.",
    )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        ok, err_msg, clean_df = _validate_company_df(raw_df)
        if ok:
            st.session_state["company_df"] = clean_df
            st.success(f"✅ {len(clean_df)} rows loaded successfully.")
        else:
            st.error(f"Validation error: {err_msg}")
            st.session_state["company_df"] = None

    # -----------------------------------------------------------------------
    # Step 2: FRED API key
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Step 2 · Connect to FRED")
    st.markdown(
        "[Get your free API key here →](https://fredaccount.stlouisfed.org/apikeys)"
    )
    fred_key = st.text_input(
        "FRED API Key",
        type="password",
        placeholder="Paste your free FRED API key",
        help="Your key stays in your browser session only and is never stored.",
    )

    connect_btn = st.button(
        "🔌 Connect to Federal Reserve",
        disabled=(not fred_key),
        use_container_width=True,
    )

    if connect_btn and fred_key:
        with st.spinner("Contacting Federal Reserve FRED API …"):
            try:
                helper = FredHelper(api_key=fred_key)
                # Quick probe: fetch the last value of GDP series to validate key
                _ = helper.get_gdp()
                st.session_state["fred_helper"] = helper
                st.success(
                    "✅ Agent has researched 10+ years of official "
                    "GDP and inflation data from the Federal Reserve."
                )
            except RuntimeError as exc:
                st.error(str(exc))
                st.session_state["fred_helper"] = None
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                st.session_state["fred_helper"] = None

    if st.session_state["fred_helper"] is not None:
        st.info("🏦 FRED connection active.", icon="✅")

    # -----------------------------------------------------------------------
    # Step 3: What-If Sliders (dynamic — populated after indicator selection)
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Step 3 · What-If Scenario")

    selected = st.session_state.get("selected_indicators")

    if not selected:
        st.caption(
            "Run **Research Historical Correlations** first. "
            "The agent will then show scenario sliders for the macro indicators "
            "most relevant to your data."
        )
        # Keep legacy defaults so forecast section doesn't crash before research
        st.session_state["scenario_values"] = {
            "GDP_growth": st.session_state["gdp_scenario"],
            "inflation":  st.session_state["inf_scenario"],
        }
    else:
        scenario_vals = st.session_state.get("scenario_values") or {}
        new_vals = {}
        for name in selected:
            cfg = FredHelper.SCENARIO_DEFAULTS.get(name)
            if cfg is None:
                continue
            lo, hi, default, step, unit, help_txt = cfg
            # Use previously set value if available, else default
            current = scenario_vals.get(name, default)
            # Clamp to valid range in case data changed
            current = max(lo, min(hi, current))
            label = FredHelper.INDICATOR_CATALOGUE[name][1]
            new_vals[name] = st.slider(
                f"{label} ({unit})",
                min_value=lo, max_value=hi,
                value=float(current),
                step=step,
                help=help_txt,
            )
        st.session_state["scenario_values"] = new_vals

        preview = "  ·  ".join(
            f"**{FredHelper.INDICATOR_CATALOGUE[n][1].split('(')[0].strip()}** = {v:+.1f}"
            for n, v in new_vals.items()
        )
        st.caption(f"**Scenario:** {preview}")

    st.divider()
    st.subheader("Step 4 · Growth Rate Basis")
    st.caption(
        "Choose how to compute the baseline growth rate used by the naive benchmark "
        "and shown as a reference trendline on the forecast chart."
    )

    if st.session_state["company_df"] is None:
        st.caption("Upload data first to see growth rate options.")
    else:
        growth_rates = _compute_growth_rates(st.session_state["company_df"])

        # Build radio options dynamically
        radio_options = []
        rate_labels = {}
        for window in ("3-Year", "5-Year", "Lifetime"):
            rate = growth_rates.get(window)
            if rate is not None:
                label = f"{window} Average ({rate * 100:+.1f}%/yr)"
                radio_options.append(label)
                rate_labels[label] = rate
        radio_options.append("Custom")

        # Ensure current basis maps to a valid option label
        current_basis = st.session_state.get("growth_rate_basis", "Lifetime")
        # Find the matching label for the current basis
        matching_label = next(
            (lbl for lbl in radio_options if lbl.startswith(current_basis)), radio_options[-1]
        )

        selected_label = st.radio(
            "Growth window",
            options=radio_options,
            index=radio_options.index(matching_label),
            key="growth_rate_basis_label",
        )

        if selected_label == "Custom":
            custom_pct = st.number_input(
                "Custom annual growth rate (%)",
                min_value=-50.0,
                max_value=200.0,
                value=20.0,
                step=1.0,
            )
            g_val = custom_pct / 100.0
            st.session_state["growth_rate_basis"] = "Custom"
        else:
            g_val = rate_labels[selected_label]
            # Store the window name (e.g. "3-Year", "5-Year", "Lifetime")
            st.session_state["growth_rate_basis"] = selected_label.split(" Average")[0]

        st.session_state["growth_rate_value"] = g_val
        st.metric("Selected Growth Rate", f"{g_val * 100:+.1f}%/yr")

    st.divider()
    st.subheader("Step 5 · Trend Flexibility")
    st.caption(
        "Controls how aggressively Prophet follows rapid changes in your demand trend. "
        "High-growth startups should use Moderate or Aggressive."
    )

    flex_options = {
        "Conservative (0.05)": 0.05,
        "Moderate (0.15)": 0.15,
        "Aggressive (0.35)": 0.35,
        "Very Aggressive (0.5)": 0.50,
        "Custom": None,
    }

    flex_choice = st.radio(
        "Trend flexibility",
        options=list(flex_options.keys()),
        index=0,
        help="Higher values let Prophet track rapid growth or decline more closely. "
             "Lower values produce smoother, more conservative trend estimates.",
    )

    if flex_choice == "Custom":
        custom_cps = st.slider(
            "Custom changepoint_prior_scale",
            min_value=0.001, max_value=1.0, value=0.15, step=0.01,
            help="Prophet default is 0.05. Values above 0.3 follow the data more aggressively.",
        )
        st.session_state["trend_flexibility"] = custom_cps
    else:
        st.session_state["trend_flexibility"] = flex_options[flex_choice]

    st.metric(
        "changepoint_prior_scale",
        f"{st.session_state['trend_flexibility']:.3f}",
    )

    st.divider()
    st.caption(
        "⚖️ **Compliance:** FRED data is used for educational purposes only. "
        "[Terms of Use](https://fred.stlouisfed.org/about/terms)"
    )


# ===========================================================================
# MAIN CONTENT AREA
# ===========================================================================

# ---------------------------------------------------------------------------
# Feature A: Data Preview
# ---------------------------------------------------------------------------

st.header("A · Company Demand Data")

if st.session_state["company_df"] is None:
    st.info(
        "Upload your demand CSV in the sidebar to get started. "
        "No file yet? Download `data/sample_demand.csv` from the repo.",
        icon="📂",
    )
else:
    df = st.session_state["company_df"]

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.rename(columns={"ds": "Date", "y": "Demand"}), height=260)

    with col2:
        st.subheader("Summary Statistics")
        stats = df["y"].describe()
        trend_pct = (df["y"].iloc[-1] / df["y"].iloc[0] - 1) * 100

        st.metric("Mean Demand",  f"{stats['mean']:,.1f}")
        st.metric("Min Demand",   f"{stats['min']:,.1f}")
        st.metric("Max Demand",   f"{stats['max']:,.1f}")
        st.metric("Overall Trend",  f"{trend_pct:+.1f}%",
                  help="% change from first to last data point")
        st.metric("Data Points",  f"{len(df)}")

    # Quick time-series chart of raw demand
    fig_raw = px.line(
        df, x="ds", y="y",
        labels={"ds": "Date", "y": "Demand"},
        title="Historical Demand Time Series",
        template="plotly_white",
    )
    fig_raw.update_traces(line=dict(color="#1E88E5"))
    st.plotly_chart(fig_raw, use_container_width=True)

# ---------------------------------------------------------------------------
# Feature B & C: FRED connection + Historical Correlation Research
# ---------------------------------------------------------------------------

st.header("B + C · Macro Research & Historical Correlations")

research_btn = st.button(
    "🔬 Research Historical Correlations",
    disabled=(
        st.session_state["company_df"] is None
        or st.session_state["fred_helper"] is None
    ),
    use_container_width=False,
)

if research_btn:
    with st.spinner(
        "Agent is scanning 10 Federal Reserve indicators to find which macro "
        "forces matter most for your data — this takes ~10 seconds …"
    ):
        try:
            helper: FredHelper = st.session_state["fred_helper"]

            # Step 1: Fetch all candidate indicators
            indicators = helper.fetch_all_indicators()
            st.session_state["all_indicators"] = indicators

            # Step 2: Align all indicators with company data
            merged = helper.align_all_indicators(
                st.session_state["company_df"], indicators
            )
            st.session_state["merged_df"] = merged

            # Step 3: Rank and select top 3 indicators
            ranking = helper.rank_indicators(merged, top_n=3)
            st.session_state["indicator_ranking"]   = ranking
            st.session_state["selected_indicators"] = ranking["selected_names"]

            # Reset downstream state so stale results don't persist
            st.session_state["forecast_df"]      = None
            st.session_state["backtest_metrics"] = None
            st.session_state["scenario_values"]  = None

        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error during macro research: {exc}")
            with st.expander("Technical details (for debugging)"):
                st.code(traceback.format_exc())

if st.session_state["indicator_ranking"] is not None:
    ranking = st.session_state["indicator_ranking"]
    merged  = st.session_state["merged_df"]
    n_fetched = len(st.session_state.get("all_indicators") or {})
    n_ranked  = len(ranking["all_rankings"])

    st.success(
        f"✅ Agent evaluated **{n_fetched} Federal Reserve indicators** and found "
        f"**{n_ranked}** with sufficient data. "
        f"The top **{len(ranking['selected'])}** were selected as macro regressors."
    )

    # --- Agent Selected Indicators -----------------------------------------
    st.subheader("Agent-Selected Indicators")
    st.markdown(
        f"After evaluating {n_ranked} FRED data series, the agent selected the "
        f"**{len(ranking['selected'])} indicators** with the strongest correlation "
        f"to your demand growth rate:"
    )

    sel_cols = st.columns(len(ranking["selected"]))
    for col, item in zip(sel_cols, ranking["selected"]):
        col.metric(
            item["label"],
            f"{item['corr']:+.3f}",
            help=f"Pearson r between YoY demand growth and {item['label']}. "
                 "Computed on detrended data to remove company growth trend.",
        )

    r2_pct = ranking["r_squared"] * 100
    selected_labels = " + ".join(s["label"] for s in ranking["selected"])
    st.metric(
        f"Joint R² ({selected_labels} → Demand Growth)",
        f"{r2_pct:.1f}%",
        help="% of YoY demand growth variance explained by the selected indicators together.",
    )

    # --- Agent Insight text ------------------------------------------------
    top = ranking["selected"][0] if ranking["selected"] else None
    if top:
        direction = "positively" if top["corr"] > 0 else "inversely"
        st.info(
            f"**Agent Insight:** The selected indicators together explain approximately "
            f"**{r2_pct:.1f}%** of your historical demand *growth rate* fluctuations "
            f"(after removing your company's underlying trend). "
            f"The strongest signal is **{top['label']}** (r = {top['corr']:+.3f}) — "
            f"your demand is {direction} correlated with this indicator."
        )

    # --- Full ranking table (collapsible) ----------------------------------
    with st.expander(f"View full ranking — all {n_ranked} evaluated indicators"):
        selected_names_set = set(ranking["selected_names"])
        rank_table = pd.DataFrame([
            {
                "Rank": i + 1,
                "Indicator": r["label"],
                "Pearson r (YoY)": f"{r['corr']:+.4f}",
                "|r|": f"{abs(r['corr']):.4f}",
                "Selected": "✅" if r["name"] in selected_names_set else "",
            }
            for i, r in enumerate(ranking["all_rankings"])
        ])
        st.dataframe(rank_table, use_container_width=True, hide_index=True)

    # --- OLS Regression Weights --------------------------------------------
    if ranking["coefficients"]:
        st.subheader("Regression Weights (OLS)")
        st.caption(
            "How much does each 1-unit change in a macro indicator shift "
            "your demand's YoY growth rate (in percentage points)?"
        )
        coef_rows = []
        for item in ranking["selected"]:
            name = item["name"]
            coef = ranking["coefficients"].get(name, 0.0)
            cfg  = FredHelper.SCENARIO_DEFAULTS.get(name, (None, None, None, None, "unit", ""))
            unit = cfg[4]
            coef_rows.append({
                "Indicator": item["label"],
                f"Coefficient (pp demand growth per 1 {unit} change)": f"{coef:+.4f}",
                "Interpretation": (
                    f"1 {unit} {'increase' if coef >= 0 else 'decrease'} in "
                    f"{item['label']} → demand YoY growth shifts by {coef:+.2f} pp"
                ),
            })
        st.dataframe(
            pd.DataFrame(coef_rows), use_container_width=True, hide_index=True
        )

    # --- Correlation bar chart (all indicators) ----------------------------
    st.subheader("Visual Analysis")
    st.plotly_chart(
        _build_correlation_bar(ranking["all_rankings"], ranking["selected_names"]),
        use_container_width=True,
    )

    # --- Scatter charts for selected indicators ----------------------------
    n_sel = len(ranking["selected"])
    if n_sel > 0:
        scatter_cols = st.columns(min(n_sel, 3))
        for col, item in zip(scatter_cols, ranking["selected"]):
            with col:
                st.plotly_chart(
                    _scatter_chart(merged, item["name"], item["label"]),
                    use_container_width=True,
                )

elif (
    st.session_state["company_df"] is None
    or st.session_state["fred_helper"] is None
):
    if st.session_state["company_df"] is None:
        st.warning("Please upload your demand data first (sidebar Step 1).")
    if st.session_state["fred_helper"] is None:
        st.warning("Please connect your FRED API key first (sidebar Step 2).")

# ---------------------------------------------------------------------------
# Feature D: Backtest / Model Validation
# ---------------------------------------------------------------------------

st.header("D · Model Validation (Backtest)")

_selected_for_backtest = st.session_state.get("selected_indicators") or []
_backtest_label = (
    ", ".join(
        FredHelper.INDICATOR_CATALOGUE[n][1] for n in _selected_for_backtest
    ) if _selected_for_backtest else "macro indicators"
)
st.markdown(
    f"Runs a **{N_FOLDS}-fold expanding-window cross-validation**, re-selects the best macro "
    f"indicators using only pre-cutoff data (no data leakage), and compares macro-augmented "
    f"Prophet vs. vanilla Prophet vs. a naive seasonal baseline. Results are averaged across "
    f"all folds for a robust accuracy estimate."
)

# --- Horizon selector ------------------------------------------------------
backtest_horizon = st.radio(
    "Backtest Horizon",
    ["1 Year (12 months)", "5 Years (60 months)", "Both"],
    horizontal=True,
    key="backtest_horizon_radio",
)

# --- Data sufficiency warning for 5-year option ----------------------------
if st.session_state["merged_df"] is not None and backtest_horizon in ("5 Years (60 months)", "Both"):
    _merged_for_check = st.session_state["merged_df"]
    _n_months_available = int(round(
        (_merged_for_check["ds"].max() - _merged_for_check["ds"].min()).days / 30.44
    ))
    _min_needed_5y = 84  # 24 train + 60 holdout
    if _n_months_available < _min_needed_5y:
        st.warning(
            f"Your dataset has only {_n_months_available} months. "
            f"The 5-year backtest needs at least {_min_needed_5y} months. "
            "Results may use fewer folds or may not be possible."
        )

# --- Dynamic button label --------------------------------------------------
_btn_label_map = {
    "1 Year (12 months)":  "🧪 Run 1-Year Backtest (Hold-Out Last 12 Months)",
    "5 Years (60 months)": "🧪 Run 5-Year Backtest (Hold-Out Last 60 Months)",
    "Both":                "🧪 Run Both Backtests (1-Year + 5-Year)",
}
backtest_btn = st.button(
    _btn_label_map.get(backtest_horizon, "🧪 Run Backtest"),
    disabled=(st.session_state["merged_df"] is None or not _selected_for_backtest),
    use_container_width=False,
)

if backtest_btn:
    _run_1y = backtest_horizon in ("1 Year (12 months)", "Both")
    _run_5y = backtest_horizon in ("5 Years (60 months)", "Both")

    _spinner_msg = (
        f"Running {N_FOLDS}-fold cross-validation — training {N_FOLDS * 2} models, "
        "this may take ~60 seconds …"
        if not _run_5y else
        f"Running {N_FOLDS}-fold cross-validation with a 5-year holdout — "
        "training many models, this may take several minutes …"
    )

    with st.spinner(_spinner_msg):
        try:
            merged = st.session_state["merged_df"]
            helper: FredHelper = st.session_state["fred_helper"]
            _trend_flex = st.session_state["trend_flexibility"]
            _g_override = st.session_state.get("growth_rate_value")
            _research_sel = st.session_state.get("selected_indicators") or []

            if _run_1y:
                bt_result, bt_err = _run_backtest(
                    merged, helper,
                    horizon=12, n_folds=N_FOLDS,
                    trend_flex=_trend_flex,
                    growth_rate_override=_g_override,
                )
                if bt_err:
                    st.error(bt_err)
                    st.session_state["backtest_metrics"] = None
                else:
                    st.session_state["backtest_metrics"] = bt_result
                    # Notify if backtest indicators differed from research phase
                    differing = [
                        f"Fold {i+1}: {', '.join(FredHelper.INDICATOR_CATALOGUE[n][1] for n in fi)}"
                        for i, fi in enumerate(bt_result["fold_indicators"])
                        if set(fi) != set(_research_sel)
                    ]
                    if differing:
                        st.info(
                            "**Note (1-Year):** To prevent data leakage, each fold re-selected "
                            "indicators using only pre-cutoff data. The following folds used "
                            f"different indicators than the research phase:  \n" +
                            "  \n".join(differing)
                        )

            if _run_5y:
                bt_result_5y, bt_err_5y = _run_backtest(
                    merged, helper,
                    horizon=60, n_folds=N_FOLDS,
                    trend_flex=_trend_flex,
                    growth_rate_override=_g_override,
                )
                if bt_err_5y:
                    st.error(bt_err_5y)
                    st.session_state["backtest_metrics_5y"] = None
                else:
                    st.session_state["backtest_metrics_5y"] = bt_result_5y
                    differing_5y = [
                        f"Fold {i+1}: {', '.join(FredHelper.INDICATOR_CATALOGUE[n][1] for n in fi)}"
                        for i, fi in enumerate(bt_result_5y["fold_indicators"])
                        if set(fi) != set(_research_sel)
                    ]
                    if differing_5y:
                        st.info(
                            "**Note (5-Year):** To prevent data leakage, each fold re-selected "
                            "indicators using only pre-cutoff data. The following folds used "
                            f"different indicators than the research phase:  \n" +
                            "  \n".join(differing_5y)
                        )

        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            with st.expander("Technical details (for debugging)"):
                st.code(traceback.format_exc())

# --- Display results -------------------------------------------------------
_bt_1y = st.session_state["backtest_metrics"]
_bt_5y = st.session_state["backtest_metrics_5y"]

if _bt_1y is not None:
    _display_backtest_results(_bt_1y, label="1-Year")

if _bt_5y is not None:
    _display_backtest_results(_bt_5y, label="5-Year")

if _bt_1y is None and _bt_5y is None and st.session_state["merged_df"] is None:
    st.info("Complete Steps 1–3 and run Historical Correlations above first.", icon="ℹ️")

# ---------------------------------------------------------------------------
# Feature E + F: What-If Sliders & Forecast Generation
# ---------------------------------------------------------------------------

st.header("E + F · What-If Scenario & 12-Month Forecast")

# Show a real-time preview of the chosen scenario
scenario_values = st.session_state.get("scenario_values") or {}

if scenario_values:
    scen_cols = st.columns(len(scenario_values))
    for col, (name, value) in zip(scen_cols, scenario_values.items()):
        label = FredHelper.INDICATOR_CATALOGUE[name][1]
        unit  = FredHelper.SCENARIO_DEFAULTS[name][4]
        col.metric(label, f"{value:+.1f} {unit}")

    # Dynamic scenario classification
    if "GDP_growth" in scenario_values and scenario_values["GDP_growth"] < 0:
        scenario_label, scenario_color = "Recession Risk ⚠️", "warning"
    elif "inflation" in scenario_values and scenario_values["inflation"] > 8.0:
        scenario_label, scenario_color = "High Inflation Stress 🔥", "warning"
    elif "GDP_growth" in scenario_values and scenario_values["GDP_growth"] > 3.0:
        scenario_label, scenario_color = "Strong Growth 🌟", "success"
    else:
        scenario_label, scenario_color = "Custom Scenario 📊", "info"
    getattr(st, scenario_color)(f"**Scenario Classification:** {scenario_label}")
else:
    st.caption("Run **Research Historical Correlations** to see your scenario here.")

# --- Forecast button -------------------------------------------------------
forecast_btn = st.button(
    "🚀 Generate 12-Month Macro-Adjusted Forecast",
    disabled=(st.session_state["merged_df"] is None),
    type="primary",
    use_container_width=True,
)

if forecast_btn:
    if st.session_state["merged_df"] is None:
        st.error("Please complete the Historical Correlation Research step first.")
    else:
        with st.spinner("Prophet is training on your macro-augmented data … this may take ~30 seconds."):
            try:
                merged: pd.DataFrame = st.session_state["merged_df"]
                helper: FredHelper   = st.session_state["fred_helper"]
                sel_names: list      = st.session_state["selected_indicators"]
                scenario_vals: dict  = st.session_state.get("scenario_values") or {}

                # --- Build training DataFrame ---------------------------------
                train_df = merged[["ds", "y"] + sel_names].copy()

                # --- Initialise and train Prophet ----------------------------
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    interval_width=0.80,
                    changepoint_prior_scale=st.session_state["trend_flexibility"],
                )
                for name in sel_names:
                    model.add_regressor(name)
                model.fit(train_df)

                # --- Build future DataFrame (12 months) ----------------------
                future = model.make_future_dataframe(periods=12, freq="MS")

                # Merge known historical regressor values
                future = future.merge(
                    train_df[["ds"] + sel_names],
                    on="ds",
                    how="left",
                )

                # Inject scenario values for future periods
                future = helper.prepare_future_regressors(
                    future,
                    scenario_values=scenario_vals,
                )

                # --- Generate forecast ----------------------------------------
                forecast = model.predict(future)

                # Persist results
                st.session_state["forecast_df"] = forecast
                st.session_state["model"]       = model
                st.session_state["train_df_for_chart"] = train_df

            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
                with st.expander("Technical details (for debugging)"):
                    st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Feature E (output): Display forecast results
# ---------------------------------------------------------------------------

if st.session_state.get("forecast_df") is not None:
    forecast: pd.DataFrame = st.session_state["forecast_df"]
    train_df: pd.DataFrame = st.session_state.get(
        "train_df_for_chart", st.session_state["merged_df"]
    )

    st.success("✅ Forecast generated successfully!")

    # --- Main forecast chart ------------------------------------------------
    st.subheader("Macro-Adjusted Demand Forecast")
    st.plotly_chart(
        _build_forecast_chart(forecast, train_df, growth_rate=st.session_state.get("growth_rate_value")),
        use_container_width=True,
    )

    # --- Component chart ----------------------------------------------------
    st.subheader("Forecast Components")
    st.plotly_chart(
        _build_component_chart(forecast, st.session_state.get("selected_indicators") or []),
        use_container_width=True,
    )

    # --- Final projected demand ---------------------------------------------
    st.subheader("12-Month Projected Demand Summary")
    future_only = forecast[forecast["ds"] > train_df["ds"].max()].copy()

    total_yhat     = future_only["yhat"].sum()
    total_lower    = future_only["yhat_lower"].sum()
    total_upper    = future_only["yhat_upper"].sum()
    avg_monthly    = future_only["yhat"].mean()
    peak_row       = future_only.loc[future_only["yhat"].idxmax()]
    trough_row     = future_only.loc[future_only["yhat"].idxmin()]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Total 12-Month Demand (forecast)",
        f"{total_yhat:,.0f}",
        f"± {(total_upper - total_lower) / 2:,.0f}  (80% CI)",
    )
    c2.metric(
        "Average Monthly Demand",
        f"{avg_monthly:,.1f}",
    )
    c3.metric(
        "Peak Month",
        f"{peak_row['ds'].strftime('%b %Y')}",
        f"{peak_row['yhat']:,.0f} units",
    )

    _scenario_str = "  ·  ".join(
        f"{FredHelper.INDICATOR_CATALOGUE[n][1].split('(')[0].strip()} {v:+.2f}"
        for n, v in (st.session_state.get("scenario_values") or {}).items()
    )
    st.caption(
        f"Trough month: **{trough_row['ds'].strftime('%b %Y')}** "
        f"at {trough_row['yhat']:,.0f} units.  "
        f"**Scenario used:** {_scenario_str or 'default'}"
    )

    # --- Download button ----------------------------------------------------
    dl_df = future_only[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    dl_df.columns = ["Date", "Forecast_Demand", "Lower_80pct", "Upper_80pct"]
    for n, v in (st.session_state.get("scenario_values") or {}).items():
        label = FredHelper.INDICATOR_CATALOGUE[n][1].replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        dl_df[f"Scenario_{label}"] = v

    csv_bytes = dl_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv_bytes,
        file_name="macroagent_forecast.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # --- Disclaimer on forecast output --------------------------------------
    st.caption(
        "⚠️ This forecast is for **educational and analytical purposes only**.  "
        "It is not financial or investment advice.  "
        "FRED data is used under personal/educational terms of use.  "
        "See [FRED Terms](https://fred.stlouisfed.org/about/terms)."
    )

elif st.session_state["merged_df"] is None:
    st.info(
        "Complete Steps 1–3 in the sidebar, run Historical Correlations above, "
        "then click **Generate 12-Month Forecast**.",
        icon="ℹ️",
    )

# ===========================================================================
# BOTTOM DISCLAIMER
# ===========================================================================

st.divider()
_disclaimer_banner("bottom")

st.caption(
    "MacroAgent Forecaster · Open Source · "
    "Built by a first-year Rutgers Supply Chain Management + BAIT student · "
    "Powered by Prophet, FRED, and Streamlit · "
    "[GitHub Repo](https://github.com/jpgluck/macroagent-forecaster-rutgers)"
)
