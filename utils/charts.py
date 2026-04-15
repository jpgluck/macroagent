import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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


def _build_segment_forecast_chart(
    combined_forecast: pd.DataFrame,
    segment_results: dict,
    segments: list,
) -> go.Figure:
    """
    Stacked area chart showing per-segment forecast contributions,
    with the combined total overlaid as a line.

    Parameters
    ----------
    combined_forecast : pd.DataFrame
        Output of run_segment_forecast() — must contain ds, yhat, and
        per-segment yhat_{seg} columns.
    segment_results : dict
        {segment: {"forecast": df, "train_df": df, "sel_names": list}}
    segments : list
        Ordered list of segment names (from segment_rankings["segments"]).
    """
    SEGMENT_COLORS = [
        "#1E88E5",  # blue
        "#43A047",  # green
        "#FB8C00",  # orange
        "#E53935",  # red
        "#8E24AA",  # purple
        "#00ACC1",  # cyan
    ]

    fig = go.Figure()

    # Stacked area — one trace per segment using their individual yhats
    for i, seg in enumerate(segments):
        col = f"yhat_{seg}"
        if col not in combined_forecast.columns:
            continue
        color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        fig.add_trace(go.Scatter(
            x=combined_forecast["ds"],
            y=combined_forecast[col],
            mode="lines",
            stackgroup="segments",
            name=seg,
            line=dict(color=color, width=0),
            fillcolor=color.replace("#", "rgba(").replace(
                "rgba(", "rgba("
            ) if False else color,  # plotly handles fill colour from line colour for stackgroup
        ))

    # Total forecast line on top
    fig.add_trace(go.Scatter(
        x=combined_forecast["ds"],
        y=combined_forecast["yhat"],
        mode="lines",
        name="Total Forecast",
        line=dict(color="#212121", width=2.5),
    ))

    # Historical actuals (summed across segments using train_dfs)
    hist_frames = []
    for seg, res in segment_results.items():
        hist_frames.append(res["train_df"][["ds", "y"]])
    if hist_frames:
        hist = (
            pd.concat(hist_frames)
            .groupby("ds", as_index=False)["y"]
            .sum()
            .sort_values("ds")
        )
        fig.add_trace(go.Scatter(
            x=hist["ds"],
            y=hist["y"],
            mode="markers+lines",
            name="Historical (Total)",
            line=dict(color="#616161", width=1.5, dash="dot"),
            marker=dict(size=5, color="#616161"),
        ))

    # Vertical line at forecast start
    if hist_frames:
        cutoff_str = hist["ds"].max().isoformat()
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
        title="12-Month Segment Forecast (Stacked by Market Segment)",
        xaxis_title="Date",
        yaxis_title="Demand",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500,
    )
    return fig


def _build_segment_correlation_chart(segment_rankings: dict) -> go.Figure:
    """
    Grouped bar chart showing, for each evaluated indicator,
    its Pearson correlation with demand growth broken out by segment.

    This makes it easy to see, e.g., that GDP correlates strongly with
    Commercial demand but weakly with Residential demand.

    Parameters
    ----------
    segment_rankings : dict
        Output of rank_indicators_by_segment().
    """
    segments = segment_rankings["segments"]
    # Collect all indicator names that appear in any segment's ranking
    all_indicator_names: list = []
    seen: set = set()
    for seg in segments:
        for entry in segment_rankings["segment_rankings"][seg]["all_rankings"]:
            if entry["name"] not in seen:
                all_indicator_names.append(entry["name"])
                seen.add(entry["name"])

    # Build a lookup: {seg: {name: corr}}
    corr_lookup: dict = {}
    for seg in segments:
        corr_lookup[seg] = {
            e["name"]: e["corr"]
            for e in segment_rankings["segment_rankings"][seg]["all_rankings"]
        }

    # Collect labels in indicator order
    # Use the label from the first segment that has the indicator
    label_map: dict = {}
    for seg in segments:
        for e in segment_rankings["segment_rankings"][seg]["all_rankings"]:
            if e["name"] not in label_map:
                label_map[e["name"]] = e["label"]

    x_labels = [label_map.get(n, n) for n in all_indicator_names]

    SEGMENT_COLORS = [
        "#1E88E5",
        "#43A047",
        "#FB8C00",
        "#E53935",
        "#8E24AA",
        "#00ACC1",
    ]

    fig = go.Figure()
    for i, seg in enumerate(segments):
        color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        y_vals = [corr_lookup[seg].get(name, 0.0) for name in all_indicator_names]
        fig.add_trace(go.Bar(
            name=seg,
            x=x_labels,
            y=y_vals,
            marker_color=color,
            text=[f"{v:+.3f}" for v in y_vals],
            textposition="outside",
        ))

    fig.add_hline(y=0, line_width=1, line_color="#888")
    fig.update_layout(
        title=(
            "Indicator Correlations with Demand Growth — by Market Segment<br>"
            "<sup>Grouped bars show how each macro indicator relates differently to each segment</sup>"
        ),
        barmode="group",
        yaxis_title="Pearson r",
        yaxis=dict(range=[-1.15, 1.15], tickformat=".2f"),
        xaxis_title=None,
        xaxis_tickangle=-30,
        template="plotly_white",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(b=120),
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
