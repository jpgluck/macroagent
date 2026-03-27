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
