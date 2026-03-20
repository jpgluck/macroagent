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
    "company_df":     None,   # uploaded & validated demand data (Prophet format)
    "fred_helper":    None,   # authenticated FredHelper instance
    "merged_df":      None,   # company data merged with FRED macro data
    "corr_stats":     None,   # dict from calculate_correlations_and_weights()
    "forecast_df":    None,   # Prophet forecast result
    "model":          None,   # trained Prophet model
    "gdp_scenario":   2.0,
    "inf_scenario":   5.0,
}
for _key, _val in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


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
        (c for c in df.columns if "demand" in c.lower() or "y" == c.lower()), None
    )
    if demand_col is None:
        return False, "Could not find a column containing 'demand'. Please rename it.", None

    # --- parse & clean ------------------------------------------------------
    try:
        df = df[[date_col, demand_col]].copy()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
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


def _build_forecast_chart(forecast: pd.DataFrame, train_df: pd.DataFrame) -> go.Figure:
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

    # Vertical line at today / forecast start
    cutoff = train_df["ds"].max()
    fig.add_vline(
        x=cutoff,
        line_width=1.5,
        line_dash="dash",
        line_color="#E53935",
        annotation_text="Forecast Start",
        annotation_position="top left",
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


def _build_component_chart(forecast: pd.DataFrame) -> go.Figure:
    """Build a stacked area chart showing Prophet's decomposed components."""
    cols_wanted = [c for c in ["trend", "yearly", "GDP_growth", "inflation"]
                   if c in forecast.columns]

    fig = go.Figure()
    colors = ["#1E88E5", "#43A047", "#FB8C00", "#E53935"]
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


def _build_correlation_heatmap(merged_df: pd.DataFrame) -> go.Figure:
    """Build a Plotly correlation heatmap for demand, GDP, and inflation."""
    cols = ["y", "GDP_growth", "inflation"]
    corr_matrix = merged_df[cols].corr()
    labels = ["Demand", "GDP Growth", "Inflation"]

    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title="Correlation Heatmap: Demand vs. Macro Indicators",
        height=380,
        template="plotly_white",
    )
    return fig


def _scatter_chart(merged_df: pd.DataFrame, x_col: str, x_label: str) -> go.Figure:
    """Simple scatter of demand vs. a macro indicator with an OLS trend line."""
    fig = px.scatter(
        merged_df, x=x_col, y="y",
        trendline="ols",
        labels={x_col: x_label, "y": "Demand"},
        template="plotly_white",
        title=f"Demand vs. {x_label}",
    )
    fig.update_traces(marker=dict(size=7, color="#1E88E5", opacity=0.75),
                      selector=dict(mode="markers"))
    return fig


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
    # Step 3: What-If Sliders
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Step 3 · What-If Scenario")

    gdp_slider = st.slider(
        "Expected GDP Growth this year (%)",
        min_value=-5.0, max_value=10.0,
        value=st.session_state["gdp_scenario"],
        step=0.1,
        help="Set to your expected annual real GDP growth rate.",
    )
    inf_slider = st.slider(
        "Expected Inflation this year (%)",
        min_value=-5.0, max_value=15.0,
        value=st.session_state["inf_scenario"],
        step=0.1,
        help="Set to your expected annual CPI inflation rate.",
    )

    # Persist slider values so they survive reruns
    st.session_state["gdp_scenario"] = gdp_slider
    st.session_state["inf_scenario"] = inf_slider

    st.caption(
        f"**Scenario preview:** GDP = **{gdp_slider:+.1f}%** · "
        f"Inflation = **{inf_slider:+.1f}%**"
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
    with st.spinner("Agent is aligning your data with Federal Reserve macro indicators …"):
        try:
            helper: FredHelper = st.session_state["fred_helper"]
            merged = helper.align_with_company_data(st.session_state["company_df"])
            st.session_state["merged_df"] = merged

            stats = helper.calculate_correlations_and_weights(merged)
            st.session_state["corr_stats"] = stats

        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error during macro alignment: {exc}")
            st.caption("Details: " + traceback.format_exc())

if st.session_state["corr_stats"] is not None:
    stats = st.session_state["corr_stats"]
    merged = st.session_state["merged_df"]

    st.success(
        "✅ Agent has successfully researched and aligned Federal Reserve macro data "
        "with your demand history."
    )

    # --- Metric cards -------------------------------------------------------
    st.subheader("Correlation Findings")
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Correlation: Demand ↔ GDP Growth",
        f"{stats['corr_gdp']:+.3f}",
        help="Pearson r. Close to +1 = strong positive link; −1 = inverse.",
    )
    m2.metric(
        "Correlation: Demand ↔ Inflation",
        f"{stats['corr_inflation']:+.3f}",
        help="Pearson r between demand and year-over-year CPI change.",
    )
    m3.metric(
        "Joint R² (GDP + Inflation → Demand)",
        f"{stats['r_squared']:.1%}",
        help="% of demand variance explained by GDP growth + inflation together.",
    )

    # --- Regression weights table ------------------------------------------
    st.subheader("Regression Weights (OLS)")
    reg_data = pd.DataFrame({
        "Macro Indicator": ["GDP Growth (%)", "Inflation (%)"],
        "Coefficient (units per 1% change)": [
            f"{stats['gdp_coef']:+.4f}",
            f"{stats['inf_coef']:+.4f}",
        ],
        "Interpretation": [
            f"GDP growth moves demand by {stats['gdp_coef']:+.2f} units per 1% change",
            f"Inflation moves demand by {stats['inf_coef']:+.2f} units per 1% change",
        ],
    })
    st.dataframe(reg_data, use_container_width=True, hide_index=True)

    # --- Insight text -------------------------------------------------------
    r2_pct = stats["r_squared"] * 100
    gdp_pct = abs(stats["corr_gdp"]) / (
        abs(stats["corr_gdp"]) + abs(stats["corr_inflation"]) + 1e-9
    ) * 100
    st.info(
        f"**Agent Insight:**  GDP growth and inflation together explain approximately "
        f"**{r2_pct:.1f}%** of your historical demand movement.  "
        f"GDP growth accounts for roughly **{gdp_pct:.0f}%** of that macro influence.  "
        f"{'Your demand appears positively correlated with economic expansion.' if stats['corr_gdp'] > 0 else 'Your demand appears counter-cyclical — it tends to rise when GDP contracts.'}"
    )

    # --- Scatter charts + Heatmap ------------------------------------------
    st.subheader("Visual Analysis")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.plotly_chart(
            _scatter_chart(merged, "GDP_growth", "GDP Growth (%)"),
            use_container_width=True,
        )
    with sc2:
        st.plotly_chart(
            _scatter_chart(merged, "inflation", "Inflation (YoY %)"),
            use_container_width=True,
        )

    st.plotly_chart(
        _build_correlation_heatmap(merged),
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
# Feature D + E: What-If Sliders & Forecast Generation
# ---------------------------------------------------------------------------

st.header("D + E · What-If Scenario & 12-Month Forecast")

# Show a real-time preview of the chosen scenario
col_a, col_b = st.columns(2)
col_a.metric(
    "Your GDP Growth Assumption",
    f"{st.session_state['gdp_scenario']:+.1f}%",
    help="Adjust with the sidebar slider.",
)
col_b.metric(
    "Your Inflation Assumption",
    f"{st.session_state['inf_scenario']:+.1f}%",
    help="Adjust with the sidebar slider.",
)

# Classify scenario
gdp_val = st.session_state["gdp_scenario"]
inf_val = st.session_state["inf_scenario"]
if gdp_val > 3.0 and inf_val < 4.0:
    scenario_label, scenario_color = "Goldilocks Growth 🌟", "success"
elif gdp_val < 0:
    scenario_label, scenario_color = "Recession Risk ⚠️", "warning"
elif inf_val > 8.0:
    scenario_label, scenario_color = "High Inflation Stress 🔥", "warning"
else:
    scenario_label, scenario_color = "Moderate Growth 📊", "info"

getattr(st, scenario_color)(f"**Scenario Classification:** {scenario_label}")

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

                # --- Build training DataFrame ---------------------------------
                train_df = merged[["ds", "y", "GDP_growth", "inflation"]].copy()

                # --- Initialise and train Prophet ----------------------------
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    interval_width=0.80,
                    changepoint_prior_scale=0.05,
                )
                model.add_regressor("GDP_growth")
                model.add_regressor("inflation")
                model.fit(train_df)

                # --- Build future DataFrame (12 months) ----------------------
                future = model.make_future_dataframe(periods=12, freq="MS")

                # Merge known historical regressor values
                future = future.merge(
                    train_df[["ds", "GDP_growth", "inflation"]],
                    on="ds",
                    how="left",
                )

                # Inject scenario values for future periods
                future = helper.prepare_future_regressors(
                    future,
                    gdp_scenario=st.session_state["gdp_scenario"],
                    inf_scenario=st.session_state["inf_scenario"],
                )

                # --- Generate forecast ----------------------------------------
                forecast = model.predict(future)

                # Persist results
                st.session_state["forecast_df"] = forecast
                st.session_state["model"]       = model
                st.session_state["train_df_for_chart"] = train_df

            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
                st.caption("Details: " + traceback.format_exc())

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
        _build_forecast_chart(forecast, train_df),
        use_container_width=True,
    )

    # --- Component chart ----------------------------------------------------
    st.subheader("Forecast Components")
    st.plotly_chart(
        _build_component_chart(forecast),
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

    st.caption(
        f"Trough month: **{trough_row['ds'].strftime('%b %Y')}** "
        f"at {trough_row['yhat']:,.0f} units.  "
        f"**Scenario used:** GDP {st.session_state['gdp_scenario']:+.1f}% · "
        f"Inflation {st.session_state['inf_scenario']:+.1f}%"
    )

    # --- Download button ----------------------------------------------------
    dl_df = future_only[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    dl_df.columns = ["Date", "Forecast_Demand", "Lower_80pct", "Upper_80pct"]
    dl_df["GDP_Scenario_%"] = st.session_state["gdp_scenario"]
    dl_df["Inflation_Scenario_%"] = st.session_state["inf_scenario"]

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
