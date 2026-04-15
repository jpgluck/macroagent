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
import traceback

# ---------------------------------------------------------------------------
# Third-party imports  (all from requirements.txt)
# ---------------------------------------------------------------------------
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.backtest import _display_backtest_results, _run_backtest
from utils.charts import (
    _build_component_chart,
    _build_correlation_bar,
    _build_forecast_chart,
    _build_segment_correlation_chart,
    _build_segment_forecast_chart,
    _scatter_chart,
)
from utils.forecasting import run_forecast, run_segment_forecast

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from utils.fred_helper import FredHelper
from utils.validation import _compute_growth_rates, _validate_company_df

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
    "has_segments":        False,       # whether uploaded data has market segments
    "segment_rankings":    None,        # output of rank_indicators_by_segment()
    "segment_results":     None,        # per-segment forecast results
    "segment_models":      None,        # per-segment Prophet models
    "combined_forecast":   None,        # combined segment forecast df
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
            has_seg = "segment" in clean_df.columns
            st.session_state["has_segments"] = has_seg
            if has_seg:
                segs = clean_df["segment"].unique()
                st.success(
                    f"✅ {len(clean_df)} rows loaded across "
                    f"**{len(segs)} market segments**: {', '.join(sorted(segs))}"
                )
            else:
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
        # For segment data, aggregate to total demand before computing growth rates
        _gr_df = st.session_state["company_df"]
        if st.session_state["has_segments"]:
            _gr_df = _gr_df.groupby("ds", as_index=False)["y"].sum()
        growth_rates = _compute_growth_rates(_gr_df)

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
    _has_seg = st.session_state["has_segments"]

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Data Preview")
        rename_map = {"ds": "Date", "y": "Demand"}
        if _has_seg:
            rename_map["segment"] = "Market Segment"
        st.dataframe(df.rename(columns=rename_map), height=260)

    with col2:
        st.subheader("Summary Statistics")
        if _has_seg:
            # Show per-segment summary
            for seg in sorted(df["segment"].unique()):
                seg_data = df[df["segment"] == seg]
                seg_total = seg_data["y"].sum()
                seg_mean = seg_data["y"].mean()
                st.metric(
                    f"{seg}",
                    f"Mean: {seg_mean:,.0f}",
                    f"Total: {seg_total:,.0f}  ·  {len(seg_data)} rows",
                )
            total_demand = df.groupby("ds")["y"].sum()
            st.metric("Total Data Points", f"{len(df)} ({len(df['segment'].unique())} segments)")
        else:
            stats = df["y"].describe()
            trend_pct = (df["y"].iloc[-1] / df["y"].iloc[0] - 1) * 100
            st.metric("Mean Demand",  f"{stats['mean']:,.1f}")
            st.metric("Min Demand",   f"{stats['min']:,.1f}")
            st.metric("Max Demand",   f"{stats['max']:,.1f}")
            st.metric("Overall Trend",  f"{trend_pct:+.1f}%",
                      help="% change from first to last data point")
            st.metric("Data Points",  f"{len(df)}")

    # Quick time-series chart of raw demand
    if _has_seg:
        fig_raw = px.line(
            df, x="ds", y="y", color="segment",
            labels={"ds": "Date", "y": "Demand", "segment": "Market Segment"},
            title="Historical Demand by Market Segment",
            template="plotly_white",
        )
        # Also show total as a dashed line
        total_ts = df.groupby("ds", as_index=False)["y"].sum()
        fig_raw.add_scatter(
            x=total_ts["ds"], y=total_ts["y"],
            mode="lines", name="Total",
            line=dict(color="#212121", width=2.5, dash="dash"),
        )
    else:
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
    _has_seg = st.session_state["has_segments"]
    _spinner_text = (
        "Agent is scanning 10 Federal Reserve indicators per market segment "
        "to find which macro forces matter most — this takes ~15 seconds …"
        if _has_seg else
        "Agent is scanning 10 Federal Reserve indicators to find which macro "
        "forces matter most for your data — this takes ~10 seconds …"
    )
    with st.spinner(_spinner_text):
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

            if _has_seg:
                # Step 3a: Rank per segment
                seg_rank = helper.rank_indicators_by_segment(merged, top_n=3)
                st.session_state["segment_rankings"]    = seg_rank
                st.session_state["selected_indicators"] = seg_rank["all_selected_names"]
                # Store a combined indicator_ranking for backward compat (use total demand)
                total_merged = merged.groupby("ds", as_index=False).agg(
                    {"y": "sum", **{c: "first" for c in merged.columns if c not in ("ds", "y", "segment")}}
                )
                ranking = helper.rank_indicators(total_merged, top_n=3)
                st.session_state["indicator_ranking"] = ranking
            else:
                # Step 3b: Rank globally (original behavior)
                ranking = helper.rank_indicators(merged, top_n=3)
                st.session_state["indicator_ranking"]   = ranking
                st.session_state["selected_indicators"] = ranking["selected_names"]
                st.session_state["segment_rankings"]    = None

            # Reset downstream state so stale results don't persist
            st.session_state["forecast_df"]        = None
            st.session_state["combined_forecast"]   = None
            st.session_state["segment_results"]     = None
            st.session_state["backtest_metrics"]    = None
            st.session_state["scenario_values"]     = None

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
    _seg_rank = st.session_state.get("segment_rankings")

    # ===== SEGMENT-AWARE CORRELATION DISPLAY ================================
    if _seg_rank is not None:
        n_all_sel = len(_seg_rank["all_selected_names"])
        st.success(
            f"✅ Agent evaluated **{n_fetched} Federal Reserve indicators** across "
            f"**{len(_seg_rank['segments'])} market segments** and selected "
            f"**{n_all_sel} unique indicators** as macro regressors."
        )

        # --- Per-Segment Breakdown -------------------------------------------
        for seg in _seg_rank["segments"]:
            seg_r = _seg_rank["segment_rankings"][seg]
            with st.expander(f"**{seg}** — Top {len(seg_r['selected'])} Indicators", expanded=True):
                if seg_r["selected"]:
                    seg_cols = st.columns(len(seg_r["selected"]))
                    for col, item in zip(seg_cols, seg_r["selected"]):
                        col.metric(
                            item["label"],
                            f"r = {item['corr']:+.3f}",
                        )
                    r2_pct = seg_r["r_squared"] * 100
                    st.metric(
                        f"Joint R² → {seg} Demand Growth",
                        f"{r2_pct:.1f}%",
                    )
                    # Insight per segment
                    top = seg_r["selected"][0]
                    direction = "positively" if top["corr"] > 0 else "inversely"
                    st.caption(
                        f"**{seg}** demand is most {direction} correlated with "
                        f"**{top['label']}** (r = {top['corr']:+.3f}). "
                        f"Selected indicators explain **{r2_pct:.1f}%** of this segment's demand growth variance."
                    )

        # --- Agent Insight (cross-segment) -----------------------------------
        insight_parts = []
        for seg in _seg_rank["segments"]:
            seg_r = _seg_rank["segment_rankings"][seg]
            if seg_r["selected"]:
                top = seg_r["selected"][0]
                insight_parts.append(
                    f"**{seg}** is driven by **{top['label']}** (r = {top['corr']:+.3f})"
                )
        if insight_parts:
            st.info(
                "**Agent Insight — Segment Differentiation:**  \n"
                + "  \n".join(f"- {p}" for p in insight_parts)
                + "\n\nDifferent market segments respond to different macro forces. "
                "The forecast model exploits these distinct relationships."
            )

        # --- Segment correlation grouped bar chart ----------------------------
        st.subheader("Visual Analysis — Correlations by Segment")
        st.plotly_chart(
            _build_segment_correlation_chart(_seg_rank),
            use_container_width=True,
        )

        # --- Also show the aggregate correlation bar -------------------------
        with st.expander("View aggregate correlation (total demand, all segments combined)"):
            st.plotly_chart(
                _build_correlation_bar(ranking["all_rankings"], ranking["selected_names"]),
                use_container_width=True,
            )

    # ===== ORIGINAL (NON-SEGMENT) CORRELATION DISPLAY =======================
    else:
        st.success(
            f"✅ Agent evaluated **{n_fetched} Federal Reserve indicators** and found "
            f"**{n_ranked}** with sufficient data. "
            f"The top **{len(ranking['selected'])}** were selected as macro regressors."
        )

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

        st.subheader("Visual Analysis")
        st.plotly_chart(
            _build_correlation_bar(ranking["all_rankings"], ranking["selected_names"]),
            use_container_width=True,
        )

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

            # For segment data, aggregate to total for backtesting
            if st.session_state["has_segments"] and "segment" in merged.columns:
                bt_merged = merged.drop(columns=["segment"]).groupby("ds", as_index=False).sum()
            else:
                bt_merged = merged

            if _run_1y:
                bt_result, bt_err = _run_backtest(
                    bt_merged, helper,
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
                            "different indicators than the research phase:  \n" +
                            "  \n".join(differing)
                        )

            if _run_5y:
                bt_result_5y, bt_err_5y = _run_backtest(
                    bt_merged, helper,
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
                            "different indicators than the research phase:  \n" +
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
        _has_seg = st.session_state["has_segments"]
        _spinner_text = (
            "Prophet is training a model per market segment … this may take ~60 seconds."
            if _has_seg else
            "Prophet is training on your macro-augmented data … this may take ~30 seconds."
        )
        with st.spinner(_spinner_text):
            try:
                merged: pd.DataFrame = st.session_state["merged_df"]
                helper: FredHelper   = st.session_state["fred_helper"]
                scenario_vals: dict  = st.session_state.get("scenario_values") or {}

                if _has_seg and st.session_state.get("segment_rankings"):
                    # Segment-aware forecast
                    combined, seg_results, seg_models = run_segment_forecast(
                        merged=merged,
                        segment_rankings=st.session_state["segment_rankings"],
                        scenario_vals=scenario_vals,
                        trend_flexibility=st.session_state["trend_flexibility"],
                        helper=helper,
                    )
                    st.session_state["combined_forecast"] = combined
                    st.session_state["segment_results"]   = seg_results
                    st.session_state["segment_models"]    = seg_models
                    st.session_state["forecast_df"]       = combined
                    st.session_state["model"]             = None
                    # Build a combined train_df for chart compatibility
                    all_train = []
                    for seg, res in seg_results.items():
                        all_train.append(res["train_df"][["ds", "y"]])
                    combined_train = pd.concat(all_train).groupby("ds", as_index=False)["y"].sum()
                    st.session_state["train_df_for_chart"] = combined_train
                else:
                    # Original single-model forecast
                    sel_names: list = st.session_state["selected_indicators"]
                    forecast, model, train_df = run_forecast(
                        merged=merged,
                        sel_names=sel_names,
                        scenario_vals=scenario_vals,
                        trend_flexibility=st.session_state["trend_flexibility"],
                        helper=helper,
                    )
                    st.session_state["forecast_df"]       = forecast
                    st.session_state["model"]             = model
                    st.session_state["train_df_for_chart"] = train_df
                    st.session_state["combined_forecast"]  = None
                    st.session_state["segment_results"]    = None

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
    _seg_results = st.session_state.get("segment_results")
    _seg_rank = st.session_state.get("segment_rankings")

    st.success("✅ Forecast generated successfully!")

    # --- Segment stacked chart (if segments) --------------------------------
    if _seg_results and _seg_rank:
        st.subheader("Combined Forecast — by Market Segment")
        st.plotly_chart(
            _build_segment_forecast_chart(
                forecast, _seg_results, _seg_rank["segments"]
            ),
            use_container_width=True,
        )

        # --- Per-segment detail -----------------------------------------------
        st.subheader("Per-Segment Forecast Detail")
        for seg in _seg_rank["segments"]:
            if seg not in _seg_results:
                continue
            seg_res = _seg_results[seg]
            seg_fc = seg_res["forecast"]
            seg_train = seg_res["train_df"]
            seg_future = seg_fc[seg_fc["ds"] > seg_train["ds"].max()]
            sel_labels = ", ".join(
                FredHelper.INDICATOR_CATALOGUE[n][1]
                for n in seg_res["sel_names"]
                if n in FredHelper.INDICATOR_CATALOGUE
            )
            with st.expander(f"**{seg}** — driven by {sel_labels}", expanded=False):
                st.plotly_chart(
                    _build_forecast_chart(seg_fc, seg_train),
                    use_container_width=True,
                )
                if len(seg_future) > 0:
                    seg_total = seg_future["yhat"].sum()
                    seg_avg = seg_future["yhat"].mean()
                    c1, c2 = st.columns(2)
                    c1.metric(f"{seg} — 12-Month Total", f"{seg_total:,.0f}")
                    c2.metric(f"{seg} — Monthly Average", f"{seg_avg:,.0f}")

    # --- Main forecast chart (combined total) --------------------------------
    st.subheader("Macro-Adjusted Demand Forecast (Total)")
    st.plotly_chart(
        _build_forecast_chart(forecast, train_df, growth_rate=st.session_state.get("growth_rate_value")),
        use_container_width=True,
    )

    # --- Component chart (only for non-segment mode) -------------------------
    if not _seg_results:
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

    # --- Per-segment summary table (if segments) ----------------------------
    if _seg_results and _seg_rank:
        st.subheader("Forecast by Segment")
        seg_summary_rows = []
        for seg in _seg_rank["segments"]:
            if seg not in _seg_results:
                continue
            seg_res = _seg_results[seg]
            seg_fc = seg_res["forecast"]
            seg_train = seg_res["train_df"]
            seg_future = seg_fc[seg_fc["ds"] > seg_train["ds"].max()]
            seg_total = seg_future["yhat"].sum() if len(seg_future) > 0 else 0
            seg_pct = (seg_total / total_yhat * 100) if total_yhat else 0
            sel_labels = ", ".join(
                FredHelper.INDICATOR_CATALOGUE[n][1].split("(")[0].strip()
                for n in seg_res["sel_names"]
                if n in FredHelper.INDICATOR_CATALOGUE
            )
            seg_summary_rows.append({
                "Segment": seg,
                "12-Month Forecast": f"{seg_total:,.0f}",
                "% of Total": f"{seg_pct:.1f}%",
                "Key Macro Drivers": sel_labels,
            })
        st.dataframe(
            pd.DataFrame(seg_summary_rows), use_container_width=True, hide_index=True
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
    dl_cols = {"ds": "Date", "yhat": "Forecast_Demand", "yhat_lower": "Lower_80pct", "yhat_upper": "Upper_80pct"}
    # Include per-segment columns in download if available
    if _seg_results and _seg_rank:
        for seg in _seg_rank["segments"]:
            col = f"yhat_{seg}"
            if col in future_only.columns:
                dl_df[col] = future_only[col]
                dl_cols[col] = f"Forecast_{seg}"
    dl_df = dl_df.rename(columns=dl_cols)
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
