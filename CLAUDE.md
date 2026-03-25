# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Setup
- Primary languages: TypeScript and Python
- When encountering dependency/build errors, check version compatibility FIRST before attempting fixes
- For Python projects, use `pip3` (not `pip`) as pip may not be in PATH

## Running the App

```bash
streamlit run app.py
```

Opens at http://localhost:8501. On first run, Streamlit may prompt for an email — skip it by pressing Enter or pre-create `~/.streamlit/credentials.toml` with `email = ""` under `[general]`.

**Python compatibility note:** Tested on Python 3.14.3. `requirements.txt` pins `streamlit==1.55.0`; earlier versions (≤1.42.0) are incompatible with Python 3.12+ due to a changed `asyncio.get_event_loop()` behavior.

## Dependencies & Environment

```bash
pip3 install -r requirements.txt
```

| Package | Pinned version | Role |
|---------|---------------|------|
| streamlit | 1.55.0 | Web UI |
| prophet | 1.3.0 | Time-series forecasting |
| fredapi | 0.5.2 | FRED API wrapper |
| pandas | 2.2.3 | Data manipulation |
| numpy | 2.4.3 | Numerical operations |
| plotly | 6.6.0 | Interactive charts |
| scikit-learn | 1.7.2 | OLS regression |
| statsmodels | 0.14.6 | Required by Prophet |

No build step, no tests, no linter configured.

- Always verify package version compatibility with the runtime (e.g., Python version) before installing or upgrading
- After any dependency change, run `streamlit run app.py` to verify it works before reporting success

## Architecture

Single-page Streamlit app (`app.py`) with one helper module (`utils/fred_helper.py`). Streamlit re-runs the entire script on every user interaction; persistent state lives in `st.session_state`.

### `app.py` — UI and orchestration

Session state keys:

| Key | Type | Description |
|-----|------|-------------|
| `company_df` | DataFrame | Uploaded CSV parsed to `ds`/`y` columns |
| `fred_helper` | FredHelper | Connected FRED API client |
| `all_indicators` | dict | `{name: pd.Series}` — all 10 fetched FRED series |
| `merged_df` | DataFrame | Company data joined with all indicator series |
| `indicator_ranking` | dict | Output of `rank_indicators()` — corr scores, R², OLS coefs |
| `selected_indicators` | list[str] | Top-N indicator names chosen by the agent |
| `scenario_values` | dict | `{indicator_name: float}` — what-if slider values |
| `forecast_df` | DataFrame | Prophet forecast output |
| `model` | Prophet | Trained Prophet model |

Key helpers defined at module level (above `main()`):
- `_validate_company_df` — parses CSV, detects date/demand columns by name substring
- `_mape`, `_mae`, `_rmse` — forecast accuracy metrics
- `_build_forecast_chart`, `_build_backtest_chart`, `_build_correlation_bar`, `_scatter_chart` — Plotly chart builders
- `_disclaimer_banner` — renders FRED compliance notice; **do not remove**

### `utils/fred_helper.py` — FRED API and indicator logic

`FredHelper` class:

- **`INDICATOR_CATALOGUE`** — 10 FRED series: GDP growth, CPI inflation, unemployment, fed funds rate, consumer sentiment, 10-yr treasury, industrial production, housing starts, retail sales, S&P 500. Each entry: `(series_id, label, frequency, transform)`.
- **`SCENARIO_DEFAULTS`** — per-indicator slider config `(min, max, default, step, unit, help_text)` used to render dynamic sidebar sliders.
- `fetch_indicator(name)` — fetches one series, applies YoY % transform if configured, resamples to monthly.
- `fetch_all_indicators()` — fetches all 10, skips failures silently.
- `align_all_indicators(company_df, indicators)` — inner-joins company data with all indicators on month-start dates.
- `rank_indicators(merged_df, top_n=3)` — **core agent logic**: computes Pearson r of each indicator vs. detrended demand (YoY %), ranks by |r|, selects top N, computes joint OLS R² and coefficients.
- `prepare_future_regressors(future_df, scenario_values)` — injects what-if values into Prophet future DataFrame.

## User Workflow (reflected in UI sections)

1. Upload CSV with `date` + `demand`/`revenue` columns → validated to Prophet format (`ds`, `y`)
2. Enter a free FRED API key → `FredHelper` tests connection
3. Click **Research Historical Correlations** → fetches all 10 FRED series, auto-selects top 3 most correlated with this dataset's detrended demand
4. Adjust per-indicator what-if scenario sliders (dynamically rendered for the selected indicators)
5. Click **Generate Forecast** → trains Prophet with the 3 selected indicators as regressors, projects 12 months
6. Review backtest metrics (MAPE/MAE/RMSE vs. naive baseline) and download forecast CSV

## Input CSV Format

Must have a date column (name contains `"date"`) and a value column (name contains `"demand"` or `"revenue"`, or exactly `"y"`). Minimum 12 rows, all values positive. Year-only integers (e.g. `2003`) are supported — parsed via `.astype(str)`. See `data/sample_demand.csv` for a working example.

## Debugging Rules
- When a runtime error occurs, search the codebase for related configuration files before proposing fixes
- If something is ambiguous, grep the codebase first before asking the user for clarification
- Never claim you cannot perform an action (like opening a browser) without first checking available tools

## FRED API Compliance

This app is for **educational and personal use only** per FRED Terms of Use. The app must never cache or store FRED data server-side, and every user must supply their own free API key. Compliance disclaimers are rendered via `_disclaimer_banner()` at both top and bottom of the page — do not remove them.
