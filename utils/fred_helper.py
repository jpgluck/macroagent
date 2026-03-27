"""
fred_helper.py
--------------
Helper class for all Federal Reserve Economic Data (FRED) API interactions.

IMPORTANT COMPLIANCE NOTE:
  This module accesses the FRED API strictly for educational and personal use
  in accordance with the Federal Reserve Bank of St. Louis Terms of Use.
  See: https://fred.stlouisfed.org/about/terms

  - Users must supply their own free API key.
  - Data retrieved must NOT be used for commercial purposes, AI training,
    or redistribution.
  - This module is part of a Rutgers student educational project.
"""

import time

import pandas as pd
from fredapi import Fred
from sklearn.linear_model import LinearRegression


class FredHelper:
    """
    Encapsulates all interactions with the FRED (Federal Reserve Economic Data)
    API for the MacroAgent Forecaster application.

    Evaluates a catalogue of 10 macro indicators, automatically selects the
    ones most correlated with the user's demand data, and uses them as
    Prophet regressors for forecasting.
    """

    # Kept for backwards compatibility — used by the FRED connection probe
    GDP_SERIES_ID = "A191RL1Q225SBEA"
    CPI_SERIES_ID = "CPIAUCSL"

    # Full catalogue of candidate indicators
    # Format: name -> (FRED series ID, human label, frequency, transform)
    #   frequency: "quarterly" | "monthly" | "daily"
    #   transform: "level" (use as-is) | "yoy_pct" (compute YoY % change)
    INDICATOR_CATALOGUE = {
        "GDP_growth":         ("A191RL1Q225SBEA", "Real GDP Growth Rate",        "quarterly", "level"),
        "inflation":          ("CPIAUCSL",        "CPI Inflation (YoY %)",       "monthly",   "yoy_pct"),
        "unemployment":       ("UNRATE",          "Unemployment Rate",            "monthly",   "level"),
        "fed_funds":          ("FEDFUNDS",        "Federal Funds Rate",           "monthly",   "level"),
        "consumer_sentiment": ("UMCSENT",         "Consumer Sentiment (UMich)",   "monthly",   "level"),
        "treasury_10y":       ("DGS10",           "10-Year Treasury Yield",       "daily",     "level"),
        "industrial_prod":    ("INDPRO",          "Industrial Production (YoY %)", "monthly",  "yoy_pct"),
        "housing_starts":     ("HOUST",           "Housing Starts (thousands)",   "monthly",   "level"),
        "retail_sales":       ("RSXFS",           "Retail Sales ex. Food (YoY %)", "monthly",  "yoy_pct"),
        "sp500":              ("SP500",           "S&P 500 Index (YoY %)",        "daily",     "yoy_pct"),
    }

    # Scenario slider config per indicator: (min, max, default, step, unit_label, help_text)
    SCENARIO_DEFAULTS = {
        "GDP_growth":         (-5.0,  10.0,   2.0,   0.1,  "%",
                               "Annual real GDP growth rate. US average ~2–3%. Negative = recession."),
        "inflation":          (-2.0,  15.0,   3.0,   0.1,  "%",
                               "Year-over-year CPI inflation. Fed target is 2%. 2022 peak was ~9%."),
        "unemployment":       ( 2.0,  15.0,   4.5,   0.1,  "%",
                               "US unemployment rate. ~4% is considered full employment. 2020 peak was ~15%."),
        "fed_funds":          ( 0.0,  10.0,   5.0,   0.25, "%",
                               "Federal Funds Rate set by the Fed. Affects borrowing costs economy-wide."),
        "consumer_sentiment": (40.0, 120.0,  70.0,   1.0,  "index",
                               "UMich Consumer Sentiment index. 100 = historical average. Below 70 = pessimism."),
        "treasury_10y":       ( 0.5,   8.0,   4.5,   0.1,  "%",
                               "10-Year US Treasury yield. Proxy for long-term borrowing costs."),
        "industrial_prod":    (-10.0, 10.0,   2.0,   0.1,  "% YoY",
                               "Year-over-year % change in US industrial output."),
        "housing_starts":     (500.0, 2000.0, 1400.0, 50.0, "K units",
                               "Monthly housing starts in thousands. ~1,400K is a normal pace."),
        "retail_sales":       (-10.0, 15.0,   4.0,   0.1,  "% YoY",
                               "Year-over-year % change in retail sales excluding food services."),
        "sp500":              (-30.0, 30.0,  10.0,   0.5,  "% YoY",
                               "Year-over-year % change in the S&P 500 index. Proxy for market sentiment."),
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.fred = Fred(api_key=api_key)

    # ------------------------------------------------------------------
    # Legacy methods — kept for backward compatibility
    # ------------------------------------------------------------------

    def get_gdp(self) -> pd.Series:
        """Fetch Real GDP Growth Rate. Used by the FRED connection probe."""
        return self.fetch_indicator("GDP_growth")

    def get_inflation(self) -> pd.Series:
        """Fetch CPI inflation (YoY %). Legacy method."""
        return self.fetch_indicator("inflation")

    # ------------------------------------------------------------------
    # Generic indicator fetching
    # ------------------------------------------------------------------

    def fetch_indicator(self, name: str) -> pd.Series:
        """
        Fetch a single indicator from FRED by catalogue name, apply its
        transform, and return a monthly pd.Series indexed by month-start.
        """
        series_id, label, freq, transform = self.INDICATOR_CATALOGUE[name]

        time.sleep(0.5)  # polite rate limiting — FRED allows 120 req/min
        try:
            raw = self.fred.get_series(series_id)
        except Exception as exc:
            _handle_fred_error(exc, label)

        raw.index = pd.to_datetime(raw.index)
        raw = raw.dropna()

        # Apply transform
        if transform == "yoy_pct":
            periods = 4 if freq == "quarterly" else 12
            raw = raw.pct_change(periods=periods) * 100
            raw = raw.dropna()

        # Resample to monthly month-start
        if freq == "quarterly":
            monthly = raw.resample("MS").ffill()
        else:  # monthly or daily
            monthly = raw.resample("MS").last()

        monthly = monthly.dropna()
        monthly.name = name
        return monthly

    def fetch_all_indicators(self) -> dict:
        """
        Fetch all 10 catalogue indicators from FRED, skipping any that fail.
        Returns dict of {name: pd.Series}.
        """
        results = {}
        for name in self.INDICATOR_CATALOGUE:
            try:
                results[name] = self.fetch_indicator(name)
            except Exception:
                pass  # skip unavailable series silently
        return results

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def align_all_indicators(
        self, company_df: pd.DataFrame, indicators: dict
    ) -> pd.DataFrame:
        """
        Merge company demand data with all successfully-fetched indicators
        on month-start dates.

        Parameters
        ----------
        company_df : pd.DataFrame
            Must contain 'ds' (datetime) and 'y' (float) columns.
        indicators : dict
            {name: pd.Series} from fetch_all_indicators().

        Returns
        -------
        pd.DataFrame with columns: ds, y, <indicator_name>, ...
        """
        company = company_df.copy()
        company["ds"] = (
            pd.to_datetime(company["ds"]).dt.to_period("M").dt.to_timestamp()
        )

        macro_df = pd.DataFrame(indicators)
        macro_df.index.name = "ds"
        macro_df = macro_df.reset_index()
        macro_df["ds"] = pd.to_datetime(macro_df["ds"])

        merged = pd.merge(company, macro_df, on="ds", how="inner")
        merged = merged.sort_values("ds").reset_index(drop=True)
        return merged

    def align_with_company_data(self, company_df: pd.DataFrame) -> pd.DataFrame:
        """Legacy wrapper — fetches only GDP + inflation. Used by old code paths."""
        indicators = {
            "GDP_growth": self.fetch_indicator("GDP_growth"),
            "inflation":  self.fetch_indicator("inflation"),
        }
        return self.align_all_indicators(company_df, indicators)

    # ------------------------------------------------------------------
    # Ranking & selection (core agent logic)
    # ------------------------------------------------------------------

    def rank_indicators(self, merged_df: pd.DataFrame, top_n: int = 3) -> dict:
        """
        Rank all available indicators by absolute Pearson r with detrended
        demand (YoY % growth). Select the top N as Prophet regressors.
        Also computes joint OLS R² using the selected indicators together.

        Parameters
        ----------
        merged_df : pd.DataFrame
            Output of align_all_indicators().
        top_n : int
            Number of indicators to select.

        Returns
        -------
        dict with keys:
          all_rankings   – full sorted list of {name, label, corr} dicts
          selected       – top N picks (same format)
          selected_names – list of indicator name strings
          r_squared      – joint OLS R² of selected indicators → demand_yoy
          coefficients   – {name: OLS coefficient} for selected indicators
        """
        df = merged_df.copy()
        df["demand_yoy"] = df["y"].pct_change(periods=12) * 100
        df = df.dropna(subset=["demand_yoy"])

        indicator_names = [
            c for c in df.columns if c not in ("ds", "y", "demand_yoy")
        ]

        rankings = []
        for name in indicator_names:
            col_data = df[["demand_yoy", name]].dropna()
            if len(col_data) < 12:
                continue
            r = col_data["demand_yoy"].corr(col_data[name])
            if pd.isna(r):
                continue
            label = self.INDICATOR_CATALOGUE.get(name, (None, name, None, None))[1]
            rankings.append({
                "name":  name,
                "label": label,
                "corr":  round(float(r), 4),
            })

        rankings.sort(key=lambda x: abs(x["corr"]), reverse=True)

        selected = rankings[:top_n]
        selected_names = [s["name"] for s in selected]

        # Joint OLS R² with selected indicators
        r_squared = 0.0
        coefficients = {}
        if selected_names:
            reg_df = df[["demand_yoy"] + selected_names].dropna()
            if len(reg_df) > len(selected_names) + 1:
                X = reg_df[selected_names].values
                y_vals = reg_df["demand_yoy"].values
                lm = LinearRegression()
                lm.fit(X, y_vals)
                r_squared = round(float(lm.score(X, y_vals)), 4)
                for n, coef in zip(selected_names, lm.coef_):
                    coefficients[n] = round(float(coef), 4)

        return {
            "all_rankings":   rankings,
            "selected":       selected,
            "selected_names": selected_names,
            "r_squared":      r_squared,
            "coefficients":   coefficients,
        }

    # ------------------------------------------------------------------
    # Future regressor injection (generic)
    # ------------------------------------------------------------------

    def prepare_future_regressors(
        self,
        future_df: pd.DataFrame,
        scenario_values: dict = None,
        # Legacy keyword args kept for backward compatibility
        gdp_scenario: float = None,
        inf_scenario: float = None,
    ) -> pd.DataFrame:
        """
        Inject scenario values into Prophet's future DataFrame for the
        forecast horizon. Preserves historical actual values where present.

        Parameters
        ----------
        future_df : pd.DataFrame
            Prophet future DataFrame.
        scenario_values : dict
            {indicator_name: scenario_float} for all selected indicators.
        gdp_scenario, inf_scenario : float (legacy)
            Kept for backwards compatibility.
        """
        future = future_df.copy()

        # Build from legacy params if scenario_values not provided
        if scenario_values is None:
            scenario_values = {}
            if gdp_scenario is not None:
                scenario_values["GDP_growth"] = gdp_scenario
            if inf_scenario is not None:
                scenario_values["inflation"] = inf_scenario

        for name, value in scenario_values.items():
            if name not in future.columns:
                future[name] = value
            else:
                future[name] = future[name].fillna(value)

        return future


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _handle_fred_error(exc: Exception, series_name: str):
    """
    Convert raw FRED API exceptions into user-friendly error messages
    and raise a clean RuntimeError so callers can handle it gracefully.
    """
    msg = str(exc).lower()
    if "api_key" in msg or "bad request" in msg or "400" in msg:
        friendly = (
            f"Invalid FRED API key — could not fetch {series_name}. "
            "Please double-check your key at https://fredaccount.stlouisfed.org/apikeys"
        )
    elif "429" in msg or "rate" in msg:
        friendly = (
            f"FRED API rate limit reached while fetching {series_name}. "
            "Please wait a moment and try again."
        )
    elif "connection" in msg or "timeout" in msg:
        friendly = (
            f"Network error while contacting FRED API for {series_name}. "
            "Please check your internet connection and retry."
        )
    else:
        friendly = (
            f"Unexpected error fetching {series_name} from FRED: {exc}. "
            "Please try again or check https://fred.stlouisfed.org for outages."
        )
    raise RuntimeError(friendly)
