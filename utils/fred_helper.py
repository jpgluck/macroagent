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
import numpy as np
import streamlit as st
from fredapi import Fred
from sklearn.linear_model import LinearRegression


class FredHelper:
    """
    Encapsulates all interactions with the FRED (Federal Reserve Economic Data)
    API for the MacroAgent Forecaster application.

    Retrieves GDP growth and inflation data, aligns it with user-provided
    company demand data, and computes macro correlations and regression weights.
    """

    # FRED series identifiers used in this app
    GDP_SERIES_ID = "A191RL1Q225SBEA"   # Real GDP Growth Rate (quarterly, % change)
    CPI_SERIES_ID = "CPIAUCSL"           # Consumer Price Index for All Urban Consumers (monthly)

    def __init__(self, api_key: str):
        """
        Initialise the FRED client.

        Parameters
        ----------
        api_key : str
            A valid, free FRED API key obtained from
            https://fredaccount.stlouisfed.org/apikeys
        """
        self.api_key = api_key
        self.fred = Fred(api_key=api_key)

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def get_gdp(self) -> pd.Series:
        """
        Fetch the Real GDP Growth Rate (annualised quarterly, % change)
        from FRED series A191RL1Q225SBEA.

        Returns
        -------
        pd.Series
            Indexed by date (quarterly), values are % growth rates.
        """
        try:
            gdp = self.fred.get_series(self.GDP_SERIES_ID)
            gdp.name = "GDP_growth"
            gdp.index = pd.to_datetime(gdp.index)
            return gdp.dropna()
        except Exception as exc:
            _handle_fred_error(exc, "GDP growth")

    def get_inflation(self) -> pd.Series:
        """
        Fetch CPI (CPIAUCSL) and compute Year-over-Year % change to proxy
        annual inflation rate.

        Returns
        -------
        pd.Series
            Monthly YoY % change in CPI (inflation proxy).
        """
        try:
            # Polite sleep to avoid hammering the API in rapid-fire calls
            time.sleep(0.25)
            cpi = self.fred.get_series(self.CPI_SERIES_ID)
            cpi.index = pd.to_datetime(cpi.index)
            cpi = cpi.dropna()
            # YoY % change: (current / 12-months-ago - 1) * 100
            inflation = cpi.pct_change(periods=12) * 100
            inflation.name = "inflation"
            return inflation.dropna()
        except Exception as exc:
            _handle_fred_error(exc, "inflation (CPI)")

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def align_with_company_data(self, company_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge FRED macro data with user-supplied company demand data.

        Both series are resampled to monthly frequency (month-start) before
        merging so that quarterly GDP and monthly CPI align cleanly with the
        user's monthly demand column.

        Parameters
        ----------
        company_df : pd.DataFrame
            Must contain a datetime column 'ds' and a numeric column 'y'
            (already renamed by app.py before this call).

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with columns: ds, y, GDP_growth, inflation.
            Rows with any NaN in macro columns are dropped.
        """
        gdp_raw = self.get_gdp()
        inflation_raw = self.get_inflation()

        # --- Resample GDP to monthly (forward-fill from quarterly) ----------
        gdp_monthly = (
            gdp_raw
            .resample("MS")          # month-start
            .ffill()                 # forward-fill quarters to months
        )

        # --- Inflation is already monthly; normalise index -------------------
        inflation_monthly = inflation_raw.resample("MS").last()

        # --- Build macro DataFrame ------------------------------------------
        macro_df = pd.DataFrame({
            "GDP_growth": gdp_monthly,
            "inflation":  inflation_monthly,
        })
        macro_df.index.name = "ds"
        macro_df = macro_df.reset_index()
        macro_df["ds"] = pd.to_datetime(macro_df["ds"])

        # --- Merge on month-start date --------------------------------------
        company_df = company_df.copy()
        company_df["ds"] = pd.to_datetime(company_df["ds"]).dt.to_period("M").dt.to_timestamp()

        merged = pd.merge(company_df, macro_df, on="ds", how="inner")
        merged = merged.dropna(subset=["GDP_growth", "inflation"])
        merged = merged.sort_values("ds").reset_index(drop=True)

        return merged

    # ------------------------------------------------------------------
    # Correlation & regression
    # ------------------------------------------------------------------

    def calculate_correlations_and_weights(self, merged_df: pd.DataFrame) -> dict:
        """
        Compute Pearson correlations and OLS regression coefficients between
        demand (y) and the two macro regressors (GDP_growth, inflation).

        Parameters
        ----------
        merged_df : pd.DataFrame
            Output of align_with_company_data().

        Returns
        -------
        dict with keys:
          corr_gdp        – Pearson r between y and GDP_growth
          corr_inflation  – Pearson r between y and inflation
          gdp_coef        – OLS coefficient for GDP_growth
          inf_coef        – OLS coefficient for inflation
          intercept       – OLS intercept
          r_squared       – R² of the joint regression
        """
        df = merged_df[["y", "GDP_growth", "inflation"]].dropna()

        # Pearson correlations
        corr_gdp = df["y"].corr(df["GDP_growth"])
        corr_inf = df["y"].corr(df["inflation"])

        # Linear regression: demand ~ GDP_growth + inflation
        X = df[["GDP_growth", "inflation"]].values
        y = df["y"].values
        model = LinearRegression()
        model.fit(X, y)

        return {
            "corr_gdp":       round(float(corr_gdp), 4),
            "corr_inflation": round(float(corr_inf), 4),
            "gdp_coef":       round(float(model.coef_[0]), 4),
            "inf_coef":       round(float(model.coef_[1]), 4),
            "intercept":      round(float(model.intercept_), 4),
            "r_squared":      round(float(model.score(X, y)), 4),
        }

    # ------------------------------------------------------------------
    # Future regressor injection
    # ------------------------------------------------------------------

    def prepare_future_regressors(
        self,
        future_df: pd.DataFrame,
        gdp_scenario: float,
        inf_scenario: float,
    ) -> pd.DataFrame:
        """
        Inject user-chosen what-if scenario values into Prophet's future
        DataFrame so macro regressors are available for the forecast period.

        For historical dates already in the training data the actual FRED
        values are preserved; for future dates the scenario constants are used.

        Parameters
        ----------
        future_df : pd.DataFrame
            Prophet future DataFrame (columns: ds, and optionally existing
            GDP_growth / inflation columns from training data).
        gdp_scenario : float
            Expected annual GDP growth rate (%) for the forecast horizon.
        inf_scenario : float
            Expected annual inflation rate (%) for the forecast horizon.

        Returns
        -------
        pd.DataFrame
            future_df with GDP_growth and inflation columns fully populated.
        """
        future = future_df.copy()

        # Fill any missing macro values with the scenario constants
        if "GDP_growth" not in future.columns:
            future["GDP_growth"] = gdp_scenario
        else:
            future["GDP_growth"] = future["GDP_growth"].fillna(gdp_scenario)

        if "inflation" not in future.columns:
            future["inflation"] = inf_scenario
        else:
            future["inflation"] = future["inflation"].fillna(inf_scenario)

        return future


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _handle_fred_error(exc: Exception, series_name: str):
    """
    Convert raw FRED API exceptions into user-friendly Streamlit error messages
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
