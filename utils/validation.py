import numpy as np
import pandas as pd


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


