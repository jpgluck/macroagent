import pandas as pd
from prophet import Prophet

from utils.fred_helper import FredHelper


def run_forecast(
    merged: pd.DataFrame,
    sel_names: list,
    scenario_vals: dict,
    trend_flexibility: float,
    helper: FredHelper,
) -> tuple[pd.DataFrame, Prophet, pd.DataFrame]:
    """
    Train a Prophet model with macro regressors and generate a 12-month forecast.

    Returns (forecast_df, model, train_df).
    """
    # Build training DataFrame
    train_df = merged[["ds", "y"] + sel_names].copy()

    # Initialise and train Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=trend_flexibility,
    )
    for name in sel_names:
        model.add_regressor(name)
    model.fit(train_df)

    # Build future DataFrame (12 months)
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

    # Generate forecast
    forecast = model.predict(future)

    return forecast, model, train_df


def run_segment_forecast(
    merged: pd.DataFrame,
    segment_rankings: dict,
    scenario_vals: dict,
    trend_flexibility: float,
    helper: FredHelper,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Train a Prophet model per segment, each with its own best indicators.

    Parameters
    ----------
    merged : pd.DataFrame
        Output of align_all_indicators() that still contains a 'segment' column.
    segment_rankings : dict
        Output of rank_indicators_by_segment().
    scenario_vals : dict
        {indicator_name: float} what-if scenario values (shared across segments).
    trend_flexibility : float
        Prophet changepoint_prior_scale passed to run_forecast.
    helper : FredHelper
        Used for prepare_future_regressors inside run_forecast.

    Returns
    -------
    combined_forecast : pd.DataFrame
        ds, yhat, yhat_lower, yhat_upper summed across segments, plus per-segment
        yhat_{seg} columns for stacked charting.
    segment_results : dict
        {segment: {"forecast": df, "train_df": df, "sel_names": list}}
    segment_models : dict
        {segment: Prophet model}
    """
    segment_results: dict = {}
    segment_models: dict = {}

    for seg in segment_rankings["segments"]:
        seg_data = merged[merged["segment"] == seg].copy()
        seg_data = seg_data.drop(columns=["segment"])
        ranking = segment_rankings["segment_rankings"][seg]
        sel_names = ranking["selected_names"]

        if not sel_names:
            continue

        # Filter scenario_vals to only this segment's indicators
        seg_scenario = {k: v for k, v in scenario_vals.items() if k in sel_names}

        forecast, model, train_df = run_forecast(
            merged=seg_data,
            sel_names=sel_names,
            scenario_vals=seg_scenario,
            trend_flexibility=trend_flexibility,
            helper=helper,
        )

        segment_results[seg] = {
            "forecast":  forecast,
            "train_df":  train_df,
            "sel_names": sel_names,
        }
        segment_models[seg] = model

    if not segment_results:
        raise RuntimeError("No segments could be forecast.")

    # Combine: sum yhat / yhat_lower / yhat_upper across segments
    all_forecasts = []
    for seg, res in segment_results.items():
        fc = res["forecast"][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        fc = fc.rename(columns={
            "yhat":       f"yhat_{seg}",
            "yhat_lower": f"yhat_lower_{seg}",
            "yhat_upper": f"yhat_upper_{seg}",
        })
        all_forecasts.append(fc)

    combined = all_forecasts[0]
    for fc in all_forecasts[1:]:
        combined = combined.merge(fc, on="ds", how="outer")

    # Sum across segments (NaN → 0 so partial overlap doesn't inflate)
    yhat_cols = [
        c for c in combined.columns
        if c.startswith("yhat_")
        and not c.startswith("yhat_lower_")
        and not c.startswith("yhat_upper_")
    ]
    lower_cols = [c for c in combined.columns if c.startswith("yhat_lower_")]
    upper_cols = [c for c in combined.columns if c.startswith("yhat_upper_")]

    combined["yhat"]       = combined[yhat_cols].fillna(0).sum(axis=1)
    combined["yhat_lower"] = combined[lower_cols].fillna(0).sum(axis=1)
    combined["yhat_upper"] = combined[upper_cols].fillna(0).sum(axis=1)

    combined = combined.sort_values("ds").reset_index(drop=True)

    return combined, segment_results, segment_models
