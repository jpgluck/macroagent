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
