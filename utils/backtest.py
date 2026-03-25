import numpy as np
import pandas as pd                                              
import streamlit as st                                    
from prophet import Prophet
from utils.fred_helper import FredHelper
from utils.validation import _mape, _mae, _rmse
from utils.charts import _build_backtest_chart 

def _run_backtest(merged, helper, horizon, n_folds, trend_flex, growth_rate_override):
    """Run n_folds expanding-window CV with given horizon.

    Returns a tuple: (dict_with_results, None) on success,
    or (None, "error message string") if insufficient data or all folds fail.

    The result dict has keys: results_df, train_df, fold_metrics, fold_indicators,
    and averaged macro/vanilla/naive MAPE/MAE/RMSE, n_test_months, n_folds.
    """
    min_train_months = 24

    max_date      = merged["ds"].max()
    earliest_date = merged["ds"].min()
    total_months  = int(round((max_date - earliest_date).days / 30.44))

    # Compute how many folds are possible
    max_possible_folds = max(0, (total_months - min_train_months) // horizon)
    actual_folds = min(n_folds, max_possible_folds)

    if actual_folds == 0:
        return (
            None,
            f"Not enough data for backtest with a {horizon}-month horizon. "
            f"Need at least {min_train_months + horizon} months; dataset has ~{total_months} months.",
        )

    cutoff_dates = sorted([
        max_date - pd.DateOffset(months=horizon * (i + 1))
        for i in range(actual_folds)
    ])  # earliest first

    # Drop folds that don't leave at least min_train_months of training data
    cutoff_dates = [
        c for c in cutoff_dates
        if (c - earliest_date).days > min_train_months * 30
    ]

    if len(cutoff_dates) == 0:
        return (
            None,
            f"No valid fold cutoffs found for {horizon}-month horizon. "
            "Upload more historical data.",
        )

    all_fold_results = []
    fold_metrics     = []
    fold_indicators  = []
    train_split      = None  # will hold last fold's train split for chart

    for fold_idx, cutoff_date in enumerate(cutoff_dates):
        # Re-select indicators using ONLY pre-cutoff data (no data leakage)
        tmp_train    = merged[merged["ds"] <= cutoff_date]
        bt_ranking   = helper.rank_indicators(tmp_train, top_n=3)
        bt_sel_names = bt_ranking["selected_names"]

        if not bt_sel_names:
            continue  # skip fold if no indicators found

        # Build fold DataFrames
        full_df = merged[["ds", "y"] + bt_sel_names].copy()
        train_split = full_df[full_df["ds"] <= cutoff_date].copy()
        test_split  = full_df[
            (full_df["ds"] > cutoff_date) &
            (full_df["ds"] <= cutoff_date + pd.DateOffset(months=horizon))
        ].copy()

        if len(test_split) < 2:
            continue

        # --- Macro-augmented Prophet -----------------------------------------
        macro_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
            changepoint_prior_scale=trend_flex,
        )
        for name in bt_sel_names:
            macro_model.add_regressor(name)
        macro_model.fit(train_split)

        future_macro = macro_model.make_future_dataframe(
            periods=len(test_split), freq="MS"
        )
        future_macro = future_macro.merge(
            full_df[["ds"] + bt_sel_names], on="ds", how="left"
        )
        for name in bt_sel_names:
            future_macro[name] = future_macro[name].fillna(
                train_split[name].mean()
            )
        forecast_macro = macro_model.predict(future_macro)

        # --- Vanilla Prophet (no regressors) -----------------------------------
        vanilla_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
            changepoint_prior_scale=trend_flex,
        )
        vanilla_model.fit(train_split[["ds", "y"]])
        future_vanilla = vanilla_model.make_future_dataframe(
            periods=len(test_split), freq="MS"
        )
        forecast_vanilla = vanilla_model.predict(future_vanilla)

        # --- Naive baseline (seasonal naive, growth-adjusted) ------------------
        if growth_rate_override is not None:
            g = growth_rate_override
        else:
            yoy_growth = (
                train_split.set_index("ds")["y"]
                .pct_change(periods=12)
                .dropna()
            )
            g = yoy_growth.median()

        if horizon > 12:
            # Multi-year: tile the last 12 months of training and apply
            # compound growth scaled by how many years ahead each point is.
            seasonal_pattern = train_split.tail(12)["y"].values
            n_repeats = int(np.ceil(len(test_split) / 12))
            tiled = np.tile(seasonal_pattern, n_repeats)[:len(test_split)]
            years_ahead = np.ceil(np.arange(1, len(test_split) + 1) / 12)
            naive_vals = tiled * (1 + g) ** years_ahead
        else:
            # 1-year: simple last-12-months × (1 + g)
            naive_source = train_split.tail(horizon)[["ds", "y"]].copy()
            naive_vals = naive_source["y"].values[:len(test_split)] * (1 + g)

        naive_preds = test_split[["ds"]].copy()
        naive_preds["yhat_naive"] = naive_vals

        # --- Align predictions to hold-out dates -------------------------------
        macro_preds = pd.merge_asof(
            test_split[["ds"]].sort_values("ds"),
            forecast_macro[["ds", "yhat"]].sort_values("ds"),
            on="ds", direction="nearest",
            tolerance=pd.Timedelta(days=15),
        ).rename(columns={"yhat": "yhat_macro"})

        vanilla_preds = pd.merge_asof(
            test_split[["ds"]].sort_values("ds"),
            forecast_vanilla[["ds", "yhat"]].sort_values("ds"),
            on="ds", direction="nearest",
            tolerance=pd.Timedelta(days=15),
        ).rename(columns={"yhat": "yhat_vanilla"})

        results = (
            test_split[["ds", "y"]]
            .merge(macro_preds,   on="ds")
            .merge(vanilla_preds, on="ds")
            .merge(naive_preds,   on="ds")
        )

        if len(results) < 2:
            continue

        a   = results["y"].values
        mp  = results["yhat_macro"].values
        vp  = results["yhat_vanilla"].values
        np_ = results["yhat_naive"].values

        all_fold_results.append(results)
        fold_metrics.append({
            "fold":          fold_idx + 1,
            "cutoff":        cutoff_date.strftime("%b %Y"),
            "n_months":      len(results),
            "macro_mape":    _mape(a, mp),
            "macro_mae":     _mae(a, mp),
            "macro_rmse":    _rmse(a, mp),
            "vanilla_mape":  _mape(a, vp),
            "vanilla_mae":   _mae(a, vp),
            "vanilla_rmse":  _rmse(a, vp),
            "naive_mape":    _mape(a, np_),
            "naive_mae":     _mae(a, np_),
            "naive_rmse":    _rmse(a, np_),
        })
        fold_indicators.append(bt_sel_names)

    if not fold_metrics:
        return (
            None,
            "All folds failed — could not align predictions to hold-out dates. "
            "Try uploading data with dates on the 1st of each month.",
        )

    combined_results = pd.concat(all_fold_results, ignore_index=True)

    avg_metrics = {
        "macro_mape":    np.mean([f["macro_mape"]   for f in fold_metrics]),
        "macro_mae":     np.mean([f["macro_mae"]    for f in fold_metrics]),
        "macro_rmse":    np.mean([f["macro_rmse"]   for f in fold_metrics]),
        "vanilla_mape":  np.mean([f["vanilla_mape"] for f in fold_metrics]),
        "vanilla_mae":   np.mean([f["vanilla_mae"]  for f in fold_metrics]),
        "vanilla_rmse":  np.mean([f["vanilla_rmse"] for f in fold_metrics]),
        "naive_mape":    np.mean([f["naive_mape"]   for f in fold_metrics]),
        "naive_mae":     np.mean([f["naive_mae"]    for f in fold_metrics]),
        "naive_rmse":    np.mean([f["naive_rmse"]   for f in fold_metrics]),
        "n_test_months": sum(f["n_months"] for f in fold_metrics),
        "n_folds":       len(fold_metrics),
    }

    bt_result = {
        "results_df":      combined_results,
        "train_df":        train_split,
        "fold_metrics":    fold_metrics,
        "fold_indicators": fold_indicators,
        **avg_metrics,
    }
    return (bt_result, None)


def _display_backtest_results(bt, label="1-Year"):
    """Render backtest metrics, charts, and insights for a completed backtest."""
    results_df  = bt["results_df"]
    train_df_bt = bt["train_df"]

    # Rank models by MAPE (lower is better)
    model_mapes = [
        ("Macro-Augmented Prophet", bt["macro_mape"]),
        ("Vanilla Prophet",        bt["vanilla_mape"]),
        ("Growth-Adjusted Naive",   bt["naive_mape"]),
    ]
    model_mapes.sort(key=lambda x: x[1])
    best_name, best_mape   = model_mapes[0]
    worst_name, worst_mape = model_mapes[-1]

    st.success(
        f"✅ {label} — {bt['n_folds']}-fold cross-validation complete over "
        f"**{bt['n_test_months']} total hold-out months**.  "
        f"Best model: **{best_name}** ({best_mape:.1f}% MAPE).  "
        f"Worst: {worst_name} ({worst_mape:.1f}% MAPE)."
    )

    # --- Error metrics table -------------------------------------------------
    st.subheader(f"{label} Accuracy Metrics: Three-Way Comparison")
    st.caption(
        f"Averaged across {bt['n_folds']} expanding-window folds "
        f"({bt['n_test_months']} total hold-out months). "
        "Each fold re-selects indicators using only pre-cutoff data (no data leakage).  "
        "The **Growth-Adjusted Naive** applies the training data's median YoY growth rate to last year's actuals — "
        "a much harder baseline to beat for high-growth companies. If a model can't beat this, it isn't adding value beyond trend."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "MAPE — Macro Prophet",
        f"{bt['macro_mape']:.1f}%",
        help="Mean Absolute Percentage Error. Lower is better.",
    )
    m2.metric(
        "MAPE — Vanilla Prophet",
        f"{bt['vanilla_mape']:.1f}%",
        help="Prophet without GDP/inflation regressors.",
    )
    m3.metric(
        "MAPE — Growth-Adj. Naive",
        f"{bt['naive_mape']:.1f}%",
        help="Seasonal naive: repeats the last 12 months of training data. The floor any real model should beat.",
    )

    # Full comparison table
    metrics_table = pd.DataFrame({
        "Metric": ["MAPE (%)", "MAE", "RMSE"],
        "Macro-Augmented Prophet": [
            f"{bt['macro_mape']:.2f}%",
            f"{bt['macro_mae']:,.0f}",
            f"{bt['macro_rmse']:,.0f}",
        ],
        "Vanilla Prophet": [
            f"{bt['vanilla_mape']:.2f}%",
            f"{bt['vanilla_mae']:,.0f}",
            f"{bt['vanilla_rmse']:,.0f}",
        ],
        "Growth-Adjusted Naive": [
            f"{bt['naive_mape']:.2f}%",
            f"{bt['naive_mae']:,.0f}",
            f"{bt['naive_rmse']:,.0f}",
        ],
    })
    st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    # --- Interpretation ------------------------------------------------------
    macro_vs_naive   = bt["naive_mape"] - bt["macro_mape"]
    vanilla_vs_naive = bt["naive_mape"] - bt["vanilla_mape"]
    macro_vs_vanilla = bt["vanilla_mape"] - bt["macro_mape"]

    insights = []
    if macro_vs_naive > 0:
        insights.append(
            f"Macro Prophet beats the growth-adjusted naive by **{macro_vs_naive:.1f} pp** MAPE — "
            f"Prophet is adding real predictive value beyond simple trend extrapolation."
        )
    else:
        insights.append(
            f"Macro Prophet underperforms the growth-adjusted naive by **{abs(macro_vs_naive):.1f} pp** MAPE — "
            f"the model is not beating a simple 'last year × growth rate' forecast. "
            f"The data may be too short, too noisy, or the trend may dominate any macro signal."
        )

    _sel_labels = ", ".join(
        FredHelper.INDICATOR_CATALOGUE[n][1]
        for n in (st.session_state.get("selected_indicators") or [])
    ) or "the selected macro indicators"
    if macro_vs_vanilla > 0:
        insights.append(
            f"Adding **{_sel_labels}** improved accuracy by **{macro_vs_vanilla:.1f} pp** MAPE "
            f"over vanilla Prophet — the agent-selected indicators carry real signal for this dataset."
        )
    elif macro_vs_vanilla < -0.5:
        insights.append(
            f"The selected macro indicators did **not** improve accuracy (vanilla was "
            f"{abs(macro_vs_vanilla):.1f} pp better) — "
            f"this company's demand may be driven more by company-specific factors than macro conditions."
        )
    else:
        insights.append(
            "The agent-selected macro indicators made negligible difference vs. vanilla Prophet for this dataset."
        )

    st.info("**Agent Insight:**  \n" + "  \n".join(insights))

    # --- Per-Fold Breakdown --------------------------------------------------
    with st.expander(f"Per-Fold Breakdown ({bt['n_folds']} folds)"):
        fold_table = pd.DataFrame(bt["fold_metrics"])
        fold_table = fold_table.rename(columns={
            "fold": "Fold", "cutoff": "Cutoff", "n_months": "Months",
            "macro_mape": "Macro MAPE%", "vanilla_mape": "Vanilla MAPE%",
            "naive_mape": "Growth-Adj Naive MAPE%",
        })
        display_cols = ["Fold", "Cutoff", "Months", "Macro MAPE%", "Vanilla MAPE%", "Naive MAPE%"]
        st.dataframe(
            fold_table[[c for c in display_cols if c in fold_table.columns]],
            use_container_width=True, hide_index=True,
        )
        st.caption("Indicators selected per fold (pre-cutoff data only — no leakage):")
        for i, (fm, fi) in enumerate(zip(bt["fold_metrics"], bt["fold_indicators"])):
            labels = [FredHelper.INDICATOR_CATALOGUE[n][1] for n in fi]
            st.caption(f"Fold {fm['fold']} (cutoff {fm['cutoff']}): {', '.join(labels)}")

    # --- Backtest chart ------------------------------------------------------
    st.subheader(f"{label} Forecast vs. Actuals (Hold-Out Period)")
    st.plotly_chart(
        _build_backtest_chart(results_df, train_df_bt),
        use_container_width=True,
    )

    # --- Month-by-month breakdown --------------------------------------------
    with st.expander("Month-by-Month Breakdown"):
        breakdown = results_df.copy()
        breakdown["Error (Macro)"] = (breakdown["yhat_macro"] - breakdown["y"]).map("{:+,.0f}".format)
        breakdown["Error (Vanilla)"] = (breakdown["yhat_vanilla"] - breakdown["y"]).map("{:+,.0f}".format)
        breakdown["Error (Growth-Adj Naive)"] = (breakdown["yhat_naive"] - breakdown["y"]).map("{:+,.0f}".format)
        breakdown["APE (Macro)"] = (
            ((breakdown["yhat_macro"] - breakdown["y"]) / breakdown["y"]).abs() * 100
        ).map("{:.1f}%".format)
        breakdown["APE (Vanilla)"] = (
            ((breakdown["yhat_vanilla"] - breakdown["y"]) / breakdown["y"]).abs() * 100
        ).map("{:.1f}%".format)
        breakdown["APE (Growth-Adj Naive)"] = (
            ((breakdown["yhat_naive"] - breakdown["y"]) / breakdown["y"]).abs() * 100
        ).map("{:.1f}%".format)
        breakdown["ds"] = breakdown["ds"].dt.strftime("%b %Y")
        st.dataframe(
            breakdown.rename(columns={
                "ds": "Month", "y": "Actual",
                "yhat_macro": "Macro Forecast",
                "yhat_vanilla": "Vanilla Forecast",
                "yhat_naive": "Growth-Adj Naive",
            }),
            use_container_width=True,
            hide_index=True,
        )