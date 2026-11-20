"""Monthly spending forecast using rolling backtesting across five models."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def monthly_totals(df, include_current=False):
    """Aggregates expenses into monthly totals. Drops current month if incomplete."""
    if df.empty:
        return pd.Series(dtype=float)
    if not pd.api.types.is_datetime64_any_dtype(df["spent_on"]):
        df = df.copy()
        df["spent_on"] = pd.to_datetime(df["spent_on"])

    monthly = (
        df.groupby(df["spent_on"].dt.to_period("M"))["amount"]
        .sum()
        .sort_index()
    )
    current_period = pd.Timestamp.today().to_period("M")
    if not include_current and len(monthly) and monthly.index[-1] == current_period:
        monthly = monthly.iloc[:-1]
    return monthly


def model_last_month(series):
    return float(series[-1])


def model_moving_avg_3(series):
    return float(np.mean(series[-3:]))


def model_linear_trend(series):
    x = np.arange(len(series)).reshape(-1, 1)
    reg = LinearRegression().fit(x, series)
    pred = reg.predict([[len(series)]])[0]
    return float(pred)


def model_holt(series):
    """Holt linear exponential smoothing with grid search over alpha and beta."""
    best_tuple = None
    for alpha in (0.2, 0.4, 0.6, 0.8):
        for beta in (0.1, 0.3, 0.5):
            level = series[0]
            trend = series[1] - series[0]
            sse = 0.0
            for val in series[1:]:
                forecast_val = level + trend
                sse += (val - forecast_val) ** 2
                new_level = alpha * val + (1 - alpha) * (level + trend)
                trend = beta * (new_level - level) + (1 - beta) * trend
                level = new_level
            if best_tuple is None or sse < best_tuple[0]:
                best_tuple = (sse, level, trend)
    return float(best_tuple[1] + best_tuple[2])


def model_gradient_boosting(series, lags=2):
    """Gradient boosting on month-to-month first differences."""
    if len(series) < 8:
        raise ValueError("Series length must be at least 8 for gradient boosting with 2 lags.")
    diffs = np.diff(series)
    x = np.array([diffs[i - lags:i] for i in range(lags, len(diffs))])
    y = diffs[lags:]
    model = GradientBoostingRegressor(
        n_estimators=50, max_depth=2, random_state=42
    )
    model.fit(x, y)
    next_diff = model.predict([diffs[-lags:]])[0]
    return float(series[-1] + next_diff)


FORECAST_MODELS = {
    "last_month": model_last_month,
    "moving_avg_3": model_moving_avg_3,
    "linear_trend": model_linear_trend,
    "holt": model_holt,
    "gradient_boosting": model_gradient_boosting,
}


def backtest(y, folds=4):
    """Calculates Mean Absolute Error (MAE) for each model across recent folds."""
    y = np.asarray(y, dtype=float)
    if len(y) < 6:
        return {}
    folds = max(1, min(folds, len(y) - 6))
    errors = {name: [] for name in FORECAST_MODELS}

    for idx in range(len(y) - folds, len(y)):
        history = y[:idx]
        actual = y[idx]
        for name, fn in FORECAST_MODELS.items():
            try:
                pred = fn(history)
                errors[name].append(abs(pred - actual))
            except ValueError:
                errors[name].append(np.nan)

    mae_scores = {}
    for name, err_list in errors.items():
        if np.isnan(err_list).any():
            mae_scores[name] = None
        else:
            mae_scores[name] = float(np.mean(err_list))
    return mae_scores


def forecast_next_month(df):
    """Selects best model via rolling backtest and forecasts overall spending."""
    monthly = monthly_totals(df)
    if len(monthly) < 6:
        return None

    y = monthly.values.astype(float)
    bt = backtest(y, folds=4)
    valid_scores = {k: v for k, v in bt.items() if v is not None}
    if not valid_scores:
        return None

    best_model_name = min(valid_scores, key=valid_scores.get)
    prediction = max(0.0, FORECAST_MODELS[best_model_name](y))

    return {
        "next_month": str(monthly.index[-1] + 1),
        "prediction": prediction,
        "history": monthly,
        "model": best_model_name,
        "backtest": bt,
        "mae_model": valid_scores[best_model_name],
        "mae_baseline": bt.get("moving_avg_3"),
    }


def forecast_by_category(df):
    """Forecasts next month spending for each category using linear regression."""
    result = {}
    for cat, group in df.groupby("category"):
        m = monthly_totals(group)
        if len(m) >= 3:
            pred = max(0.0, model_linear_trend(m.values.astype(float)))
            result[cat] = pred
    return dict(sorted(result.items(), key=lambda item: -item[1]))
