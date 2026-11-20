"""Data formatting service layer shared between API and Dashboard."""
from . import anomaly, forecast


def df_records(df, limit=None):
    """Converts a DataFrame of expenses into serialized dict records."""
    if df.empty:
        return []
    if limit:
        df = df.head(limit)
    out = df.copy()
    if "spent_on" in out.columns:
        out["spent_on"] = out["spent_on"].dt.strftime("%Y-%m-%d")
    return out.to_dict("records")


def forecast_summary(df):
    """Returns a serialized dictionary of monthly and per-category forecasts."""
    if df.empty:
        return None
    f = forecast.forecast_next_month(df)
    if not f:
        return None
    return {
        "next_month": f["next_month"],
        "prediction": f["prediction"],
        "model": f["model"],
        "backtest": f["backtest"],
        "history": {str(k): float(v) for k, v in f["history"].items()},
        "by_category": forecast.forecast_by_category(df),
    }


def anomaly_records(df):
    """Returns list of serialized anomaly records."""
    if df.empty:
        return []
    flagged = anomaly.detect_anomalies(df)
    return df_records(flagged)
