"""Unsupervised anomaly detection for expenses using Isolation Forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df, contamination=0.03, min_samples=20):
    """Fits an Isolation Forest per category on log(1 + amount) and returns high outliers."""
    if df.empty:
        return df.copy()

    df = df.reset_index(drop=True)
    flagged_dfs = []

    for category, group in df.groupby("category"):
        if len(group) < min_samples:
            continue
        model = IsolationForest(contamination=contamination, random_state=42)
        log_amounts = np.log1p(group[["amount"]].values)
        predictions = model.fit_predict(log_amounts)

        # Flagged by Isolation Forest is -1; filter only those strictly above category median
        is_outlier = predictions == -1
        is_above_median = group["amount"] > group["amount"].median()
        flagged_dfs.append(group[is_outlier & is_above_median])

    if not flagged_dfs:
        return df.iloc[0:0]

    return pd.concat(flagged_dfs).sort_values("amount", ascending=False)
