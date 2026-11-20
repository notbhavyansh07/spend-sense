# Week 7 Viva Notes: Spending Forecast

### 1. Why do we require a rolling backtest over the last 4 months instead of a simple train/test split?
Time series data possesses temporal ordering, meaning future values depend on historical sequences. Standard randomized train/test splits cause data leakage by training on future observations to predict past ones. A rolling backtest evaluates each fold by training strictly on observations prior to month \(t\) to predict month \(t\), accurately mimicking real-world forecasting conditions.

### 2. How does Holt exponential smoothing handle trend compared to simple moving averages?
A simple moving average lags behind upward or downward spending trends because it treats all recent observations with equal weighting and assumes a stationary mean. Holt linear exponential smoothing introduces two distinct equations: a level component updated with smoothing parameter \(\alpha\), and a trend component updated with parameter \(\beta\):
\[ \ell_t = \alpha y_t + (1 - \alpha)(\ell_{t-1} + b_{t-1}) \]
\[ b_t = \beta(\ell_t - \ell_{t-1}) + (1 - \beta)b_{t-1} \]
This explicitly projects the trajectory forward, following compounding expenditure trends.

### 3. Why is Gradient Boosting trained on month-to-month changes (first differences) rather than raw monthly sums?
Tree-based regression algorithms like gradient boosting cannot extrapolate linear trends beyond the minimum and maximum target values observed in the training data. If monthly spending trends upward from Rs 30,000 to Rs 45,000, decision trees cannot predict Rs 50,000. By training on differenced values \(\Delta y_t = y_t - y_{t-1}\) using lag features, the model learns the expected rate of change and projects it onto the latest known value \(y_{t} + \hat{\Delta y}_{t+1}\).

### 4. Why does forecast_next_month return None when fewer than 6 full months of data are available?
Evaluating five distinct models across a 4-month rolling backtest requires sufficient historical depth. If a user has only 2 or 3 months of logs, backtesting would have zero degrees of freedom, and complex models like gradient boosting with lag features would fail. Enforcing a minimum threshold of 6 full months prevents misleading or mathematically degenerate forecasts.

### 5. Why is the current month dropped in monthly_totals?
If a user runs SpendSense halfway through the month (e.g. on the 15th), that month's cumulative spending will appear artificially low (e.g. 50% of normal). If included in time series regression or moving averages, this incomplete month would register as an abrupt false crash in spending, distorting the trend. Dropping the current incomplete month ensures all input series represent complete, comparable billing cycles.

---

### One Thing That Could Break
If a student takes an extended semester break or leaves campus for two months, resulting in months with zero or near-zero recorded expenses followed by sudden resumption, the sudden variance can disrupt Holt smoothing and differenced gradient boosting, causing inflated MAE scores.
