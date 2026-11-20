# Week 8 Viva Notes: Anomaly Detection

### 1. Why do we fit one Isolation Forest per category rather than a single global model?
Expense amounts vary by orders of magnitude across categories. For example, a normal monthly shopping transaction might be Rs 4,500, whereas a normal canteen meal is Rs 80. A global model would falsely flag almost every legitimate shopping expense as anomalous while failing to detect an unusually high Rs 1,500 food transaction. Training separate Isolation Forests per category establishes category-relative baselines.

### 2. Why is log(1 + amount) applied before training the Isolation Forest?
Financial expenditure distributions exhibit strong positive skewness with long right-hand tails. A linear scale can cause tree partitions to isolate moderately high expenses purely due to skew rather than true anomaly status. The log-transform \(\ln(1 + x)\) compresses the scale and stabilizes variance, allowing the recursive splitting algorithm of Isolation Forest to isolate true distributional deviations more reliably.

### 3. Why are flagged expenses filtered to retain only those strictly above the category median?
Isolation Forest is a non-directional outlier detector: it assigns anomaly scores to isolated points in both extremes of the distribution. Without filtering, an uncharacteristically small expense (such as a Rs 5 test transaction or coupon discount) would also be flagged as anomalous. For an expense tracker, users are concerned with overspending; requiring `amount > median` filters out small-value outliers.

### 4. What is the contamination parameter (0.03) and how does it influence sensitivity?
The `contamination` parameter defines the expected proportion of outliers in the dataset. Setting `contamination=0.03` configures the decision threshold such that approximately 3% of observations in each category with the shortest average path lengths across isolation trees are marked as anomalies. Higher contamination increases recall at the cost of more false positives.

### 5. Why do we skip categories with fewer than 20 rows?
Statistical density estimation requires a minimum sample size to form a reliable baseline. In sparse categories with fewer than 20 records, normal variance is indistinguishable from anomalous behavior, leading to unstable decision boundaries. Skipping small categories prevents false alarms during early usage.

---

### One Thing That Could Break
If an expense category consists almost entirely of identical fixed payments (such as a fixed monthly hostel rent of Rs 8,000 every month), the log-transformed amounts have near-zero variance. An Isolation Forest may struggle to determine sensible tree splits, potentially producing erratic anomaly scores for minor variations.
