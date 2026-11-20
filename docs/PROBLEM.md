# Problem Analysis and Planning

## 1. Problem Statement

College students frequently struggle to manage personal finances. Pocket money and monthly allowances are quickly depleted on daily expenses such as mess meals, canteens, auto rides, printing, and online orders. Because expenses are paid across multiple channels (UPI, cash, cards), students typically do not log transactions as they happen. They notice overspending only at the end of the month when funds run dry. Manual expense tracking apps fail because manual entry creates too much friction.

## 2. Project Objectives

SpendSense is designed to reduce tracking friction through automation and machine learning:
1. Automatically assign expense categories from plain text descriptions and merchant names.
2. Parse unstructured bank and UPI transaction SMS alerts into structured records.
3. Import historical bank statements from CSV files.
4. Forecast spending for the upcoming month to warn users early.
5. Identify anomalous transactions that deviate significantly from category norms.
6. Provide natural language summaries of spending records.

## 3. The Three Machine Learning Tasks

### Task 1: Category Classification (Supervised NLP)
- **Goal**: Predict one of 7 expense categories (`food`, `travel`, `education`, `entertainment`, `bills`, `health`, `shopping`) from short merchant or transaction text.
- **Approach**: Character and word n-gram TF-IDF representations evaluated across Naive Bayes, Logistic Regression, and Linear Support Vector Classifiers. An optional sentence embedding pipeline will be evaluated for generalization on unseen vendors.
- **Evaluation**: Stratified cross-validation accuracy, precision, recall, confusion matrix, and out-of-distribution evaluation on unseen merchant names.

### Task 2: Spending Forecasting (Time Series Regression)
- **Goal**: Forecast total and per-category spending for the upcoming month using historical monthly totals.
- **Approach**: Compare five models: naive last month baseline, three-month moving average, linear regression on time index, Holt exponential smoothing, and gradient boosting on lag features.
- **Evaluation**: Rolling backtest over the last four months evaluated with Mean Absolute Error (MAE).

### Task 3: Anomaly Detection (Unsupervised Learning)
- **Goal**: Flag transactions that are unusually high for their specific category.
- **Approach**: Per-category Isolation Forest models fitted on log-transformed transaction amounts.
- **Evaluation**: Precision and recall against synthetic history with planted high-value outliers.

## 4. Data Sources

1. **Synthetic Labeled Dataset**: Text descriptions generated from category-specific vendors and item templates with injected typographical errors and case variations.
2. **Hard Evaluation Dataset**: 42 hand-written real-world transactions containing vendors not present in training data to evaluate zero-shot generalizability.
3. **Synthetic Monthly History**: 12 months of synthetic student transactions simulating upward trend and seasonal spending.
4. **User Feedback**: User corrections stored in a local CSV to periodically retrain models.

## 5. Technology Stack

- **Language**: Python 3.12
- **Core Libraries**: pandas, numpy, scikit-learn, joblib
- **Database**: SQLite (built-in `sqlite3`)
- **Backend API**: FastAPI, uvicorn, pydantic
- **Dashboard**: Streamlit, matplotlib
- **HTTP Clients**: requests, httpx
- **Testing**: Python `unittest` framework
- **Containerization**: Docker, docker-compose

## 6. Technical Risks and Mitigations

1. **Vocabulary Mismatch**: Students use informal slang, regional phrasing, or UPI handles. Mitigation: character n-grams and feedback-based retraining.
2. **Short Time Series**: With only 12 months of monthly aggregated data, complex time series models overfit. Mitigation: rely on simple linear and smoothing baselines with rolling backtest validation.
3. **SMS Format Diversity**: Indian banks use diverse SMS formats for debits, credits, and UPI references. Mitigation: conservative regex patterns that require explicit amount matches and return None when uncertain.
4. **SQL Injection via Natural Language**: Asking freeform questions must not generate raw SQL strings. Mitigation: map questions to a strict, whitelisted parameter specification executed directly in pandas.
