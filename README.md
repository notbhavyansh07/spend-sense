# SpendSense

SpendSense is an expense tracker with machine learning built in. I developed it as my 2nd-year B.Tech AI/ML specialization project over 12 weekly milestones. It is designed to solve a problem that college students face every month: losing track of daily pocket money across fragmented channels (cash, canteens, UPI, online cards) and realizing they overspent only after their money is gone.

SpendSense provides three automated interfaces: a command line tool (`main.py`), a REST API built with FastAPI (`api.py`), and a web dashboard built with Streamlit (`app.py`).

---

## What the Product Does

1. **Guesses Expense Categories from Text**: Automatically categorizes short descriptions and merchant names (for example, `"zomato dinner"` becomes `food`).
2. **Parses Bank Transaction SMS**: Reads transaction amounts, debit versus credit types, merchant names, UPI handles, and transaction dates directly from bank notification text.
3. **Imports Bank Statement CSVs**: Parses exported bank statements with heuristic column mapping, day-first date normalization, and credit row filtering.
4. **Forecasts Next Month's Budget**: Evaluates five forecasting algorithms using rolling backtests over historical months and selects the lowest-error model to project future spending.
5. **Flags Unusual Expenses**: Trains an Isolation Forest per category on log-transformed amounts to highlight unusually large expenditures above category medians.
6. **Answers Spending Questions in English**: Converts natural language queries into whitelisted query specifications executed with pandas without generating raw SQL.

---

## Machine Learning Results

All figures below are real measurements recorded from running the model pipelines and test suites:

### 1. Category Classification
Trained on a balanced 700-row synthetic dataset (100 rows across 7 categories: `food`, `travel`, `education`, `entertainment`, `bills`, `health`, `shopping`) using character n-gram TF-IDF (`char_wb`, n-grams 2 to 4).

| Model Candidate | 5-Fold Cross-Validation Accuracy | Test Split Accuracy (80/20) | Hard Test Set Accuracy (Unseen Merchants) |
| :--- | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 0.988 | 0.993 | 0.643 |
| **Logistic Regression** | 0.996 | 0.993 | 0.548 |
| **LinearSVC (Selected)** | **0.998** | **1.000** | **0.619** |

*Note on Generalization*: The test split accuracy reaches 1.000 because synthetic templates share vocabulary. On `data/hard_test.csv` (42 hand-labeled transactions with merchants never seen in training data), accuracy drops to 0.619 due to unseen brand names and keyword polysemy. See [docs/ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md) for detailed error analysis.

### 2. Time Series Spending Forecast
Evaluated over 12 months of synthetic student spending with a compound 1.5% monthly trend using a 4-fold rolling backtest:

| Forecasting Model | Backtest Mean Absolute Error (MAE) | Status |
| :--- | :---: | :---: |
| **3-Month Moving Average (`moving_avg_3`)** | **Rs 9,575.89** | **Selected (Lowest Error)** |
| **Linear Trend (`linear_trend`)** | Rs 12,521.97 | Evaluated |
| **Naive Last Month (`last_month`)** | Rs 19,723.91 | Baseline |
| **Holt Exponential Smoothing (`holt`)** | Rs 19,736.98 | Evaluated |
| **Gradient Boosting Differenced (`gradient_boosting`)** | Rs 29,966.43 | Evaluated |

- **Next Month Total Spending Forecast**: **Rs 61,570.45**

### 3. Anomaly Detection
Category-specific Isolation Forests fitted on \(\ln(1 + \text{amount})\) with `contamination=0.03`:
- **Planted Outliers Detected**: 4 of 6 planted outliers successfully flagged.
- **Normal Expenses Flagged (False Alarms)**: 2 legitimate transactions flagged.
- **Total Flagged Outliers**: 6 transactions surfaced in the anomaly report.

---

## Project Structure

```
├── main.py                     # Command line app and interactive terminal menu
├── api.py                      # FastAPI REST backend with Pydantic validation
├── app.py                      # Streamlit dashboard (supports Local and Remote backends)
├── Dockerfile                  # Container build recipe
├── docker-compose.yml          # Multi-container orchestration (API + Dashboard)
├── requirements.txt            # Python dependencies
├── spendsense/                 # Core library modules
│   ├── __init__.py             # Package definitions and category constants
│   ├── database.py             # SQLite storage layer with CHECK constraints
│   ├── categorizer.py          # Character n-gram classification pipeline
│   ├── embed.py                # Optional sentence-transformer pipeline
│   ├── sms.py                  # Regex parser for Indian bank and UPI SMS
│   ├── importer.py             # Bank statement CSV importer
│   ├── forecast.py             # 5 time-series models with rolling backtesting
│   ├── anomaly.py              # Isolation Forest anomaly detector
│   ├── chat.py                 # Query parser with whitelist validation
│   ├── service.py              # Serialization helpers for API and UI
│   └── synthetic.py            # Synthetic dataset and demo history generator
├── data/
│   ├── generate_dataset.py     # Generates expenses_labeled.csv (700 rows)
│   ├── expenses_labeled.csv    # Balanced synthetic training dataset
│   └── hard_test.csv           # 42 hand-written out-of-distribution test cases
├── tests/
│   └── test_spendsense.py      # Comprehensive unit tests (38 tests)
└── docs/
    ├── PROBLEM.md              # Problem analysis and architecture design
    ├── DEPLOY.md               # Streamlit Cloud and Docker deployment guide
    ├── ERROR_ANALYSIS.md       # Root cause breakdown of hard test set failures
    ├── weekly_updates.md       # Weekly milestone updates (Weeks 1 to 12)
    ├── class_counts.png        # Category distribution chart
    ├── confusion_matrix.png    # Classifier confusion matrix
    ├── index.html              # Static browser prototype for GitHub Pages
    └── viva/                   # Viva examination preparation notes
        ├── week_1.md ... week_12.md
```

---

## How to Run

### 1. Local Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/notbhavyansh07/spend-sense.git
   cd spend-sense
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate the dataset and train the model:
   ```bash
   python data/generate_dataset.py
   python main.py train
   ```

4. Run the full unit test suite:
   ```bash
   python -m unittest discover -s tests -v
   ```

### 2. Running the Command Line Interface

- **Interactive Menu**:
  ```bash
  python main.py
  ```
- **CLI Commands**:
  ```bash
  python main.py add 150 food "lunch at canteen"
  python main.py view
  python main.py total
  python main.py forecast
  python main.py anomalies
  python main.py ask "how much did I spend on food last month"
  python main.py parse-sms "Rs 450.00 debited from A/c XX1234 on 05-11-26 to ZOMATO. UPI Ref 123456"
  ```

### 3. Running the REST API

Launch the FastAPI backend:
```bash
uvicorn api:app --reload --port 8000
```
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

### 4. Running the Streamlit Dashboard

Launch the web dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. To connect the dashboard to the FastAPI backend instead of local modules, run:
```bash
set API_URL=http://localhost:8000
streamlit run app.py
```

### 5. Running with Docker Compose

Run the complete multi-container setup with persistent volume storage:
```bash
docker compose up --build
```
- API Docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

---

## Static Browser Demo (GitHub Pages)

A static zero-dependency browser prototype is available in `docs/index.html`. It runs the category classifier directly inside client-side JavaScript. When GitHub Pages is enabled on the `main` branch under `/docs`, it is hosted at:
`https://notbhavyansh07.github.io/spend-sense/`

---

## Limitations

1. **Synthetic Training Data**: Training data is generated from templates with injected typographical errors. While class-balanced, it lacks the colloquial slang and varied merchant names of real student spending, resulting in ~60% accuracy on unseen vendors.
2. **Short Forecasting Time Series**: Forecasting is evaluated across 12 monthly data points. This horizon is sufficient for moving averages and linear trends, but insufficient for modeling complex multi-year seasonality.
3. **Regex SMS Extraction**: The SMS parser relies on deterministic regular expressions tailored to Indian bank notification formats. Novel or non-standard bank SMS templates require pattern updates.
4. **Constrained Query Language**: The natural language question system intentionally executes only 5 whitelisted aggregation intents using pandas to eliminate SQL injection vulnerabilities.

---

## Next Steps

- Expand the active feedback loop by allowing users to export corrected receipts into a crowdsourced open expense dataset.
- Integrate direct Open Banking APIs (such as the RBI Account Aggregator framework) for automated receipt ingestion with user consent.
- Deploy an on-device small language model (SLM) for offline conversational question answering.

---

## License

MIT
