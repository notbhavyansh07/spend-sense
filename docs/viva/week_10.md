# Week 10 Viva Notes: Dashboard

### 1. How does app.py implement dual-backend abstraction (Local vs Remote)?
`app.py` implements a common interface through two classes: `LocalBackend` and `RemoteBackend`. When the environment variable `API_URL` is omitted, `LocalBackend` invokes Python modules (`ExpenseDB`, `categorizer`, `service`, `sms`, `chat`) in-process. When `API_URL` is set (such as in a distributed Docker container or microservice architecture), `RemoteBackend` issues HTTP REST calls (`GET /expenses`, `POST /expenses`, `POST /ask`) to the FastAPI backend. The Streamlit UI code interacts strictly with the unified backend interface, requiring zero UI code changes between standalone and client-server deployments.

### 2. How does the dashboard handle cold-start deployment when models/categorizer.joblib is missing?
Because `.gitignore` strictly excludes `models/` from version control, a clean repository clone or fresh cloud container starts without a trained model file. At top-level execution in `app.py`, an initialization check detects if `categorizer.MODEL_PATH.exists()`. If missing, it invokes `data/generate_dataset.py` and runs `categorizer.train()` synchronously before serving the UI, ensuring the dashboard boots with a working classifier without manual intervention.

### 3. How does Streamlit handle state updates and reactivity when expenses are added?
Streamlit follows an execution model where the entire script re-runs from top to bottom whenever an interactive widget triggers an event. When a user clicks `"Save Expense"` or `"Load 12-Month Demo Data"`, the backend updates SQLite and `st.rerun()` triggers an immediate re-render, recalculating metric cards, refreshing charts, and re-querying recent transactions.

### 4. What visualizations are used in the dashboard and what insights do they provide?
- **Category Bar Chart**: Aggregates total expenditure across the 7 categories (`groupby("category")["amount"].sum()`), illustrating which category consumes the largest fraction of budget.
- **Monthly Spending Line Chart**: Aggregates historical expenditure across monthly billing cycles (`spent_on.dt.to_period("M")`), visualizing spending trajectory and inflation trend over 12 months.
- **Backtest Comparison Table**: Displays MAE across all five forecasting algorithms, confirming which model achieved the lowest backtest error.

### 5. Why are anomalies presented in a dedicated table rather than embedded in the main expense list?
In typical financial tracking, users record hundreds of routine transactions. Embedding anomaly flags solely as icons within a large 500-row table forces users to scroll and search. A dedicated anomaly table surfaces only high-value deviations above category medians, immediately drawing attention to suspected overspending or billing errors.

---

### One Thing That Could Break
If a user deploys `app.py` to Streamlit Community Cloud and runs with the `LocalBackend`, the SQLite database is written to the ephemeral container disk. If Streamlit restarts or puts the container to sleep after inactivity, local database modifications will be lost upon reboot.
