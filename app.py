"""SpendSense Streamlit dashboard.

Run: streamlit run app.py
If API_URL is set (e.g. http://localhost:8000), the dashboard communicates via the REST API.
Otherwise, it executes using local SpendSense Python modules directly.
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from spendsense import categorizer, chat, service, sms
from spendsense.database import ExpenseDB
from spendsense.synthetic import generate_expenses

API_URL = os.getenv("API_URL")

# Auto-train category model on first startup if missing
if not API_URL and not categorizer.MODEL_PATH.exists():
    if not categorizer.DATA_PATH.exists():
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "data" / "generate_dataset.py")],
            check=True,
        )
    categorizer.train()


class LocalBackend:
    def __init__(self):
        self.db = ExpenseDB()

    def expenses(self):
        return self.db.all()

    def predict(self, text):
        try:
            return categorizer.predict(text)
        except FileNotFoundError:
            return "other", 0.0

    def add(self, amount, category, note, spent_on=None):
        self.db.add(amount, category, note, spent_on)

    def parse_sms(self, text):
        return sms.parse_sms(text)

    def forecast(self):
        return service.forecast_summary(self.db.all())

    def anomalies(self):
        records = service.anomaly_records(self.db.all())
        return pd.DataFrame(records) if records else pd.DataFrame()

    def ask(self, question):
        return chat.ask(self.db.all(), question)["answer"]

    def demo(self):
        self.db.clear()
        self.db.add_many(generate_expenses(months=12, seed=42))


class RemoteBackend:
    def __init__(self, base_url):
        import requests

        self.client = requests
        self.base_url = base_url.rstrip("/")

    def _get(self, path):
        try:
            resp = self.client.get(self.base_url + path, timeout=30)
            return resp.json() if resp.ok else None
        except Exception:
            return None

    def _post(self, path, body):
        try:
            resp = self.client.post(self.base_url + path, json=body, timeout=60)
            return resp.json() if resp.ok else None
        except Exception:
            return None

    def expenses(self):
        data = self._get("/expenses?limit=100000") or []
        df = pd.DataFrame(data)
        if not df.empty and "spent_on" in df.columns:
            df["spent_on"] = pd.to_datetime(df["spent_on"])
        return df

    def predict(self, text):
        data = self._post("/predict-category", {"text": text}) or {
            "category": "other",
            "confidence": 0.0,
        }
        return data.get("category", "other"), data.get("confidence", 0.0)

    def add(self, amount, category, note, spent_on=None):
        self._post(
            "/expenses",
            {
                "amount": float(amount),
                "category": category,
                "note": note,
                "spent_on": spent_on,
            },
        )

    def parse_sms(self, text):
        return self._post("/parse-sms", {"text": text})

    def forecast(self):
        return self._get("/forecast")

    def anomalies(self):
        records = self._get("/anomalies") or []
        return pd.DataFrame(records)

    def ask(self, question):
        data = self._post("/ask", {"question": question}) or {}
        return data.get("answer", "The API did not answer.")

    def demo(self):
        self._post("/demo", {})


st.set_page_config(page_title="SpendSense", layout="wide")
backend = RemoteBackend(API_URL) if API_URL else LocalBackend()
st.title("SpendSense: Expense Tracker & ML Insights")

with st.sidebar:
    st.header("Expense Management")
    tab_type, tab_sms = st.tabs(["Add Expense", "Paste SMS"])

    with tab_type:
        note_input = st.text_input("Description", placeholder="e.g. zomato dinner")
        amount_input = st.number_input("Amount (Rs)", min_value=0.0, step=10.0)
        suggested_category = ""
        if note_input.strip():
            suggested_category, confidence = backend.predict(note_input)
            st.caption(f"Suggested: **{suggested_category}** ({confidence:.0%} confidence)")
        chosen_category = st.text_input("Category", value=suggested_category)
        if st.button("Save Expense") and amount_input > 0 and chosen_category:
            backend.add(amount_input, chosen_category, note_input)
            st.success("Expense saved successfully.")
            st.rerun()

    with tab_sms:
        sms_text = st.text_area(
            "Bank SMS",
            placeholder="Rs 450.00 debited from A/c XX1234 on 05-11-26 to ZOMATO.",
        )
        if sms_text.strip():
            parsed_sms = backend.parse_sms(sms_text)
            if parsed_sms and parsed_sms.get("type") == "debit":
                merchant_desc = parsed_sms.get("merchant") or sms_text
                cat_guess, _ = backend.predict(merchant_desc)
                st.write(
                    f"**Extracted**: Rs {parsed_sms['amount']:,.0f} at '{merchant_desc}' on {parsed_sms['date']}"
                )
                if st.button("Save this SMS"):
                    backend.add(
                        parsed_sms["amount"],
                        cat_guess,
                        merchant_desc,
                        parsed_sms["date"],
                    )
                    st.success("SMS transaction saved.")
                    st.rerun()
            else:
                st.caption("No debit transaction found in message.")

    st.divider()
    if st.button("Load 12-Month Demo Data"):
        backend.demo()
        st.success("Demo history loaded.")
        st.rerun()

expenses_df = backend.expenses()
if expenses_df.empty:
    st.info("No expense data recorded yet. Use the sidebar to add expenses or click 'Load 12-Month Demo Data'.")
    st.stop()

forecast_data = backend.forecast()

# Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("Total Spent", f"Rs {expenses_df['amount'].sum():,.0f}")
col2.metric("Total Expenses", len(expenses_df))
if forecast_data:
    col3.metric(
        f"Forecast ({forecast_data['next_month']})",
        f"Rs {forecast_data['prediction']:,.0f}",
        help=f"Selected model: {forecast_data['model']}",
    )

# Charts Row
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Spending by Category")
    category_totals = expenses_df.groupby("category")["amount"].sum()
    st.bar_chart(category_totals)

with col_right:
    st.subheader("Monthly Total Trend")
    monthly_series = expenses_df.groupby(
        expenses_df["spent_on"].dt.to_period("M").astype(str)
    )["amount"].sum()
    st.line_chart(monthly_series)

# Forecast Model Comparison Table
if forecast_data and forecast_data.get("backtest"):
    st.subheader("Forecast Models Comparison (Rolling Backtest MAE)")
    bt_records = [
        {
            "Model": name,
            "Error (Rs, lower is better)": f"{err:,.2f}" if err is not None else "N/A",
            "Selected": "Yes (Best)" if name == forecast_data["model"] else "",
        }
        for name, err in sorted(
            forecast_data["backtest"].items(),
            key=lambda kv: (kv[1] is None, kv[1] if kv[1] is not None else 999999),
        )
    ]
    st.dataframe(pd.DataFrame(bt_records), use_container_width=True, hide_index=True)

# Question Answering Box
st.subheader("Ask SpendSense About Your Spending")
query_text = st.text_input(
    "Ask a question in plain English",
    placeholder="e.g. How much did I spend on food last month?",
)
if query_text.strip():
    answer_text = backend.ask(query_text)
    st.info(answer_text)

# Unusual Expenses (Anomalies)
st.subheader("Unusual Expenses Flagged by Isolation Forest")
anomalies_df = backend.anomalies()
if not anomalies_df.empty:
    cols_to_show = [c for c in ["id", "spent_on", "category", "amount", "note"] if c in anomalies_df.columns]
    st.dataframe(anomalies_df[cols_to_show], use_container_width=True, hide_index=True)
else:
    st.write("No anomalous transactions detected.")

# Recent Expenses Table
st.subheader("Recent Expenses Log")
st.dataframe(expenses_df.head(50), use_container_width=True, hide_index=True)
