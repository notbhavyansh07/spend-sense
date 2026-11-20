"""SQLite storage for SpendSense expenses."""
import os
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("SPENDSENSE_DB", ROOT / "data" / "expenses.db"))


class ExpenseDB:
    def __init__(self, path=DB_PATH):
        self.path = str(path)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount REAL NOT NULL CHECK(amount > 0),
                    category TEXT NOT NULL,
                    note TEXT,
                    spent_on TEXT NOT NULL
                )
                """
            )
            conn.commit()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
        finally:
            conn.close()

    def add(self, amount, category, note="", spent_on=None):
        spent_on = spent_on or date.today().isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO expenses (amount, category, note, spent_on) VALUES (?, ?, ?, ?)",
                (float(amount), category, note, str(spent_on)),
            )
            conn.commit()
            return cur.lastrowid

    def add_many(self, rows):
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO expenses (amount, category, note, spent_on) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()

    def all(self):
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM expenses ORDER BY spent_on DESC, id DESC",
                conn,
            )
        if not df.empty:
            df["spent_on"] = pd.to_datetime(df["spent_on"])
        return df

    def update_category(self, expense_id, category):
        with self._conn() as conn:
            conn.execute(
                "UPDATE expenses SET category = ? WHERE id = ?",
                (category, expense_id),
            )
            conn.commit()

    def delete(self, expense_id):
        with self._conn() as conn:
            conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
            conn.commit()

    def clear(self):
        with self._conn() as conn:
            conn.execute("DELETE FROM expenses")
            conn.commit()
