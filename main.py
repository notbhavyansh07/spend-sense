"""Command line interface and baseline implementation for SpendSense."""
import argparse
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DB = ROOT_DIR / "spendsense.db"
MODEL_PATH = ROOT_DIR / "models" / "categorizer.joblib"
DATA_PATH = ROOT_DIR / "data" / "expenses_labeled.csv"

VALID_CATEGORIES = [
    "food",
    "travel",
    "education",
    "entertainment",
    "bills",
    "health",
    "shopping",
]


def get_db_path():
    env_path = os.environ.get("SPENDSENSE_DB")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB


def get_connection(db_path=None):
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
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
    return conn


def add_expense(amount, category, note="", spent_on=None, db_path=None):
    category = category.strip().lower()
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {', '.join(VALID_CATEGORIES)}")
    if amount <= 0:
        raise ValueError("Amount must be greater than zero.")
    if spent_on is None:
        spent_on = date.today().isoformat()

    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO expenses (amount, category, note, spent_on) VALUES (?, ?, ?, ?)",
        (float(amount), category, note, str(spent_on)),
    )
    conn.commit()
    inserted_id = cur.lastrowid
    conn.close()
    return inserted_id


def list_expenses(db_path=None):
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, amount, category, note, spent_on FROM expenses ORDER BY spent_on DESC, id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_total(db_path=None):
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(SUM(amount), 0.0) FROM expenses")
    total = cur.fetchone()[0]
    conn.close()
    return float(total)


def train_baseline(data_file=None, model_out=None):
    if data_file is None:
        data_file = DATA_PATH
    if model_out is None:
        model_out = MODEL_PATH

    data_file = Path(data_file)
    if not data_file.exists():
        from data.generate_dataset import generate_csv
        generate_csv(data_file)

    df = pd.read_csv(data_file)
    if len(df) < 14:
        raise ValueError("Dataset is too small to train and evaluate.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["category"],
        test_size=0.2,
        random_state=42,
        stratify=df["category"],
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200, random_state=42)),
    ])
    pipeline.fit(x_train, y_train)

    score = pipeline.score(x_test, y_test)

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)

    return score, len(df), len(x_train), len(x_test)


def predict_category(text, model_file=None):
    if model_file is None:
        model_file = MODEL_PATH
    model_file = Path(model_file)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found at {model_file}. Run 'python main.py train' first."
        )
    pipeline = joblib.load(model_file)
    predicted = pipeline.predict([text])[0]
    return predicted


def cmd_add(args):
    inserted_id = add_expense(args.amount, args.category, args.note or "")
    print(f"Added expense #{inserted_id}: Rs {args.amount:.2f} [{args.category}] {args.note or ''}")


def cmd_view(args):
    rows = list_expenses()
    if not rows:
        print("No expenses recorded yet.")
        return
    print(f"{'ID':<4} {'Date':<12} {'Category':<15} {'Amount (Rs)':<12} {'Note'}")
    print("-" * 60)
    for row in rows:
        print(f"{row[0]:<4} {row[4]:<12} {row[2]:<15} {row[1]:<12.2f} {row[3] or ''}")


def cmd_total(args):
    total = get_total()
    print(f"Total spending: Rs {total:.2f}")


def cmd_train(args):
    score, total_rows, train_rows, test_rows = train_baseline()
    print(f"Training dataset: {total_rows} rows ({train_rows} train, {test_rows} test)")
    print(f"Baseline test accuracy: {score:.3f}")
    print("Note: Test accuracy is low on this small 75-row baseline dataset. More training data will be generated in Week 2.")


def cmd_predict(args):
    try:
        category = predict_category(args.text)
        print(f"Text: '{args.text}' -> Predicted category: {category}")
    except FileNotFoundError as err:
        print(err)
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(description="SpendSense expense tracker")
    subparsers = parser.add_subparsers(dest="command")

    add_parser = subparsers.add_parser("add", help="Add an expense")
    add_parser.add_argument("amount", type=float, help="Expense amount in Rs")
    add_parser.add_argument("category", type=str, help="Expense category")
    add_parser.add_argument("note", nargs="?", default="", help="Optional note or merchant")
    add_parser.set_defaults(func=cmd_add)

    view_parser = subparsers.add_parser("view", help="View all expenses")
    view_parser.set_defaults(func=cmd_view)

    total_parser = subparsers.add_parser("total", help="Show total spending")
    total_parser.set_defaults(func=cmd_total)

    train_parser = subparsers.add_parser("train", help="Train baseline category model")
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Predict category for text")
    predict_parser.add_argument("text", type=str, help="Expense description")
    predict_parser.set_defaults(func=cmd_predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
