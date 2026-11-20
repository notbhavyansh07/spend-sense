"""SpendSense command line interface and interactive menu."""
import argparse
import sys
from datetime import date
from pathlib import Path

from spendsense import CATEGORIES, anomaly, categorizer, chat, forecast, importer, sms
from spendsense.database import ExpenseDB
from spendsense.synthetic import generate_expenses

db = ExpenseDB()


def cmd_add(args):
    cat = args.category.strip().lower()
    if cat not in CATEGORIES:
        print(f"Error: Invalid category '{cat}'. Valid: {', '.join(CATEGORIES)}")
        sys.exit(1)
    new_id = db.add(args.amount, cat, args.note or "")
    print(f"Added expense #{new_id}: Rs {args.amount:.2f} [{cat}] {args.note or ''}")
    return new_id


def suggest_category(text):
    """Predicts category and presents prompt to user; Enter accepts."""
    try:
        guess, conf = categorizer.predict(text)
        prompt_msg = f"Suggested category: '{guess}' ({conf:.0%} confidence). Enter to accept, or enter another: "
        user_input = input(prompt_msg).strip().lower()
        if user_input and user_input != guess:
            categorizer.add_feedback(text, user_input)
            return user_input
        return guess
    except FileNotFoundError:
        user_input = input("Category: ").strip().lower()
        return user_input or "other"


def menu_add_expense():
    try:
        amount_input = input("Amount (Rs): ").strip()
        amount = float(amount_input)
        if amount <= 0:
            print("Amount must be greater than zero.")
            return
    except ValueError:
        print("Invalid amount format.")
        return

    note = input("Description (e.g. 'zomato dinner'): ").strip()
    category = suggest_category(note)
    spent_on = input("Date YYYY-MM-DD (Enter = today): ").strip() or None
    try:
        new_id = db.add(amount, category, note, spent_on)
        print(f"Saved expense #{new_id}: Rs {amount:.2f} [{category}]")
    except Exception as err:
        print(f"Error saving expense: {err}")


def menu_paste_sms():
    sms_text = input("Paste the bank SMS: ").strip()
    parsed = sms.parse_sms(sms_text)
    if not parsed:
        print("No valid transaction amount found in that message.")
        return
    if parsed["type"] != "debit":
        print(f"Detected a {parsed['type']} transaction (Rs {parsed['amount']:.2f}). Only debits are logged.")
        return

    desc = parsed["merchant"] or "unknown merchant"
    print(f"\nExtracted: Rs {parsed['amount']:.2f} at '{desc}' on {parsed['date']}")
    category = suggest_category(desc)
    new_id = db.add(parsed["amount"], category, desc, parsed["date"])
    print(f"Saved expense #{new_id}: Rs {parsed['amount']:.2f} [{category}]")


def menu_view_expenses():
    df = db.all()
    if df.empty:
        print("No expenses recorded yet.")
        return
    print(f"\n{'ID':<5} {'Date':<12} {'Category':<15} {'Amount (Rs)':<12} {'Note'}")
    print("-" * 65)
    for _, row in df.iterrows():
        spent_str = (
            row["spent_on"].strftime("%Y-%m-%d")
            if hasattr(row["spent_on"], "strftime")
            else str(row["spent_on"])
        )
        print(
            f"{row['id']:<5} {spent_str:<12} {row['category']:<15} {row['amount']:<12.2f} {row['note'] or ''}"
        )


def menu_summary():
    df = db.all()
    if df.empty:
        print("No expenses recorded.")
        return
    total = float(df["amount"].sum())
    print(f"\nTotal spending: Rs {total:.2f}")
    print("\nSpending by category:")
    grouped = df.groupby("category")["amount"].agg(["sum", "count"]).rename(
        columns={"sum": "Total (Rs)", "count": "Transactions"}
    )
    print(grouped.to_string())


def menu_fix_category():
    df = db.all()
    if df.empty:
        print("No expenses recorded.")
        return
    try:
        exp_id_str = input("Enter expense ID to correct: ").strip()
        exp_id = int(exp_id_str)
        matched = df[df["id"] == exp_id]
        if matched.empty:
            print(f"Expense #{exp_id} not found.")
            return
        row = matched.iloc[0]
    except ValueError:
        print("Invalid ID.")
        return

    print(f"Current expense #{exp_id}: '{row['note']}' is categorized as '{row['category']}'")
    new_cat = input(f"New category ({', '.join(CATEGORIES)}): ").strip().lower()
    if not new_cat:
        print("No change made.")
        return
    if new_cat not in CATEGORIES:
        print(f"Warning: '{new_cat}' is not in standard categories.")
    db.update_category(exp_id, new_cat)
    categorizer.add_feedback(row["note"], new_cat)
    print(f"Updated expense #{exp_id} to '{new_cat}' and appended to data/feedback.csv.")


def menu_demo():
    db.clear()
    rows = generate_expenses(months=12, seed=42)
    db.add_many(rows)
    print(f"Loaded {len(rows)} demo expenses spanning 12 months.")


def cmd_train(embeddings=False):
    print("Training category models with 5-fold cross-validation...")
    result = categorizer.train(use_embeddings=embeddings)
    print(f"Dataset size: {result['dataset_size']} rows")
    print("5-Fold Cross-Validation Scores:")
    for name, score in result["cv_scores"].items():
        print(f"  {name:<22}: {score:.3f}")
    print(f"Best model selected   : {result['best_model']}")
    print(f"Test split accuracy   : {result['test_accuracy']:.3f}")
    if result["hard_test_accuracy"] is not None:
        print(f"Hard test set accuracy: {result['hard_test_accuracy']:.3f}")


def cmd_compare(embeddings=False):
    print("Evaluating models on data/hard_test.csv (unseen merchants):")
    scores = categorizer.compare_hard(use_embeddings=embeddings)
    for name, score in scores.items():
        val_str = f"{score:.3f}" if score is not None else "N/A"
        print(f"  {name:<22}: {val_str}")


def cmd_forecast():
    df = db.all()
    res = forecast.forecast_next_month(df)
    if not res:
        print("At least 6 full months of historical data are required to generate a forecast.")
        print("Tip: Load demo data first via 'python main.py demo'")
        return

    print(f"\nForecast for {res['next_month']}: Rs {res['prediction']:.2f}")
    print(f"Selected model: {res['model']} (Backtest MAE: Rs {res['mae_model']:.2f})")
    print(f"Baseline 3-month moving average MAE: Rs {res['mae_baseline']:.2f}")

    print("\nModel Comparison (Rolling Backtest MAE):")
    print(f"{'Model':<25} {'MAE (Rs)':<12}")
    print("-" * 38)
    for model_name, mae in sorted(res["backtest"].items(), key=lambda kv: (kv[1] is None, kv[1])):
        val_str = f"Rs {mae:.2f}" if mae is not None else "N/A"
        mark = " <-- Best" if model_name == res["model"] else ""
        print(f"{model_name:<25} {val_str:<12}{mark}")

    cat_forecasts = forecast.forecast_by_category(df)
    if cat_forecasts:
        print("\nForecast by Category (Linear Trend):")
        for cat, amount in cat_forecasts.items():
            print(f"  {cat:<15}: Rs {amount:.2f}")


def cmd_anomalies():
    df = db.all()
    if df.empty:
        print("No expenses recorded.")
        return
    flagged = anomaly.detect_anomalies(df)
    if flagged.empty:
        print("No unusual expenses flagged across categories.")
        return
    print(f"\nFlagged {len(flagged)} Unusually Large Expenses:")
    print(f"{'ID':<5} {'Date':<12} {'Category':<15} {'Amount (Rs)':<12} {'Note'}")
    print("-" * 65)
    for _, row in flagged.iterrows():
        spent_str = (
            row["spent_on"].strftime("%Y-%m-%d")
            if hasattr(row["spent_on"], "strftime")
            else str(row["spent_on"])
        )
        print(
            f"{row['id']:<5} {spent_str:<12} {row['category']:<15} {row['amount']:<12.2f} {row['note'] or ''}"
        )


def cmd_ask(question=None):
    df = db.all()
    if not question:
        question = input("Ask a question about your spending: ").strip()
    if not question:
        return
    res = chat.ask(df, question)
    print(f"\n{res['answer']}")


def run_menu():
    menu_options = {
        "1": ("Add expense manually", menu_add_expense),
        "2": ("Paste a bank transaction SMS", menu_paste_sms),
        "3": ("View expenses", menu_view_expenses),
        "4": ("Summary and totals", menu_summary),
        "5": ("Forecast next month spending", cmd_forecast),
        "6": ("Find unusual expenses (anomalies)", cmd_anomalies),
        "7": ("Ask a question about your spending", cmd_ask),
        "8": ("Fix a wrong category (feedback)", menu_fix_category),
        "9": ("Load demo data (12 months)", menu_demo),
        "10": ("Train category model", lambda: cmd_train(False)),
    }
    while True:
        print("\n=== SpendSense CLI Menu ===")
        for key, (label, _) in menu_options.items():
            print(f"{key}. {label}")
        print("0. Exit")
        choice = input("Select an option: ").strip()
        if choice in ("0", "exit", "q"):
            print("Goodbye.")
            break
        if choice in menu_options:
            menu_options[choice][1]()
        else:
            print("Invalid option. Please try again.")


def build_parser():
    parser = argparse.ArgumentParser(description="SpendSense expense tracker")
    subparsers = parser.add_subparsers(dest="command")

    add_p = subparsers.add_parser("add", help="Add an expense")
    add_p.add_argument("amount", type=float, help="Expense amount in Rs")
    add_p.add_argument("category", type=str, help="Expense category")
    add_p.add_argument("note", nargs="?", default="", help="Note or merchant")

    view_p = subparsers.add_parser("view", help="View all expenses")
    total_p = subparsers.add_parser("total", help="Show total spending")
    subparsers.add_parser("forecast", help="Forecast next month spending")
    subparsers.add_parser("anomalies", help="Detect unusually large expenses")
    ask_p = subparsers.add_parser("ask", help="Ask a question about your spending")
    ask_p.add_argument("question", type=str, help="Question text in English")

    train_p = subparsers.add_parser("train", help="Train categorizer models")
    train_p.add_argument("--embeddings", action="store_true", help="Include embeddings")

    comp_p = subparsers.add_parser("compare", help="Compare models on hard test set")
    comp_p.add_argument("--embeddings", action="store_true", help="Include embeddings")

    pred_p = subparsers.add_parser("predict", help="Predict expense category")
    pred_p.add_argument("text", type=str, help="Expense text")

    sms_p = subparsers.add_parser("parse-sms", help="Parse bank transaction SMS")
    sms_p.add_argument("text", type=str, help="SMS text")

    demo_p = subparsers.add_parser("demo", help="Load 12 months demo data")

    imp_p = subparsers.add_parser("import-bank", help="Import bank statement CSV")
    imp_p.add_argument("file", type=str, help="Path to bank CSV file")

    return parser


def main():
    if len(sys.argv) == 1:
        run_menu()
        return

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args)
    elif args.command == "view":
        menu_view_expenses()
    elif args.command == "total":
        menu_summary()
    elif args.command == "forecast":
        cmd_forecast()
    elif args.command == "anomalies":
        cmd_anomalies()
    elif args.command == "ask":
        cmd_ask(args.question)
    elif args.command == "train":
        cmd_train(args.embeddings)
    elif args.command == "compare":
        cmd_compare(args.embeddings)
    elif args.command == "predict":
        cat, conf = categorizer.predict(args.text)
        print(f"Text      : '{args.text}'\nCategory  : {cat}\nConfidence: {conf:.1%}")
    elif args.command == "parse-sms":
        res = sms.parse_sms(args.text)
        if not res:
            print("No transaction amount detected.")
        else:
            print(f"Amount  : Rs {res['amount']:.2f}")
            print(f"Type    : {res['type']}")
            print(f"Merchant: {res['merchant']}")
            print(f"Date    : {res['date']}")
    elif args.command == "demo":
        menu_demo()
    elif args.command == "import-bank":
        csv_file = Path(args.file)
        if not csv_file.exists():
            print(f"File not found: {csv_file}")
            sys.exit(1)
        try:
            m = categorizer.load()
        except FileNotFoundError:
            m = None
        count = importer.import_bank_csv(csv_file, db, model=m)
        print(f"Imported {count} expenses from {csv_file.name}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
