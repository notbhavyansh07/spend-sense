"""SpendSense command line interface and interactive menu."""
import argparse
import sys
from datetime import date
from pathlib import Path

from spendsense import CATEGORIES, categorizer, importer
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


def run_menu():
    menu_options = {
        "1": ("Add expense", menu_add_expense),
        "2": ("View expenses", menu_view_expenses),
        "3": ("Summary and totals", menu_summary),
        "4": ("Fix a wrong category (feedback)", menu_fix_category),
        "5": ("Load demo data (12 months)", menu_demo),
        "6": ("Train category model", lambda: cmd_train(False)),
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

    train_p = subparsers.add_parser("train", help="Train categorizer models")
    train_p.add_argument("--embeddings", action="store_true", help="Include embeddings")

    comp_p = subparsers.add_parser("compare", help="Compare models on hard test set")
    comp_p.add_argument("--embeddings", action="store_true", help="Include embeddings")

    pred_p = subparsers.add_parser("predict", help="Predict expense category")
    pred_p.add_argument("text", type=str, help="Expense text")

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
    elif args.command == "train":
        cmd_train(args.embeddings)
    elif args.command == "compare":
        cmd_compare(args.embeddings)
    elif args.command == "predict":
        cat, conf = categorizer.predict(args.text)
        print(f"Text      : '{args.text}'\nCategory  : {cat}\nConfidence: {conf:.1%}")
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
