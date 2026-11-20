"""SpendSense command line interface."""
import argparse
import sys
from datetime import date
from pathlib import Path

from spendsense import CATEGORIES, categorizer, importer
from spendsense.database import ExpenseDB
from spendsense.synthetic import generate_expenses

db = ExpenseDB()


def cmd_add(args):
    amount = float(args.amount)
    category = args.category.strip().lower()
    if category not in CATEGORIES:
        print(f"Error: Invalid category '{category}'. Valid: {', '.join(CATEGORIES)}")
        sys.exit(1)
    new_id = db.add(amount, category, args.note or "")
    print(f"Added expense #{new_id}: Rs {amount:.2f} [{category}] {args.note or ''}")


def cmd_view(args):
    df = db.all()
    if df.empty:
        print("No expenses recorded yet.")
        return
    print(f"{'ID':<5} {'Date':<12} {'Category':<15} {'Amount (Rs)':<12} {'Note'}")
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


def cmd_total(args):
    df = db.all()
    total = float(df["amount"].sum()) if not df.empty else 0.0
    print(f"Total spending: Rs {total:.2f}")


def cmd_train(args):
    print("Training category models with 5-fold cross-validation...")
    result = categorizer.train(use_embeddings=args.embeddings)
    print(f"Dataset size: {result['dataset_size']} rows")
    print("5-Fold Cross-Validation Scores:")
    for name, score in result["cv_scores"].items():
        print(f"  {name:<22}: {score:.3f}")
    print(f"Best model selected   : {result['best_model']}")
    print(f"Test split accuracy   : {result['test_accuracy']:.3f}")
    if result["hard_test_accuracy"] is not None:
        print(f"Hard test set accuracy: {result['hard_test_accuracy']:.3f}")


def cmd_compare(args):
    print("Evaluating models on data/hard_test.csv (unseen merchants):")
    scores = categorizer.compare_hard(use_embeddings=args.embeddings)
    for name, score in scores.items():
        val_str = f"{score:.3f}" if score is not None else "N/A"
        print(f"  {name:<22}: {val_str}")


def cmd_predict(args):
    try:
        category, confidence = categorizer.predict(args.text)
        print(f"Text      : '{args.text}'")
        print(f"Category  : {category}")
        print(f"Confidence: {confidence:.1%}")
    except FileNotFoundError as err:
        print(err)
        sys.exit(1)


def cmd_demo(args):
    rows = generate_expenses(months=12, seed=42)
    db.clear()
    db.add_many(rows)
    print(f"Loaded {len(rows)} demo expenses spanning 12 months.")


def cmd_import_bank(args):
    csv_file = Path(args.file)
    if not csv_file.exists():
        print(f"File not found: {csv_file}")
        sys.exit(1)
    try:
        model = categorizer.load()
    except FileNotFoundError:
        model = None
    count = importer.import_bank_csv(csv_file, db, model=model)
    print(f"Imported {count} expenses from {csv_file.name}")


def build_parser():
    parser = argparse.ArgumentParser(description="SpendSense expense tracker")
    subparsers = parser.add_subparsers(dest="command")

    add_p = subparsers.add_parser("add", help="Add an expense")
    add_p.add_argument("amount", type=float, help="Expense amount in Rs")
    add_p.add_argument("category", type=str, help="Expense category")
    add_p.add_argument("note", nargs="?", default="", help="Note or merchant")
    add_p.set_defaults(func=cmd_add)

    view_p = subparsers.add_parser("view", help="View all expenses")
    view_p.set_defaults(func=cmd_view)

    total_p = subparsers.add_parser("total", help="Show total spending")
    total_p.set_defaults(func=cmd_total)

    train_p = subparsers.add_parser("train", help="Train categorizer models")
    train_p.add_argument(
        "--embeddings",
        action="store_true",
        help="Include sentence embeddings pipeline",
    )
    train_p.set_defaults(func=cmd_train)

    comp_p = subparsers.add_parser(
        "compare", help="Compare models on hard test set"
    )
    comp_p.add_argument(
        "--embeddings",
        action="store_true",
        help="Include sentence embeddings",
    )
    comp_p.set_defaults(func=cmd_compare)

    pred_p = subparsers.add_parser("predict", help="Predict expense category")
    pred_p.add_argument("text", type=str, help="Expense text")
    pred_p.set_defaults(func=cmd_predict)

    demo_p = subparsers.add_parser("demo", help="Load 12 months demo data")
    demo_p.set_defaults(func=cmd_demo)

    imp_p = subparsers.add_parser("import-bank", help="Import bank statement CSV")
    imp_p.add_argument("file", type=str, help="Path to bank CSV file")
    imp_p.set_defaults(func=cmd_import_bank)

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
