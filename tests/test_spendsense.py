"""Unit tests for SpendSense (Weeks 1 and 2)."""
import csv
import os
import tempfile
import unittest
from pathlib import Path

import main
from spendsense.database import ExpenseDB
from spendsense.importer import clean_text, import_bank_csv
from spendsense.synthetic import CATS, generate_expenses, generate_labeled


class TestExpenseDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db = ExpenseDB(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_list_expense(self):
        expense_id = self.db.add(
            amount=250.0,
            category="food",
            note="dinner at canteen",
            spent_on="2026-11-20",
        )
        self.assertEqual(expense_id, 1)

        df = self.db.all()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["amount"], 250.0)
        self.assertEqual(df.iloc[0]["category"], "food")
        self.assertEqual(df.iloc[0]["note"], "dinner at canteen")

    def test_get_total(self):
        self.assertEqual(main.get_total(db_path=self.db_path), 0.0)
        self.db.add(amount=100.0, category="travel", note="auto")
        self.db.add(amount=450.0, category="food", note="groceries")
        total = main.get_total(db_path=self.db_path)
        self.assertAlmostEqual(total, 550.0)

    def test_invalid_category_rejected_in_cli(self):
        with self.assertRaises(ValueError):
            main.add_expense(amount=100.0, category="unknown_category", db_path=self.db_path)

    def test_non_positive_amount_rejected(self):
        with self.assertRaises(Exception):
            self.db.add(amount=-50.0, category="bills")


class TestBaselineModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.model_path = Path(self.temp_dir.name) / "test_model.joblib"
        self.data_path = Path(self.temp_dir.name) / "test_data.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_and_predict(self):
        score, total_rows, train_rows, test_rows = main.train_baseline(
            data_file=self.data_path,
            model_out=self.model_path,
        )
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertTrue(self.model_path.exists())

        pred = main.predict_category("canteen chai", model_file=self.model_path)
        self.assertIn(pred, main.VALID_CATEGORIES)


class TestSyntheticData(unittest.TestCase):
    def test_generate_labeled_distribution(self):
        rows = generate_labeled(per_category=100, seed=42)
        self.assertEqual(len(rows), 700)
        counts = {}
        for text, cat in rows:
            counts[cat] = counts.get(cat, 0) + 1
        self.assertEqual(len(counts), 7)
        for cat in CATS.keys():
            self.assertEqual(counts[cat], 100)

    def test_generate_expenses_outliers(self):
        history = generate_expenses(months=12, seed=42)
        self.assertEqual(len(history), 12 * 41)
        outliers = [r for r in history if "(outlier)" in r[2]]
        self.assertEqual(len(outliers), 6)


class TestBankImporter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "import_test.db"
        self.db = ExpenseDB(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_clean_text(self):
        raw = "UPI/23498/ZOMATO_DELIVERY!#12"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "upi zomato delivery")

    def test_import_bank_csv(self):
        csv_path = Path(self.temp_dir.name) / "statement.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Narration", "Withdrawal Amount", "Deposit Amount"])
            writer.writerow(["05/11/2026", "ZOMATO RESTAURANT", "450.00", ""])
            writer.writerow(["06/11/2026", "SALARY CREDIT", "", "50000.00"])
            writer.writerow(["07/11/2026", "UBER RIDE MUMBAI", "320.50", ""])
            writer.writerow(["invalid_date", "BROKEN ROW", "100.00", ""])

        count = import_bank_csv(csv_path, self.db)
        self.assertEqual(count, 2)
        df = self.db.all()
        self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()
