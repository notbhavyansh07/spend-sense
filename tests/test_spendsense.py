"""Unit tests for SpendSense Week 1 baseline."""
import os
import tempfile
import unittest
from pathlib import Path

import main


class TestExpenseDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_list_expense(self):
        expense_id = main.add_expense(
            amount=250.0,
            category="food",
            note="dinner at canteen",
            spent_on="2026-11-20",
            db_path=self.db_path,
        )
        self.assertEqual(expense_id, 1)

        rows = main.list_expenses(db_path=self.db_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], 250.0)
        self.assertEqual(rows[0][2], "food")
        self.assertEqual(rows[0][3], "dinner at canteen")
        self.assertEqual(rows[0][4], "2026-11-20")

    def test_get_total(self):
        self.assertEqual(main.get_total(db_path=self.db_path), 0.0)
        main.add_expense(amount=100.0, category="travel", note="auto", db_path=self.db_path)
        main.add_expense(amount=450.0, category="food", note="groceries", db_path=self.db_path)
        total = main.get_total(db_path=self.db_path)
        self.assertAlmostEqual(total, 550.0)

    def test_invalid_category_rejected(self):
        with self.assertRaises(ValueError):
            main.add_expense(amount=100.0, category="unknown_category", db_path=self.db_path)

    def test_non_positive_amount_rejected(self):
        with self.assertRaises(ValueError):
            main.add_expense(amount=-50.0, category="bills", db_path=self.db_path)


class TestBaselineModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
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
        self.assertEqual(total_rows, 75)
        self.assertTrue(self.model_path.exists())

        pred = main.predict_category("canteen chai", model_file=self.model_path)
        self.assertIn(pred, main.VALID_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
