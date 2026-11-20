"""Unit tests for SpendSense (Weeks 1, 2, and 3)."""
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import main
from spendsense import categorizer, forecast, sms
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
        self.db.add(amount=100.0, category="travel", note="auto")
        self.db.add(amount=450.0, category="food", note="groceries")
        df = self.db.all()
        total = float(df["amount"].sum())
        self.assertAlmostEqual(total, 550.0)

    def test_invalid_category_rejected_in_cli(self):
        with self.assertRaises(SystemExit):
            # Using CLI cmd_add with bad category
            class DummyArgs:
                amount = 100.0
                category = "unknown_category"
                note = "bad"
            main.cmd_add(DummyArgs())

    def test_non_positive_amount_rejected(self):
        with self.assertRaises(Exception):
            self.db.add(amount=-50.0, category="bills")

    def test_update_category_and_delete(self):
        eid = self.db.add(100.0, "food", "dinner")
        self.db.update_category(eid, "entertainment")
        df = self.db.all()
        self.assertEqual(df.iloc[0]["category"], "entertainment")

        self.db.delete(eid)
        self.assertTrue(self.db.all().empty)

    def test_clear_database(self):
        self.db.add(50.0, "food", "tea")
        self.db.add(80.0, "travel", "bus")
        self.assertEqual(len(self.db.all()), 2)
        self.db.clear()
        self.assertEqual(len(self.db.all()), 0)


class TestCategorizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Train once for tests
        cls.train_result = categorizer.train(save=True)
        cls.model = categorizer.load()

    def test_obvious_cases(self):
        cat_netflix, conf_netflix = categorizer.predict("netflix subscription", self.model)
        self.assertEqual(cat_netflix, "entertainment")
        self.assertGreater(conf_netflix, 0.3)

        cat_uber, conf_uber = categorizer.predict("uber ride", self.model)
        self.assertEqual(cat_uber, "travel")
        self.assertGreater(conf_uber, 0.3)

    def test_hard_test_evaluation(self):
        hard_acc = categorizer.hard_accuracy(self.model)
        self.assertIsNotNone(hard_acc)
        self.assertTrue(0.0 <= hard_acc <= 1.0)

    def test_metrics_json_generated(self):
        self.assertTrue(categorizer.METRICS_PATH.exists())
        data = json.loads(categorizer.METRICS_PATH.read_text(encoding="utf-8"))
        self.assertIn("cv_scores", data)
        self.assertIn("best_model", data)
        self.assertIn("test_accuracy", data)

    def test_feedback_loop(self):
        categorizer.add_feedback("special tea stall", "food")
        self.assertTrue(categorizer.FEEDBACK_PATH.exists())
        df = categorizer.load_dataset()
        self.assertIn("special tea stall", df["text"].values)

    def test_confusion_matrix_created(self):
        cm_path = Path("docs") / "confusion_matrix.png"
        self.assertTrue(cm_path.exists())
        self.assertGreater(cm_path.stat().st_size, 1000)

    def test_embed_module(self):
        from spendsense import embed
        self.assertFalse(embed.available())
        pipe = embed.make_pipeline()
        self.assertIsNotNone(pipe)


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


class TestSMS(unittest.TestCase):
    def test_upi_debit_zomato(self):
        text = "Rs 450.00 debited from A/c XX1234 on 05-11-26 to ZOMATO. UPI Ref 123456"
        res = sms.parse_sms(text)
        self.assertIsNotNone(res)
        self.assertEqual(res["amount"], 450.0)
        self.assertEqual(res["type"], "debit")
        self.assertEqual(res["merchant"], "zomato")
        self.assertEqual(res["date"], "2026-11-05")

    def test_card_spent_amazon(self):
        text = "INR 1,299.00 spent on HDFC Bank Card XX4321 at AMAZON on 03-Nov-2026"
        res = sms.parse_sms(text)
        self.assertIsNotNone(res)
        self.assertEqual(res["amount"], 1299.0)
        self.assertEqual(res["type"], "debit")
        self.assertEqual(res["merchant"], "amazon")
        self.assertEqual(res["date"], "2026-11-03")

    def test_kotak_sent_upi_handle(self):
        text = "Sent Rs.250.00 from Kotak Bank AC X1234 to ramesh@upi on 02-11-26"
        res = sms.parse_sms(text)
        self.assertIsNotNone(res)
        self.assertEqual(res["amount"], 250.0)
        self.assertEqual(res["type"], "debit")
        self.assertEqual(res["merchant"], "ramesh")

    def test_debited_and_credited_to(self):
        text = "Your a/c no. XX1234 is debited for Rs.89.00 on 04-11-2026 and credited to SWIGGY"
        res = sms.parse_sms(text)
        self.assertIsNotNone(res)
        self.assertEqual(res["amount"], 89.0)
        self.assertEqual(res["type"], "debit")
        self.assertEqual(res["merchant"], "swiggy")

    def test_account_credited(self):
        text = "Rs 5000 credited to your account on 01-11-26"
        res = sms.parse_sms(text)
        self.assertIsNotNone(res)
        self.assertEqual(res["amount"], 5000.0)
        self.assertEqual(res["type"], "credit")

    def test_otp_returns_none(self):
        text = "Your OTP is 123456"
        res = sms.parse_sms(text)
        self.assertIsNone(res)


class TestForecast(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = ExpenseDB(Path(self.temp_dir.name) / "forecast.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_insufficient_history_returns_none(self):
        self.db.add(500, "food", "dinner", "2026-01-15")
        self.db.add(700, "travel", "flight", "2026-02-15")
        res = forecast.forecast_next_month(self.db.all())
        self.assertIsNone(res)

    def test_forecast_on_demo_data(self):
        rows = generate_expenses(months=12, seed=42)
        self.db.add_many(rows)
        df = self.db.all()
        res = forecast.forecast_next_month(df)
        self.assertIsNotNone(res)
        self.assertIn("prediction", res)
        self.assertGreater(res["prediction"], 0)
        self.assertIn(res["model"], forecast.FORECAST_MODELS)
        self.assertIn("backtest", res)
        self.assertIn("last_month", res["backtest"])
        self.assertIn("holt", res["backtest"])

        cat_f = forecast.forecast_by_category(df)
        self.assertGreaterEqual(len(cat_f), 5)


if __name__ == "__main__":
    unittest.main()
