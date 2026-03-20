import os
import shutil
import unittest
import uuid

import pandas as pd

from engine.analysis_service import run_analysis


class RunAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = os.path.join(os.getcwd(), "tests", "_tmp", str(uuid.uuid4()))
        os.makedirs(self.temp_dir, exist_ok=True)
        self.data_path = os.path.join(self.temp_dir, "transactions.csv")

        df = pd.DataFrame(
            [
                {
                    "client_id": 1,
                    "card_type": "Credit",
                    "merchant_id": 100,
                    "merchant_category": "Grocery",
                    "merchant_city": "New York",
                    "amount": 120.0,
                    "is_fraud": 1,
                    "timestamp": "2024-01-10 10:00:00",
                    "credit_score": 720,
                    "yearly_income": 50000,
                },
                {
                    "client_id": 1,
                    "card_type": "Credit",
                    "merchant_id": 101,
                    "merchant_category": "Travel",
                    "merchant_city": "Boston",
                    "amount": 80.0,
                    "is_fraud": 0,
                    "timestamp": "2024-01-25 15:00:00",
                    "credit_score": 720,
                    "yearly_income": 50000,
                },
                {
                    "client_id": 2,
                    "card_type": "Debit",
                    "merchant_id": 200,
                    "merchant_category": "Grocery",
                    "merchant_city": "New York",
                    "amount": 200.0,
                    "is_fraud": 1,
                    "timestamp": "2024-02-03 08:30:00",
                    "credit_score": 680,
                    "yearly_income": 40000,
                },
                {
                    "client_id": 3,
                    "card_type": "Debit",
                    "merchant_id": 201,
                    "merchant_category": "Travel",
                    "merchant_city": "Chicago",
                    "amount": 300.0,
                    "is_fraud": 20000,
                    "timestamp": "2024-02-15 09:00:00",
                    "credit_score": 650,
                    "yearly_income": 45000,
                },
            ]
        )
        df.to_csv(self.data_path, index=False)
        self.previous_data_path = os.environ.get("ANALYTICS_DATA_PATH")
        os.environ["ANALYTICS_DATA_PATH"] = self.data_path

    def tearDown(self):
        if self.previous_data_path is None:
            os.environ.pop("ANALYTICS_DATA_PATH", None)
        else:
            os.environ["ANALYTICS_DATA_PATH"] = self.previous_data_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_grouped_sum_with_filters(self):
        result = run_analysis(
            filters={"client_id": {"eq": 1}},
            group_by=["merchant_city"],
            metrics=["sum(amount)"],
            sort_by="sum(amount)",
            order="desc",
        )

        self.assertEqual(result["meta"]["row_count"], 2)
        self.assertEqual(result["meta"]["invalid_records"], 1)
        self.assertEqual(result["data"][0]["merchant_city"], "New York")
        self.assertEqual(result["data"][0]["sum(amount)"], 120.0)

    def test_fraud_rate_ignores_invalid_labels(self):
        result = run_analysis(
            group_by=["card_type"],
            metrics=["count(*)", "fraud_rate"],
            sort_by="fraud_rate",
            order="desc",
        )

        self.assertEqual(result["meta"]["invalid_records"], 1)
        by_card_type = {row["card_type"]: row for row in result["data"]}
        self.assertAlmostEqual(by_card_type["Credit"]["fraud_rate"], 0.5)
        self.assertAlmostEqual(by_card_type["Debit"]["fraud_rate"], 1.0)

    def test_time_grain_groups_by_month(self):
        result = run_analysis(
            metrics=["sum(amount)", "count(*)"],
            time_grain="month",
            sort_by="time_grain",
            order="asc",
        )

        self.assertEqual([row["time_grain"] for row in result["data"]], ["2024-01", "2024-02"])
        self.assertEqual(result["data"][0]["count(*)"], 2)
        self.assertEqual(result["data"][1]["sum(amount)"], 500.0)

    def test_recent_months_filters_to_latest_window(self):
        result = run_analysis(
            metrics=["sum(amount)", "count(*)", "distinct_count(client_id)"],
            recent_months=1,
        )

        self.assertEqual(result["meta"]["recent_months"], 1)
        self.assertEqual(result["meta"]["recent_window_anchor"], "2024-02-15")
        self.assertEqual(result["meta"]["recent_window_cutoff"], "2024-01-16")
        self.assertEqual(result["data"][0]["count(*)"], 3)
        self.assertEqual(result["data"][0]["distinct_count(client_id)"], 3)
        self.assertEqual(result["data"][0]["sum(amount)"], 580.0)

    def test_invalid_metric_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported metric"):
            run_analysis(metrics=["median(amount)"])

    def test_limit_validation_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "limit must be <= 1000"):
            run_analysis(metrics=["count(*)"], limit=1001)


if __name__ == "__main__":
    unittest.main()
