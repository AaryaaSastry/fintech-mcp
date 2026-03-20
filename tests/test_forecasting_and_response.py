import os
import shutil
import unittest
import uuid

import pandas as pd

from backend.ai_response_generator import AIResponseGenerator
from engine.forecasting import run_forecast_analysis


class ForecastingAndResponseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = os.path.join(os.getcwd(), "tests", "_tmp", str(uuid.uuid4()))
        os.makedirs(self.temp_dir, exist_ok=True)
        self.data_path = os.path.join(self.temp_dir, "transactions.csv")
        self.previous_data_path = os.environ.get("ANALYTICS_DATA_PATH")

    def tearDown(self):
        if self.previous_data_path is None:
            os.environ.pop("ANALYTICS_DATA_PATH", None)
        else:
            os.environ["ANALYTICS_DATA_PATH"] = self.previous_data_path
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_dataset(self, rows):
        pd.DataFrame(rows).to_csv(self.data_path, index=False)
        os.environ["ANALYTICS_DATA_PATH"] = self.data_path

    def test_sparse_stale_history_is_marked_low_confidence(self):
        self._write_dataset(
            [
                {"client_id": 3, "amount": 980.0, "timestamp": "2016-05-23 17:35:00"},
                {"client_id": 3, "amount": 18.25, "timestamp": "2017-02-18 14:50:00"},
                {"client_id": 3, "amount": 120.0, "timestamp": "2017-11-11 12:10:00"},
                {"client_id": 3, "amount": 430.75, "timestamp": "2019-02-15 18:20:00"},
            ]
        )

        result = run_forecast_analysis(client_id=3, months=3)

        self.assertEqual(result["avg_historical_monthly"], 387.25)
        self.assertEqual(result["total_months_data"], 4)
        self.assertEqual(result["confidence"], "low")
        self.assertIn("limited_history", result["warnings"])
        self.assertIn("sparse_month_coverage", result["warnings"])
        self.assertIn("stale_history", result["warnings"])
        self.assertEqual(result["history_start_month"], "2016-05")
        self.assertEqual(result["history_end_month"], "2019-02")
        self.assertEqual(result["latest_transaction_date"], "2019-02-15")

    def test_fallback_response_explains_low_confidence_forecast(self):
        generator = AIResponseGenerator()

        response = generator._fallback_response(
            {
                "run_forecast_analysis": {
                    "forecast": {"month_1": 387.25, "month_2": 387.25, "month_3": 387.25},
                    "confidence": "low",
                    "total_months_data": 4,
                    "latest_transaction_date": "2019-02-15",
                    "warnings": ["limited_history", "sparse_month_coverage", "stale_history"],
                }
            },
            {"client_id": 3, "is_all_users": False, "user_name": None},
        )

        self.assertIn("low-confidence estimate", response)
        self.assertIn("4 historical spending months", response)
        self.assertIn("2019-02-15", response)
        self.assertIn("spread out rather than continuous", response)

    def test_fallback_response_formats_all_users_per_client(self):
        generator = AIResponseGenerator()

        response = generator._fallback_response(
            {
                "run_aggregate_analysis": {
                    "data": [
                        {"client_id": 1, "sum(amount)": 120.0, "count(*)": 2},
                        {"client_id": 2, "sum(amount)": 300.0, "count(*)": 1},
                    ],
                    "meta": {"row_count": 2},
                }
            },
            {"client_id": None, "is_all_users": True, "user_name": None},
        )

        self.assertIn("client 1: total $120.00, 2 transactions", response)
        self.assertIn("client 2: total $300.00, 1 transactions", response)

    def test_fallback_response_structures_fraud_categories_by_client(self):
        generator = AIResponseGenerator()

        response = generator._fallback_response(
            {
                "run_fraud_analysis": {
                    "total_transactions": 100,
                    "fraud_count": 27,
                    "fraud_rate": 0.27,
                    "top_fraudulent_clients": [
                        {"client_id": 3, "fraud_rate": 0.5, "fraud_count": 2},
                        {"client_id": 6, "fraud_rate": 0.5, "fraud_count": 2},
                    ],
                },
                "run_aggregate_analysis": {
                    "data": [
                        {"client_id": 3, "merchant_category": "Miscellaneous Retail Stores", "sum(amount)": 1410.75, "count(*)": 2},
                        {"client_id": 3, "merchant_category": "Eating Places Restaurants", "sum(amount)": 120.0, "count(*)": 1},
                        {"client_id": 6, "merchant_category": "Miscellaneous Retail Stores", "sum(amount)": 750.0, "count(*)": 2},
                        {"client_id": 6, "merchant_category": "Grocery Stores", "sum(amount)": 65.5, "count(*)": 1},
                        {"client_id": 20, "merchant_category": "Fast Food Restaurants", "sum(amount)": 12.75, "count(*)": 1},
                    ],
                    "meta": {"row_count": 5},
                },
            },
            {"client_id": None, "is_all_users": True, "user_name": None},
        )

        self.assertIn("Top clients with flagged activity by category:", response)
        self.assertIn("client 3: fraud rate 50.00%, 2 flagged transactions;", response)
        self.assertIn("Miscellaneous Retail Stores ($1,410.75, 2 transactions)", response)
        self.assertIn("client 6: fraud rate 50.00%, 2 flagged transactions;", response)
        self.assertNotIn("client 20", response)


if __name__ == "__main__":
    unittest.main()
