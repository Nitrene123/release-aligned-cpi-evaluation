"""Model-origin feature/lag/reference input eligibility contract tests."""
from __future__ import annotations

import copy
import importlib.util
import math
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
MODULE_CANDIDATES = [HERE.with_name("audit_point_in_time.py"),
                     HERE.parents[1] / "src" / "rae" / "audit_point_in_time.py"]
MODULE_PATH = next(path for path in MODULE_CANDIDATES if path.exists())
SPEC = importlib.util.spec_from_file_location("audit_point_in_time", MODULE_PATH)
pit = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(pit)


def exact_bundle() -> dict:
    return {
        "schema_version": "1.1",
        "origins": [{"origin_id": "O1", "origin_month": "2024-01",
                     "origin_time_utc": "2024-02-10T01:30:00Z"}],
        "required_inputs_by_model_origin": {
            "M": {"O1": [{"feature": "CPI", "lag": 1, "reference_month": "2023-12"}]}
        },
        "inputs": [{"origin_id": "O1", "model": "M", "feature": "CPI", "lag": 1,
                    "reference_month": "2023-12", "available_at_utc": "2024-01-15T00:00:00Z",
                    "vintage_id": "v1", "expected_vintage_id": "v1", "vintage_status": "ACTIVE",
                    "masked": False, "value": 0.4}],
        "planned_forecasts": [{"model": "M", "origin_id": "O1", "horizon_months": 1,
                               "target_month": "2024-02"}],
        "forecasts": [{"model": "M", "origin_id": "O1", "horizon_months": 1,
                       "target_month": "2024-02", "status": "FORECAST_COMPLETE", "prediction": 0.5}],
    }


def reason_set(row: dict) -> set[str]:
    return set(filter(None, row["reason_codes"].split(";")))


class ModelInputEligibilityTests(unittest.TestCase):
    def test_exact_feature_lag_reference_contract_passes(self):
        report, rows = pit.audit_bundle(exact_bundle())
        self.assertEqual(report["status"], "PASS", report["issues"])
        self.assertTrue(rows[0]["input_eligible"])
        self.assertTrue(rows[0]["evaluation_eligible"])
        self.assertEqual(report["coverage_by_model_horizon"][0]["planned_origin_count"], 1)

    def test_missing_required_lag_is_ineligible(self):
        b = exact_bundle()
        b["required_inputs_by_model_origin"]["M"]["O1"].append(
            {"feature": "CPI", "lag": 2, "reference_month": "2023-11"})
        b["forecasts"][0]["status"] = "INPUT_INELIGIBLE"
        b["forecasts"][0].pop("prediction")
        report, rows = pit.audit_bundle(b)
        self.assertFalse(rows[0]["input_eligible"])
        self.assertIn("CPI@lag2[2023-11]", rows[0]["reason_codes"])

    def test_required_input_row_without_lag_metadata_is_rejected(self):
        b = exact_bundle()
        b["inputs"][0].pop("lag")
        b["forecasts"][0]["status"] = "INPUT_INELIGIBLE"
        b["forecasts"][0].pop("prediction")
        report, rows = pit.audit_bundle(b)
        self.assertIn("INPUT_LAG_MISSING", {x["code"] for x in report["issues"]})
        self.assertFalse(rows[0]["input_eligible"])

    def test_model_scoped_input_cannot_satisfy_another_models_contract(self):
        b = exact_bundle()
        b["required_inputs_by_model_origin"]["M2"] = {
            "O1": [{"feature": "CPI", "lag": 1, "reference_month": "2023-12"}]
        }
        b["planned_forecasts"].append({"model": "M2", "origin_id": "O1", "horizon_months": 1,
                                       "target_month": "2024-02"})
        b["forecasts"].append({"model": "M2", "origin_id": "O1", "horizon_months": 1,
                               "target_month": "2024-02", "status": "INPUT_INELIGIBLE"})
        report, rows = pit.audit_bundle(b)
        row = next(r for r in rows if r["model"] == "M2")
        self.assertFalse(row["input_eligible"])
        self.assertIn("CPI@lag1[2023-12]", row["reason_codes"])

    def test_unrelated_row_is_not_subject_to_another_origins_lag_contract(self):
        b = exact_bundle()
        b["origins"].append({"origin_id": "O2", "origin_month": "2024-02",
                             "origin_time_utc": "2024-03-10T01:30:00Z"})
        b["required_inputs_by_model_origin"]["M"]["O2"] = [
            {"feature": "CPI", "lag": 1, "reference_month": "2024-01"}]
        b["inputs"].append({"origin_id": "O1", "model": "M", "feature": "CPI",
                            "reference_month": "2024-01", "available_at_utc": "2024-02-01T00:00:00Z",
                            "masked": False, "value": 0.3})
        report, rows = pit.audit_bundle(b)
        self.assertNotIn("INPUT_LAG_MISSING", {x["code"] for x in report["issues"]})
        self.assertTrue(rows[0]["evaluation_eligible"])

    def test_nan_and_infinity_are_not_eligible(self):
        for bad_value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=bad_value):
                b = exact_bundle()
                b["inputs"][0]["value"] = bad_value
                b["forecasts"][0]["status"] = "INPUT_INELIGIBLE"
                b["forecasts"][0].pop("prediction")
                report, rows = pit.audit_bundle(b)
                self.assertIn("INVALID_INPUT_VALUE", {x["code"] for x in report["issues"]})
                self.assertFalse(rows[0]["input_eligible"])

    def test_masked_required_feature_is_not_eligible(self):
        b = exact_bundle()
        b["inputs"][0].update(masked=True, value=None)
        b["forecasts"][0]["status"] = "INPUT_INELIGIBLE"
        b["forecasts"][0].pop("prediction")
        report, rows = pit.audit_bundle(b)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(rows[0]["input_eligible"])
        self.assertIn("CPI@lag1[2023-12]", rows[0]["reason_codes"])

    def test_duplicate_feature_lag_is_rejected(self):
        b = exact_bundle()
        duplicate = copy.deepcopy(b["inputs"][0])
        b["inputs"].append(duplicate)
        report, rows = pit.audit_bundle(b)
        self.assertIn("DUPLICATE_INPUT_FEATURE_LAG", {x["code"] for x in report["issues"]})
        self.assertFalse(rows[0]["input_eligible"])

    def test_lag_reference_inconsistency_is_rejected(self):
        b = exact_bundle()
        b["inputs"][0]["reference_month"] = "2023-11"
        b["forecasts"][0]["status"] = "INPUT_INELIGIBLE"
        b["forecasts"][0].pop("prediction")
        report, rows = pit.audit_bundle(b)
        self.assertIn("INPUT_LAG_REFERENCE_MISMATCH", {x["code"] for x in report["issues"]})
        self.assertFalse(rows[0]["input_eligible"])

    def test_late_required_input_is_not_eligible(self):
        b = exact_bundle()
        b["inputs"][0]["available_at_utc"] = "2024-02-10T01:30:00Z"
        b["forecasts"][0]["status"] = "INPUT_INELIGIBLE"
        b["forecasts"][0].pop("prediction")
        report, rows = pit.audit_bundle(b)
        self.assertIn("INPUT_AVAILABLE_AT_ORIGIN", {x["code"] for x in report["issues"]})
        self.assertFalse(rows[0]["input_eligible"])

    def test_forecast_from_missing_required_input_is_reported_and_not_scored(self):
        b = exact_bundle()
        b["inputs"] = []
        report, rows = pit.audit_bundle(b)
        self.assertIn("FORECAST_FROM_INELIGIBLE_INPUT", {x["code"] for x in report["issues"]})
        self.assertFalse(rows[0]["input_eligible"])
        self.assertFalse(rows[0]["evaluation_eligible"])

    def test_ineligible_origin_remains_in_planned_coverage_denominator(self):
        b = exact_bundle()
        b["origins"].append({"origin_id": "O2", "origin_month": "2024-02",
                             "origin_time_utc": "2024-03-10T01:30:00Z"})
        b["required_inputs_by_model_origin"]["M"]["O2"] = [
            {"feature": "CPI", "lag": 1, "reference_month": "2024-01"}]
        second = copy.deepcopy(b["inputs"][0])
        second.update(origin_id="O2", reference_month="2024-01", available_at_utc="2024-02-15T00:00:00Z",
                      masked=True, value=None)
        b["inputs"].append(second)
        b["planned_forecasts"].append({"model": "M", "origin_id": "O2", "horizon_months": 1,
                                       "target_month": "2024-03"})
        b["forecasts"].append({"model": "M", "origin_id": "O2", "horizon_months": 1,
                               "target_month": "2024-03", "status": "FORECAST_COMPLETE", "prediction": 0.6})
        report, rows = pit.audit_bundle(b)
        coverage = report["coverage_by_model_horizon"][0]
        self.assertEqual(coverage["planned_origin_count"], 2)
        self.assertEqual(coverage["input_eligible_origin_count"], 1)
        self.assertEqual(coverage["completed_forecasts_on_eligible_inputs"], 1)
        self.assertEqual(coverage["completed_forecasts_on_ineligible_inputs"], 1)
        self.assertEqual(coverage["evaluation_eligible_origin_count"], 1)
        self.assertEqual(coverage["eligible_forecast_coverage_rate_over_planned"], 0.5)
        self.assertIn("FORECAST_FROM_INELIGIBLE_INPUT", {x["code"] for x in report["issues"]})
        self.assertFalse(next(r for r in rows if r["origin_id"] == "O2")["evaluation_eligible"])

    def test_rich_model_must_declare_each_planned_origin(self):
        b = exact_bundle()
        b["required_inputs_by_model_origin"]["M"].clear()
        report, rows = pit.audit_bundle(b)
        self.assertIn("MODEL_ORIGIN_REQUIREMENTS_MISSING", {x["code"] for x in report["issues"]})
        self.assertFalse(rows[0]["input_eligible"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
