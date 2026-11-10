"""Synthetic contract tests for audit_point_in_time.py (stdlib only)."""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "rae" / "audit_point_in_time.py"
SPEC = importlib.util.spec_from_file_location("audit_point_in_time", MODULE_PATH)
pit = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(pit)


def bundle() -> dict:
    return {
        "schema_version": "1.0",
        "origins": [{"origin_id": "O1", "origin_month": "2024-01", "origin_time_utc": "2024-02-10T01:30:00Z"}],
        "required_features_by_model": {"M": ["CPI"]},
        "inputs": [{"origin_id": "O1", "feature": "CPI", "reference_month": "2023-12",
                    "available_at_utc": "2024-01-15T00:00:00Z", "vintage_id": "v1",
                    "expected_vintage_id": "v1", "vintage_status": "ACTIVE", "masked": False, "value": 0.4}],
        "planned_forecasts": [{"model": "M", "origin_id": "O1", "horizon_months": 1, "target_month": "2024-02"}],
        "forecasts": [{"model": "M", "origin_id": "O1", "horizon_months": 1, "target_month": "2024-02",
                       "status": "FORECAST_COMPLETE", "prediction": 0.5}],
        "expected_origin_ids_by_model_horizon": {"M|1": ["O1"]},
    }


def codes(report: dict) -> set[str]:
    return {item["code"] for item in report["issues"]}


class PointInTimeAuditTests(unittest.TestCase):
    def test_01_valid_bundle_passes(self):
        report, rows = pit.audit_bundle(bundle())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["policy"]["input_time_rule"], "unmasked availability strictly before origin")
        self.assertTrue(rows[0]["evaluation_eligible"])

    def test_02_late_availability_is_rejected(self):
        b = bundle(); b["inputs"][0]["available_at_utc"] = "2024-02-10T01:31:00Z"
        self.assertIn("INPUT_AVAILABLE_AFTER_ORIGIN", codes(pit.audit_bundle(b)[0]))

    def test_03_future_context_month_is_rejected(self):
        b = bundle(); b["inputs"][0]["reference_month"] = "2024-02"
        self.assertIn("INPUT_REFERENCE_AFTER_ORIGIN", codes(pit.audit_bundle(b)[0]))

    def test_04_stale_vintage_identifier_is_rejected(self):
        b = bundle(); b["inputs"][0]["vintage_id"] = "old-v0"
        self.assertIn("VINTAGE_ID_MISMATCH", codes(pit.audit_bundle(b)[0]))

    def test_05_overwritten_vintage_is_rejected(self):
        b = bundle(); b["inputs"][0]["vintage_status"] = "OVERWRITTEN"
        self.assertIn("VINTAGE_OVERWRITTEN", codes(pit.audit_bundle(b)[0]))

    def test_06_target_horizon_mismatch_is_rejected(self):
        b = bundle(); b["planned_forecasts"][0]["target_month"] = "2024-03"
        self.assertIn("TARGET_HORIZON_MISMATCH", codes(pit.audit_bundle(b)[0]))

    def test_07_model_origin_set_mismatch_is_rejected(self):
        b = bundle(); b["expected_origin_ids_by_model_horizon"]["M|1"] = ["O1", "O2"]
        self.assertIn("MODEL_ORIGIN_SET_MISMATCH", codes(pit.audit_bundle(b)[0]))

    def test_08_missing_forecast_row_is_rejected(self):
        b = bundle(); b["forecasts"] = []
        self.assertIn("MISSING_FORECAST_ROW", codes(pit.audit_bundle(b)[0]))

    def test_09_explicit_fit_failure_is_retained_and_not_scored(self):
        b = bundle(); b["forecasts"][0] = {"model": "M", "origin_id": "O1", "horizon_months": 1,
            "target_month": "2024-02", "status": "FIT_FAILED", "failure_code": "NON_CONVERGENCE"}
        report, rows = pit.audit_bundle(b)
        self.assertIn("EXPLICIT_FIT_OR_PREDICTION_FAILURE", codes(report))
        self.assertFalse(rows[0]["evaluation_eligible"])
        self.assertIn("FIT_FAILED", rows[0]["reason_codes"])

    def test_10_expected_masked_input_is_ineligible_without_error(self):
        b = bundle(); b["inputs"][0].update(masked=True, value=None)
        b["forecasts"][0] = {"model": "M", "origin_id": "O1", "horizon_months": 1,
            "target_month": "2024-02", "status": "INPUT_INELIGIBLE"}
        report, rows = pit.audit_bundle(b)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(rows[0]["evaluation_eligible"])
        self.assertIn("REQUIRED_INPUT_MISSING_OR_MASKED:CPI", rows[0]["reason_codes"])

    def test_11_output_overwrite_is_refused(self):
        report, rows = pit.audit_bundle(bundle())
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); (out / "audit_report.json").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                pit.write_outputs(report, rows, out)
            self.assertEqual((out / "audit_report.json").read_text(encoding="utf-8"), "keep")

    def test_12_target_value_fields_are_forbidden(self):
        b = bundle(); b["forecasts"][0]["target_value"] = 0.6
        self.assertIn("TARGET_VALUE_FIELD_FORBIDDEN", codes(pit.audit_bundle(b)[0]))

    def test_13_equal_availability_is_rejected(self):
        b = bundle(); b["inputs"][0]["available_at_utc"] = b["origins"][0]["origin_time_utc"]
        report, rows = pit.audit_bundle(b)
        self.assertIn("INPUT_AVAILABLE_AT_ORIGIN", codes(report))
        self.assertFalse(rows[0]["input_eligible"])
        self.assertFalse(rows[0]["evaluation_eligible"])

    def test_14_late_masked_input_is_expected_ineligibility(self):
        b = bundle(); b["inputs"][0].update(masked=True, value=None, available_at_utc="2024-02-11T01:30:00Z")
        b["forecasts"][0] = {"model": "M", "origin_id": "O1", "horizon_months": 1,
            "target_month": "2024-02", "status": "INPUT_INELIGIBLE"}
        report, rows = pit.audit_bundle(b)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(rows[0]["evaluation_eligible"])

    def test_15_masked_input_with_unknown_time_is_expected_ineligibility(self):
        b = bundle(); b["inputs"][0].update(masked=True, value=None)
        del b["inputs"][0]["available_at_utc"]
        b["forecasts"][0] = {"model": "M", "origin_id": "O1", "horizon_months": 1,
            "target_month": "2024-02", "status": "INPUT_INELIGIBLE"}
        report, rows = pit.audit_bundle(b)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(rows[0]["evaluation_eligible"])

    def test_16_availability_just_before_origin_is_eligible(self):
        b = bundle(); b["inputs"][0]["available_at_utc"] = "2024-02-10T01:29:59.999999Z"
        report, rows = pit.audit_bundle(b)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(rows[0]["evaluation_eligible"])

    def test_17_nonfinite_input_is_ineligible(self):
        b = bundle(); b["inputs"][0]["value"] = float("nan")
        report, rows = pit.audit_bundle(b)
        self.assertIn("INVALID_INPUT_VALUE", codes(report))
        self.assertFalse(rows[0]["evaluation_eligible"])

    def test_18_undeclared_model_requirements_are_rejected(self):
        b = bundle(); b["required_features_by_model"] = {}
        report, rows = pit.audit_bundle(b)
        self.assertIn("MODEL_REQUIREMENTS_UNDECLARED", codes(report))
        self.assertFalse(rows[0]["evaluation_eligible"])

    def test_19_invalid_forecast_is_ineligible(self):
        b = bundle(); b["forecasts"][0]["prediction"] = float("nan")
        report, rows = pit.audit_bundle(b)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("INVALID_FORECAST_VALUE", codes(report))
        self.assertFalse(rows[0]["forecast_complete"])
        self.assertFalse(rows[0]["evaluation_eligible"])
        self.assertIn("INVALID_FORECAST_VALUE", rows[0]["reason_codes"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
