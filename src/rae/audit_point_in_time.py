#!/usr/bin/env python3
"""Target-free point-in-time contract audit for forecast bundles (stdlib only).

Input is JSON with origins, model-specific required features, vintage-tagged
inputs, planned forecast keys, and forecast/status rows. No numeric target
values are accepted. See PIT_AUDIT_README.md for the schema and limits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def _month_index(month: str) -> int:
    if len(month) != 7 or month[4] != "-":
        raise ValueError(f"invalid YYYY-MM month: {month!r}")
    year, mon = int(month[:4]), int(month[5:])
    if not 1 <= mon <= 12:
        raise ValueError(f"invalid YYYY-MM month: {month!r}")
    return year * 12 + mon - 1


def add_month(month: str, offset: int) -> str:
    value = _month_index(month) + int(offset)
    return f"{value // 12:04d}-{value % 12 + 1:02d}"


def _aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include a UTC offset: {value!r}")
    return parsed


def _finite_value(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def audit_bundle(bundle: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a target-free report and one eligibility row per planned key."""
    issues: list[dict[str, Any]] = []

    def issue(code: str, severity: str, message: str, key: str = "") -> None:
        issues.append({"code": code, "severity": severity, "key": key, "message": message})

    def forbid_target_values(rows: list[dict[str, Any]], label: str) -> None:
        for i, row in enumerate(rows):
            if any(k.lower() in {"target_value", "observed_target", "actual_target", "label_value"}
                   for k in row):
                issue("TARGET_VALUE_FIELD_FORBIDDEN", "error",
                      f"{label}[{i}] contains a numeric target-value field", str(i))

    origins_list = bundle.get("origins", [])
    inputs = bundle.get("inputs", [])
    planned = bundle.get("planned_forecasts", [])
    forecasts = bundle.get("forecasts", [])
    forbid_target_values(inputs, "inputs")
    forbid_target_values(forecasts, "forecasts")

    origins: dict[str, dict[str, Any]] = {}
    for row in origins_list:
        oid = str(row.get("origin_id", ""))
        if not oid or oid in origins:
            issue("DUPLICATE_OR_INVALID_ORIGIN", "error", "origin_id is blank or duplicated", oid)
            continue
        try:
            _month_index(str(row["origin_month"]))
            origin_time = _aware_datetime(str(row["origin_time_utc"]))
        except (KeyError, ValueError, TypeError) as exc:
            issue("INVALID_ORIGIN_CLOCK", "error", str(exc), oid)
            continue
        origins[oid] = {**row, "_origin_time": origin_time}

    requirements = bundle.get("required_features_by_model", {})
    model_names = set(requirements)
    inputs_by_origin: dict[str, list[dict[str, Any]]] = {}
    seen_inputs: set[tuple[str, str, str]] = set()
    for row in inputs:
        oid, feature = str(row.get("origin_id", "")), str(row.get("feature", ""))
        ref_month = str(row.get("reference_month", ""))
        key = (oid, feature, ref_month)
        if key in seen_inputs:
            issue("DUPLICATE_INPUT_KEY", "error", "duplicate origin/feature/reference-month key", "|".join(key))
        seen_inputs.add(key)
        origin = origins.get(oid)
        if origin is None:
            issue("INPUT_ORIGIN_UNKNOWN", "error", "input references an unknown origin", oid)
            continue
        inputs_by_origin.setdefault(oid, []).append(row)
        try:
            if _month_index(ref_month) > _month_index(str(origin["origin_month"])):
                issue("INPUT_REFERENCE_AFTER_ORIGIN", "error",
                      "context reference month is later than the forecast-origin month", "|".join(key))
            masked = bool(row.get("masked", False))
            available_raw = row.get("available_at_utc")
            if available_raw in (None, ""):
                if not masked:
                    raise ValueError("unmasked input has no availability timestamp")
            else:
                available = _aware_datetime(str(available_raw))
                if not masked and available > origin["_origin_time"]:
                    issue("INPUT_AVAILABLE_AFTER_ORIGIN", "error",
                          "unmasked input becomes available after the forecast origin", "|".join(key))
                elif not masked and available == origin["_origin_time"]:
                    issue("INPUT_AVAILABLE_AT_ORIGIN", "error",
                          "unmasked input must become available strictly before the forecast origin", "|".join(key))
        except (KeyError, ValueError, TypeError) as exc:
            issue("INVALID_INPUT_CLOCK", "error", str(exc), "|".join(key))
        if row.get("expected_vintage_id") not in (None, "") and str(row.get("vintage_id")) != str(row.get("expected_vintage_id")):
            issue("VINTAGE_ID_MISMATCH", "error", "selected vintage differs from the frozen expected vintage", "|".join(key))
        if str(row.get("vintage_status", "ACTIVE")).upper() in {"OVERWRITTEN", "SUPERSEDED", "REVOKED"}:
            issue("VINTAGE_OVERWRITTEN", "error", "selected vintage is marked overwritten/superseded", "|".join(key))
        if not bool(row.get("masked", False)):
            try:
                if row.get("value") is None or not math.isfinite(float(row["value"])):
                    issue("INVALID_INPUT_VALUE", "error", "unmasked input value is missing or non-finite", "|".join(key))
            except (KeyError, ValueError, TypeError):
                issue("INVALID_INPUT_VALUE", "error", "unmasked input value is missing or non-numeric", "|".join(key))

    planned_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in planned:
        try:
            model, oid = str(row["model"]), str(row["origin_id"])
            h = int(row["horizon_months"])
            target_month = str(row["target_month"])
            key = (model, oid, h)
            if key in planned_by_key:
                issue("DUPLICATE_PLANNED_KEY", "error", "planned model/origin/horizon key is duplicated", repr(key))
            planned_by_key[key] = row
            if model not in model_names:
                issue("MODEL_REQUIREMENTS_UNDECLARED", "error",
                      "planned model has no declared input requirements", model)
            if oid not in origins:
                issue("PLANNED_ORIGIN_UNKNOWN", "error", "planned forecast references an unknown origin", repr(key))
            elif target_month != add_month(str(origins[oid]["origin_month"]), h):
                issue("TARGET_HORIZON_MISMATCH", "error", "target month does not equal origin month plus horizon", repr(key))
        except (KeyError, ValueError, TypeError) as exc:
            issue("INVALID_PLANNED_KEY", "error", str(exc), str(row))

    forecast_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in forecasts:
        try:
            key = (str(row["model"]), str(row["origin_id"]), int(row["horizon_months"]))
        except (KeyError, ValueError, TypeError) as exc:
            issue("INVALID_FORECAST_KEY", "error", str(exc), str(row))
            continue
        if key in forecast_by_key:
            issue("DUPLICATE_FORECAST_KEY", "error", "forecast model/origin/horizon key is duplicated", repr(key))
        forecast_by_key[key] = row
        if key not in planned_by_key:
            issue("UNPLANNED_FORECAST", "error", "forecast has no matching planned key", repr(key))
        elif str(row.get("target_month", "")) != str(planned_by_key[key].get("target_month", "")):
            issue("FORECAST_TARGET_MISMATCH", "error", "forecast target month differs from its planned key", repr(key))
        if str(row.get("status", "")).upper() == "FORECAST_COMPLETE":
            try:
                if not math.isfinite(float(row["prediction"])):
                    raise ValueError("prediction is non-finite")
            except (KeyError, ValueError, TypeError) as exc:
                issue("INVALID_FORECAST_VALUE", "error", str(exc), repr(key))
        elif str(row.get("status", "")).upper() in {"FIT_FAILED", "PREDICTION_FAILED"}:
            issue("EXPLICIT_FIT_OR_PREDICTION_FAILURE", "info",
                  "failed forecast is retained with an explicit status; it is not a zero error", repr(key))
        elif str(row.get("status", "")).upper() not in {"INPUT_INELIGIBLE"}:
            issue("INVALID_FORECAST_STATUS", "error", "forecast status is missing or unrecognized", repr(key))

    for key in planned_by_key:
        if key not in forecast_by_key:
            issue("MISSING_FORECAST_ROW", "error", "planned forecast key has no corresponding output/status row", repr(key))

    # Optional, frozen origin-set contract keyed as "model|horizon".
    expected_sets = bundle.get("expected_origin_ids_by_model_horizon", {})
    for label, expected in expected_sets.items():
        try:
            model, h_text = label.rsplit("|", 1)
            h = int(h_text)
            actual = {k[1] for k in forecast_by_key if k[0] == model and k[2] == h}
            if actual != set(map(str, expected)):
                issue("MODEL_ORIGIN_SET_MISMATCH", "error",
                      f"forecast origins differ from frozen expected set (expected {len(expected)}, found {len(actual)})", label)
        except (ValueError, TypeError):
            issue("INVALID_EXPECTED_ORIGIN_SET", "error", "expected-origin-set key must be model|horizon", str(label))

    eligibility_rows: list[dict[str, Any]] = []
    for key, plan in sorted(planned_by_key.items()):
        model, oid, h = key
        required = list(requirements.get(model, []))
        available_features = {
            str(row.get("feature")) for row in inputs_by_origin.get(oid, [])
            if not bool(row.get("masked", False))
            and _finite_value(row.get("value"))
            and str(row.get("vintage_status", "ACTIVE")).upper() not in {"OVERWRITTEN", "SUPERSEDED", "REVOKED"}
            and (row.get("expected_vintage_id") in (None, "") or str(row.get("vintage_id")) == str(row.get("expected_vintage_id")))
            and oid in origins
            and _input_time_valid(row, origins[oid])
        }
        missing = [feature for feature in required if feature not in available_features]
        forecast = forecast_by_key.get(key, {})
        status = str(forecast.get("status", "MISSING")).upper()
        reasons = []
        if model not in model_names:
            reasons.append("MODEL_REQUIREMENTS_UNDECLARED")
        if missing:
            reasons.append("REQUIRED_INPUT_MISSING_OR_MASKED:" + ",".join(missing))
        if status in {"FIT_FAILED", "PREDICTION_FAILED"}:
            reasons.append(status)
        if status == "MISSING":
            reasons.append("MISSING_FORECAST_ROW")
        complete = status == "FORECAST_COMPLETE" and _finite_value(forecast.get("prediction"))
        if status == "FORECAST_COMPLETE" and not complete:
            reasons.append("INVALID_FORECAST_VALUE")
        input_eligible = model in model_names and not missing
        eligible = input_eligible and complete
        eligibility_rows.append({
            "model": model, "origin_id": oid, "origin_month": origins.get(oid, {}).get("origin_month", ""),
            "horizon_months": h, "target_month": plan.get("target_month", ""),
            "input_eligible": input_eligible, "forecast_status": status,
            "forecast_complete": complete, "evaluation_eligible": eligible,
            "reason_codes": ";".join(reasons),
        })

    errors = sum(i["severity"] == "error" for i in issues)
    report = {
        "schema_version": "1.0",
        "status": "FAIL" if errors else "PASS",
        "target_values_read_or_required": False,
        "policy": {
            "input_time_rule": "unmasked availability strictly before origin",
            "masked_input_rule": "masked inputs do not satisfy required features; availability may be late or unknown",
            "vintage_rule": "checks supplied selected and expected vintage identifiers; does not enumerate vintage histories",
            "feature_completeness": "at least one eligible record per required feature; lag-chain completeness is external",
        },
        "counts": {
            "origins": len(origins_list), "input_rows": len(inputs),
            "planned_forecast_keys": len(planned_by_key), "forecast_rows": len(forecasts),
            "evaluation_eligible_keys": sum(bool(r["evaluation_eligible"]) for r in eligibility_rows),
            "issues_error": errors,
            "issues_info": sum(i["severity"] == "info" for i in issues),
        },
        "issues": issues,
        "limitations": [
            "Checks only the selected-vintage records supplied in the bundle; it cannot establish a complete historical vintage census.",
            "Metadata checks do not independently authenticate external publication timestamps.",
            "Feature-presence checks do not establish task-specific lag-chain completeness.",
            "The audit is target-free and does not compute forecast errors or scores.",
        ],
    }
    return report, eligibility_rows


def _input_time_valid(row: dict[str, Any], origin: dict[str, Any]) -> bool:
    try:
        return _aware_datetime(str(row["available_at_utc"])) < origin["_origin_time"] and _month_index(str(row["reference_month"])) <= _month_index(str(origin["origin_month"]))
    except (KeyError, ValueError, TypeError):
        return False


def write_outputs(report: dict[str, Any], eligibility: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "audit_report.json"
    eligibility_path = out_dir / "eligibility.csv"
    existing = [p.name for p in (report_path, eligibility_path) if p.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite audit outputs: " + ", ".join(existing))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = list(eligibility[0]) if eligibility else ["model", "origin_id", "origin_month", "horizon_months", "target_month", "input_eligible", "forecast_status", "forecast_complete", "evaluation_eligible", "reason_codes"]
    with eligibility_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(eligibility)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="target-free PIT contract JSON")
    parser.add_argument("--out-dir", type=Path, required=True, help="new output directory (never overwritten)")
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    report, eligibility = audit_bundle(bundle)
    write_outputs(report, eligibility, args.out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
