#!/usr/bin/env python3
"""Target-free point-in-time audit for versioned rolling forecast bundles.

The bundle contains metadata and forecasts, not target values. Audit output is
written outside the immutable input bundle so the checker cannot invalidate
the source manifest by overwriting one of its own inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditError(Exception):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stamp(value: str, field: str) -> datetime:
    if not value or not value.strip():
        raise AuditError(f"missing timestamp: {field}")
    try:
        out = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as e:
        raise AuditError(f"invalid timestamp for {field}: {value!r}") from e
    if out.tzinfo is None:
        raise AuditError(f"timestamp must include an explicit UTC offset: {field}")
    return out.astimezone(timezone.utc)


def month_index(value: str, field: str) -> int:
    try:
        year, month = (int(x) for x in value.split("-", 1))
        if not 1 <= month <= 12:
            raise ValueError
        return year * 12 + month - 1
    except Exception as e:
        raise AuditError(f"invalid YYYY-MM value for {field}: {value!r}") from e


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_input_hash(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(sorted(rows, key=lambda r: (r["feature"], r["observation_month"],
                                                        r.get("source_observation_month", ""))),
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def month_label(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def is_masked(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise AuditError(f"{field} must be true or false, got {value!r}")
    return normalized == "true"


AUDIT_MODULES = ("release", "vintage", "dependency", "eligibility", "mapping")


def run_audit(config_path: Path, output_path: Path,
              enabled_modules: list[str] | tuple[str, ...] | set[str] | None = None,
              write_outputs: bool = True) -> dict[str, Any]:
    config_path = config_path.resolve()
    bundle = config_path.parent.resolve()
    output_path = output_path.resolve()
    if output_path == bundle or output_path.is_relative_to(bundle):
        raise AuditError("audit output must be outside the immutable input bundle")
    eligibility_path = output_path.with_name(f"{output_path.stem}_eligibility.csv")
    if output_path.exists() or eligibility_path.exists():
        raise AuditError("audit output already exists; refusing overwrite")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    active_modules = set(AUDIT_MODULES if enabled_modules is None else enabled_modules)
    unknown_modules = active_modules - set(AUDIT_MODULES)
    if unknown_modules:
        raise AuditError(f"unknown audit modules: {sorted(unknown_modules)}")

    def module_on(name: str) -> bool:
        return name in active_modules

    module_check_calls = {name: 0 for name in AUDIT_MODULES}

    def count_module_check(name: str) -> None:
        module_check_calls[name] += 1

    required_path_names = ("origins_file", "inputs_file", "vintages_file", "targets_file", "forecasts_file")
    optional_path_names = ("forecast_statuses_file",)
    paths = {name: (bundle / config[name]).resolve()
             for name in required_path_names if config.get(name)}
    paths.update({name: (bundle / config[name]).resolve()
                  for name in optional_path_names if config.get(name)})
    for name, path in paths.items():
        if not path.is_relative_to(bundle) or not path.is_file():
            raise AuditError(f"missing or out-of-bundle {name}: {path}")

    errors: list[str] = []
    checks: dict[str, Any] = {}
    origins = read_csv(paths["origins_file"])
    inputs = read_csv(paths["inputs_file"])
    # The vintage registry is not even loaded in the vintage ablation arm.
    vintages = read_csv(paths["vintages_file"]) if module_on("vintage") else []
    targets = read_csv(paths["targets_file"])
    forecasts = read_csv(paths["forecasts_file"])
    forecast_status_rows = (read_csv(paths["forecast_statuses_file"])
                            if module_on("eligibility") and "forecast_statuses_file" in paths else [])

    # Optional source byte-locks are verified before interpreting the rows.
    source_checks = []
    for item in config.get("source_files", []):
        p = (bundle / item["path"]).resolve()
        if not p.is_relative_to(bundle) or not p.is_file():
            errors.append(f"source file missing/outside bundle: {item['path']}")
            continue
        actual = sha256(p)
        ok = actual == item["sha256"].lower()
        source_checks.append({"path": item["path"], "expected_sha256": item["sha256"],
                              "actual_sha256": actual, "pass": ok})
        if not ok:
            errors.append(f"source hash mismatch: {item['path']}")
    checks["source_hash_checks"] = source_checks

    origin_map: dict[tuple[str, str], dict[str, str]] = {}
    for i, row in enumerate(origins, 2):
        key = (row.get("model", ""), row.get("origin_id", ""))
        if not all(key) or key in origin_map:
            errors.append(f"duplicate/empty model-origin key at origins row {i}: {key}")
            continue
        origin_map[key] = row
        try:
            stamp(row.get("origin_timestamp", ""), f"origins[{i}].origin_timestamp")
            month_index(row.get("origin_month", ""), f"origins[{i}].origin_month")
        except AuditError as e:
            errors.append(str(e))
    if module_on("eligibility") and config.get("expected_models") is not None:
        actual_models = sorted({m for m, _ in origin_map})
        expected_models = sorted(config["expected_models"])
        count_module_check("eligibility")
        if actual_models != expected_models:
            errors.append(f"origin model set mismatch: expected {expected_models}, got {actual_models}")
    checks["origin_rows"] = len(origins)

    vintage_map: dict[tuple[str, str, str, str], dict[str, str]] = {}
    versions_by_series_month: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for i, row in enumerate(vintages, 2):
        count_module_check("vintage")
        key = (row.get("source_id", ""), row.get("series_id", ""),
               row.get("observation_month", ""), row.get("vintage_id", ""))
        if not all(key):
            errors.append(f"empty vintage identity at vintages row {i}")
            continue
        try:
            month_index(key[2], f"vintages[{i}].observation_month")
            stamp(row.get("available_at", ""), f"vintages[{i}].available_at")
        except AuditError as e:
            errors.append(str(e))
        if key in vintage_map:
            previous = vintage_map[key]
            if previous.get("value") != row.get("value") or previous.get("source_sha256") != row.get("source_sha256"):
                errors.append(f"vintage identifier reused with changed value/source hash: {key}")
            else:
                errors.append(f"duplicate vintage identity row: {key}")
        else:
            vintage_map[key] = row
        versions_by_series_month[key[:3]].append(row)
    checks["vintage_rows"] = len(vintages)

    target_map: dict[str, dict[str, str]] = {}
    for i, row in enumerate(targets, 2):
        month = row.get("target_month", "")
        if not month or month in target_map:
            errors.append(f"duplicate/empty target month at targets row {i}: {month!r}")
            continue
        try:
            month_index(month, f"targets[{i}].target_month")
            stamp(row.get("first_release_at", ""), f"targets[{i}].first_release_at")
        except AuditError as e:
            errors.append(str(e))
        target_map[month] = row
    checks["target_calendar_rows"] = len(targets)

    inputs_by_origin: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen_input_keys = set()
    for i, row in enumerate(inputs, 2):
        origin_key = (row.get("model", ""), row.get("origin_id", ""))
        if origin_key not in origin_map:
            if module_on("eligibility"):
                count_module_check("eligibility")
                errors.append(f"input references unknown model-origin at row {i}: {origin_key}")
            continue
        key = (*origin_key, row.get("feature", ""), row.get("observation_month", ""))
        if key in seen_input_keys:
            errors.append(f"duplicate model-origin-feature-observation input: {key}")
        seen_input_keys.add(key)
        inputs_by_origin[origin_key].append(row)
        origin_at = (stamp(origin_map[origin_key]["origin_timestamp"], f"origin {origin_key}")
                     if module_on("release") else None)
        input_month = month_index(row.get("observation_month", ""), f"inputs[{i}].observation_month")
        if module_on("release"):
            origin_month = month_index(origin_map[origin_key].get("origin_month", ""), f"origin {origin_key}.origin_month")
            count_module_check("release")
            if input_month > origin_month:
                errors.append(f"input observation month is after forecast origin at row {i}")
        source_observation_month = row.get("source_observation_month", "").strip() or row.get("observation_month", "")
        try:
            source_month = month_index(source_observation_month, f"inputs[{i}].source_observation_month")
            if module_on("dependency"):
                count_module_check("dependency")
            if module_on("dependency") and source_month > input_month:
                errors.append(f"source observation month is after input context month at row {i}")
        except AuditError as e:
            errors.append(str(e))
        masked = is_masked(row.get("masked", ""), f"inputs[{i}].masked")
        value_text = row.get("value", "").strip()
        if masked:
            if value_text or row.get("vintage_id", "").strip():
                errors.append(f"masked input must not carry value/vintage_id at row {i}")
            if not row.get("mask_reason", "").strip():
                errors.append(f"masked input missing mask_reason at row {i}")
            continue
        try:
            value = float(value_text)
            if not math.isfinite(value):
                raise ValueError
            available_at = stamp(row.get("available_at", ""), f"inputs[{i}].available_at")
            month_index(row.get("observation_month", ""), f"inputs[{i}].observation_month")
        except (ValueError, AuditError) as e:
            errors.append(f"invalid unmasked input row {i}: {e}")
            continue
        if module_on("release"):
            count_module_check("release")
        if module_on("release") and not available_at < origin_at:
            errors.append(f"input available_at is not strictly before origin at row {i}")
        raw_hash = row.get("source_sha256", "").lower()
        if len(raw_hash) != 64 or any(c not in "0123456789abcdef" for c in raw_hash):
            errors.append(f"missing/invalid source_sha256 at row {i}")
        if module_on("vintage"):
            count_module_check("vintage")
            vintage_key = (row.get("source_id", ""), row.get("series_id", ""),
                           source_observation_month, row.get("vintage_id", ""))
            chosen = vintage_map.get(vintage_key)
            if chosen is None:
                errors.append(f"selected input vintage absent from vintages table at row {i}: {vintage_key}")
            else:
                if module_on("release"):
                    chosen_at = stamp(chosen["available_at"], f"vintage {vintage_key}")
                    count_module_check("release")
                    if not chosen_at < origin_at:
                        errors.append(f"selected vintage was not available before origin at row {i}")
                if chosen.get("value") != value_text or chosen.get("source_sha256", "").lower() != raw_hash:
                    errors.append(f"input value/hash differs from selected vintage at row {i}")
                if module_on("release"):
                    # As-of vintage selection needs the release cutoff. When
                    # release checks are ablated, skip this dependent comparison
                    # rather than silently consulting the origin clock.
                    series_month = vintage_key[:3]
                    candidates = versions_by_series_month.get(series_month, [])
                    candidates = [v for v in candidates
                                  if stamp(v["available_at"], f"vintage {series_month}") < origin_at]
                    if candidates:
                        latest_at = max(stamp(v["available_at"], f"vintage {series_month}") for v in candidates)
                        latest_ids = {v["vintage_id"] for v in candidates
                                      if stamp(v["available_at"], f"vintage {series_month}") == latest_at}
                        if len(latest_ids) != 1:
                            errors.append(f"ambiguous latest vintage before origin for {series_month}")
                        elif row.get("vintage_id") not in latest_ids:
                            errors.append(f"input selected stale vintage for series/month at row {i}: {series_month}")
    required_features = config.get("required_features_by_model", {})
    # Missing required rows are computed only when eligibility is enabled.
    missing_features_by_origin = ({
        origin_key: sorted(set(required_features.get(origin_key[0], [])) -
                           {r.get("feature", "") for r in inputs_by_origin.get(origin_key, [])})
        for origin_key in origin_map
    } if module_on("eligibility") else {})
    checks["input_rows"] = len(inputs)
    checks["model_origin_feature_cells"] = len(seen_input_keys)
    checks["masked_input_rows"] = sum(1 for r in inputs if r.get("masked", "").strip().lower() == "true")
    checks["model_origin_panels_missing_required_rows"] = sum(bool(v) for v in missing_features_by_origin.values()) if module_on("eligibility") else "SKIPPED"
    checks["missing_required_feature_rows_by_model"] = ({
        model: sum(bool(missing_features_by_origin.get((model, origin_id), []))
                   for origin_id in {origin_id for m, origin_id in origin_map if m == model})
        for model in sorted({model for model, _ in origin_map})
    } if module_on("eligibility") else "SKIPPED")

    forecast_keys: dict[tuple[str, str, int], dict[str, str]] = {}
    statuses_by_key: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for i, row in enumerate(forecast_status_rows, 2):
        try:
            h = int(row.get("horizon_months", ""))
            key = (row.get("model", ""), row.get("origin_id", ""), h)
            if not all(key[:2]):
                raise ValueError
        except ValueError:
            errors.append(f"invalid forecast status key at row {i}")
            continue
        statuses_by_key[key].append(row)
    origins_by_model_h: dict[tuple[str, int], set[str]] = defaultdict(set)
    input_hash_mismatches = 0
    execution_hash_format_failures = 0
    expected_horizons = {int(h) for h in config.get("expected_horizons", [])}
    for i, row in enumerate(forecasts, 2):
        model = row.get("model", "")
        origin_id = row.get("origin_id", "")
        try:
            h = int(row.get("horizon_months", ""))
            pred = float(row.get("prediction", ""))
            if not math.isfinite(pred):
                raise ValueError
        except ValueError:
            errors.append(f"invalid horizon/prediction at forecasts row {i}")
            continue
        key = (model, origin_id, h)
        if key in forecast_keys:
            errors.append(f"duplicate forecast key at row {i}: {key}")
            continue
        forecast_keys[key] = row
        origin_key = (model, origin_id)
        origin = origin_map.get(origin_key)
        if origin is None:
            if module_on("eligibility"):
                count_module_check("eligibility")
                errors.append(f"forecast references unknown model-origin at row {i}: {origin_key}")
            continue
        if module_on("eligibility"):
            count_module_check("eligibility")
            missing_for_forecast = missing_features_by_origin.get(origin_key, [])
            if missing_for_forecast:
                errors.append(f"forecast emitted despite missing required input rows at row {i}: {key}: {missing_for_forecast}")
        if module_on("mapping") and expected_horizons and h not in expected_horizons:
            count_module_check("mapping")
            errors.append(f"unregistered horizon at row {i}: {h}")
        try:
            origin_month = month_index(origin["origin_month"], f"forecast {origin_key} origin_month")
            target_month = month_index(row.get("target_month", ""), f"forecasts[{i}].target_month")
            if module_on("mapping"):
                count_module_check("mapping")
            if module_on("mapping") and target_month - origin_month != h:
                errors.append(f"target-origin-horizon mismatch at row {i}: {key}")
            target = target_map.get(row.get("target_month", ""))
            if module_on("mapping") and target is None:
                count_module_check("mapping")
                errors.append(f"forecast target absent from target calendar at row {i}")
            elif module_on("release") and target is not None:
                origin_at = stamp(origin["origin_timestamp"], f"origin {origin_key}")
                target_at = stamp(target["first_release_at"], f"target {row.get('target_month')}")
                count_module_check("release")
                if not target_at > origin_at:
                    errors.append(f"target was released by origin time at row {i}")
        except AuditError as e:
            errors.append(str(e))
        origin_inputs = inputs_by_origin.get(origin_key, [])
        expected_input_hash = canonical_input_hash(origin_inputs)
        if row.get("input_hash", "").lower() != expected_input_hash:
            input_hash_mismatches += 1
            errors.append(f"forecast input hash mismatch at row {i}: {key}")
        execution_hash = row.get("execution_input_hash", "").strip().lower()
        if execution_hash and (len(execution_hash) != 64 or any(c not in "0123456789abcdef" for c in execution_hash)):
            execution_hash_format_failures += 1
            errors.append(f"invalid execution input hash at row {i}: {key}")
        if module_on("eligibility"):
            origins_by_model_h[(model, h)].add(origin_id)
    checks["forecast_rows"] = len(forecasts)
    checks["unique_forecast_keys"] = len(forecast_keys)
    checks["forecast_input_hash_mismatches"] = input_hash_mismatches
    checks["execution_input_hash_format_failures"] = execution_hash_format_failures
    checks["execution_input_hash_rows_present"] = sum(bool(r.get("execution_input_hash", "").strip()) for r in forecasts)

    models = config.get("expected_models") or sorted({m for m, _, _ in forecast_keys})
    horizons = sorted(expected_horizons or {h for _, _, h in forecast_keys})
    coverage = {}
    all_equal = True
    if module_on("eligibility"):
        for h in horizons:
            count_module_check("eligibility")
            sets = {m: origins_by_model_h.get((m, h), set()) for m in models}
            union = set().union(*sets.values()) if sets else set()
            intersection = set.intersection(*sets.values()) if sets else set()
            same = all(s == sets[models[0]] for s in sets.values()) if models else True
            all_equal &= same
            coverage[str(h)] = {"origin_count_by_model": {m: len(sets[m]) for m in models},
                                "union_count": len(union), "all_model_intersection_count": len(intersection),
                                "identical_origin_sets": same}
            if config.get("require_identical_model_origin_sets", False) and not same:
                errors.append(f"model origin sets differ at horizon {h}")
        checks["coverage_by_horizon"] = coverage
        checks["all_models_have_identical_origin_sets"] = all_equal
    else:
        checks["coverage_by_horizon"] = "SKIPPED"
        checks["all_models_have_identical_origin_sets"] = "SKIPPED"

    # Emit an explicit target-origin-horizon eligibility matrix. Targets are
    # represented by release metadata only; no target values are loaded.
    # This crosswalk includes missing model-origin panels and missing forecast
    # rows so that failures cannot disappear from the denominator.
    origin_by_model_month = {}
    origin_rows_by_month: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (model, _origin_id), origin in origin_map.items():
        if not module_on("eligibility"):
            break
        origin_by_model_month[(model, origin["origin_month"])] = origin
        origin_rows_by_month[origin["origin_month"]].append(origin)
    for origin_month, rows in origin_rows_by_month.items():
        clocks = {stamp(r["origin_timestamp"], f"origin clock {origin_month}") for r in rows}
        if len(clocks) > 1:
            if module_on("eligibility"):
                count_module_check("eligibility")
                errors.append(f"origin timestamps differ between models for {origin_month}")

    eligibility_rows = []
    for target_month, target in (sorted(target_map.items(), key=lambda x: month_index(x[0], "target"))
                                 if module_on("eligibility") else []):
        count_module_check("eligibility")
        target_idx = month_index(target_month, "target")
        target_at = stamp(target["first_release_at"], f"target {target_month}")
        for h in horizons:
            origin_month = month_label(target_idx - h)
            reference_rows = origin_rows_by_month.get(origin_month, [])
            reference_origin = reference_rows[0] if reference_rows else None
            origin_at = (stamp(reference_origin["origin_timestamp"], f"origin {origin_month}")
                         if reference_origin else None)
            if module_on("release") and origin_at is not None and not target_at > origin_at:
                count_module_check("release")
                errors.append(f"target release is not after derived origin: {origin_month}/h{h}/{target_month}")
            for model in models:
                origin = origin_by_model_month.get((model, origin_month))
                origin_id = origin.get("origin_id", "") if origin else (
                    reference_origin.get("origin_id", "") if reference_origin else "")
                panel_key = (model, origin_id) if origin_id else (model, "")
                panel_rows = inputs_by_origin.get(panel_key, [])
                found_features = {r.get("feature", "") for r in panel_rows}
                required = set(required_features.get(model, []))
                missing_features = missing_features_by_origin.get(panel_key, sorted(required - found_features))
                masked_required = sum(
                    1 for r in panel_rows
                    if r.get("feature", "") in required
                    and r.get("masked", "").strip().lower() == "true"
                )
                panel_present = origin is not None
                if not panel_present:
                    panel_status = "NO_MODEL_ORIGIN_PANEL"
                elif missing_features:
                    panel_status = "REQUIRED_FEATURE_ROWS_MISSING"
                else:
                    panel_status = "PANEL_PRESENT_REQUIRED_ROWS_PRESENT"
                forecast = forecast_keys.get((model, origin_id, h)) if origin_id else None
                eligibility_rows.append({
                    "model": model,
                    "target_month": target_month,
                    "first_release_at": target["first_release_at"],
                    "horizon_months": h,
                    "origin_id": origin_id,
                    "origin_month": origin_month,
                    "origin_timestamp": (origin.get("origin_timestamp", "") if origin
                                         else reference_origin.get("origin_timestamp", "") if reference_origin else ""),
                    "target_release_after_origin": (
                        str(origin_at is not None and target_at > origin_at).lower()
                        if module_on("release") else "NOT_ASSESSED"
                    ),
                    "origin_clock_record_present": str(panel_present).lower(),
                    "required_features_present": str(not missing_features).lower(),
                    "missing_required_features": ";".join(missing_features),
                    "masked_required_feature_rows": masked_required,
                    "input_panel_status": panel_status,
                    "forecast_record_status": (
                        "PRESENT" if forecast else
                        (";".join(sorted({r.get("status", "UNKNOWN") for r in statuses_by_key.get((model, origin_id, h), [])}))
                         if statuses_by_key.get((model, origin_id, h), []) else
                         ("NOT_ELIGIBLE_MISSING_REQUIRED_INPUTS" if missing_features else "NO_OUTPUT_ROW"))
                    ),
                    "forecast_failure_reason": "; ".join(sorted({r.get("failure_reason", "").strip()
                                                                      for r in statuses_by_key.get((model, origin_id, h), [])
                                                                      if r.get("failure_reason", "").strip()})),
                    "input_hash": forecast.get("input_hash", "") if forecast else "",
                })
    checks["eligibility_table_rows"] = len(eligibility_rows) if module_on("eligibility") else "SKIPPED"
    checks["eligibility_table_missing_forecast_rows"] = (
        sum(row["forecast_record_status"] != "PRESENT" for row in eligibility_rows)
        if module_on("eligibility") else "SKIPPED"
    )

    status = "PASS" if not errors else "FAIL"
    checks["module_execution"] = {
        name: {"enabled": name in active_modules, "checks_executed": module_check_calls[name]}
        for name in AUDIT_MODULES
    }
    report = {
        "audit_schema": "pit_forecast_bundle_audit_v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(config_path),
        "bundle_root_name": bundle.name,
        "target_values_read": False,
        "scope_note": config.get("scope_note", ""),
        "vintage_registry_scope": config.get("vintage_registry_scope", "selected versions represented in the supplied bundle"),
        "eligibility_table_file": eligibility_path.name if module_on("eligibility") else "",
        "eligibility_table_sha256": "",
        "checks": checks,
        "errors": errors,
    }
    if write_outputs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if write_outputs and module_on("eligibility"):
        with eligibility_path.open("x", encoding="utf-8", newline="") as f:
            fields = list(eligibility_rows[0]) if eligibility_rows else [
                "model", "target_month", "first_release_at", "horizon_months", "origin_id",
                "origin_month", "origin_timestamp", "target_release_after_origin",
                "origin_clock_record_present", "required_features_present", "missing_required_features",
                "masked_required_feature_rows", "input_panel_status", "forecast_record_status", "input_hash",
                "forecast_failure_reason",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(eligibility_rows)
        report["eligibility_table_sha256"] = sha256(eligibility_path)
    if write_outputs:
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Bundle audit configuration JSON")
    parser.add_argument("--out", type=Path, required=True, help="Report path outside the immutable bundle")
    args = parser.parse_args()
    try:
        report = run_audit(args.config, args.out)
    except (AuditError, OSError, json.JSONDecodeError) as e:
        print(json.dumps({"status": "ERROR", "error": str(e)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": report["status"], "errors": len(report["errors"]),
                      "report": str(args.out.resolve())}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
