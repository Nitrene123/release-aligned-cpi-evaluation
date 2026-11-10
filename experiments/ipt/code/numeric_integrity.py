#!/usr/bin/env python3
"""Portable numeric fingerprints and tolerance-based replay checks for IPT outputs.

This tool reads point-in-time contexts, forecasts, and target-free derived score
tables only. It never opens the target workbook. Raw SHA-256 values remain exact
file-integrity checks; normalized numeric fingerprints and tolerances are used
for cross-platform numerical replay comparisons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

SCHEMA = "ipt_numeric_integrity_v1"
HASH_DECIMALS = 8
CONTEXT_FILE = "contexts_unscored.jsonl"
CSV_SPECS: dict[str, dict[str, Any]] = {
    "statistical_forecasts_unscored.csv": {
        "keys": ["model", "policy", "origin_month", "horizon", "target_month"],
        "exact": ["status", "converged"],
        "numeric": ["prediction"],
        "atol": 1e-6,
    },
    "timesfm_forecasts_unscored.csv": {
        "keys": ["model", "policy", "origin_month", "horizon", "target_month"],
        "exact": ["status"],
        "numeric": ["prediction"],
        "atol": 1e-6,
    },
    "derived_losses_target_free.csv": {
        "keys": ["model", "policy", "origin_month", "horizon", "target_month"],
        "exact": ["status", "converged", "valid"],
        "numeric": ["prediction", "abs_error", "squared_error"],
        "atol": 1e-6,
    },
    "score_summary.csv": {
        "keys": ["model", "policy", "horizon"],
        "exact_numeric": ["n_native", "n_common"],
        "numeric": ["mae", "rmse"],
        "atol": 1e-6,
    },
    "paired_intervals.csv": {
        "keys": ["model", "horizon", "contrast"],
        "exact_numeric": ["n"],
        "numeric": ["mean_abs_loss_difference", "lower", "upper"],
        "atol": 1e-6,
    },
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decimal_value(value: Any, *, field: str) -> Decimal | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{field} is not a decimal number: {text!r}") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be finite, got {text!r}")
    return result


def normalized_decimal(value: Any, *, field: str, places: int = HASH_DECIMALS) -> str | None:
    number = decimal_value(value, field=field)
    if number is None:
        return None
    quantum = Decimal(1).scaleb(-places)
    rounded = number.quantize(quantum, rounding=ROUND_HALF_EVEN)
    if rounded == 0:
        rounded = abs(rounded)
    return format(rounded, f".{places}f")


def canonical_exact_decimal(value: Any, *, field: str) -> str | None:
    """Normalize numeric identifiers/counts without rounding away differences."""
    number = decimal_value(value, field=field)
    if number is None:
        return None
    number = number.normalize()
    if number == 0:
        number = abs(number)
    return format(number, "f")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_contexts(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            key = (str(raw["origin_month"]), str(raw["policy"]))
            if key in rows:
                raise ValueError(f"duplicate context key at {path}:{line_number}: {key}")
            rows[key] = raw
    return rows


def context_fingerprint(rows: dict[tuple[str, str], dict[str, Any]]) -> str:
    records = []
    exact_fields = (
        "origin_month", "origin_timestamp_utc", "policy", "vintage",
        "context_start", "context_end", "context_length", "months",
    )
    for key in sorted(rows):
        raw = rows[key]
        records.append({
            "key": list(key),
            "metadata": {field: raw[field] for field in exact_fields},
            "values": [normalized_decimal(v, field=f"context[{key}]") for v in raw["values"]],
        })
    return canonical_hash({"schema": SCHEMA, "decimal_places": HASH_DECIMALS, "records": records})


def read_csv_rows(path: Path, spec: dict[str, Any]) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(spec["keys"] + spec.get("exact", []) + spec.get("exact_numeric", []) + spec["numeric"])
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path.name} is missing required columns: {sorted(required - set(reader.fieldnames or []))}")
        for row_number, row in enumerate(reader, 2):
            key = tuple(str(row[k]).strip() for k in spec["keys"])
            if key in result:
                raise ValueError(f"duplicate row key in {path.name}:{row_number}: {key}")
            result[key] = row
    return result


def csv_fingerprint(rows: dict[tuple[str, ...], dict[str, str]], spec: dict[str, Any]) -> str:
    records = []
    for key in sorted(rows):
        row = rows[key]
        record: dict[str, Any] = {"key": list(key)}
        record["exact"] = {field: row.get(field, "").strip() for field in spec.get("exact", [])}
        record["exact_numeric"] = {
            field: canonical_exact_decimal(row.get(field, ""), field=field)
            for field in spec.get("exact_numeric", [])
        }
        record["numeric"] = {
            field: normalized_decimal(row.get(field, ""), field=field)
            for field in spec["numeric"]
        }
        records.append(record)
    return canonical_hash({"schema": SCHEMA, "decimal_places": HASH_DECIMALS, "records": records})


def compare_contexts(reference: dict[tuple[str, str], dict[str, Any]],
                     candidate: dict[tuple[str, str], dict[str, Any]], atol: float) -> dict[str, Any]:
    errors: list[str] = []
    if set(reference) != set(candidate):
        errors.append("context key sets differ")
    max_diff = 0.0
    compared = 0
    exact_fields = (
        "origin_timestamp_utc", "vintage", "context_start", "context_end",
        "context_length", "months",
    )
    for key in sorted(set(reference) & set(candidate)):
        ref, cand = reference[key], candidate[key]
        for field in exact_fields:
            if ref.get(field) != cand.get(field):
                errors.append(f"context metadata mismatch for {key}: {field}")
        ref_values, cand_values = ref.get("values", []), cand.get("values", [])
        if len(ref_values) != len(cand_values):
            errors.append(f"context length mismatch for {key}")
            continue
        for i, (a, b) in enumerate(zip(ref_values, cand_values)):
            da = decimal_value(a, field=f"reference context {key}[{i}]")
            db = decimal_value(b, field=f"candidate context {key}[{i}]")
            if da is None or db is None:
                if da != db:
                    errors.append(f"missing-value mismatch for context {key}[{i}]")
                continue
            difference = float(abs(da - db))
            max_diff = max(max_diff, difference)
            compared += 1
            if difference > atol:
                errors.append(f"context value outside tolerance for {key}[{i}]")
    return {"rows_reference": len(reference), "rows_candidate": len(candidate),
            "numeric_cells_compared": compared, "max_abs_difference": max_diff,
            "atol": atol, "within_tolerance": not errors, "errors": errors[:20]}


def compare_csv(reference: dict[tuple[str, ...], dict[str, str]],
                candidate: dict[tuple[str, ...], dict[str, str]],
                spec: dict[str, Any], atol: float) -> dict[str, Any]:
    errors: list[str] = []
    if set(reference) != set(candidate):
        errors.append("row key sets differ")
    max_by_field = {field: 0.0 for field in spec["numeric"]}
    compared_by_field = {field: 0 for field in spec["numeric"]}
    exact_fields = spec.get("exact", []) + spec.get("exact_numeric", [])
    for key in sorted(set(reference) & set(candidate)):
        left, right = reference[key], candidate[key]
        for field in exact_fields:
            if field in spec.get("exact_numeric", []):
                a = decimal_value(left.get(field, ""), field=f"reference {field}")
                b = decimal_value(right.get(field, ""), field=f"candidate {field}")
                same = a == b
            else:
                same = left.get(field, "").strip() == right.get(field, "").strip()
            if not same:
                errors.append(f"exact field mismatch for {key}: {field}")
        for field in spec["numeric"]:
            a = decimal_value(left.get(field, ""), field=f"reference {field}")
            b = decimal_value(right.get(field, ""), field=f"candidate {field}")
            if a is None or b is None:
                if a != b:
                    errors.append(f"missing numeric mismatch for {key}: {field}")
                continue
            difference = float(abs(a - b))
            max_by_field[field] = max(max_by_field[field], difference)
            compared_by_field[field] += 1
            if difference > atol:
                errors.append(f"numeric field outside tolerance for {key}: {field}")
    return {"rows_reference": len(reference), "rows_candidate": len(candidate),
            "numeric_cells_compared_by_field": compared_by_field,
            "max_abs_difference_by_field": max_by_field,
            "atol": atol, "within_tolerance": not errors, "errors": errors[:20]}


def artifact_record(folder: Path, name: str) -> tuple[dict[str, Any], Any]:
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(path)
    if name == CONTEXT_FILE:
        rows = read_contexts(path)
        normalized = context_fingerprint(rows)
        record = {"rows": len(rows), "raw_sha256": file_sha256(path), "normalized_numeric_sha256": normalized}
        return record, rows
    spec = CSV_SPECS[name]
    rows = read_csv_rows(path, spec)
    normalized = csv_fingerprint(rows, spec)
    record = {"rows": len(rows), "raw_sha256": file_sha256(path), "normalized_numeric_sha256": normalized}
    return record, rows


def build_report(reference_dir: Path, candidate_dir: Path | None,
                 context_atol: float, prediction_atol: float, score_atol: float) -> dict[str, Any]:
    names = [CONTEXT_FILE, *CSV_SPECS.keys()]
    reference_artifacts: dict[str, Any] = {}
    reference_objects: dict[str, Any] = {}
    for name in names:
        reference_artifacts[name], reference_objects[name] = artifact_record(reference_dir, name)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "hash_normalization": {
            "method": "Exact numeric fields use lossless canonical Decimal text; measured values use ROUND_HALF_EVEN fixed-place quantization",
            "decimal_places": HASH_DECIMALS,
            "sort_order": "lexicographic by documented row keys",
            "raw_sha256_retained": True,
        },
        "tolerances": {
            "context_values_pp_abs": context_atol,
            "forecast_predictions_pp_abs": prediction_atol,
            "derived_scores_abs_native_units": score_atol,
            "relative_tolerance": 0,
        },
        "scope": "Target-free contexts, forecasts, and derived numeric score outputs; the target workbook is not opened.",
        "reference_dir": str(reference_dir),
        "reference_artifacts": reference_artifacts,
    }
    if candidate_dir is None:
        report["comparison"] = {
            "status": "REFERENCE_FINGERPRINTS_RECORDED",
            "cross_platform_replay_performed": False,
            "note": "Supply --candidate-dir from a separate replay to execute row-key and absolute-tolerance validation.",
        }
        return report

    candidate_artifacts: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for name in names:
        candidate_artifacts[name], candidate_obj = artifact_record(candidate_dir, name)
        if name == CONTEXT_FILE:
            comparison = compare_contexts(reference_objects[name], candidate_obj, context_atol)
        else:
            atol = prediction_atol if "forecasts_unscored" in name else score_atol
            comparison = compare_csv(reference_objects[name], candidate_obj, CSV_SPECS[name], atol)
        comparisons[name] = comparison
    all_pass = all(row["within_tolerance"] for row in comparisons.values())
    report["candidate_dir"] = str(candidate_dir)
    report["candidate_artifacts"] = candidate_artifacts
    report["comparison"] = {
        "status": "PASS_WITHIN_TOLERANCE" if all_pass else "FAIL_OUTSIDE_TOLERANCE",
        "cross_platform_replay_performed": reference_dir.resolve() != candidate_dir.resolve(),
        "within_tolerance": all_pass,
        "artifacts": comparisons,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parent
    parser.add_argument("--reference-dir", type=Path, default=default_dir)
    parser.add_argument("--candidate-dir", type=Path, default=None,
                        help="directory containing a replay to compare with the reference")
    parser.add_argument("--output", type=Path, default=None,
                        help="new JSON path; existing files are never overwritten")
    parser.add_argument("--context-atol", type=float, default=1e-9, help="absolute input tolerance in percentage points")
    parser.add_argument("--prediction-atol", type=float, default=1e-6, help="absolute forecast tolerance in percentage points")
    parser.add_argument("--score-atol", type=float, default=1e-6,
                        help="absolute tolerance for derived score fields in each field's native units")
    args = parser.parse_args()
    reference_dir = args.reference_dir.resolve()
    candidate_dir = args.candidate_dir.resolve() if args.candidate_dir else None
    output = args.output.resolve() if args.output else (
        (candidate_dir / "numeric_integrity_comparison.json") if candidate_dir
        else (reference_dir / "numeric_integrity_reference.json")
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite numeric-integrity output: {output}")
    for name, value in (("context-atol", args.context_atol), ("prediction-atol", args.prediction_atol), ("score-atol", args.score_atol)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be a finite non-negative number")
    report = build_report(reference_dir, candidate_dir, args.context_atol, args.prediction_atol, args.score_atol)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "comparison": report["comparison"]}, ensure_ascii=False, indent=2))
    return 0 if report["comparison"].get("within_tolerance", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
