#!/usr/bin/env python3
"""Create a post-run correction for the fixed-candidate resource count.

This changes resource metadata only. Forecast rows and their sealed hashes are
read-only. The original metadata and original sealed manifest are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="directory containing model_forecasts_unscored.csv and resource_metrics.json")
    args = ap.parse_args()
    root = args.run_dir.resolve()
    csv_path = root / "model_forecasts_unscored.csv"
    metric_path = root / "resource_metrics.json"
    manifest_path = root / "sealed_manifest.json"
    original_metric_path = root / "resource_metrics_as_run_v1.json"
    original_manifest_path = root / "sealed_manifest_as_run_v1.json"
    correction_path = root / "resource_metrics_correction_v1.json"
    note_path = root / "resource_metrics_correction_v1.md"
    if not all(p.is_file() for p in (csv_path, metric_path, manifest_path)):
        raise FileNotFoundError("run_dir must contain the source CSV, resource metrics, and sealed manifest")
    if any(p.exists() for p in (original_metric_path, original_manifest_path, correction_path, note_path)):
        raise FileExistsError("refusing to overwrite an existing resource-metrics amendment artifact")

    origins: set[str] = set()
    by_horizon: Counter[str] = Counter()
    total = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") != "Fixed candidate" or row.get("status") != "FORECAST_COMPLETE":
                continue
            origins.add(row["origin_id"])
            by_horizon[str(int(row["horizon_months"]))] += 1
            total += 1
    by_horizon = Counter({str(h): by_horizon[str(h)] for h in (1, 3, 6, 12)})
    if (len(origins), total, dict(by_horizon)) != (116, 420, {"1": 105, "3": 105, "6": 105, "12": 105}):
        raise ValueError(f"sealed forecast table has unexpected fixed-candidate counts: origins={len(origins)}, rows={total}, horizons={dict(by_horizon)}")

    original_metric_bytes = metric_path.read_bytes()
    original_manifest_bytes = manifest_path.read_bytes()
    original_metric_path.write_bytes(original_metric_bytes)
    original_manifest_path.write_bytes(original_manifest_bytes)
    metrics = json.loads(original_metric_bytes)
    fixed = metrics.setdefault("fixed_candidate", {})
    fixed["origin_count"] = len(origins)
    fixed["forecast_record_count"] = total
    fixed["forecast_records_by_horizon"] = dict(by_horizon)
    fixed["count_basis"] = "unique origin_id values and FORECAST_COMPLETE rows in model_forecasts_unscored.csv"
    metrics["resource_metadata_amendment"] = {
        "amendment_id": "FIXED_CANDIDATE_COUNT_V1",
        "amended_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "count metadata only; no forecast, score, model, or elapsed-time value changed",
        "original_resource_metrics_sha256": sha256(original_metric_path),
        "forecast_table_sha256": sha256(csv_path),
    }
    metric_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = json.loads(original_manifest_bytes)
    files = manifest.setdefault("files", {})
    files["resource_metrics.json"] = sha256(metric_path)
    files["resource_metrics_as_run_v1.json"] = sha256(original_metric_path)
    files["sealed_manifest_as_run_v1.json"] = sha256(original_manifest_path)
    correction = {
        "amendment_id": "FIXED_CANDIDATE_COUNT_V1",
        "status": "PASS_RECOUNTED_FROM_SEALED_FORECAST_ROWS",
        "amended_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin_count": len(origins),
        "forecast_record_count": total,
        "forecast_records_by_horizon": dict(by_horizon),
        "prediction_wall_seconds_retained": fixed.get("prediction_wall_seconds"),
        "source_forecast_table": csv_path.name,
        "source_forecast_table_sha256": sha256(csv_path),
        "original_resource_metrics_sha256": sha256(original_metric_path),
        "corrected_resource_metrics_sha256": sha256(metric_path),
        "original_sealed_manifest_sha256": sha256(original_manifest_path),
        "scope_note": "Only resource-count metadata is corrected. The original metric and manifest are retained; forecast and score records are not modified.",
    }
    correction_path.write_text(json.dumps(correction, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note_path.write_text(
        "# Fixed-candidate resource-count correction\n\n"
        "The original `origin_count = len(cand_rows) // 4` divided 420 scalar forecast records by four and reported 105 as an origin count. "
        "The sealed forecast table contains 116 distinct origin IDs and 420 completed scalar forecasts: 105 for each of h=1, 3, 6, and 12. "
        "The recorded 0.01580483-second candidate-stage wall time is retained. No prediction or score is changed. "
        "The original metadata and original sealed manifest are retained as `resource_metrics_as_run_v1.json` and `sealed_manifest_as_run_v1.json`.\n",
        encoding="utf-8",
    )
    manifest["amendment"] = {
        "amendment_id": "FIXED_CANDIDATE_COUNT_V1",
        "reason": "correct resource origin count from the sealed forecast table",
        "original_manifest_sha256": sha256(original_manifest_path),
        "forecast_rows_changed": False,
        "scores_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Include amendment record and note in the amended verification manifest.
    manifest["files"][correction_path.name] = sha256(correction_path)
    manifest["files"][note_path.name] = sha256(note_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(correction, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
