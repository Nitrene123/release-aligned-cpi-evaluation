#!/usr/bin/env python3
"""One-pass, frozen scoring for the sealed U.S. CPI cross-country forecast archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

HORIZONS = (1, 3, 6, 12)
SEEDS = ("417", "42", "123", "2027", "3407")
METHODS = (
    "No-change", "Seasonal-naive", "AR(1)", "SARIMA",
    "TimesFM-3 target-only", "B3 multivariate",
    "BEAH rolling", "Direct rolling", "Fixed candidate",
    "Fixed candidate + gate",
)
SEEDED_MODELS = {"BEAH rolling", "Direct rolling", "Fixed candidate + gate"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_utc(value: str) -> datetime:
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"naive timestamp: {value}")
    return moment.astimezone(timezone.utc)


def finite_prediction(row: dict[str, str]) -> float:
    value = float(row["prediction"])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite prediction: {row}")
    return value


def add_months(month: str, offset: int) -> str:
    year, mon = map(int, month.split("-"))
    serial = year * 12 + mon - 1 + offset
    return f"{serial // 12:04d}-{serial % 12 + 1:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--target-validation", type=Path, required=True)
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--forecast-root", type=Path, required=True)
    parser.add_argument("--forecast-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite score output: {args.output_dir}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    target_validation = json.loads(args.target_validation.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "US_CPI_CROSSCOUNTRY_ROLLING_2018_2024_V1":
        raise ValueError("unexpected frozen protocol")
    if authorization.get("status") != "AUTHORIZED_ONE_PASS_US_SCORE":
        raise ValueError("scoring authorization record is missing or invalid")
    if authorization.get("scope") != "US CPI-U NSA first-release targets 2018-01 through 2024-12; ten frozen methods; horizons 1,3,6,12; one scoring pass; no retraining, tuning, or P1 access":
        raise ValueError("authorization scope differs from the requested one-pass scoring scope")
    if target_validation.get("status") != "PASS":
        raise ValueError("official target/source validation did not pass")
    if target_validation.get("target_months") != 84 or target_validation.get("raw_release_response_bytes_archived") is not False:
        raise ValueError("target validation report does not match the declared scope")

    protocol_targets = protocol["target"]
    if tuple(protocol_targets["horizons_months"]) != HORIZONS:
        raise ValueError("horizon set differs from frozen protocol")
    if protocol["scoring"].get("p_values") is not False:
        raise ValueError("protocol must prohibit p-values")

    targets_rows = read_csv(args.target_file)
    if len(targets_rows) != 84:
        raise ValueError(f"expected 84 official first-release targets; found {len(targets_rows)}")
    targets = {}
    target_release = {}
    for row in targets_rows:
        month = row["target_reference_month"]
        if month in targets:
            raise ValueError(f"duplicate target month: {month}")
        if row["country"] != "US" or row["series_id"] != "CUUR0000SA0":
            raise ValueError(f"wrong target series: {month}")
        targets[month] = float(row["target_first_release_yoy_pp"])
        target_release[month] = parse_utc(row["release_timestamp_utc"])
        if not math.isfinite(targets[month]):
            raise ValueError(f"nonfinite first-release target: {month}")
    if sorted(targets) != [f"{y}-{m:02d}" for y in range(2018, 2025) for m in range(1, 13)]:
        raise ValueError("target-month coverage is not the frozen 2018-01..2024-12 range")

    plan_path = args.forecast_root / "audit_context" / "planned_target_origin_horizons.csv"
    plan_rows = read_csv(plan_path)
    plan = {}
    for row in plan_rows:
        key = (row["origin_id"], int(row["horizon_months"]), row["target_month"])
        if key in plan:
            raise ValueError(f"duplicate planned key: {key}")
        if row["country"] != "US":
            raise ValueError(f"non-US planned row: {key}")
        if row["origin_id"] != f"US_{row['origin_month']}":
            raise ValueError(f"origin id/month mismatch: {key}")
        if key[1] not in HORIZONS or add_months(row["origin_month"], key[1]) != row["target_month"]:
            raise ValueError(f"target/origin/horizon mapping mismatch: {key}")
        if key[2] not in target_release:
            raise ValueError(f"planned target absent from release ledger: {key}")
        if parse_utc(row["target_release_timestamp_utc"]) != target_release[key[2]]:
            raise ValueError(f"planned/official target release timestamp mismatch: {key}")
        if target_release[key[2]] <= parse_utc(row["origin_clock_utc"]):
            raise ValueError(f"target release is not after the frozen origin: {key}")
        plan[key] = row
    expected_plan = {
        (f"US_{add_months(month, -h)}", h, month)
        for month in targets
        for h in HORIZONS
    }
    if set(plan) != expected_plan:
        raise ValueError(f"planned key set differs from the frozen target/horizon mapping: {len(plan)}/{len(expected_plan)}")
    if len(plan) != 336:
        raise ValueError(f"expected 336 planned target-origin-horizon keys; found {len(plan)}")

    predictions: dict[str, dict[tuple[str, int, str], list[tuple[str, float]]]] = {m: {} for m in METHODS}
    forecast_files = [
        args.forecast_root / "statistics" / "statistical_forecasts_unscored.csv",
        args.forecast_root / "foundation_and_heads" / "foundation_evaluation_forecasts_unscored.csv",
        args.forecast_root / "foundation_and_heads" / "fixed_candidate_forecasts_unscored.csv",
        args.forecast_root / "foundation_and_heads" / "adapter_forecasts_unscored.csv",
    ]
    for file_path in forecast_files:
        for row in read_csv(file_path):
            model = row["model"]
            if model not in predictions:
                raise ValueError(f"unregistered model in forecast file {file_path.name}: {model}")
            key = (row["origin_id"], int(row["horizon_months"]), row["target_month"])
            if key not in plan:
                raise ValueError(f"forecast key not in frozen planned set: {model}/{key}")
            frozen = plan[key]
            if row["origin_month"] != frozen["origin_month"]:
                raise ValueError(f"origin-month mismatch: {model}/{key}")
            if row["origin_timestamp_utc"] != frozen["origin_clock_utc"]:
                raise ValueError(f"origin-clock mismatch: {model}/{key}")
            if row["status"] != "FORECAST_COMPLETE":
                raise ValueError(f"forecast not complete: {model}/{key}/{row['status']}")
            if key[2] not in targets:
                raise ValueError(f"target month missing from score file: {key[2]}")
            if target_release[key[2]] <= parse_utc(frozen["origin_clock_utc"]):
                raise ValueError(f"target was not released after origin: {model}/{key}")
            seed = str(row.get("seed", "") or "")
            if model in SEEDED_MODELS:
                if seed not in SEEDS:
                    raise ValueError(f"unexpected seed for {model}: {seed}")
            elif seed:
                raise ValueError(f"unexpected seed for single-forecast method {model}: {seed}")
            predictions[model].setdefault(key, []).append((seed, finite_prediction(row)))

    # Exact method-specific coverage and seed-replication checks before score construction.
    for model in METHODS:
        observed = set(predictions[model])
        if observed != set(plan):
            raise ValueError(f"planned coverage mismatch for {model}: {len(observed)}/336")
        for key, values in predictions[model].items():
            seeds = [seed for seed, _ in values]
            if model in SEEDED_MODELS:
                if len(values) != 5 or set(seeds) != set(SEEDS) or len(seeds) != len(set(seeds)):
                    raise ValueError(f"seed set mismatch for {model}/{key}: {seeds}")
            elif len(values) != 1:
                raise ValueError(f"duplicate single forecast for {model}/{key}")

    per_origin = []
    loss_maps: dict[str, dict[tuple[str, int, str], float]] = {m: {} for m in METHODS}
    for model in METHODS:
        for key in sorted(plan, key=lambda k: (k[2], k[1], k[0])):
            values = predictions[model][key]
            y = targets[key[2]]
            abs_errors = [abs(prediction - y) for _, prediction in values]
            origin_loss = sum(abs_errors) / len(abs_errors)
            loss_maps[model][key] = origin_loss
            frozen = plan[key]
            per_origin.append({
                "model": model, "origin_id": key[0], "origin_month": frozen["origin_month"],
                "origin_timestamp_utc": frozen["origin_clock_utc"], "target_month": key[2],
                "horizon_months": key[1], "n_seed_predictions": len(values),
                "origin_mean_absolute_error_pp": f"{origin_loss:.12g}",
            })

    coverage = []
    mae_native = []
    common_mae = []
    common_keys_by_h = {}
    for h in HORIZONS:
        planned_h = {key for key in plan if key[1] == h}
        common = set.intersection(*(set(k for k in loss_maps[m] if k[1] == h) for m in METHODS))
        common_keys_by_h[h] = common
        if len(planned_h) != 84:
            raise ValueError(f"expected 84 planned keys at h={h}; found {len(planned_h)}")
        for model in METHODS:
            native = {k for k in loss_maps[model] if k[1] == h}
            coverage.append({"model": model, "horizon_months": h, "planned_origins": 84,
                "eligible_forecasts": len(native), "native_coverage": f"{len(native)/84:.12g}",
                "ten_method_common_origins": len(common)})
            mae_native.append({"model": model, "horizon_months": h, "n_origins": len(native),
                "mae_pp": f"{sum(loss_maps[model][k] for k in native)/len(native):.12g}"})
            common_mae.append({"model": model, "horizon_months": h, "n_common_origins": len(common),
                "mae_pp": f"{sum(loss_maps[model][k] for k in common)/len(common):.12g}"})

    pairwise = []
    for h in HORIZONS:
        for a, b in itertools.combinations(METHODS, 2):
            common = set(k for k in loss_maps[a] if k[1] == h) & set(k for k in loss_maps[b] if k[1] == h)
            differences = [loss_maps[a][k] - loss_maps[b][k] for k in common]
            wins_a = sum(d < -1e-12 for d in differences)
            wins_b = sum(d > 1e-12 for d in differences)
            ties = len(differences) - wins_a - wins_b
            pairwise.append({"method_a": a, "method_b": b, "horizon_months": h,
                "n_paired_origins": len(common),
                "mae_a_on_pair": f"{sum(loss_maps[a][k] for k in common)/len(common):.12g}",
                "mae_b_on_pair": f"{sum(loss_maps[b][k] for k in common)/len(common):.12g}",
                "mean_loss_difference_a_minus_b_pp": f"{sum(differences)/len(differences):.12g}",
                "origins_a_lower_loss": wins_a, "origins_b_lower_loss": wins_b, "ties_1e-12": ties})

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "origin_method_losses.csv", per_origin,
        ["model", "origin_id", "origin_month", "origin_timestamp_utc", "target_month", "horizon_months", "n_seed_predictions", "origin_mean_absolute_error_pp"])
    write_csv(args.output_dir / "coverage_by_method_horizon.csv", coverage,
        ["model", "horizon_months", "planned_origins", "eligible_forecasts", "native_coverage", "ten_method_common_origins"])
    write_csv(args.output_dir / "mae_by_method_horizon_native.csv", mae_native,
        ["model", "horizon_months", "n_origins", "mae_pp"])
    write_csv(args.output_dir / "mae_by_method_horizon_common10.csv", common_mae,
        ["model", "horizon_months", "n_common_origins", "mae_pp"])
    write_csv(args.output_dir / "pairwise_loss_contrasts.csv", pairwise,
        ["method_a", "method_b", "horizon_months", "n_paired_origins", "mae_a_on_pair", "mae_b_on_pair",
         "mean_loss_difference_a_minus_b_pp", "origins_a_lower_loss", "origins_b_lower_loss", "ties_1e-12"])

    score_files = [args.output_dir / n for n in (
        "origin_method_losses.csv", "coverage_by_method_horizon.csv", "mae_by_method_horizon_native.csv",
        "mae_by_method_horizon_common10.csv", "pairwise_loss_contrasts.csv")]
    metadata = {
        "protocol_id": protocol["protocol_id"], "status": "US_CPI_SCORE_COMPLETE_RETROSPECTIVE_SUPPLEMENTARY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_authorization_sha256": sha256(args.authorization), "target_validation_report_sha256": sha256(args.target_validation),
        "target_file_sha256": sha256(args.target_file), "forecast_archive_sha256": sha256(args.forecast_archive),
        "protocol_sha256": sha256(args.protocol), "scorer_sha256": sha256(Path(__file__)),
        "scoring_target_rows_read": len(targets), "scored_origin_method_horizon_rows": len(per_origin),
        "methods": list(METHODS), "horizons_months": list(HORIZONS), "seeded_methods": sorted(SEEDED_MODELS),
        "seed_rule": "Compute each seed-specific absolute error, average the five absolute errors within origin, then average over origins; seeds are not independent time samples.",
        "primary_metric": "per-horizon MAE in percentage points",
        "paired_comparisons": "Same target-origin-horizon keys; report pairwise and ten-method common samples.",
        "p_values_computed": False,
        "bootstrap_or_other_intervals_computed": False,
        "interval_note": "The frozen protocol did not specify a resampling algorithm or block length; no post-score interval method was introduced.",
        "target_values_exported_to_score_outputs": False,
        "retrospective": True, "blind": False, "prospective": False,
        "p1_accessed": False, "retraining_or_tuning": False,
        "forecast_source_files": [{"path": str(p), "sha256": sha256(p)} for p in forecast_files],
        "output_files": [{"name": p.name, "sha256": sha256(p)} for p in score_files],
        "forecast_archive_manifest_includes_target_file": False,
    }
    (args.output_dir / "scoring_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("status", "scoring_target_rows_read", "scored_origin_method_horizon_rows",
        "methods", "p_values_computed", "bootstrap_or_other_intervals_computed", "target_values_exported_to_score_outputs")}, indent=2))


if __name__ == "__main__":
    main()
