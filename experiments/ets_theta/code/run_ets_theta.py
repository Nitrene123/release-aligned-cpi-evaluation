#!/usr/bin/env python3
"""Separate forecast/score steps for the frozen ETS and Theta analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel

PROJECT = Path(__file__).resolve().parents[2]
HOME = Path(__file__).resolve().parent
US = PROJECT / "stage6_us_cpi_crosscountry_v1"
US_RUN = US / "runs/us_cpi_crosscountry_run_v1_attempt2"
CN = PROJECT / "manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a"
CN_RUN = CN / "runs/run_20261030_unified_full_v2"
HORIZONS = (1, 3, 6, 12)
NEW_METHODS = ("ETS(A,A,N)", "Theta")
CFG = {
    "US": {
        "planned": US_RUN / "audit_context/planned_target_origin_horizons.csv",
        "training": US / "data/us_cpi_first_release_training_label_panel_2012_2024.csv",
        "old_forecasts": US_RUN / "statistics/statistical_forecasts_unscored.csv",
        "old_losses": US_RUN / "score_us_2018_2024_once_v1/origin_method_losses.csv",
        "targets": US / "data/us_cpi_first_release_targets_2018_2024.csv",
        "n_targets": 84,
        "n_origins": 95,
    },
    "CN": {
        "planned": CN_RUN / "planned_origin_horizon_keys.csv",
        "training": CN / "source_data/unified_cpi_panel.csv",
        "old_forecasts": CN_RUN / "source/statistical_forecast_predictions.csv",
        "old_losses": CN / "score_v2_clean_20261031/origin_method_losses.csv",
        "targets": CN / "score_inputs/score_target_extension_2025_2026.csv",
        "n_targets": 105,
        "n_origins": 116,
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_write(path: Path, obj: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def stamp(value: str, *, china_local: bool = False) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        if not china_local:
            raise ValueError(f"Missing timezone: {value}")
        result = result.replace(tzinfo=timezone(timedelta(hours=8)))
    return result.astimezone(timezone.utc)


def add_month(month: str, offset: int) -> str:
    y, m = map(int, month.split("-"))
    total = y * 12 + m - 1 + offset
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def normkey(row: dict, country: str) -> tuple[str, str, str, str, int]:
    clock = row.get("origin_clock_utc") or row["origin_timestamp_utc"]
    return (row["origin_id"], row["origin_month"], stamp(clock).isoformat(),
            row["target_month"], int(row["horizon_months"]))


def input_panel(country: str) -> dict[str, dict]:
    rows = csv_read(CFG[country]["training"])
    panel = {}
    for row in rows:
        month = row["reference_month"][:7]
        if month in panel:
            raise ValueError(f"duplicate reference month: {country} {month}")
        value = row["first_release_yoy_pp"] if country == "US" else row["first_release_yoy_pct"]
        release = row["available_at_utc"] if country == "US" else row["available_at_conservative_local"]
        panel[month] = {"value": float(value) if value.strip() else None,
                        "release": stamp(release, china_local=(country == "CN"))}
    return panel


def existing_source_check(country: str) -> None:
    config = CFG[country]
    if country == "US":
        j = json.loads((US_RUN / "statistics/statistical_run_metadata.json").read_text(encoding="utf8"))
        assert sha(config["training"]) == j["source_sha256"]["label_panel"]
        assert sha(config["planned"]) == j["source_sha256"]["planned_keys"]
        assert sha(config["old_forecasts"]) == j["outputs"]["forecasts_sha256"]
    else:
        sealed = json.loads((CN_RUN / "sealed_manifest.json").read_text(encoding="utf8"))
        assert sha(config["planned"]) == sealed["files"]["planned_origin_horizon_keys.csv"]
        assert sha(config["old_forecasts"]) == sealed["files"]["source/statistical_forecast_predictions.csv"]
        bundle = json.loads((CN_RUN / "source/source_bundle_manifest.json").read_text(encoding="utf8"))
        records = [v for v in bundle.values() if isinstance(v, dict) and "unified_cpi_panel.csv" in v]
        assert len(records) == 1
        assert sha(config["training"]) == records[0]["unified_cpi_panel.csv"]["sha256"]


def fit_path(method: str, y: np.ndarray, steps: int) -> tuple[np.ndarray | None, dict]:
    started = time.perf_counter()
    status, reason, converged, iterations, parameter_dict = "FIT_FAILED", "", "", "", {}
    path = None
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            if method == "ETS(A,A,N)":
                result = ETSModel(y, error="add", trend="add", damped_trend=False,
                                  seasonal=None, initialization_method="estimated", missing="none").fit(
                                      maxiter=1000, full_output=True, disp=False)
                ret = result.mle_retvals or {}
                converged = bool(ret.get("converged", False))
                iterations = int(ret.get("iterations", 0))
                parameter_dict = {str(k): float(v) for k, v in zip(result.param_names, result.params)}
                if not converged:
                    raise RuntimeError("ETS optimizer did not converge")
                candidate = np.asarray(result.forecast(steps), dtype=float)
            elif method == "Theta":
                result = ThetaModel(y, period=12, deseasonalize=False, use_test=False,
                                    method="additive", difference=False).fit(use_mle=False, disp=False)
                parameter_dict = {str(k): float(v) for k, v in result.params.items()}
                candidate = np.asarray(result.forecast(steps, theta=2), dtype=float)
                if any(issubclass(x.category, ConvergenceWarning) for x in captured):
                    raise RuntimeError("Theta fitting issued ConvergenceWarning")
            else:
                raise ValueError(f"undeclared model: {method}")
            if len(candidate) != steps or not np.isfinite(candidate).all() or not all(
                    math.isfinite(v) for v in parameter_dict.values()):
                raise ValueError("nonfinite parameters or forecast")
            path, status = candidate, "FIT_COMPLETE"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            parameter_dict = {k: v if math.isfinite(v) else str(v) for k, v in parameter_dict.items()}
    return path, {
        "fit_status": status, "failure_reason": reason,
        "optimizer_converged": converged, "optimizer_iterations": iterations,
        "params_json": json.dumps(parameter_dict, sort_keys=True, allow_nan=False),
        "fit_warnings_json": json.dumps([f"{w.category.__name__}: {w.message}" for w in captured]),
        "wall_seconds": format(time.perf_counter() - started, ".6f"),
    }


def forecast(country: str) -> None:
    assert statsmodels.__version__ == "0.14.4", statsmodels.__version__
    config = CFG[country]
    existing_source_check(country)
    planned, panel = csv_read(config["planned"]), input_panel(country)
    if len(planned) != config["n_targets"] * 4:
        raise ValueError("Unexpected planned forecast count")
    by_origin = defaultdict(list)
    seen, target_horizon = set(), set()
    for row in planned:
        key = normkey(row, country)
        if key in seen or (key[3], key[4]) in target_horizon or add_month(key[1], key[4]) != key[3]:
            raise ValueError(f"Duplicate/invalid planned key: {key}")
        seen.add(key); target_horizon.add((key[3], key[4])); by_origin[key[0]].append((key, row))
    assert len(by_origin) == config["n_origins"]
    assert {h: sum(k[-1] == h for k in seen) for h in HORIZONS} == {h: config["n_targets"] for h in HORIZONS}
    old = {normkey(r, country): r for r in csv_read(config["old_forecasts"]) if r["model"] == "No-change"}
    assert set(old) == seen, "Archived No-change keys do not match planned keys"

    # Build and validate every input window before running the first fit.
    prepared = []
    for oid, pairs in sorted(by_origin.items()):
        origin_month, clock = pairs[0][0][1], stamp(pairs[0][0][2])
        if any(x[0][1] != origin_month or stamp(x[0][2]) != clock for x in pairs):
            raise ValueError(f"Inconsistent origin clock/month: {oid}")
        eligible = [(month, x["value"], x["release"]) for month, x in sorted(panel.items())
                    if month <= origin_month and x["value"] is not None and x["release"] < clock]
        months = [x[0] for x in eligible]
        if (not months or months[0] != "2012-01" or months[-1] != origin_month or len(months) < 60
                or months != [add_month(months[0], i) for i in range(len(months))]):
            raise ValueError(f"Noncontiguous or short eligible history at {oid}")
        y = np.asarray([x[1] for x in eligible], dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError(f"Nonfinite training history at {oid}")
        for key, _ in pairs:
            reference = old[key]
            if (reference["status"] != "FORECAST_COMPLETE"
                    or reference["training_start"] != months[0]
                    or reference["training_end"] != months[-1]
                    or int(reference["training_n_months"]) != len(y)
                    or not math.isclose(float(reference["prediction"]), float(y[-1]), abs_tol=1e-10, rel_tol=0)):
                raise ValueError(f"Training window differs from original No-change at {key}")
        serialized = json.dumps([[m, format(v, ".17g"), t.isoformat()] for m, v, t in eligible],
                                separators=(",", ":"), ensure_ascii=True)
        prepared.append((oid, origin_month, clock, pairs, months, y,
                         hashlib.sha256(serialized.encode()).hexdigest()))
    output = HOME / "output" / country / "forecast"
    output.mkdir(parents=True, exist_ok=False)
    rows, audits = [], []
    start = time.perf_counter()
    for index, (oid, origin_month, clock, pairs, months, y, input_hash) in enumerate(prepared, 1):
        for method in NEW_METHODS:
            pred, details = fit_path(method, y, max(k[-1] for k, _ in pairs))
            audits.append({"country": country, "model": method, "origin_id": oid,
                           "origin_month": origin_month, "origin_timestamp_utc": clock.isoformat(),
                           "training_start": months[0], "training_end": months[-1],
                           "training_n_months": len(y), "input_sha256": input_hash, **details})
            for key, _ in pairs:
                h = key[-1]
                rows.append({"country": country, "model": method, "origin_id": oid,
                             "origin_month": origin_month, "origin_timestamp_utc": clock.isoformat(),
                             "target_month": key[3], "horizon_months": h,
                             "prediction": format(float(pred[h - 1]), ".17g") if pred is not None else "",
                             "status": "FORECAST_COMPLETE" if pred is not None else "FIT_FAILED",
                             "failure_reason": details["failure_reason"],
                             "input_sha256": input_hash})
        if index % 20 == 0 or index == len(prepared):
            print(f"{country}: {index}/{len(prepared)} origin fits complete", flush=True)
    assert len(rows) == len(seen) * len(NEW_METHODS)
    assert len({(r["model"], r["origin_id"], r["target_month"], r["horizon_months"]) for r in rows}) == len(rows)
    csv_write(output / "forecasts.csv", rows)
    csv_write(output / "fit_audit.csv", audits)
    fit_summary = []
    for method in NEW_METHODS:
        group = [r for r in audits if r["model"] == method]
        fit_summary.append({
            "country": country, "model": method, "planned_origins": len(group),
            "fit_complete": sum(r["fit_status"] == "FIT_COMPLETE" for r in group),
            "fit_failed": sum(r["fit_status"] != "FIT_COMPLETE" for r in group),
            "optimizer_converged_true": sum(r["optimizer_converged"] is True for r in group),
            "optimizer_converged_false": sum(r["optimizer_converged"] is False for r in group),
            "optimizer_convergence_not_exposed": sum(r["optimizer_converged"] == "" for r in group),
            "origins_with_any_warning": sum(bool(json.loads(r["fit_warnings_json"])) for r in group),
            "origins_with_convergence_warning": sum("ConvergenceWarning:" in r["fit_warnings_json"] for r in group),
        })
    csv_write(output / "fit_summary.csv", fit_summary)
    json_write(output / "forecast_manifest.json", {
        "status": "FROZEN_FORECASTS_GENERATED_HISTORICALLY", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "country": country, "pre_registered": False, "prospective": False,
        "protocol_sha256": sha(HOME / "PROTOCOL.md"), "code_sha256": sha(Path(__file__)),
        "source_sha256": {name: sha(config[name]) for name in ("planned", "training", "old_forecasts")},
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
        "target_keys": len(seen), "origin_count": len(prepared), "forecast_rows": len(rows),
        "fit_status_counts": {f"{m}|{status}": n for (m, status), n in Counter(
            (r["model"], r["fit_status"]) for r in audits).items()},
        "wall_seconds": round(time.perf_counter() - start, 6),
        "outputs": {name: sha(output / name) for name in ("forecasts.csv", "fit_audit.csv", "fit_summary.csv")},
        "target_values_exported": False, "P1_accessed": False,
    })
    print(f"{country}: frozen forecast manifest {output / 'forecast_manifest.json'}", flush=True)


def target_map(country: str) -> dict[str, tuple[float, datetime]]:
    config = CFG[country]
    if country == "US":
        metadata = json.loads((US_RUN / "score_us_2018_2024_once_v1/scoring_metadata.json").read_text(encoding="utf8"))
        assert sha(config["targets"]) == metadata["target_file_sha256"]
        return {r["target_reference_month"]: (float(r["target_first_release_yoy_pp"]),
                                                stamp(r["release_timestamp_utc"]))
                for r in csv_read(config["targets"])}
    metadata = json.loads((CN / "score_v2_clean_20261031/score_run_metadata.json").read_text(encoding="utf8"))
    assert sha(config["targets"]) == metadata["score_target_extension_sha256"]
    panel = input_panel("CN")
    targets = {}
    for month, row in panel.items():
        if "2018-01" <= month <= "2024-12":
            targets[month] = (float(Decimal(str(row["value"])).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)),
                              row["release"])
    for row in csv_read(config["targets"]):
        month = row["reference_month"]
        if not row["source_url"].startswith("https://www.stats.gov.cn/"):
            raise ValueError(f"Non-NBS target source: {month}")
        targets[month] = (float(Decimal(row["first_release_yoy_pct"]).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)),
                          stamp(row["available_at_utc"]))
    return targets


def score(country: str) -> None:
    config = CFG[country]
    forecast_dir = HOME / "output" / country / "forecast"
    manifest = json.loads((forecast_dir / "forecast_manifest.json").read_text(encoding="utf8"))
    if sha(HOME / "PROTOCOL.md") != manifest["protocol_sha256"] or sha(Path(__file__)) != manifest["code_sha256"]:
        raise ValueError("Protocol or code changed after forecast sealing")
    for name, digest in manifest["outputs"].items():
        if sha(forecast_dir / name) != digest:
            raise ValueError(f"Forecast archive hash mismatch: {name}")
    for name, digest in manifest["source_sha256"].items():
        if sha(config[name]) != digest:
            raise ValueError(f"Input source changed since forecast generation: {name}")
    if country == "US":
        metadata = json.loads((US_RUN / "score_us_2018_2024_once_v1/scoring_metadata.json").read_text(encoding="utf8"))
        expected = {x["name"]: x["sha256"] for x in metadata["output_files"]}
        assert sha(config["old_losses"]) == expected["origin_method_losses.csv"]
    else:
        metadata = json.loads((CN / "score_v2_clean_20261031/score_manifest.json").read_text(encoding="utf8"))
        assert sha(config["old_losses"]) == metadata["files"]["origin_method_losses.csv"]
    targets, predictions = target_map(country), csv_read(forecast_dir / "forecasts.csv")
    old_losses = csv_read(config["old_losses"])
    planned_rows = csv_read(config["planned"])
    planned = {normkey(r, country) for r in planned_rows}
    if country == "US":
        for row in planned_rows:
            if stamp(row["target_release_timestamp_utc"]) != targets[row["target_month"]][1]:
                raise ValueError(f"Target release timestamp differs from planned metadata: {row['target_month']}")
    forecast_keys = {normkey(r, country) for r in predictions}
    assert planned == forecast_keys
    assert len({(r["model"], normkey(r, country)) for r in predictions}) == len(predictions)

    old_by_key = {}
    for row in old_losses:
        key = normkey(row, country)
        if key not in planned:
            raise ValueError(f"Unplanned old loss key: {key}")
        method_key = (row["model"], key)
        if method_key in old_by_key:
            raise ValueError(f"Duplicate old model/key: {method_key}")
        field = "origin_mean_absolute_error_pp" if country == "US" else "mean_seed_absolute_error_pp"
        value = float(row[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid old loss: {method_key}")
        old_by_key[method_key] = value
    methods_old = sorted({m for m, _ in old_by_key})
    assert len(methods_old) == 10
    # Independently validate target mapping using the archived No-change predictions.
    no_change = {normkey(r, country): r for r in csv_read(config["old_forecasts"]) if r["model"] == "No-change"}
    max_reconstruction_error = 0.0
    for key, row in no_change.items():
        target, target_release = targets[key[3]]
        if target_release <= stamp(key[2]):
            raise ValueError(f"Target published before origin: {key}")
        error = abs(abs(float(row["prediction"]) - target) - old_by_key[("No-change", key)])
        max_reconstruction_error = max(max_reconstruction_error, error)
    if max_reconstruction_error > 1e-9:
        raise ValueError(f"Target/No-change loss reconstruction mismatch: {max_reconstruction_error}")

    new_losses = []
    loss_index = dict(old_by_key)
    for row in predictions:
        key = normkey(row, country)
        target, target_release = targets[key[3]]
        if target_release <= stamp(key[2]):
            raise ValueError(f"Target published before origin: {key}")
        if row["status"] != "FORECAST_COMPLETE":
            continue
        value = abs(float(row["prediction"]) - target)
        loss_index[(row["model"], key)] = value
        new_losses.append({"country": country, "model": row["model"], "origin_id": key[0],
                           "origin_month": key[1], "origin_timestamp_utc": key[2],
                           "target_month": key[3], "horizon_months": key[4],
                           "absolute_error_pp": format(value, ".17g")})
    all_methods = methods_old + list(NEW_METHODS)
    score_rows, coverage, pairwise = [], [], []
    for h in HORIZONS:
        sets = {m: {k for method, k in loss_index if method == m and k[-1] == h} for m in all_methods}
        common = set.intersection(*(sets[m] for m in all_methods))
        ten_common = set.intersection(*(sets[m] for m in methods_old))
        for m in all_methods:
            native = sets[m]
            existing_common = ten_common & native
            coverage.append({"country": country, "model": m, "horizon_months": h,
                             "planned_keys": config["n_targets"], "native_valid": len(native),
                             "old_ten_common": len(ten_common),
                             "model_in_old_ten_common": len(existing_common), "all_twelve_common": len(common)})
            score_rows.append({"country": country, "model": m, "horizon_months": h,
                               "n_native": len(native),
                               "mae_native_pp": format(np.mean([loss_index[(m, k)] for k in native]), ".12f") if native else "",
                               "n_old_common10": len(existing_common),
                               "mae_old_common10_pp": format(np.mean([loss_index[(m, k)] for k in existing_common]), ".12f") if existing_common else "",
                               "n_common12": len(common),
                               "mae_common12_pp": format(np.mean([loss_index[(m, k)] for k in common]), ".12f") if common else ""})
        for method in NEW_METHODS:
            for ref in methods_old:
                overlap = sorted(sets[method] & sets[ref], key=lambda k: (k[1], k[0]))
                delta = [loss_index[(method, k)] - loss_index[(ref, k)] for k in overlap]
                pairwise.append({"country": country, "model": method, "reference": ref, "horizon_months": h,
                                 "n_pairs": len(overlap),
                                 "mean_loss_difference_pp": format(float(np.mean(delta)), ".12f") if delta else "",
                                 "model_lower_count": sum(x < -1e-12 for x in delta),
                                 "reference_lower_count": sum(x > 1e-12 for x in delta),
                                 "tie_count": sum(abs(x) <= 1e-12 for x in delta)})
    out = HOME / "output" / country / "score"
    out.mkdir(parents=True, exist_ok=False)
    csv_write(out / "new_origin_losses.csv", new_losses)
    csv_write(out / "coverage.csv", coverage)
    csv_write(out / "mae_summary.csv", score_rows)
    csv_write(out / "new_vs_existing_pairwise.csv", pairwise)
    files = ("new_origin_losses.csv", "coverage.csv", "mae_summary.csv", "new_vs_existing_pairwise.csv")
    json_write(out / "score_manifest.json", {
        "status": "RETROSPECTIVE_SUPPLEMENTARY_SCORE_COMPLETE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "country": country, "forecast_manifest_sha256": sha(forecast_dir / "forecast_manifest.json"),
        "old_origin_losses_sha256": sha(config["old_losses"]), "score_targets_sha256": sha(config["targets"]),
        "planned_keys_sha256": sha(config["planned"]), "old_methods": methods_old,
        "additional_methods": list(NEW_METHODS), "target_units": "percentage points",
        "maximum_no_change_loss_reconstruction_abs_difference_pp": max_reconstruction_error,
        "new_scored_rows": len(new_losses), "p_values_computed": False,
        "pre_registered": False, "prospective": False, "P1_accessed": False,
        "target_values_exported": False,
        "outputs": {name: sha(out / name) for name in files},
    })
    print(f"{country}: scored result manifest {out / 'score_manifest.json'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("forecast", "score"))
    p.add_argument("--country", choices=("US", "CN"), required=True)
    args = p.parse_args()
    (forecast if args.action == "forecast" else score)(args.country)


if __name__ == "__main__":
    main()
