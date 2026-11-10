#!/usr/bin/env python3
"""Independent arithmetic and origin-key check of sealed ETS/Theta results."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CN = ROOT / "manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a"
US = ROOT / "stage6_us_cpi_crosscountry_v1"
US_RUN = US / "runs/us_cpi_crosscountry_run_v1_attempt2"
HORIZONS = (1, 3, 6, 12)


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clock(x, *, cn_local=False):
    d = datetime.fromisoformat(x.replace("Z", "+00:00"))
    if d.tzinfo is None:
        if not cn_local:
            raise ValueError("timezone missing")
        d = d.replace(tzinfo=timezone(timedelta(hours=8)))
    return d.astimezone(timezone.utc).isoformat()


def key(row):
    return (row["origin_id"], row["origin_month"], clock(row["origin_timestamp_utc"]),
            row["target_month"], int(row["horizon_months"]))


def verify(country):
    out = HERE / "output" / country
    forecast_dir, score_dir = out / "forecast", out / "score"
    fm = json.loads((forecast_dir / "forecast_manifest.json").read_text(encoding="utf8"))
    sm = json.loads((score_dir / "score_manifest.json").read_text(encoding="utf8"))
    assert fm["protocol_sha256"] == hash_file(HERE / "PROTOCOL.md")
    assert fm["code_sha256"] == hash_file(HERE / "run_ets_theta.py")
    assert sm["forecast_manifest_sha256"] == hash_file(forecast_dir / "forecast_manifest.json")
    for name, expected in fm["outputs"].items():
        assert hash_file(forecast_dir / name) == expected
    for name, expected in sm["outputs"].items():
        assert hash_file(score_dir / name) == expected

    if country == "US":
        original = US_RUN / "score_us_2018_2024_once_v1/origin_method_losses.csv"
        target_file = US / "data/us_cpi_first_release_targets_2018_2024.csv"
        targets = {r["target_reference_month"]: (float(r["target_first_release_yoy_pp"]),
                                                  clock(r["release_timestamp_utc"])) for r in read(target_file)}
        loss_column = "origin_mean_absolute_error_pp"
        expected_original_n = 84
    else:
        original = CN / "score_v2_clean_20261031/origin_method_losses.csv"
        target_file = CN / "score_inputs/score_target_extension_2025_2026.csv"
        source_panel = CN / "source_data/unified_cpi_panel.csv"
        targets = {r["reference_month"]: (
            float(Decimal(r["first_release_yoy_pct"]).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)),
            clock(r["available_at_conservative_local"], cn_local=True))
            for r in read(source_panel) if "2018-01" <= r["reference_month"] <= "2024-12"}
        targets.update({r["reference_month"]: (
            float(Decimal(r["first_release_yoy_pct"]).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)),
            clock(r["available_at_utc"])) for r in read(target_file)})
        loss_column = "mean_seed_absolute_error_pp"
        expected_original_n = 99
    assert hash_file(original) == sm["old_origin_losses_sha256"]
    assert hash_file(target_file) == sm["score_targets_sha256"]
    archived_rows = read(original)
    old = {}
    for r in archived_rows:
        k = (r["model"], key(r))
        assert k not in old
        old[k] = float(r[loss_column])
    all_old_models = set(r["model"] for r in archived_rows)
    assert len(all_old_models) == 10

    prediction_rows = read(forecast_dir / "forecasts.csv")
    loss_rows = read(score_dir / "new_origin_losses.csv")
    predicted = {(r["model"], key(r)): r for r in prediction_rows if r["status"] == "FORECAST_COMPLETE"}
    observed = {(r["model"], key(r)): float(r["absolute_error_pp"]) for r in loss_rows}
    assert len(predicted) == len(loss_rows) == len(observed)
    assert set(predicted) == set(observed)
    max_error = 0.0
    for mk, r in predicted.items():
        k = mk[1]
        target, release_clock = targets[k[3]]
        assert release_clock > k[2]
        error = abs(float(r["prediction"]) - target)
        max_error = max(max_error, abs(error - observed[mk]))
    assert max_error <= 1e-12

    sets = defaultdict(set)
    combined = {**old, **observed}
    for (model, k), value in combined.items():
        assert np.isfinite(value) and value >= 0
        sets[(model, k[-1])].add(k)
    summaries = {(r["model"], int(r["horizon_months"])): r
                 for r in read(score_dir / "mae_summary.csv")}
    methods = sorted({m for m, _ in combined})
    assert len(methods) == 12 and len(summaries) == 48
    per_h = {}
    max_summary_error = 0.0
    for h in HORIZONS:
        old_common = set.intersection(*(sets[(m, h)] for m in all_old_models))
        b3 = sets[("B3 multivariate", h)]
        common12 = set.intersection(*(sets[(m, h)] for m in methods))
        assert len(old_common) == len(b3) == len(common12) == expected_original_n
        assert old_common == b3 == common12
        per_h[str(h)] = {"common_keys": len(common12), "b3_key_set_exact_match": True,
                         "ets_native": len(sets[("ETS(A,A,N)", h)]),
                         "theta_native": len(sets[("Theta", h)])}
        for model in methods:
            row = summaries[(model, h)]
            for label, sample in (("native", sets[(model, h)]),
                                  ("old_common10", old_common & sets[(model, h)]),
                                  ("common12", common12)):
                assert int(row["n_" + label]) == len(sample)
                expected = sum(combined[(model, k)] for k in sample) / len(sample)
                max_summary_error = max(max_summary_error, abs(float(row["mae_" + label + "_pp"]) - expected))
    assert max_summary_error <= 1e-11

    pair_rows = read(score_dir / "new_vs_existing_pairwise.csv")
    assert len(pair_rows) == 80
    max_pair_error = 0.0
    for r in pair_rows:
        h, a, b = int(r["horizon_months"]), r["model"], r["reference"]
        pairs = sets[(a, h)] & sets[(b, h)]
        differences = [combined[(a, k)] - combined[(b, k)] for k in pairs]
        assert int(r["n_pairs"]) == len(differences)
        assert int(r["model_lower_count"]) == sum(v < -1e-12 for v in differences)
        assert int(r["reference_lower_count"]) == sum(v > 1e-12 for v in differences)
        assert int(r["tie_count"]) == sum(abs(v) <= 1e-12 for v in differences)
        max_pair_error = max(max_pair_error, abs(float(r["mean_loss_difference_pp"]) - np.mean(differences)))
    assert max_pair_error <= 1e-11

    report = {"status": "PASS", "country": country, "horizon_keys": per_h,
              "new_origin_losses_checked": len(loss_rows), "old_loss_rows_checked": len(archived_rows),
              "mae_summary_cells_rebuilt": len(summaries), "pairwise_cells_rebuilt": len(pair_rows),
              "max_origin_loss_reconstruction_error_pp": max_error,
              "max_mae_reconstruction_error_pp": max_summary_error,
              "max_pairwise_reconstruction_error_pp": max_pair_error,
              "historical_reconstruction_not_prospective": True,
              "verify_code_sha256": hash_file(Path(__file__)),
              "target_values_exported": False, "P1_accessed": False}
    with (out / "independent_verification.json").open("x", encoding="utf8") as f:
        json.dump(report, f, indent=2); f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    for country in ("US", "CN"):
        verify(country)
