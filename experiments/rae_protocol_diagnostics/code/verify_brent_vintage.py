from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "vintage_scores"
AUDIT = HERE / "vintage_audit"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    paths = {"forecasts": HERE / "vintage_forecasts/forecast_predictions.csv",
             "inference_metadata": HERE / "vintage_forecasts/inference_metadata.json",
             "replacement_map": HERE / "vintage_map/brent_vintage_replacement_map.csv",
             "baseline_losses": ROOT / "manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a/score_v2_clean_20261031/origin_method_losses.csv",
             "s11_paired_losses": ROOT / "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/derived_scores_v2/paired_origin_losses.csv",
             "reference_metrics": HERE / "results/model_scenario_metrics.csv"}
    assert all(sha(paths[k]) == v for k, v in manifest["source_sha256"].items())
    assert all(sha(OUT / k) == v for k, v in manifest["outputs_sha256"].items())
    assert sha(HERE / "SCORING_PARTIAL_VINTAGE.md") == manifest["scoring_protocol_sha256"]
    assert sha(HERE / "score_brent_vintage.py") == manifest["scoring_code_sha256"]
    imeta = json.loads(paths["inference_metadata"].read_text(encoding="utf-8"))
    mmap = json.loads((HERE / "vintage_map/map_metadata.json").read_text(encoding="utf-8"))
    assert imeta["input_file_sha256"]["mask_recovery_map.csv"] == sha(paths["replacement_map"]) == mmap["map_sha256"]
    assert imeta["input_file_sha256"]["PROTOCOL_PARTIAL_VINTAGE.md"] == sha(HERE / "PROTOCOL_PARTIAL_VINTAGE.md") == mmap["protocol_sha256"]
    assert imeta["weights_sha256"] == "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da"
    assert imeta["official_source_commit"] == "e51928e27119cb17bebc005be2696b75e0a9e688"
    pred = pd.read_csv(paths["forecasts"])
    ref = pd.read_csv(ROOT / "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1/inputs/archived_b3_q50.csv")
    key = ["origin_id", "origin_month", "origin_timestamp_utc", "horizon_months"]
    strict = pred[pred.variant.eq("strict_replay")].merge(ref, on=key, validate="one_to_one")
    assert len(strict) == 1884
    q50_max = float((strict.point_forecast_q50 - strict.archived_q50).abs().max())
    assert q50_max <= 1e-6
    strict = pred[pred.variant.eq("strict_replay")][key + ["point_forecast_q50"]]
    alt = pred[pred.variant.eq("latest_archived_brent")][key + ["point_forecast_q50"]]
    both = strict.merge(alt, on=key, validate="one_to_one", suffixes=("_strict", "_alt"))
    paired = pd.read_csv(OUT / "paired_losses.csv")
    baseline = pd.read_csv(paths["baseline_losses"])
    baseline = baseline[baseline.model.eq("B3 multivariate")]
    baseline["origin_timestamp_utc"] = pd.to_datetime(baseline.origin_timestamp_utc, utc=True)
    both["origin_timestamp_utc"] = pd.to_datetime(both.origin_timestamp_utc, utc=True)
    k = ["origin_id", "origin_month", "target_month", "horizon_months"]
    rows = paired.merge(baseline, on=k, validate="one_to_one").merge(both, on=["origin_id", "origin_month", "origin_timestamp_utc", "horizon_months"], validate="one_to_one")
    assert len(rows) == 396
    delta = rows.point_forecast_q50_alt - rows.point_forecast_q50_strict
    recomputed = np.abs(rows.seed_mean_prediction_minus_target_pp + delta)
    max_loss = float(np.max(np.abs(recomputed - rows.vintage_loss_pp)))
    assert max_loss < 1e-10
    summ = pd.read_csv(OUT / "summary.csv"); intervals = pd.read_csv(OUT / "paired_block_intervals.csv")
    rng = np.random.default_rng(417); max_summary = max_interval = 0.0
    for h in (1, 3, 6, 12):
        q = rows[rows.horizon_months.eq(h)].sort_values("target_month")
        s = summ[summ.horizon_months.eq(h)].iloc[0]
        vals = q.vintage_loss_pp.to_numpy(float)
        max_summary = max(max_summary, abs(float(s.vintage_mae_pp) - float(vals.mean())), abs(float(s.vintage_rmse_pp) - math.sqrt(float(np.mean(vals**2)))))
        assert int(s.changed_forecasts) == int((q.forecast_change_pp.abs() > 1e-12).sum())
        calendar = pd.period_range(q.target_month.min(), q.target_month.max(), freq="M")
        y = pd.Series(q.loss_change_pp.to_numpy(float), index=pd.PeriodIndex(q.target_month, freq="M")).reindex(calendar).to_numpy(float)
        for block in (3, 6, 12):
            starts = rng.integers(len(y), size=(30000, math.ceil(len(y) / block)))
            idx = (starts[:, :, None] + np.arange(block)) % len(y)
            draws = y[idx.reshape(30000, -1)[:, :len(y)]]
            means = np.nansum(draws, axis=1) / np.isfinite(draws).sum(axis=1)
            lo, hi = np.quantile(means, [.025, .975]); row = intervals[(intervals.horizon_months.eq(h)) & intervals.block_months.eq(block)].iloc[0]
            max_interval = max(max_interval, abs(lo - row.interval_low_pp), abs(hi - row.interval_high_pp))
    assert max_summary < 1e-10 and max_interval < 1e-10
    AUDIT.mkdir()
    result = {"status": "PASS", "strict_forecasts_checked": 1884, "scored_keys": 396, "intervals_recomputed": 12,
              "max_strict_q50_difference_pp": q50_max, "max_loss_difference_pp": max_loss,
              "max_summary_difference_pp": max_summary, "max_interval_difference_pp": max_interval,
              "input_and_output_hashes": "PASS", "verifier_sha256": sha(Path(__file__))}
    (AUDIT / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
