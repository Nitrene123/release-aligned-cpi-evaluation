from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S11 = ROOT / "manuscript/ars_full_v2_4/reproducibility/pit_mask_counterfactual_2018_2026_v1"
OUT = HERE / "combined_scores"
AUDIT = HERE / "combined_audit"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    paths = {"base": ROOT / "manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a/score_v2_clean_20261031/origin_method_losses.csv",
             "paired": S11 / "derived_scores_v2/paired_origin_losses.csv", "timing_forecasts": S11 / "server_outputs_v1/forecast_predictions.csv",
             "vintage_forecasts": HERE / "vintage_forecasts/forecast_predictions.csv", "combined_forecasts": HERE / "combined_forecasts/forecast_predictions.csv",
             "combined_metadata": HERE / "combined_forecasts/inference_metadata.json", "combined_map": HERE / "combined_map/mask_recovery_map.csv",
             "vintage_scores": HERE / "vintage_scores/paired_losses.csv", "reference_metrics": HERE / "results/model_scenario_metrics.csv"}
    assert all(sha(paths[k]) == v for k, v in manifest["source_sha256"].items())
    assert all(sha(OUT / k) == v for k, v in manifest["outputs_sha256"].items())
    assert sha(HERE / "PROTOCOL_COMBINED_PARTIAL.md") == manifest["protocol_sha256"]
    assert sha(HERE / "score_combined_partial.py") == manifest["script_sha256"]
    meta = json.loads(paths["combined_metadata"].read_text(encoding="utf-8"))
    assert meta["input_file_sha256"]["mask_recovery_map.csv"] == sha(paths["combined_map"])
    assert meta["input_file_sha256"]["PROTOCOL_COMBINED_PARTIAL.md"] == sha(HERE / "PROTOCOL_COMBINED_PARTIAL.md")
    assert meta["weights_sha256"] == "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da"
    assert meta["official_source_commit"] == "e51928e27119cb17bebc005be2696b75e0a9e688"
    mapping = pd.read_csv(paths["combined_map"])
    inputs = pd.read_csv(S11 / "inputs/b3_input_snapshots.csv", low_memory=False)
    cells = mapping.merge(inputs, on=["origin_id", "context_reference_month", "feature_name"], validate="one_to_one")
    assert len(cells) == 118 and cells[cells.modification_type.eq("timing")].masked.all()
    assert not cells[cells.modification_type.eq("vintage")].masked.any()
    f = pd.read_csv(paths["combined_forecasts"])
    keys = ["origin_id", "origin_month", "origin_timestamp_utc", "horizon_months"]
    a = f[f.variant.eq("strict_replay")][keys + ["point_forecast_q50"]]
    b = f[f.variant.eq("partial_combined")][keys + ["point_forecast_q50"]]
    forecasts = a.merge(b, on=keys, validate="one_to_one", suffixes=("_strict", "_combined"))
    ref = pd.read_csv(S11 / "inputs/archived_b3_q50.csv")
    z = a.merge(ref, on=keys, validate="one_to_one")
    assert len(z) == 1884
    qmax = float((z.point_forecast_q50 - z.archived_q50).abs().max())
    assert qmax <= 1e-6
    unaffected = ~forecasts.origin_id.isin(set(mapping.origin_id))
    assert np.max(np.abs(forecasts.loc[unaffected, "point_forecast_q50_combined"] - forecasts.loc[unaffected, "point_forecast_q50_strict"])) < 1e-6
    forecasts["origin_timestamp_utc"] = pd.to_datetime(forecasts.origin_timestamp_utc, utc=True)
    base = pd.read_csv(paths["base"]); base = base[base.model.eq("B3 multivariate")].copy()
    base["origin_timestamp_utc"] = pd.to_datetime(base.origin_timestamp_utc, utc=True)
    paired = pd.read_csv(OUT / "paired_origin_losses.csv")
    k = ["origin_id", "origin_month", "target_month", "horizon_months"]
    rows = paired.merge(base, on=k, validate="one_to_one").merge(forecasts, on=keys, validate="one_to_one")
    assert len(rows) == 396
    loss = np.abs(rows.seed_mean_prediction_minus_target_pp + rows.point_forecast_q50_combined - rows.point_forecast_q50_strict)
    max_loss = float(np.max(np.abs(loss - rows.combined_loss_pp)))
    algebra = rows.combined_loss_pp - rows.timing_loss_pp - rows.vintage_loss_pp + rows.strict_loss_pp
    assert max_loss < 1e-10 and np.max(np.abs(algebra - rows.interaction)) < 1e-10
    summaries = pd.read_csv(OUT / "scenario_summary.csv"); intervals = pd.read_csv(OUT / "paired_block_intervals.csv")
    reference = pd.read_csv(paths["reference_metrics"]); reference = reference[reference.scenario.eq("strict")]
    rng = np.random.default_rng(417); max_metric = max_interval = 0.0
    contrasts = ["timing_minus_strict", "vintage_minus_strict", "combined_minus_strict", "combined_minus_timing", "combined_minus_vintage", "interaction"]
    for h in (1, 3, 6, 12):
        q = rows[rows.horizon_months.eq(h)].sort_values("target_month")
        assert len(q) == 99 and q.target_month.nunique() == 99
        other = reference[reference.horizon_months.eq(h) & ~reference.model.eq("B3 multivariate")]
        assert len(other) == 12
        for scenario in ("strict", "timing", "vintage", "combined"):
            values = q[f"{scenario}_loss_pp"].to_numpy(float)
            s = summaries[(summaries.horizon_months.eq(h)) & summaries.scenario.eq(scenario)].iloc[0]
            max_metric = max(max_metric, abs(s.mae_pp - values.mean()), abs(s.rmse_pp - math.sqrt(float(np.mean(values**2)))))
            assert int(s.rank_among_13) == 1 + int((other.mae_pp < s.mae_pp).sum())
        calendar = pd.period_range(q.target_month.min(), q.target_month.max(), freq="M")
        for name in contrasts:
            x = pd.Series(q[name].to_numpy(float), index=pd.PeriodIndex(q.target_month, freq="M")).reindex(calendar).to_numpy(float)
            for block in (3, 6, 12):
                starts = rng.integers(len(x), size=(30000, math.ceil(len(x) / block)))
                idx = (starts[:, :, None] + np.arange(block)) % len(x)
                draws = x[idx.reshape(30000, -1)[:, :len(x)]]
                low, high = np.quantile(np.nansum(draws, axis=1) / np.isfinite(draws).sum(axis=1), [.025, .975])
                row = intervals[intervals.horizon_months.eq(h) & intervals.contrast.eq(name) & intervals.block_months.eq(block)].iloc[0]
                max_interval = max(max_interval, abs(low - row.interval_low_pp), abs(high - row.interval_high_pp))
    assert max_metric < 1e-10 and max_interval < 1e-10 and len(intervals) == 72
    AUDIT.mkdir()
    report = {"status": "PASS", "cells_checked": len(cells), "strict_forecasts_checked": len(z), "scored_keys": len(rows),
              "intervals_recomputed": len(intervals), "max_strict_q50_difference_pp": qmax, "max_loss_difference_pp": max_loss,
              "max_metric_difference_pp": max_metric, "max_interval_difference_pp": max_interval, "verifier_sha256": sha(Path(__file__))}
    (AUDIT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
